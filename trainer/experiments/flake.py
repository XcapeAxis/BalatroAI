from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _safe_float(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(value)
    except Exception:
        return None


def _trace_hash(path: Path) -> str:
    if not path.exists():
        return ""
    data = path.read_bytes()
    return hashlib.sha256(data).hexdigest()


def _extract_metric(summary: dict[str, Any], key: str) -> float | None:
    seed_metrics = summary.get("seed_metrics") if isinstance(summary.get("seed_metrics"), dict) else {}
    if key in seed_metrics:
        return _safe_float(seed_metrics.get(key))
    if key == "avg_ante_reached":
        return _safe_float(seed_metrics.get("mean"))
    if key == "median_ante":
        return _safe_float(seed_metrics.get("median_ante"))
    return None


def _calc_std(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    return float(statistics.pstdev(values))


def _calc_span(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(max(values) - min(values))


def build_flake_report(
    *,
    run_roots: list[Path],
    exp_id: str,
    avg_ante_std_threshold: float,
    median_span_threshold: float,
    deterministic_profile: bool,
) -> dict[str, Any]:
    repeats: list[dict[str, Any]] = []
    statuses: list[str] = []
    avg_antes: list[float] = []
    median_antes: list[float] = []
    win_rates: list[float] = []
    trace_hashes: list[str] = []

    for run_root in run_roots:
        exp_dir = run_root / exp_id
        summary = read_json(exp_dir / "exp_summary.json")
        status = str(summary.get("status") or "unknown")
        statuses.append(status)

        avg_ante = _extract_metric(summary, "avg_ante_reached")
        median_ante = _extract_metric(summary, "median_ante")
        win_rate = _extract_metric(summary, "win_rate")
        if avg_ante is not None:
            avg_antes.append(avg_ante)
        if median_ante is not None:
            median_antes.append(median_ante)
        if win_rate is not None:
            win_rates.append(win_rate)

        progress_path = exp_dir / "progress.jsonl"
        trace_h = _trace_hash(progress_path) if deterministic_profile else ""
        if trace_h:
            trace_hashes.append(trace_h)
        repeats.append(
            {
                "run_root": str(run_root),
                "exp_dir": str(exp_dir),
                "status": status,
                "avg_ante_reached": avg_ante,
                "median_ante": median_ante,
                "win_rate": win_rate,
                "trace_hash": trace_h if deterministic_profile else "",
            }
        )

    status_consistent = len(set(statuses)) <= 1 if statuses else False
    avg_std = _calc_std(avg_antes)
    median_span = _calc_span(median_antes)
    win_std = _calc_std(win_rates)
    trace_mismatch = 0
    if deterministic_profile and trace_hashes:
        trace_mismatch = len(set(trace_hashes)) - 1 if len(set(trace_hashes)) > 1 else 0

    checks = {
        "status_consistent": status_consistent,
        "avg_ante_std_ok": avg_std <= avg_ante_std_threshold,
        "median_ante_span_ok": median_span <= median_span_threshold,
        "deterministic_trace_ok": (trace_mismatch == 0) if deterministic_profile else True,
    }
    passed = all(bool(v) for v in checks.values())

    return {
        "schema": "p23_flake_report_v1",
        "generated_at": now_iso(),
        "exp_id": exp_id,
        "repeat_count": len(run_roots),
        "thresholds": {
            "avg_ante_std_max": avg_ante_std_threshold,
            "median_ante_span_max": median_span_threshold,
            "deterministic_trace_mismatch_max": 0,
        },
        "stats": {
            "statuses": statuses,
            "avg_ante_values": avg_antes,
            "median_ante_values": median_antes,
            "win_rate_values": win_rates,
            "avg_ante_std": avg_std,
            "median_ante_span": median_span,
            "win_rate_std": win_std,
            "trace_mismatch": trace_mismatch,
        },
        "checks": checks,
        "status": "PASS" if passed else "FAIL",
        "repeats": repeats,
    }


def write_flake_outputs(report: dict[str, Any], out_dir: Path, *, file_prefix: str = "flake_report") -> dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"{file_prefix}.json"
    md_path = out_dir / f"{file_prefix}.md"
    diffs_dir = out_dir / "flake_diffs"

    write_json(json_path, report)

    lines = [
        "# Flake Report",
        "",
        f"- status: `{report.get('status')}`",
        f"- exp_id: `{report.get('exp_id')}`",
        f"- repeats: `{report.get('repeat_count')}`",
        "",
        "## Checks",
    ]
    checks = report.get("checks") or {}
    for key, value in checks.items():
        lines.append(f"- {key}: `{value}`")
    stats = report.get("stats") or {}
    lines += [
        "",
        "## Stats",
        f"- avg_ante_std: `{stats.get('avg_ante_std')}`",
        f"- median_ante_span: `{stats.get('median_ante_span')}`",
        f"- win_rate_std: `{stats.get('win_rate_std')}`",
        f"- trace_mismatch: `{stats.get('trace_mismatch')}`",
        "",
    ]

    if str(report.get("status")) == "FAIL":
        diffs_dir.mkdir(parents=True, exist_ok=True)
        diff_path = diffs_dir / "summary.txt"
        diff_path.write_text(
            "Flake checks failed.\n"
            + json.dumps({"checks": checks, "stats": stats}, ensure_ascii=False, indent=2)
            + "\n",
            encoding="utf-8",
        )
        lines.append(f"- diagnostics: `{diff_path}`")

    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"json": str(json_path), "md": str(md_path), "diffs_dir": str(diffs_dir)}


def _infer_policy(candidate: dict[str, Any]) -> tuple[str, str, str, str]:
    exp_id = str(candidate.get("exp_id") or candidate.get("key") or "candidate")
    strategy = str(candidate.get("strategy") or "")
    model = str(candidate.get("model") or candidate.get("model_path") or "")
    rl_model = str(candidate.get("rl_model") or candidate.get("rl_model_path") or "")
    risk_config = str(candidate.get("risk_config") or "")

    if not strategy:
        low = exp_id.lower()
        if "distill" in low or "deploy" in low:
            strategy = "deploy_student"
        elif "risk" in low:
            strategy = "risk_aware"
        elif "hybrid" in low:
            strategy = "hybrid"
        elif "pv" in low or "bc" in low:
            strategy = "pv"
        else:
            strategy = "heuristic"

    # Guard unsupported risk_aware without model.
    if strategy == "risk_aware" and not (model or rl_model):
        strategy = "heuristic"

    return exp_id, strategy, model, rl_model or model


def _run_candidate_flake(
    *,
    candidate_from: Path,
    seeds_file: Path,
    repeats: int,
    out_dir: Path,
    avg_ante_std_threshold: float,
    median_span_threshold: float,
    stake: str,
) -> dict[str, Any]:
    ranking = read_json(candidate_from)
    recs = ranking.get("recommendations") if isinstance(ranking.get("recommendations"), dict) else {}
    candidate = recs.get("recommended_default_candidate") if isinstance(recs.get("recommended_default_candidate"), dict) else None
    if candidate is None:
        candidate = recs.get("top_candidate") if isinstance(recs.get("top_candidate"), dict) else None
    if candidate is None:
        raise SystemExit("candidate-from missing recommended_default_candidate/top_candidate")

    exp_id, strategy, model, rl_model = _infer_policy(candidate)
    risk_config = str(candidate.get("risk_config") or "trainer/config/p19_risk_controller.yaml")

    seed_count = len([line for line in seeds_file.read_text(encoding="utf-8").splitlines() if line.strip()])
    episodes = seed_count if seed_count > 0 else 100

    run_rows: list[dict[str, Any]] = []
    avg_vals: list[float] = []
    median_vals: list[float] = []
    win_vals: list[float] = []
    statuses: list[str] = []

    for idx in range(int(repeats)):
        rep_dir = out_dir / f"repeat_{idx+1}"
        rep_dir.mkdir(parents=True, exist_ok=True)
        out_json = rep_dir / "eval.json"
        out_logs = rep_dir / "episodes.jsonl"

        cmd = [
            sys.executable,
            "-B",
            "trainer/eval_long_horizon.py",
            "--backend",
            "sim",
            "--stake",
            stake,
            "--episodes",
            str(episodes),
            "--seeds-file",
            str(seeds_file),
            "--policy",
            strategy,
        ]
        if strategy in {"pv", "bc", "hybrid", "deploy_student"} and model:
            cmd.extend(["--model", model])
        if strategy == "risk_aware":
            if model:
                cmd.extend(["--model", model])
            if rl_model:
                cmd.extend(["--rl-model", rl_model])
            cmd.extend(["--risk-config", risk_config])
        cmd.extend(["--out", str(out_json), "--save-episode-logs", str(out_logs), "--max-steps-per-episode", "120"])

        proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", check=False)
        payload = read_json(out_json)
        avg = float(payload.get("avg_ante_reached") or 0.0)
        med = float(payload.get("median_ante") or 0.0)
        win = float(payload.get("win_rate") or 0.0)
        status = "passed" if proc.returncode == 0 else "failed"

        statuses.append(status)
        avg_vals.append(avg)
        median_vals.append(med)
        win_vals.append(win)

        run_rows.append(
            {
                "repeat": idx + 1,
                "status": status,
                "returncode": int(proc.returncode),
                "avg_ante_reached": avg,
                "median_ante": med,
                "win_rate": win,
                "eval_json": str(out_json),
                "episode_logs": str(out_logs),
                "stderr_tail": "\n".join((proc.stderr or "").splitlines()[-12:]),
            }
        )

    avg_std = _calc_std(avg_vals)
    med_span = _calc_span(median_vals)
    win_std = _calc_std(win_vals)

    checks = {
        "all_runs_success": all(s == "passed" for s in statuses),
        "avg_ante_std_ok": avg_std <= avg_ante_std_threshold,
        "median_ante_span_ok": med_span <= median_span_threshold,
    }
    passed = all(checks.values())

    return {
        "schema": "p29_best_candidate_flake_report_v1",
        "generated_at": now_iso(),
        "exp_id": exp_id,
        "strategy": strategy,
        "repeat_count": int(repeats),
        "thresholds": {
            "avg_ante_std_max": avg_ante_std_threshold,
            "median_ante_span_max": median_span_threshold,
        },
        "stats": {
            "statuses": statuses,
            "avg_ante_values": avg_vals,
            "median_ante_values": median_vals,
            "win_rate_values": win_vals,
            "avg_ante_std": avg_std,
            "median_ante_span": med_span,
            "win_rate_std": win_std,
        },
        "checks": checks,
        "status": "PASS" if passed else "FAIL",
        "repeats": run_rows,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Flake harness (legacy + P29 candidate mode)")
    p.add_argument("--mode", choices=["legacy", "candidate"], default="legacy")

    # legacy mode
    p.add_argument("--run-roots", default="", help="Comma-separated run roots")
    p.add_argument("--exp-id", default="")
    p.add_argument("--deterministic-profile", action="store_true")

    # candidate mode
    p.add_argument("--candidate-from", default="", help="ranking_summary.json path")
    p.add_argument("--seeds-file", default="")
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--stake", default="gold")

    p.add_argument("--out-dir", required=True)
    p.add_argument("--avg-ante-std-threshold", type=float, default=0.2)
    p.add_argument("--median-ante-span-threshold", type=float, default=1.0)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir).resolve()

    if args.mode == "candidate":
        if not args.candidate_from:
            raise SystemExit("--candidate-from is required for mode=candidate")
        if not args.seeds_file:
            raise SystemExit("--seeds-file is required for mode=candidate")
        report = _run_candidate_flake(
            candidate_from=Path(args.candidate_from).resolve(),
            seeds_file=Path(args.seeds_file).resolve(),
            repeats=int(args.repeats),
            out_dir=out_dir,
            avg_ante_std_threshold=float(args.avg_ante_std_threshold),
            median_span_threshold=float(args.median_ante_span_threshold),
            stake=str(args.stake),
        )
        paths = write_flake_outputs(report, out_dir, file_prefix="best_candidate_flake_report")
        print(json.dumps({"status": report.get("status"), "paths": paths}, ensure_ascii=False))
        return 0 if str(report.get("status")) == "PASS" else 1

    if not args.run_roots or not args.exp_id:
        raise SystemExit("legacy mode requires --run-roots and --exp-id")
    run_roots = [Path(part.strip()).resolve() for part in str(args.run_roots).split(",") if part.strip()]
    if not run_roots:
        raise SystemExit("run_roots is empty")

    report = build_flake_report(
        run_roots=run_roots,
        exp_id=str(args.exp_id),
        avg_ante_std_threshold=float(args.avg_ante_std_threshold),
        median_span_threshold=float(args.median_ante_span_threshold),
        deterministic_profile=bool(args.deterministic_profile),
    )
    paths = write_flake_outputs(report, out_dir, file_prefix="flake_report")
    print(json.dumps({"status": report.get("status"), "paths": paths}, ensure_ascii=False))
    return 0 if str(report.get("status")) == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
