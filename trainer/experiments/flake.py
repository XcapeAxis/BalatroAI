from __future__ import annotations

import argparse
import hashlib
import json
import statistics
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


def write_flake_outputs(report: dict[str, Any], out_dir: Path) -> dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "flake_report.json"
    md_path = out_dir / "flake_report.md"
    diffs_dir = out_dir / "flake_diffs"

    write_json(json_path, report)

    lines = [
        "# P23 Flake Report",
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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="P23 flake harness")
    p.add_argument("--run-roots", required=True, help="Comma-separated run roots")
    p.add_argument("--exp-id", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--avg-ante-std-threshold", type=float, default=0.2)
    p.add_argument("--median-ante-span-threshold", type=float, default=1.0)
    p.add_argument("--deterministic-profile", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    run_roots = [Path(part.strip()).resolve() for part in args.run_roots.split(",") if part.strip()]
    if not run_roots:
        raise SystemExit("run_roots is empty")
    report = build_flake_report(
        run_roots=run_roots,
        exp_id=args.exp_id,
        avg_ante_std_threshold=float(args.avg_ante_std_threshold),
        median_span_threshold=float(args.median_ante_span_threshold),
        deterministic_profile=bool(args.deterministic_profile),
    )
    paths = write_flake_outputs(report, Path(args.out_dir).resolve())
    print(json.dumps({"status": report.get("status"), "paths": paths}, ensure_ascii=False))
    return 0 if str(report.get("status")) == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
