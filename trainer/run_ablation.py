from __future__ import annotations

if __package__ is None or __package__ == "":
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).resolve().parent.parent))

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


@dataclass
class CandidateSpec:
    exp_id: str
    strategy: str
    model: str = ""
    rl_model: str = ""
    risk_config: str = ""


def _build_eval_args(
    *,
    backend: str,
    stake: str,
    episodes: int,
    seeds_file: str,
    strategy: str,
    model: str | None,
    rl_model: str | None,
    risk_config: str | None,
    out_json: Path,
    logs_jsonl: Path,
    max_steps_per_episode: int = 120,
) -> list[str]:
    policy = "heuristic"
    args = [
        sys.executable,
        "-B",
        "trainer/eval_long_horizon.py",
        "--backend",
        backend,
        "--stake",
        stake,
        "--episodes",
        str(int(episodes)),
        "--seeds-file",
        seeds_file,
        "--policy",
    ]

    if strategy in {"pv", "champion"}:
        policy = "pv"
    elif strategy == "rl":
        policy = "pv"
    elif strategy == "hybrid":
        policy = "hybrid"
    elif strategy == "risk_aware":
        policy = "risk_aware"
    elif strategy == "deploy_student":
        policy = "deploy_student"
    elif strategy == "search":
        policy = "search"
    elif strategy == "bc":
        policy = "bc"
    else:
        policy = "heuristic"

    args.append(policy)
    if policy in {"bc", "pv", "hybrid", "deploy_student"} and model:
        args.extend(["--model", str(model)])
    if policy == "risk_aware":
        if model:
            args.extend(["--model", str(model)])
        if rl_model:
            args.extend(["--rl-model", str(rl_model)])
        elif model:
            args.extend(["--rl-model", str(model)])
        if risk_config:
            args.extend(["--risk-config", str(risk_config)])

    args.extend(
        [
            "--max-steps-per-episode",
            str(int(max_steps_per_episode)),
            "--out",
            str(out_json),
            "--save-episode-logs",
            str(logs_jsonl),
        ]
    )
    return args


def _extract_metrics(payload: dict[str, Any]) -> dict[str, float]:
    return {
        "win_rate": float(payload.get("win_rate") or 0.0),
        "avg_ante_reached": float(payload.get("avg_ante_reached") or 0.0),
        "median_ante_reached": float(payload.get("median_ante") or 0.0),
        "runtime_seconds": float(payload.get("elapsed_sec") or payload.get("runtime_seconds") or 0.0),
    }


def _infer_strategy(exp_id: str, strategy_hint: str = "") -> str:
    token = str(strategy_hint or "").strip().lower()
    if token:
        return token
    exp = str(exp_id or "").lower()
    if "deploy" in exp or "distill" in exp:
        return "deploy_student"
    if "risk" in exp:
        return "risk_aware"
    if "hybrid" in exp:
        return "hybrid"
    if "search" in exp:
        return "search"
    if "pv" in exp or "bc" in exp:
        return "pv"
    if "champion" in exp:
        return "champion"
    return "heuristic"


def _load_candidates_from_manifest(path: Path) -> list[CandidateSpec]:
    payload = _read_json(path)
    rows = payload.get("candidates") if isinstance(payload.get("candidates"), list) else []
    out: list[CandidateSpec] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        if str(row.get("status") or "").lower() not in {"success", "passed", "pass", "ok"}:
            continue
        exp_id = str(row.get("exp_id") or "")
        if not exp_id:
            continue
        out.append(
            CandidateSpec(
                exp_id=exp_id,
                strategy=_infer_strategy(exp_id, str(row.get("strategy") or "")),
                model=str(row.get("model_path") or ""),
                rl_model=str(row.get("rl_model_path") or ""),
                risk_config=str(row.get("risk_config") or ""),
            )
        )
    return out


def _load_candidates_from_ranking(path: Path, topk: int) -> list[CandidateSpec]:
    payload = _read_json(path)
    rows = payload.get("rows") if isinstance(payload.get("rows"), list) else []
    filtered = [r for r in rows if isinstance(r, dict) and bool(r.get("gate_pass", True))]
    if not filtered:
        # In degraded regimes all candidates may fail hard filters; still evaluate the top scored rows.
        filtered = [r for r in rows if isinstance(r, dict)]
    filtered = sorted(filtered, key=lambda r: float(r.get("weighted_score") or 0.0), reverse=True)
    if topk > 0:
        filtered = filtered[:topk]
    out: list[CandidateSpec] = []
    for row in filtered:
        exp_id = str(row.get("exp_id") or "")
        if not exp_id:
            continue
        out.append(
            CandidateSpec(
                exp_id=exp_id,
                strategy=_infer_strategy(exp_id, str(row.get("strategy") or "")),
                model=str(row.get("model") or row.get("model_path") or ""),
                rl_model=str(row.get("rl_model") or row.get("rl_model_path") or ""),
                risk_config=str(row.get("risk_config") or ""),
            )
        )
    return out


def _load_champion_candidate() -> CandidateSpec | None:
    candidates = [
        Path("docs/artifacts/p24/runs/latest/stages/validation_b/champion.json"),
        Path("docs/artifacts/p24/champion.json"),
    ]
    path = next((p for p in candidates if p.exists()), None)
    if path is None:
        return None
    payload = _read_json(path)
    exp_id = str(payload.get("exp_id") or "champion")
    model = str(payload.get("model_path") or payload.get("model") or "")
    # current champion artifacts often lack model checkpoint; fallback to heuristic policy.
    strategy = "champion" if model else "heuristic"
    return CandidateSpec(exp_id=f"champion::{exp_id}", strategy=strategy, model=model)


def _dedupe_candidates(rows: list[CandidateSpec]) -> list[CandidateSpec]:
    out: list[CandidateSpec] = []
    seen: set[str] = set()
    for row in rows:
        key = f"{row.exp_id}|{row.strategy}|{row.model}|{row.rl_model}|{row.risk_config}"
        if key in seen:
            continue
        seen.add(key)
        out.append(row)
    return out


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run multi-strategy long-horizon ablation.")
    p.add_argument("--backend", choices=["sim"], default="sim")
    p.add_argument("--stake", default="gold")
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--seeds-file", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--heuristic", action="store_true")
    p.add_argument("--pv-model", default="")
    p.add_argument("--hybrid-model", default="")
    p.add_argument("--rl-model", default="")
    p.add_argument("--risk-aware-config", default="")
    p.add_argument("--champion-model", default="")
    p.add_argument("--deploy-student-model", default="")
    p.add_argument("--strategies", default="")
    p.add_argument("--max-steps-per-episode", type=int, default=120)

    p.add_argument("--from-train-batch-manifest", default="")
    p.add_argument("--from-ranking", default="")
    p.add_argument("--topk", type=int, default=0)
    p.add_argument("--include-champion", action="store_true")
    return p


def main() -> int:
    args = _build_parser().parse_args()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    candidates: list[CandidateSpec] = []

    if args.from_train_batch_manifest:
        candidates.extend(_load_candidates_from_manifest(Path(args.from_train_batch_manifest).resolve()))

    if args.from_ranking:
        candidates.extend(_load_candidates_from_ranking(Path(args.from_ranking).resolve(), int(args.topk)))

    if args.include_champion:
        champion = _load_champion_candidate()
        if champion is not None:
            candidates.append(champion)

    if not candidates:
        strategies: list[str] = []
        if args.strategies:
            strategies = [x.strip() for x in str(args.strategies).split(",") if x.strip()]
        else:
            if args.heuristic:
                strategies.append("heuristic")
            if args.champion_model:
                strategies.append("champion")
            if args.pv_model:
                if "champion" not in strategies:
                    strategies.append("champion")
                strategies.append("pv")
            if args.hybrid_model:
                strategies.append("hybrid")
            if args.rl_model:
                strategies.append("rl")
            if args.risk_aware_config:
                strategies.append("risk_aware")
            if args.deploy_student_model:
                strategies.append("deploy_student")
        if not strategies:
            raise RuntimeError("no strategies selected")
        model_map = {
            "pv": args.pv_model,
            "hybrid": args.hybrid_model or args.pv_model,
            "rl": args.rl_model,
            "champion": args.champion_model or args.pv_model,
            "risk_aware": args.pv_model or args.champion_model,
            "deploy_student": args.deploy_student_model,
        }
        for s in strategies:
            candidates.append(
                CandidateSpec(
                    exp_id=s,
                    strategy=s,
                    model=str(model_map.get(s) or ""),
                    rl_model=str(args.rl_model if s == "risk_aware" else ""),
                    risk_config=str(args.risk_aware_config if s == "risk_aware" else ""),
                )
            )

    candidates = _dedupe_candidates(candidates)
    if not candidates:
        raise RuntimeError("candidate set is empty")

    rows: list[dict[str, Any]] = []
    for c in candidates:
        out_json = out_dir / f"eval_gold_{c.exp_id.replace(':', '_')}.json"
        logs = out_dir / f"episodes_{c.exp_id.replace(':', '_')}.jsonl"
        cmd = _build_eval_args(
            backend=args.backend,
            stake=str(args.stake),
            episodes=int(args.episodes),
            seeds_file=str(args.seeds_file),
            strategy=c.strategy,
            model=c.model or None,
            rl_model=c.rl_model or None,
            risk_config=c.risk_config or None,
            out_json=out_json,
            logs_jsonl=logs,
            max_steps_per_episode=int(args.max_steps_per_episode),
        )
        status = "passed"
        err = ""
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as exc:
            status = "failed"
            err = str(exc)

        payload = _read_json(out_json)
        metrics = _extract_metrics(payload)
        rows.append(
            {
                "exp_id": c.exp_id,
                "strategy": c.strategy,
                "status": status,
                "error": err,
                "win_rate": metrics["win_rate"],
                "avg_ante_reached": metrics["avg_ante_reached"],
                "median_ante_reached": metrics["median_ante_reached"],
                "runtime_seconds": metrics["runtime_seconds"],
                "model": c.model,
                "rl_model": c.rl_model,
                "risk_config": c.risk_config,
                "eval_json": str(out_json),
                "episode_logs": str(logs),
            }
        )

    passing = [r for r in rows if r["status"] == "passed"]
    baseline_row = next((r for r in rows if "champion::" in str(r.get("exp_id"))), None)
    if baseline_row is None:
        baseline_row = passing[0] if passing else rows[0]
    base_metrics = _extract_metrics(baseline_row)

    for r in rows:
        r["delta_vs_baseline_avg_ante"] = float(r["avg_ante_reached"]) - base_metrics["avg_ante_reached"]
        r["delta_vs_baseline_median_ante"] = float(r["median_ante_reached"]) - base_metrics["median_ante_reached"]
        r["delta_vs_baseline_win_rate"] = float(r["win_rate"]) - base_metrics["win_rate"]
        r["elapsed_sec"] = float(r.get("runtime_seconds") or 0.0)
        r["std"] = 0.0
        r["gate_pass"] = r["status"] == "passed"
        r["weighted_score"] = (
            0.45 * float(r["avg_ante_reached"])
            + 0.25 * float(r["median_ante_reached"])
            + 0.20 * float(r["win_rate"])
            - 0.10 * float(r["runtime_seconds"])
        )

    summary = {
        "schema": "p29_ablation_summary_v2",
        "generated_at": _now_iso(),
        "backend": args.backend,
        "stake": args.stake,
        "episodes": int(args.episodes),
        "seeds_file": str(args.seeds_file),
        "baseline": str(baseline_row.get("exp_id") or ""),
        "rows": rows,
        "source": {
            "from_train_batch_manifest": str(args.from_train_batch_manifest or ""),
            "from_ranking": str(args.from_ranking or ""),
            "include_champion": bool(args.include_champion),
            "topk": int(args.topk),
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (out_dir / "summary_table.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    csv_lines = [
        "exp_id,strategy,status,win_rate,avg_ante_reached,median_ante_reached,runtime_seconds,delta_avg,delta_median,delta_win"
    ]
    for r in rows:
        csv_lines.append(
            f"{r['exp_id']},{r['strategy']},{r['status']},{r['win_rate']:.6f},{r['avg_ante_reached']:.6f},{r['median_ante_reached']:.6f},{r['runtime_seconds']:.6f},{r['delta_vs_baseline_avg_ante']:.6f},{r['delta_vs_baseline_median_ante']:.6f},{r['delta_vs_baseline_win_rate']:.6f}"
        )
    tables = out_dir / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    (tables / "ablation.csv").write_text("\n".join(csv_lines) + "\n", encoding="utf-8")

    ranked = sorted(passing, key=lambda r: (r["win_rate"], r["avg_ante_reached"], r["median_ante_reached"]), reverse=True)
    best_row = ranked[0] if ranked else baseline_row
    recommended = str(best_row.get("exp_id") or "")
    rec_md = [
        "# Ablation Recommendation",
        "",
        f"- recommended_default: {recommended}",
        f"- baseline: {summary['baseline']}",
        f"- episodes: {args.episodes}",
        "",
        "## Rationale",
        f"Best passing row by win_rate/avg_ante/median_ante: {recommended}.",
    ]
    (out_dir / "recommendation.md").write_text("\n".join(rec_md) + "\n", encoding="utf-8")

    md = [
        "# P29 Ablation Summary",
        "",
        f"- baseline: {summary['baseline']}",
        f"- episodes: {args.episodes}",
        f"- stake: {args.stake}",
        "",
        "## Rows",
    ]
    for r in rows:
        md.append(
            f"- {r['exp_id']} ({r['strategy']}): status={r['status']} win_rate={r['win_rate']:.4f} avg_ante={r['avg_ante_reached']:.3f} median_ante={r['median_ante_reached']:.3f}"
        )
    (out_dir / "summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    failed_count = len([r for r in rows if r["status"] != "passed"])
    print(
        json.dumps(
            {
                "status": "ok",
                "out_dir": str(out_dir),
                "baseline": summary["baseline"],
                "failed_count": failed_count,
                "row_count": len(rows),
            },
            ensure_ascii=False,
        )
    )
    return 0 if passing else 1


if __name__ == "__main__":
    raise SystemExit(main())
