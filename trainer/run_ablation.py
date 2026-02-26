from __future__ import annotations

if __package__ is None or __package__ == "":
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).resolve().parent.parent))

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _run_eval(
    *,
    backend: str,
    stake: str,
    episodes: int,
    seeds_file: str,
    strategy: str,
    model: str | None,
    out_json: Path,
    logs_jsonl: Path,
    max_steps_per_episode: int = 120,
) -> None:
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
    if strategy in {"pv", "rl", "champion"}:
        policy = "pv"
    elif strategy == "hybrid":
        policy = "hybrid"
    elif strategy == "search":
        policy = "search"
    else:
        policy = "heuristic"
    args.append(policy)
    if policy in {"pv", "hybrid"} and model:
        args.extend(["--model", model])
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
    subprocess.run(args, check=True)


def _extract_metrics(payload: dict[str, Any]) -> dict[str, float]:
    return {
        "win_rate": float(payload.get("win_rate") or 0.0),
        "avg_ante_reached": float(payload.get("avg_ante_reached") or 0.0),
        "median_ante_reached": float(payload.get("median_ante") or 0.0),
    }


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run multi-strategy long-horizon ablation for P18.")
    p.add_argument("--backend", choices=["sim"], default="sim")
    p.add_argument("--stake", default="gold")
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--seeds-file", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--heuristic", action="store_true")
    p.add_argument("--pv-model", default="")
    p.add_argument("--hybrid-model", default="")
    p.add_argument("--rl-model", default="")
    p.add_argument("--champion-model", default="")
    p.add_argument("--strategies", default="")
    p.add_argument("--max-steps-per-episode", type=int, default=120)
    return p


def main() -> int:
    args = _build_parser().parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

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
    if not strategies:
        raise RuntimeError("no strategies selected")

    model_map = {
        "pv": args.pv_model,
        "hybrid": args.hybrid_model or args.pv_model,
        "rl": args.rl_model,
        "champion": args.champion_model or args.pv_model,
    }

    results: dict[str, dict[str, Any]] = {}
    for s in strategies:
        out_json = out_dir / f"eval_gold_{s}.json"
        logs = out_dir / f"episodes_{s}.jsonl"
        _run_eval(
            backend=args.backend,
            stake=str(args.stake),
            episodes=int(args.episodes),
            seeds_file=str(args.seeds_file),
            strategy=s,
            model=model_map.get(s),
            out_json=out_json,
            logs_jsonl=logs,
            max_steps_per_episode=int(args.max_steps_per_episode),
        )
        results[s] = json.loads(out_json.read_text(encoding="utf-8"))

    summary_rows: list[dict[str, Any]] = []
    base_key = "champion" if "champion" in results else ("pv" if "pv" in results else strategies[0])
    base_metrics = _extract_metrics(results[base_key])
    for s in strategies:
        m = _extract_metrics(results[s])
        summary_rows.append(
            {
                "strategy": s,
                "win_rate": m["win_rate"],
                "avg_ante_reached": m["avg_ante_reached"],
                "median_ante_reached": m["median_ante_reached"],
                "delta_vs_baseline_avg_ante": m["avg_ante_reached"] - base_metrics["avg_ante_reached"],
                "delta_vs_baseline_median_ante": m["median_ante_reached"] - base_metrics["median_ante_reached"],
                "delta_vs_baseline_win_rate": m["win_rate"] - base_metrics["win_rate"],
                "failure_breakdown": results[s].get("failure_breakdown") or {},
                "eval_json": str(out_dir / f"eval_gold_{s}.json"),
                "episode_logs": str(out_dir / f"episodes_{s}.jsonl"),
            }
        )

    summary = {
        "schema": "p18_ablation_summary_v1",
        "generated_at": _now_iso(),
        "backend": args.backend,
        "stake": args.stake,
        "episodes": int(args.episodes),
        "seeds_file": str(args.seeds_file),
        "baseline": base_key,
        "rows": summary_rows,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    csv_lines = ["strategy,win_rate,avg_ante_reached,median_ante_reached,delta_avg,delta_median,delta_win"]
    for r in summary_rows:
        csv_lines.append(
            f"{r['strategy']},{r['win_rate']:.6f},{r['avg_ante_reached']:.6f},{r['median_ante_reached']:.6f},{r['delta_vs_baseline_avg_ante']:.6f},{r['delta_vs_baseline_median_ante']:.6f},{r['delta_vs_baseline_win_rate']:.6f}"
        )
    tables = out_dir / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    (tables / "ablation.csv").write_text("\n".join(csv_lines) + "\n", encoding="utf-8")

    md = [
        "# P18 Ablation Summary",
        "",
        f"- baseline: {base_key}",
        f"- episodes: {args.episodes}",
        f"- stake: {args.stake}",
        "",
        "## Strategies",
    ]
    for r in summary_rows:
        md.append(
            "- {strategy}: win_rate={win_rate:.4f}, avg_ante={avg_ante_reached:.3f}, "
            "median_ante={median_ante_reached:.3f}, d_avg={delta_vs_baseline_avg_ante:.3f}, "
            "d_median={delta_vs_baseline_median_ante:.3f}".format(**r)
        )
    (out_dir / "summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    print(json.dumps({"status": "ok", "out_dir": str(out_dir), "baseline": base_key}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
