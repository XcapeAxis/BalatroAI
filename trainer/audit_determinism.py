"""Determinism / Replay Audit: run eval twice with fixed seeds and compare action traces."""
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
    policy: str,
    model: str | None,
    rl_model: str | None,
    risk_config: str | None,
    out_json: Path,
    logs_jsonl: Path,
    max_steps: int,
) -> int:
    cmd = [
        sys.executable, "-B", "trainer/eval_long_horizon.py",
        "--backend", backend,
        "--stake", stake,
        "--episodes", str(episodes),
        "--seeds-file", seeds_file,
        "--policy", policy,
        "--max-steps-per-episode", str(max_steps),
        "--out", str(out_json),
        "--save-episode-logs", str(logs_jsonl),
    ]
    if model:
        cmd += ["--model", model]
    if rl_model and policy == "risk_aware":
        cmd += ["--rl-model", rl_model]
    if risk_config and policy == "risk_aware":
        cmd += ["--risk-config", risk_config]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    return proc.returncode


def _load_episode_logs(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except Exception:
            continue
    return rows


def _extract_action_trace(episodes: list[dict[str, Any]]) -> list[list[int | str]]:
    """Extract per-episode action sequences for determinism comparison."""
    def _normalize_actions(value: Any) -> list[Any]:
        if value is None:
            return []
        if isinstance(value, (list, tuple)):
            return list(value)
        # Some pipelines persist only top1 action id per step/episode.
        return [value]

    traces: list[list[int | str]] = []
    for ep in episodes:
        actions = _normalize_actions(ep.get("actions"))
        if not actions:
            actions = _normalize_actions(ep.get("action_trace"))
        if not actions:
            steps = ep.get("steps") or []
            actions = [s.get("action") for s in steps if s.get("action") is not None]
        traces.append([str(a) for a in actions])
    return traces


def _compare_traces(t1: list[list[str]], t2: list[list[str]], float_tol: float) -> dict[str, Any]:
    n = min(len(t1), len(t2))
    matches = 0
    diffs: list[dict[str, Any]] = []
    for i in range(n):
        if t1[i] == t2[i]:
            matches += 1
        else:
            diff_detail = {
                "episode": i,
                "len_run1": len(t1[i]),
                "len_run2": len(t2[i]),
                "first_divergence": -1,
            }
            for j in range(min(len(t1[i]), len(t2[i]))):
                if t1[i][j] != t2[i][j]:
                    diff_detail["first_divergence"] = j
                    diff_detail["run1_action"] = t1[i][j]
                    diff_detail["run2_action"] = t2[i][j]
                    break
            if diff_detail["first_divergence"] == -1 and len(t1[i]) != len(t2[i]):
                diff_detail["first_divergence"] = min(len(t1[i]), len(t2[i]))
            diffs.append(diff_detail)

    return {
        "total_episodes": n,
        "matching_episodes": matches,
        "differing_episodes": len(diffs),
        "trace_match_rate": round(matches / max(n, 1), 4),
        "diffs": diffs[:20],
    }


def _compare_summaries(s1: dict, s2: dict, tol: float) -> dict[str, Any]:
    keys = ["win_rate", "avg_ante_reached", "median_ante"]
    diffs: dict[str, Any] = {}
    all_match = True
    for k in keys:
        v1 = float(s1.get(k) or 0.0)
        v2 = float(s2.get(k) or 0.0)
        delta = abs(v1 - v2)
        match = delta <= tol
        if not match:
            all_match = False
        diffs[k] = {"run1": v1, "run2": v2, "delta": round(delta, 8), "match": match}
    return {"all_match": all_match, "fields": diffs}


def main() -> int:
    p = argparse.ArgumentParser(description="Determinism / replay audit for fixed seeds + model.")
    p.add_argument("--backend", default="sim")
    p.add_argument("--stake", default="gold")
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--seeds-file", required=True)
    p.add_argument("--policy", default="risk_aware")
    p.add_argument("--pv-model", default="")
    p.add_argument("--rl-model", default="")
    p.add_argument("--risk-config", default="")
    p.add_argument("--max-steps", type=int, default=120)
    p.add_argument("--float-tol", type=float, default=1e-6)
    p.add_argument("--runs", type=int, default=2)
    p.add_argument("--out-dir", required=True)
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    diffs_dir = out_dir / "nondeterminism_diffs"

    model = args.pv_model or None
    run_results: list[dict[str, Any]] = []
    run_traces: list[list[list[str]]] = []

    for run_idx in range(args.runs):
        out_json = out_dir / f"eval_run{run_idx}.json"
        logs_jsonl = out_dir / f"episodes_run{run_idx}.jsonl"
        rc = _run_eval(
            backend=args.backend,
            stake=args.stake,
            episodes=args.episodes,
            seeds_file=args.seeds_file,
            policy=args.policy,
            model=model,
            rl_model=args.rl_model or None,
            risk_config=args.risk_config or None,
            out_json=out_json,
            logs_jsonl=logs_jsonl,
            max_steps=args.max_steps,
        )
        if out_json.exists():
            run_results.append(json.loads(out_json.read_text(encoding="utf-8")))
        else:
            run_results.append({"error": f"run {run_idx} failed rc={rc}"})
        episodes = _load_episode_logs(logs_jsonl)
        run_traces.append(_extract_action_trace(episodes))

    # Compare runs pairwise
    trace_comparisons: list[dict[str, Any]] = []
    summary_comparisons: list[dict[str, Any]] = []
    for i in range(len(run_traces)):
        for j in range(i + 1, len(run_traces)):
            tc = _compare_traces(run_traces[i], run_traces[j], args.float_tol)
            tc["run_pair"] = [i, j]
            trace_comparisons.append(tc)
            if i < len(run_results) and j < len(run_results):
                sc = _compare_summaries(run_results[i], run_results[j], args.float_tol)
                sc["run_pair"] = [i, j]
                summary_comparisons.append(sc)

    all_traces_match = all(tc["differing_episodes"] == 0 for tc in trace_comparisons)
    all_summaries_match = all(sc["all_match"] for sc in summary_comparisons)
    audit_pass = all_traces_match and all_summaries_match

    potential_sources: list[str] = []
    if not all_traces_match:
        potential_sources.extend([
            "concurrent workers / thread scheduling",
            "unfixed random seed in sim backend",
            "dict iteration order (Python <3.7 or external lib)",
            "model eval vs train mode",
            "floating point nondeterminism (GPU)",
            "service-side state changes between runs",
        ])
        diffs_dir.mkdir(parents=True, exist_ok=True)
        for tc in trace_comparisons:
            pair = tc["run_pair"]
            (diffs_dir / f"trace_diff_run{pair[0]}_vs_run{pair[1]}.json").write_text(
                json.dumps(tc, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
            )

    report = {
        "schema": "determinism_audit_v1",
        "generated_at": _now_iso(),
        "audit_pass": audit_pass,
        "runs": args.runs,
        "episodes": args.episodes,
        "seeds_file": args.seeds_file,
        "policy": args.policy,
        "float_tol": args.float_tol,
        "trace_comparisons": trace_comparisons,
        "summary_comparisons": summary_comparisons,
        "all_traces_match": all_traces_match,
        "all_summaries_match": all_summaries_match,
        "potential_nondeterminism_sources": potential_sources,
    }

    (out_dir / "determinism_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    md_lines = [
        "# Determinism Audit Report",
        "",
        f"- audit_pass: {audit_pass}",
        f"- runs: {args.runs}",
        f"- episodes: {args.episodes}",
        f"- policy: {args.policy}",
        f"- all_traces_match: {all_traces_match}",
        f"- all_summaries_match: {all_summaries_match}",
    ]
    if potential_sources:
        md_lines.append("")
        md_lines.append("## Potential Nondeterminism Sources")
        for src in potential_sources:
            md_lines.append(f"- {src}")
    for tc in trace_comparisons:
        md_lines.append("")
        md_lines.append(f"## Trace Comparison: run {tc['run_pair'][0]} vs {tc['run_pair'][1]}")
        md_lines.append(f"- matching: {tc['matching_episodes']}/{tc['total_episodes']}")
        md_lines.append(f"- match_rate: {tc['trace_match_rate']}")
    (out_dir / "determinism_report.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    print(json.dumps({"audit_pass": audit_pass, "out_dir": str(out_dir)}, ensure_ascii=False))
    return 0 if audit_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
