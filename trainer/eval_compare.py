from __future__ import annotations

import argparse
import json
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _parse_named_path(expr: str) -> tuple[str, Path]:
    if "=" not in expr:
        raise ValueError(f"invalid named path: {expr}")
    k, v = expr.split("=", 1)
    return k.strip(), Path(v.strip())


def _metric(payload: dict[str, Any], key: str) -> float:
    return float(payload.get(key) or 0.0)


def _bootstrap_ci_for_delta(
    a: list[float],
    b: list[float],
    *,
    rounds: int = 1000,
    seed: int = 7,
) -> dict[str, float]:
    if not a or not b:
        return {"low": 0.0, "high": 0.0}
    n = min(len(a), len(b))
    ra = a[:n]
    rb = b[:n]
    rng = random.Random(seed)
    vals: list[float] = []
    for _ in range(rounds):
        idx = [rng.randrange(0, n) for _ in range(n)]
        ma = sum(ra[i] for i in idx) / n
        mb = sum(rb[i] for i in idx) / n
        vals.append(mb - ma)
    vals.sort()
    lo_idx = int(0.025 * (len(vals) - 1))
    hi_idx = int(0.975 * (len(vals) - 1))
    return {"low": float(vals[lo_idx]), "high": float(vals[hi_idx])}


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Compare baseline vs candidates for P17 perf gate.")
    p.add_argument("--baseline", required=True, help="name=path_to_eval_json")
    p.add_argument("--candidate", action="append", required=True, help="name=path_to_eval_json (repeatable)")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--median-threshold", type=float, default=0.5)
    p.add_argument("--avg-threshold", type=float, default=0.3)
    p.add_argument("--bootstrap-rounds", type=int, default=1000)
    p.add_argument("--bootstrap-seed", type=int, default=7)
    return p


def main() -> int:
    args = _build_parser().parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    base_name, base_path = _parse_named_path(args.baseline)
    base = _read_json(base_path)
    if not base:
        print(f"missing baseline: {base_path}")
        return 2

    base_metrics = {
        "win_rate": _metric(base, "win_rate"),
        "avg_ante_reached": _metric(base, "avg_ante_reached"),
        "median_ante": _metric(base, "median_ante"),
    }

    rows: list[dict[str, Any]] = []
    best_name = None
    best_score = None
    for cexpr in args.candidate:
        cname, cpath = _parse_named_path(cexpr)
        cp = _read_json(cpath)
        if not cp:
            continue
        cm = {
            "win_rate": _metric(cp, "win_rate"),
            "avg_ante_reached": _metric(cp, "avg_ante_reached"),
            "median_ante": _metric(cp, "median_ante"),
        }
        delta = {
            "win_rate": cm["win_rate"] - base_metrics["win_rate"],
            "avg_ante_reached": cm["avg_ante_reached"] - base_metrics["avg_ante_reached"],
            "median_ante": cm["median_ante"] - base_metrics["median_ante"],
        }

        # CI from episode logs if present.
        ci = None
        b_log = base.get("episode_log_values", {}).get("ante_values")
        c_log = cp.get("episode_log_values", {}).get("ante_values")
        if isinstance(b_log, list) and isinstance(c_log, list) and b_log and c_log:
            ci = _bootstrap_ci_for_delta(
                [float(x) for x in b_log],
                [float(x) for x in c_log],
                rounds=int(args.bootstrap_rounds),
                seed=int(args.bootstrap_seed),
            )

        perf_pass = bool(
            (delta["median_ante"] >= float(args.median_threshold))
            or (delta["avg_ante_reached"] >= float(args.avg_threshold))
            or (ci is not None and float(ci.get("low", 0.0)) > 0.0)
        )
        row = {
            "baseline": base_name,
            "candidate": cname,
            "baseline_metrics": base_metrics,
            "candidate_metrics": cm,
            "delta": {
                **delta,
                "avg_ante_ci95": ci,
            },
            "perf_gate_pass": perf_pass,
        }
        rows.append(row)
        score = delta["avg_ante_reached"] + 0.25 * delta["median_ante"]
        if best_score is None or score > best_score:
            best_score = score
            best_name = cname

    summary = {
        "schema": "p17_eval_compare_v2",
        "generated_at": _now_iso(),
        "baseline": {"name": base_name, "path": str(base_path)},
        "candidates": rows,
        "best_candidate": best_name,
        "perf_gate_pass": any(bool(r.get("perf_gate_pass")) for r in rows),
        "milestone_eval_present": any(int(_read_json(_parse_named_path(c)[1]).get("episodes") or 0) >= 500 for c in args.candidate),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    csv_lines = ["candidate,delta_avg_ante,delta_median_ante,delta_win_rate,perf_gate_pass,ci95_low,ci95_high"]
    for r in rows:
        ci = (r.get("delta") or {}).get("avg_ante_ci95") or {}
        csv_lines.append(
            f"{r['candidate']},{r['delta']['avg_ante_reached']:.6f},{r['delta']['median_ante']:.6f},{r['delta']['win_rate']:.6f},{str(r['perf_gate_pass']).lower()},{ci.get('low','')},{ci.get('high','')}"
        )
    tables_dir = out_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    (tables_dir / "compare.csv").write_text("\n".join(csv_lines) + "\n", encoding="utf-8")

    md = [
        "# P17 Eval Compare",
        "",
        f"- baseline: {base_name}",
        f"- best_candidate: {best_name}",
        f"- perf_gate_pass: {summary['perf_gate_pass']}",
        f"- milestone_eval_present: {summary['milestone_eval_present']}",
        "",
        "## Candidates",
    ]
    for r in rows:
        ci = (r.get("delta") or {}).get("avg_ante_ci95")
        md += [
            f"- {r['candidate']}: "
            f"delta_avg={r['delta']['avg_ante_reached']:.4f}, "
            f"delta_median={r['delta']['median_ante']:.4f}, "
            f"delta_win={r['delta']['win_rate']:.4f}, "
            f"perf_gate_pass={r['perf_gate_pass']}, "
            f"ci95={ci}",
        ]
    (out_dir / "summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    print(json.dumps({"out_dir": str(out_dir), "perf_gate_pass": summary["perf_gate_pass"], "best_candidate": best_name}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

