from __future__ import annotations

import argparse
import json
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


def _exp_dirs(run_root: Path) -> list[Path]:
    return sorted(
        [p for p in run_root.iterdir() if p.is_dir() and (p / "run_manifest.json").exists()],
        key=lambda p: p.name,
    )


def _seed_metric(seed_row: dict[str, Any]) -> float | None:
    value = seed_row.get("avg_ante_reached")
    if isinstance(value, (int, float)):
        return float(value)
    value = seed_row.get("primary_metric")
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _ddmin(items: list[str], predicate) -> list[str]:
    if not items:
        return []
    n = 2
    current = list(items)
    while len(current) >= 2:
        chunk = max(1, len(current) // n)
        subsets = [current[i : i + chunk] for i in range(0, len(current), chunk)]
        reduced = False
        for subset in subsets:
            if predicate(subset):
                current = subset
                n = 2
                reduced = True
                break
        if reduced:
            continue
        for subset in subsets:
            complement = [x for x in current if x not in subset]
            if complement and predicate(complement):
                current = complement
                n = max(2, n - 1)
                reduced = True
                break
        if not reduced:
            if n >= len(current):
                break
            n = min(len(current), n * 2)
    return current


def run_seed_bisect(run_root: Path) -> dict[str, Any]:
    exp_dirs = _exp_dirs(run_root)
    if not exp_dirs:
        return {
            "mode": "seed_bisect",
            "status": "no_data",
            "message": "no experiment directories found in run_root",
        }

    target_exp = exp_dirs[0]
    target_summary = {}
    target_per_seed: list[dict[str, Any]] = []
    for exp_dir in exp_dirs:
        summary = read_json(exp_dir / "exp_summary.json")
        per_seed = (
            (summary.get("seed_metrics") or {}).get("per_seed")
            if isinstance((summary.get("seed_metrics") or {}).get("per_seed"), list)
            else []
        )
        if per_seed:
            target_exp = exp_dir
            target_summary = summary
            target_per_seed = per_seed
            break

    if not target_per_seed:
        return {
            "mode": "seed_bisect",
            "status": "no_seed_metrics",
            "exp_id": target_exp.name,
            "message": "per_seed metrics unavailable; cannot bisect",
        }

    failed = [str(r.get("seed")) for r in target_per_seed if str(r.get("status")).lower() != "ok"]
    candidates = [str(r.get("seed")) for r in target_per_seed if str(r.get("seed") or "").strip()]
    used_synthetic_failure = False
    threshold = None
    if not failed:
        metric_values = [_seed_metric(r) for r in target_per_seed]
        metric_values = [v for v in metric_values if v is not None]
        if metric_values:
            mean_v = sum(metric_values) / len(metric_values)
            threshold = mean_v - 0.15
            failed = [
                str(r.get("seed"))
                for r in target_per_seed
                if (_seed_metric(r) is not None and float(_seed_metric(r) or 0.0) < threshold)
            ]
            used_synthetic_failure = True
        if not failed and candidates:
            failed = [candidates[0]]
            used_synthetic_failure = True

    failed_set = set(failed)

    def predicate(subset: list[str]) -> bool:
        if not subset:
            return False
        if not used_synthetic_failure:
            return any(x in failed_set for x in subset)
        rows = [r for r in target_per_seed if str(r.get("seed")) in set(subset)]
        vals = [_seed_metric(r) for r in rows]
        vals = [v for v in vals if v is not None]
        if not vals:
            return False
        avg = sum(vals) / len(vals)
        if threshold is not None:
            return avg < float(threshold)
        return any(str(r.get("seed")) in failed_set for r in rows)

    minimal_subset = _ddmin(list(dict.fromkeys(failed)), predicate) if failed else []
    return {
        "mode": "seed_bisect",
        "status": "pass",
        "exp_id": target_exp.name,
        "candidate_seed_count": len(candidates),
        "failing_seed_count": len(failed),
        "failing_seed_subset": failed,
        "minimal_trigger_subset": minimal_subset or failed[:1],
        "used_synthetic_failure": used_synthetic_failure,
        "threshold": threshold,
        "evidence_paths": [
            str(target_exp / "exp_summary.json"),
            str(target_exp / "seeds_used.json"),
        ],
    }


def run_config_bisect(run_root: Path) -> dict[str, Any]:
    exp_dirs = _exp_dirs(run_root)
    rows: list[dict[str, Any]] = []
    for exp_dir in exp_dirs:
        manifest = read_json(exp_dir / "run_manifest.json")
        summary = read_json(exp_dir / "exp_summary.json")
        exp = manifest.get("experiment") if isinstance(manifest.get("experiment"), dict) else {}
        params = exp.get("parameters") if isinstance(exp.get("parameters"), dict) else {}
        score = (summary.get("seed_metrics") or {}).get("avg_ante_reached")
        if not isinstance(score, (int, float)):
            score = summary.get("seed_metrics", {}).get("mean") if isinstance(summary.get("seed_metrics"), dict) else None
        rows.append(
            {
                "exp_id": str(manifest.get("exp_id") or exp_dir.name),
                "score": float(score) if isinstance(score, (int, float)) else 0.0,
                "params": params,
                "policy": exp.get("policy"),
                "stake": exp.get("stake"),
                "ante": exp.get("ante"),
                "run_dir": str(exp_dir),
            }
        )
    if len(rows) < 2:
        return {
            "mode": "config_bisect",
            "status": "insufficient_data",
            "message": "need at least 2 experiments for config bisect",
        }

    rows_sorted = sorted(rows, key=lambda r: float(r.get("score") or 0.0), reverse=True)
    best = rows_sorted[0]
    worst = rows_sorted[-1]
    best_params = best.get("params") if isinstance(best.get("params"), dict) else {}
    worst_params = worst.get("params") if isinstance(worst.get("params"), dict) else {}

    differing: dict[str, dict[str, Any]] = {}
    for key in sorted(set(list(best_params.keys()) + list(worst_params.keys()))):
        if best_params.get(key) != worst_params.get(key):
            differing[key] = {"best": best_params.get(key), "worst": worst_params.get(key)}

    suspected_key = ""
    max_spread = -1.0
    for key in differing.keys():
        groups: dict[str, list[float]] = {}
        for row in rows:
            params = row.get("params") if isinstance(row.get("params"), dict) else {}
            value = str(params.get(key))
            groups.setdefault(value, []).append(float(row.get("score") or 0.0))
        means = [sum(vals) / len(vals) for vals in groups.values() if vals]
        if len(means) >= 2:
            spread = max(means) - min(means)
            if spread > max_spread:
                max_spread = spread
                suspected_key = key

    return {
        "mode": "config_bisect",
        "status": "pass",
        "best": best,
        "worst": worst,
        "differing_params": differing,
        "suspected_parameter": suspected_key,
        "suspected_spread": max_spread if max_spread >= 0 else None,
        "evidence_paths": [str(p / "run_manifest.json") for p in exp_dirs],
    }


def write_outputs(report: dict[str, Any], out_dir: Path) -> dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "bisect_report.json"
    md_path = out_dir / "bisect_report.md"
    write_json(json_path, report)
    lines = [
        "# P24 Bisect-lite Report",
        "",
        f"- mode: `{report.get('mode')}`",
        f"- status: `{report.get('status')}`",
    ]
    for key in (
        "exp_id",
        "failing_seed_count",
        "minimal_trigger_subset",
        "used_synthetic_failure",
        "suspected_parameter",
        "suspected_spread",
    ):
        if key in report:
            lines.append(f"- {key}: `{report.get(key)}`")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"json": str(json_path), "md": str(md_path)}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="P24 bisect-lite regression localization")
    p.add_argument("--run-root", required=True)
    p.add_argument("--mode", choices=["seed_bisect", "config_bisect"], default="seed_bisect")
    p.add_argument("--out-dir", required=True)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    run_root = Path(args.run_root).resolve()
    out_dir = Path(args.out_dir).resolve()

    if args.mode == "seed_bisect":
        report = run_seed_bisect(run_root)
    else:
        report = run_config_bisect(run_root)
    report["generated_at"] = now_iso()
    report["run_root"] = str(run_root)
    paths = write_outputs(report, out_dir)
    print(json.dumps({"status": "PASS", "mode": args.mode, "paths": paths}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

