from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
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


def _normalize_phase(raw: Any) -> list[str]:
    if isinstance(raw, str):
        return [raw.upper()]
    if isinstance(raw, list):
        return [str(x).upper() for x in raw if str(x).strip()]
    return []


def build_coverage_summary(run_root: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    strategy_counter: Counter[str] = Counter()
    phase_counter: Counter[str] = Counter()
    stake_counter: Counter[str] = Counter()
    ante_counter: Counter[str] = Counter()
    failure_counter: Counter[str] = Counter()
    param_values: dict[str, set[str]] = defaultdict(set)
    seed_set_counter: Counter[str] = Counter()
    all_seeds: set[str] = set()
    rows: list[dict[str, Any]] = []

    for exp_dir in sorted([p for p in run_root.iterdir() if p.is_dir()], key=lambda p: p.name):
        manifest = read_json(exp_dir / "run_manifest.json")
        summary = read_json(exp_dir / "exp_summary.json")
        seeds_used = read_json(exp_dir / "seeds_used.json")
        if not manifest:
            continue

        exp = manifest.get("experiment") if isinstance(manifest.get("experiment"), dict) else {}
        strategy = str(exp.get("policy_type") or exp.get("policy") or "unknown")
        phases = _normalize_phase(exp.get("phases") or exp.get("phase"))
        stake = str(exp.get("stake") or "unknown")
        ante = str(exp.get("ante") or "unknown")
        params = exp.get("parameters") if isinstance(exp.get("parameters"), dict) else {}
        seed_list = [str(s) for s in (seeds_used.get("seeds") or manifest.get("seeds_used") or [])]
        seed_set_name = str(manifest.get("seed_set_name") or "unknown")

        strategy_counter[strategy] += 1
        stake_counter[stake] += 1
        ante_counter[ante] += 1
        seed_set_counter[seed_set_name] += 1
        for seed in seed_list:
            all_seeds.add(seed)
        for phase in phases:
            phase_counter[phase] += 1
        for key, value in params.items():
            param_values[str(key)].add(str(value))

        failures = ((summary.get("seed_metrics") or {}).get("catastrophic_failures") or [])
        if isinstance(failures, list):
            for failure in failures:
                if not isinstance(failure, dict):
                    continue
                bucket = str(failure.get("error") or failure.get("stage") or "unknown")
                failure_counter[bucket] += 1

        rows.append(
            {
                "exp_id": str(manifest.get("exp_id") or exp_dir.name),
                "status": str(summary.get("status") or "unknown"),
                "strategy_type": strategy,
                "phases": ",".join(phases) if phases else "unknown",
                "stake": stake,
                "ante": ante,
                "seed_set_name": seed_set_name,
                "seed_count": len(seed_list),
                "failure_count": len(failures) if isinstance(failures, list) else 0,
            }
        )

    summary_payload = {
        "schema": "p23_coverage_summary_v1",
        "generated_at": now_iso(),
        "run_root": str(run_root),
        "experiments": len(rows),
        "strategy_type_coverage": dict(strategy_counter),
        "phase_coverage": dict(phase_counter),
        "stake_coverage": dict(stake_counter),
        "ante_coverage": dict(ante_counter),
        "failure_reason_coverage": dict(failure_counter),
        "seed_set_coverage": dict(seed_set_counter),
        "seed_coverage": {
            "unique_seed_count": len(all_seeds),
            "total_seed_observations": int(sum(row["seed_count"] for row in rows)),
        },
        "matrix_parameter_coverage": {
            key: {
                "unique_count": len(values),
                "values": sorted(values),
            }
            for key, values in sorted(param_values.items())
        },
    }
    return summary_payload, rows


def write_coverage_outputs(
    summary: dict[str, Any],
    rows: list[dict[str, Any]],
    out_dir: Path,
) -> dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "coverage_summary.json"
    md_path = out_dir / "coverage_summary.md"
    csv_path = out_dir / "coverage_table.csv"

    write_json(json_path, summary)

    fieldnames = [
        "exp_id",
        "status",
        "strategy_type",
        "phases",
        "stake",
        "ante",
        "seed_set_name",
        "seed_count",
        "failure_count",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    md_lines = [
        "# P23 Coverage Summary",
        "",
        f"- generated_at: `{summary.get('generated_at')}`",
        f"- experiments: `{summary.get('experiments')}`",
        f"- unique_seeds: `{((summary.get('seed_coverage') or {}).get('unique_seed_count'))}`",
        "",
        "## Dimensions",
        f"- strategy_type: `{summary.get('strategy_type_coverage')}`",
        f"- phase: `{summary.get('phase_coverage')}`",
        f"- stake: `{summary.get('stake_coverage')}`",
        f"- ante: `{summary.get('ante_coverage')}`",
        f"- failure_reason: `{summary.get('failure_reason_coverage')}`",
        "",
        "## Matrix Parameters",
    ]
    params = summary.get("matrix_parameter_coverage") or {}
    if isinstance(params, dict) and params:
        for key, value in params.items():
            md_lines.append(f"- `{key}`: `{value}`")
    else:
        md_lines.append("- (none)")

    md_lines += [
        "",
        "## Per Experiment",
        "",
        "| exp_id | status | strategy_type | phases | stake | ante | seed_set | seeds | failures |",
        "|---|---|---|---|---|---|---|---:|---:|",
    ]
    for row in rows:
        md_lines.append(
            "| {exp_id} | {status} | {strategy_type} | {phases} | {stake} | {ante} | {seed_set_name} | {seed_count} | {failure_count} |".format(
                **row
            )
        )
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    return {"json": str(json_path), "md": str(md_path), "csv": str(csv_path)}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="P23 coverage aggregator")
    p.add_argument("--run-root", required=True, help="Path to runs/<run_id>")
    p.add_argument("--out-dir", default="", help="Output directory (defaults to run-root)")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    run_root = Path(args.run_root).resolve()
    if not run_root.exists():
        raise SystemExit(f"run root not found: {run_root}")
    out_dir = Path(args.out_dir).resolve() if args.out_dir else run_root
    summary, rows = build_coverage_summary(run_root)
    paths = write_coverage_outputs(summary, rows, out_dir)
    print(json.dumps({"status": "PASS", "paths": paths}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

