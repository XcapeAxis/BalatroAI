from __future__ import annotations

if __package__ is None or __package__ == "":
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze P11 oracle/sim mismatch and emit categorized tables.")
    parser.add_argument("--fixtures-dir", required=True)
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _load_first_jsonl(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8-sig") as fp:
        for line in fp:
            text = line.strip()
            if not text:
                continue
            obj = json.loads(text)
            if isinstance(obj, dict):
                return obj
    return {}


def _f(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _outcome_count(line: dict[str, Any]) -> int:
    replay = line.get("rng_replay") if isinstance(line.get("rng_replay"), dict) else {}
    outcomes = replay.get("outcomes") if isinstance(replay.get("outcomes"), list) else []
    return len(outcomes)


def _classify(row: dict[str, Any], first_action: dict[str, Any]) -> str:
    status = str(row.get("status") or "")
    if status == "pass":
        return "aligned"

    path = str(row.get("first_diff_path") or "")
    action_type = str(first_action.get("action_type") or "").upper()

    if "rng_replay" in path:
        return "outcome_missing_or_desync"
    if "score_observed" in path:
        return "prob_trigger_mismatch"
    if "economy" in path or "money" in path:
        return "econ_delta_mismatch"
    if "shop" in path or "packs" in path or "vouchers" in path:
        return "shop_offer_mismatch"
    if action_type not in {"PLAY", "DISCARD", "BUY", "SELL", "REROLL", "PACK", "USE"}:
        return "non_play_action"
    return "other_mismatch"


def main() -> int:
    args = parse_args()
    fixtures_dir = Path(args.fixtures_dir)
    report_path = fixtures_dir / "report_p11.json"
    if not report_path.exists():
        print(f"ERROR: report not found: {report_path}")
        return 2

    report = _load_json(report_path)
    results = report.get("results") if isinstance(report.get("results"), list) else []

    rows: list[dict[str, Any]] = []
    cls_counter: Counter[str] = Counter()
    status_counter: Counter[str] = Counter()

    for result in results:
        if not isinstance(result, dict):
            continue
        target = str(result.get("target") or "")
        status = str(result.get("status") or "")
        status_counter[status] += 1

        oracle_path = fixtures_dir / f"oracle_trace_{target}.jsonl"
        sim_path = fixtures_dir / f"sim_trace_{target}.jsonl"
        action_path = fixtures_dir / f"action_trace_{target}.jsonl"

        oracle0 = _load_first_jsonl(oracle_path)
        sim0 = _load_first_jsonl(sim_path)
        action0 = _load_first_jsonl(action_path)

        score_obs_oracle = oracle0.get("score_observed") if isinstance(oracle0.get("score_observed"), dict) else {}
        score_obs_sim = sim0.get("score_observed") if isinstance(sim0.get("score_observed"), dict) else {}

        oracle_delta = _f(score_obs_oracle.get("delta"))
        sim_delta = _f(score_obs_sim.get("delta"))
        delta_gap = oracle_delta - sim_delta

        classification = _classify(result, action0)
        cls_counter[classification] += 1

        rows.append(
            {
                "target": target,
                "category": str(result.get("category") or ""),
                "template": str(result.get("template") or ""),
                "status": status,
                "action_type": str(action0.get("action_type") or "").upper(),
                "oracle_delta": oracle_delta,
                "sim_delta": sim_delta,
                "delta_gap": delta_gap,
                "oracle_outcomes": _outcome_count(oracle0),
                "sim_outcomes": _outcome_count(sim0),
                "classification": classification,
                "first_diff_step": result.get("first_diff_step"),
                "first_diff_path": result.get("first_diff_path"),
                "dumped_oracle": result.get("dumped_oracle"),
                "dumped_sim": result.get("dumped_sim"),
            }
        )

    rows.sort(key=lambda x: str(x.get("target") or ""))

    csv_path = fixtures_dir / "mismatch_table_p11.csv"
    md_path = fixtures_dir / "mismatch_table_p11.md"

    fieldnames = [
        "target",
        "category",
        "template",
        "status",
        "action_type",
        "oracle_delta",
        "sim_delta",
        "delta_gap",
        "oracle_outcomes",
        "sim_outcomes",
        "classification",
        "first_diff_step",
        "first_diff_path",
        "dumped_oracle",
        "dumped_sim",
    ]

    with csv_path.open("w", encoding="utf-8-sig", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    lines: list[str] = []
    lines.append("# P11 Mismatch Table")
    lines.append("")
    lines.append("| " + " | ".join(fieldnames) + " |")
    lines.append("| " + " | ".join(["---"] * len(fieldnames)) + " |")
    for row in rows:
        vals: list[str] = []
        for key in fieldnames:
            value = row.get(key)
            if isinstance(value, float):
                vals.append(f"{value:.6f}")
            else:
                vals.append(str(value).replace("|", "\\|"))
        lines.append("| " + " | ".join(vals) + " |")

    lines.append("")
    lines.append("## Classification Summary")
    for key, value in sorted(cls_counter.items(), key=lambda x: x[0]):
        lines.append(f"- {key}: {value}")

    lines.append("")
    lines.append("## Status Summary")
    for key, value in sorted(status_counter.items(), key=lambda x: x[0]):
        lines.append(f"- {key}: {value}")

    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"wrote: {csv_path}")
    print(f"wrote: {md_path}")
    print("classification_summary:")
    for key, value in sorted(cls_counter.items(), key=lambda x: x[0]):
        print(f"  {key}: {value}")
    print("status_summary:")
    for key, value in sorted(status_counter.items(), key=lambda x: x[0]):
        print(f"  {key}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

