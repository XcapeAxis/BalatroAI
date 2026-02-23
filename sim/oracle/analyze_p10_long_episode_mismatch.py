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
    parser = argparse.ArgumentParser(description="Analyze P10 long-episode oracle/sim mismatch and emit categorized tables.")
    parser.add_argument("--fixtures-dir", required=True)
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _load_first_jsonl(path: Path) -> dict[str, Any]:
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


def _classify(result: dict[str, Any], oracle0: dict[str, Any], sim0: dict[str, Any]) -> str:
    status = str(result.get("status") or "")
    if status == "pass":
        return "aligned"

    path = str(result.get("first_diff_path") or "")
    if "score_observed" in path:
        return "hand_score_mismatch"
    if "shop" in path or "vouchers" in path or "packs" in path:
        return "shop_offer_mismatch"
    if "pack_choices" in path:
        return "pack_choice_mismatch"
    if "blind" in path or "tag" in path:
        return "blind_tag_mismatch"
    if "stake" in path or "rules" in path:
        return "stake_rule_mismatch"

    oracle_rng = oracle0.get("rng_replay") if isinstance(oracle0.get("rng_replay"), dict) else {}
    sim_rng = sim0.get("rng_replay") if isinstance(sim0.get("rng_replay"), dict) else {}
    if oracle_rng != sim_rng:
        return "rng_replay_mismatch"

    return "other_mismatch"


def main() -> int:
    args = parse_args()
    fixtures_dir = Path(args.fixtures_dir)
    report_path = fixtures_dir / "report_p10.json"
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

        oracle0 = _load_first_jsonl(oracle_path) if oracle_path.exists() else {}
        sim0 = _load_first_jsonl(sim_path) if sim_path.exists() else {}
        action0 = _load_first_jsonl(action_path) if action_path.exists() else {}

        score_obs_oracle = oracle0.get("score_observed") if isinstance(oracle0.get("score_observed"), dict) else {}
        score_obs_sim = sim0.get("score_observed") if isinstance(sim0.get("score_observed"), dict) else {}
        oracle_delta = _f(score_obs_oracle.get("delta"))
        sim_delta = _f(score_obs_sim.get("delta"))

        classification = _classify(result, oracle0, sim0)
        cls_counter[classification] += 1

        rows.append(
            {
                "target": target,
                "status": status,
                "stake": str(result.get("stake") or ""),
                "category": str(result.get("category") or ""),
                "first_action_type": str(action0.get("action_type") or "").upper(),
                "oracle_delta": oracle_delta,
                "sim_delta": sim_delta,
                "delta_gap": oracle_delta - sim_delta,
                "classification": classification,
                "first_diff_step": result.get("first_diff_step"),
                "first_diff_path": result.get("first_diff_path"),
                "dumped_oracle": result.get("dumped_oracle"),
                "dumped_sim": result.get("dumped_sim"),
            }
        )

    rows.sort(key=lambda x: str(x.get("target") or ""))
    csv_path = fixtures_dir / "episode_mismatch_table_p10.csv"
    md_path = fixtures_dir / "episode_mismatch_table_p10.md"

    fieldnames = [
        "target",
        "status",
        "stake",
        "category",
        "first_action_type",
        "oracle_delta",
        "sim_delta",
        "delta_gap",
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
    lines.append("# P10 Long Episode Mismatch Table")
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

