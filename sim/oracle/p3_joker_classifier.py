from __future__ import annotations

if __package__ is None or __package__ == "":
    import sys
    from pathlib import Path as _Path

    sys.path.append(str(_Path(__file__).resolve().parent.parent.parent))

import argparse
import csv
import json
import re
import shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SUPPORTED_TEMPLATES = [
    "flat_mult",
    "suit_mult_per_scoring_card",
    "face_chips_per_scoring_card",
    "odd_chips_per_scoring_card",
    "even_mult_per_scoring_card",
    "fibonacci_mult_per_scoring_card",
    "banner_discards_to_chips",
    "photograph_first_face_xmult",
    "baron_held_kings_xmult",
]

ALL_TEMPLATE_CATALOG = [
    "flat_mult",
    "suit_mult_per_scoring_card",
    "face_chips_per_scoring_card",
    "odd_chips_per_scoring_card",
    "even_mult_per_scoring_card",
    "fibonacci_mult_per_scoring_card",
    "banner_discards_to_chips",
    "photograph_first_face_xmult",
    "baron_held_kings_xmult",
    "joker_planet_hand_level_bonus",
    "card_modifier_bonus",
    "stacked_combo",
]

JOKER_NAME_TO_KEY = {
    "Joker": "j_joker",
    "Greedy Joker": "j_greedy_joker",
    "Lusty Joker": "j_lusty_joker",
    "Wrathful Joker": "j_wrathful_joker",
    "Gluttonous Joker": "j_gluttenous_joker",
    "Banner": "j_banner",
    "Odd Todd": "j_odd_todd",
    "Even Steven": "j_even_steven",
    "Fibonacci": "j_fibonacci",
    "Scary Face": "j_scary_face",
    "Photograph": "j_photograph",
    "Baron": "j_baron",
}


def _slug(text: str) -> str:
    low = str(text or "").strip().lower()
    low = re.sub(r"[^a-z0-9]+", "_", low)
    low = re.sub(r"_+", "_", low).strip("_")
    return low


def _joker_key_from_name(name: str) -> str:
    if name in JOKER_NAME_TO_KEY:
        return JOKER_NAME_TO_KEY[name]
    slug = _slug(name)
    return f"j_{slug}" if slug else "j_unknown"


def ensure_balatro_mechanics_root(project_root: Path) -> Path:
    dest = project_root / "balatro_mechanics"
    dest.mkdir(parents=True, exist_ok=True)

    source_final = project_root / "Balatro_Mechanics_CSV_UPDATED_20260219" / "final"
    dest_jokers = dest / "jokers.csv"

    if not dest_jokers.exists() and source_final.exists() and (source_final / "jokers.csv").exists():
        for csv_path in source_final.glob("*.csv"):
            shutil.copy2(csv_path, dest / csv_path.name)
        audit = source_final / "SOURCES_AUDIT.md"
        if audit.exists():
            shutil.copy2(audit, dest / audit.name)

    manifest = dest / "manifest.json"
    if not manifest.exists():
        files = sorted([p.name for p in dest.glob("*.csv")])
        manifest.write_text(
            json.dumps(
                {
                    "generated_at": datetime.now(timezone.utc).isoformat(),
                    "source": str(source_final),
                    "files": files,
                    "note": "Final curated mechanics CSV copied into balatro_mechanics for stable local use.",
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    return dest


def _parse_float_from_text(pattern: str, text: str) -> float | None:
    m = re.search(pattern, text, flags=re.IGNORECASE)
    if not m:
        return None
    try:
        return float(m.group(1))
    except Exception:
        return None


def _classify_row(row: dict[str, str]) -> tuple[str | None, float, dict[str, Any], list[str], str | None]:
    name = str(row.get("joker_name") or "").strip()
    effect = str(row.get("effect_summary") or "").strip()
    key_mech = str(row.get("key_mechanics") or "").strip().lower()
    trigger = str(row.get("trigger_timing") or "").strip().lower()

    low = effect.lower()
    matched_rules: list[str] = []
    params: dict[str, Any] = {}

    # 1) flat_mult
    if name.lower() == "joker":
        value = _parse_float_from_text(r"\+(\d+(?:\.\d+)?)\s*mult\b", effect)
        if value is not None and "for each" not in low:
            matched_rules.append("name=joker && +N mult")
            params["mult_add"] = value
            return "flat_mult", 0.99, params, matched_rules, None

    # 2) suit mult
    m = re.search(r"played cards with\s+(heart|diamond|spade|club)\s+suit\s+give\s*\+(\d+(?:\.\d+)?)\s*mult", low)
    if m:
        suit_word = m.group(1)
        suit = {"heart": "H", "diamond": "D", "spade": "S", "club": "C"}[suit_word]
        params["suit"] = suit
        params["mult_add_per_card"] = float(m.group(2))
        matched_rules.append("played cards with <suit> suit give +N mult")
        return "suit_mult_per_scoring_card", 0.98, params, matched_rules, None

    # 3) face chips
    m = re.search(r"played face cards\s+give\s*\+(\d+(?:\.\d+)?)\s*chips", low)
    if m:
        params["chips_add_per_card"] = float(m.group(1))
        matched_rules.append("played face cards give +N chips")
        return "face_chips_per_scoring_card", 0.98, params, matched_rules, None

    # 4) odd chips
    m = re.search(r"played cards with odd rank\s+give\s*\+(\d+(?:\.\d+)?)\s*chips", low)
    if m:
        params["chips_add_per_card"] = float(m.group(1))
        matched_rules.append("odd rank give +N chips")
        return "odd_chips_per_scoring_card", 0.98, params, matched_rules, None

    # 5) even mult
    m = re.search(r"played cards with even rank\s+give\s*\+(\d+(?:\.\d+)?)\s*mult", low)
    if m:
        params["mult_add_per_card"] = float(m.group(1))
        matched_rules.append("even rank give +N mult")
        return "even_mult_per_scoring_card", 0.98, params, matched_rules, None

    # 6) fibonacci mult
    if "ace" in low and "2" in low and "3" in low and "5" in low and "8" in low and "mult" in low:
        m = re.search(r"\+(\d+(?:\.\d+)?)\s*mult", low)
        if m:
            params["mult_add_per_card"] = float(m.group(1))
            params["ranks"] = ["A", "2", "3", "5", "8"]
            matched_rules.append("ace/2/3/5/8 gives +N mult")
            return "fibonacci_mult_per_scoring_card", 0.97, params, matched_rules, None

    # 7) banner discards->chips
    m = re.search(r"\+(\d+(?:\.\d+)?)\s*chips\s*for each remaining discard", low)
    if m:
        params["chips_add_per_discard"] = float(m.group(1))
        matched_rules.append("+N chips for each remaining discard")
        return "banner_discards_to_chips", 0.99, params, matched_rules, None

    # 8) photograph first face xmult
    m = re.search(r"first played face card\s+gives\s*x\s*(\d+(?:\.\d+)?)\s*mult", low)
    if m:
        params["mult_scale"] = float(m.group(1))
        matched_rules.append("first played face card gives xN mult")
        return "photograph_first_face_xmult", 0.99, params, matched_rules, None

    # 9) baron held kings xmult
    m = re.search(r"each king held in hand\s+gives\s*x\s*(\d+(?:\.\d+)?)\s*mult", low)
    if m:
        params["rank"] = "K"
        params["mult_scale_per_card"] = float(m.group(1))
        matched_rules.append("each king held in hand gives xN mult")
        return "baron_held_kings_xmult", 0.99, params, matched_rules, None

    # Conservative unsupported reasons
    if "chance" in low or "1 in " in low:
        return None, 0.0, {}, matched_rules, "probabilistic_trigger"
    if "end of round" in low or "final hand" in low or "this round" in low:
        return None, 0.0, {}, matched_rules, "cross_round_state"
    if "money" in low or "$" in low or "dollar" in low:
        return None, 0.0, {}, matched_rules, "economy_or_shop_related"
    if trigger in {"on_discard", "on_buy", "on_sell", "unknown"} and key_mech in {"", "unknown"}:
        return None, 0.0, {}, matched_rules, "insufficient_structured_fields"
    return None, 0.0, {}, matched_rules, "no_safe_template_match"


def classify_jokers(mechanics_root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    jokers_path = mechanics_root / "jokers.csv"
    if not jokers_path.exists():
        raise FileNotFoundError(f"missing jokers.csv at {jokers_path}")

    rows: list[dict[str, Any]] = []
    unsupported: list[dict[str, Any]] = []

    with jokers_path.open("r", encoding="utf-8-sig", newline="") as fp:
        reader = csv.DictReader(fp)
        for row in reader:
            name = str(row.get("joker_name") or "").strip()
            key = _joker_key_from_name(name)
            template, confidence, params, matched_rules, unsupported_reason = _classify_row(row)

            entry = {
                "joker_name": name,
                "joker_key": key,
                "template": template,
                "confidence": confidence,
                "matched_rules": matched_rules,
                "raw_fields_used": {
                    "effect_summary": row.get("effect_summary"),
                    "key_mechanics": row.get("key_mechanics"),
                    "trigger_timing": row.get("trigger_timing"),
                    "constraints_or_conditions": row.get("constraints_or_conditions"),
                },
                "params": params,
                "target": f"p3_{key}",
            }
            rows.append(entry)

            if template is None:
                unsupported.append(
                    {
                        "joker_name": name,
                        "joker_key": key,
                        "reason": unsupported_reason or "unknown",
                        "effect_summary": row.get("effect_summary"),
                        "trigger_timing": row.get("trigger_timing"),
                        "key_mechanics": row.get("key_mechanics"),
                    }
                )

    return rows, unsupported


def write_outputs(project_root: Path, mechanics_root: Path, mapping: list[dict[str, Any]], unsupported: list[dict[str, Any]]) -> dict[str, Any]:
    derived_dir = mechanics_root / "derived"
    derived_dir.mkdir(parents=True, exist_ok=True)

    map_path = derived_dir / "joker_template_map.json"
    unsupported_path = derived_dir / "joker_template_unsupported.json"
    map_path.write_text(json.dumps(mapping, ensure_ascii=False, indent=2), encoding="utf-8")
    unsupported_path.write_text(json.dumps(unsupported, ensure_ascii=False, indent=2), encoding="utf-8")

    total = len(mapping)
    supported = sum(1 for x in mapping if x.get("template"))

    template_counter: Counter[str] = Counter()
    for x in mapping:
        template = x.get("template")
        if isinstance(template, str) and template:
            template_counter[template] += 1

    unsupported_counter: Counter[str] = Counter(str(x.get("reason") or "unknown") for x in unsupported)

    md_lines: list[str] = []
    md_lines.append("# P3 Joker Template Coverage")
    md_lines.append("")
    md_lines.append(f"- Total jokers: **{total}**")
    md_lines.append(f"- Recognized (supported template): **{supported}**")
    md_lines.append(f"- Unsupported: **{len(unsupported)}**")
    md_lines.append("")
    md_lines.append("## Template Counts")
    for tpl in ALL_TEMPLATE_CATALOG:
        md_lines.append(f"- `{tpl}`: {template_counter.get(tpl, 0)}")
    md_lines.append("")
    md_lines.append("## Unsupported Top Reasons")
    for reason, count in unsupported_counter.most_common(20):
        md_lines.append(f"- `{reason}`: {count}")

    cov_doc = project_root / "docs" / "COVERAGE_P3_JOKERS.md"
    cov_doc.parent.mkdir(parents=True, exist_ok=True)
    cov_doc.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    return {
        "map_path": map_path,
        "unsupported_path": unsupported_path,
        "coverage_doc": cov_doc,
        "total": total,
        "supported": supported,
        "unsupported": len(unsupported),
        "template_counts": dict(template_counter),
        "unsupported_reasons": dict(unsupported_counter),
    }


def build_and_write(project_root: Path) -> dict[str, Any]:
    mechanics_root = ensure_balatro_mechanics_root(project_root)
    mapping, unsupported = classify_jokers(mechanics_root)
    summary = write_outputs(project_root, mechanics_root, mapping, unsupported)
    summary["mechanics_root"] = mechanics_root
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build conservative P3 joker template map and coverage reports.")
    parser.add_argument("--project-root", default=".")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    project_root = Path(args.project_root).resolve()
    summary = build_and_write(project_root)

    print(f"mechanics_root={summary['mechanics_root']}")
    print(f"total={summary['total']} supported={summary['supported']} unsupported={summary['unsupported']}")
    print(f"map={summary['map_path']}")
    print(f"unsupported={summary['unsupported_path']}")
    print(f"coverage_doc={summary['coverage_doc']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
