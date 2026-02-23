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
    "suit_chips_per_scoring_card",
    "face_chips_per_scoring_card",
    "face_mult_per_scoring_card",
    "odd_chips_per_scoring_card",
    "even_mult_per_scoring_card",
    "fibonacci_mult_per_scoring_card",
    "banner_discards_to_chips",
    "photograph_first_face_xmult",
    "baron_held_kings_xmult",
    "hand_contains_mult_add",
    "hand_contains_chips_add",
    "hand_contains_xmult",
    "rank_set_chips_mult_per_scoring_card",
    "held_rank_mult_add",
    "held_lowest_rank_mult_add",
    "all_held_suits_xmult",
    "scoring_club_and_other_xmult",
    "scoring_all_suits_xmult",
    "discards_zero_mult_add",
    "final_hand_xmult",
    "hand_size_lte_mult_add",
    "rank_set_xmult_per_scoring_card",
    "observed_noop",
]

ALL_TEMPLATE_CATALOG = [
    "flat_mult",
    "suit_mult_per_scoring_card",
    "suit_chips_per_scoring_card",
    "face_chips_per_scoring_card",
    "face_mult_per_scoring_card",
    "odd_chips_per_scoring_card",
    "even_mult_per_scoring_card",
    "fibonacci_mult_per_scoring_card",
    "banner_discards_to_chips",
    "photograph_first_face_xmult",
    "baron_held_kings_xmult",
    "hand_contains_mult_add",
    "hand_contains_chips_add",
    "hand_contains_xmult",
    "rank_set_chips_mult_per_scoring_card",
    "held_rank_mult_add",
    "held_lowest_rank_mult_add",
    "all_held_suits_xmult",
    "scoring_club_and_other_xmult",
    "scoring_all_suits_xmult",
    "discards_zero_mult_add",
    "final_hand_xmult",
    "hand_size_lte_mult_add",
    "rank_set_xmult_per_scoring_card",
    "joker_planet_hand_level_bonus",
    "card_modifier_bonus",
    "stacked_combo",
    "observed_noop",
]

JOKER_NAME_TO_KEY = {
    "Joker": "j_joker",
    "Greedy Joker": "j_greedy_joker",
    "Lusty Joker": "j_lusty_joker",
    "Wrathful Joker": "j_wrathful_joker",
    "Gluttonous Joker": "j_gluttenous_joker",
    "Onyx Agate": "j_onyx_agate",
    "Banner": "j_banner",
    "Odd Todd": "j_odd_todd",
    "Even Steven": "j_even_steven",
    "Fibonacci": "j_fibonacci",
    "Scary Face": "j_scary_face",
    "Photograph": "j_photograph",
    "Baron": "j_baron",
    "Jolly Joker": "j_jolly",
    "Zany Joker": "j_zany",
    "Mad Joker": "j_mad",
    "Crazy Joker": "j_crazy",
    "Droll Joker": "j_droll",
    "Sly Joker": "j_sly",
    "Wily Joker": "j_wily",
    "Clever Joker": "j_clever",
    "Devious Joker": "j_devious",
    "Crafty Joker": "j_crafty",
    "The Duo": "j_duo",
    "The Trio": "j_trio",
    "The Family": "j_family",
    "The Order": "j_order",
    "The Tribe": "j_tribe",
    "Arrowhead": "j_arrowhead",
    "Smiley Face": "j_smiley",
    "Scholar": "j_scholar",
    "Walkie Talkie": "j_walkie_talkie",
    "Shoot the Moon": "j_shoot_the_moon",
    "Raised Fist": "j_raised_fist",
    "Blackboard": "j_blackboard",
    "Flower Pot": "j_flower_pot",
    "Seeing Double": "j_seeing_double",
    "Mystic Summit": "j_mystic_summit",
    "Acrobat": "j_acrobat",
    "Half Joker": "j_half",
    "Triboulet": "j_triboulet",
}

HAND_TYPE_ALIASES = {
    "PAIR": "PAIR",
    "TWO_PAIR": "TWO_PAIR",
    "THREE_OF_A_KIND": "THREE_OF_A_KIND",
    "STRAIGHT": "STRAIGHT",
    "FLUSH": "FLUSH",
    "FOUR_OF_A_KIND": "FOUR_OF_A_KIND",
}

P5_NOOP_NAME_OVERRIDES = {
    "Blueprint",
    "Brainstorm",
    "Burglar",
    "Card Sharp",
    "Cartomancer",
    "Chicot",
    "Drunkard",
    "Dusk",
    "Four Fingers",
    "Hiker",
    "Juggler",
    "Luchador",
    "Midas Mask",
    "Mime",
    "Pareidolia",
    "Shortcut",
    "Sock and Buskin",
}

P6_NOOP_REASON_OVERRIDES = {
    "economy_or_shop_related",
    "cross_round_state",
    "probabilistic_trigger",
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


def _normalize_hand_type_from_text(text: str) -> str | None:
    low = str(text or "").strip().lower()
    if "two pair" in low:
        return "TWO_PAIR"
    if "three of a kind" in low:
        return "THREE_OF_A_KIND"
    if "four of a kind" in low:
        return "FOUR_OF_A_KIND"
    if low.endswith("pair") or " a pair" in low:
        return "PAIR"
    if "straight" in low:
        return "STRAIGHT"
    if "flush" in low:
        return "FLUSH"
    return None


def _unsupported_reason(low_effect: str, trigger: str, key_mech: str) -> str:
    if "chance" in low_effect or "1 in " in low_effect or "random" in low_effect:
        return "probabilistic_trigger"
    if "this joker gains" in low_effect or "currently" in low_effect or "resets" in low_effect:
        return "cross_round_state"
    if "money" in low_effect or "$" in low_effect or "shop" in low_effect:
        return "economy_or_shop_related"
    if trigger in {"on_discard", "on_buy", "on_sell", "unknown"} and key_mech in {"", "unknown"}:
        return "insufficient_structured_fields"
    return "no_safe_template_match"


def _classify_row(row: dict[str, str]) -> tuple[str | None, float, dict[str, Any], list[str], str | None]:
    name = str(row.get("joker_name") or "").strip()
    effect = str(row.get("effect_summary") or "").strip()
    key_mech = str(row.get("key_mechanics") or "").strip().lower()
    trigger = str(row.get("trigger_timing") or "").strip().lower()

    low = effect.lower()
    matched_rules: list[str] = []
    params: dict[str, Any] = {}

    # Existing P3 v1 templates
    if name.lower() == "joker":
        value = _parse_float_from_text(r"\+(\d+(?:\.\d+)?)\s*mult\b", effect)
        if value is not None and "for each" not in low:
            matched_rules.append("name=joker && +N mult")
            params["mult_add"] = value
            return "flat_mult", 0.99, params, matched_rules, None

    m = re.search(r"played cards with\s+(heart|diamond|spade|club)\s+suit\s+give\s*\+(\d+(?:\.\d+)?)\s*mult", low)
    if m:
        suit = {"heart": "H", "diamond": "D", "spade": "S", "club": "C"}[m.group(1)]
        params["suit"] = suit
        params["mult_add_per_card"] = float(m.group(2))
        matched_rules.append("played cards with <suit> suit give +N mult")
        return "suit_mult_per_scoring_card", 0.98, params, matched_rules, None

    m = re.search(r"played cards with\s+(heart|diamond|spade|club)\s+suit\s+give\s*\+(\d+(?:\.\d+)?)\s*chips", low)
    if m:
        suit = {"heart": "H", "diamond": "D", "spade": "S", "club": "C"}[m.group(1)]
        params["suit"] = suit
        params["chips_add_per_card"] = float(m.group(2))
        matched_rules.append("played cards with <suit> suit give +N chips")
        return "suit_chips_per_scoring_card", 0.98, params, matched_rules, None

    m = re.search(r"played face cards\s+give\s*\+(\d+(?:\.\d+)?)\s*chips", low)
    if m:
        params["chips_add_per_card"] = float(m.group(1))
        matched_rules.append("played face cards give +N chips")
        return "face_chips_per_scoring_card", 0.98, params, matched_rules, None

    m = re.search(r"played face cards\s+give\s*\+(\d+(?:\.\d+)?)\s*mult", low)
    if m:
        params["mult_add_per_card"] = float(m.group(1))
        matched_rules.append("played face cards give +N mult")
        return "face_mult_per_scoring_card", 0.98, params, matched_rules, None

    m = re.search(r"played cards with odd rank\s+give\s*\+(\d+(?:\.\d+)?)\s*chips", low)
    if m:
        params["chips_add_per_card"] = float(m.group(1))
        matched_rules.append("odd rank give +N chips")
        return "odd_chips_per_scoring_card", 0.98, params, matched_rules, None

    m = re.search(r"played cards with even rank\s+give\s*\+(\d+(?:\.\d+)?)\s*mult", low)
    if m:
        params["mult_add_per_card"] = float(m.group(1))
        matched_rules.append("even rank give +N mult")
        return "even_mult_per_scoring_card", 0.98, params, matched_rules, None

    if "ace" in low and "2" in low and "3" in low and "5" in low and "8" in low and "mult" in low:
        m = re.search(r"\+(\d+(?:\.\d+)?)\s*mult", low)
        if m:
            params["mult_add_per_card"] = float(m.group(1))
            params["ranks"] = ["A", "2", "3", "5", "8"]
            matched_rules.append("ace/2/3/5/8 gives +N mult")
            return "fibonacci_mult_per_scoring_card", 0.97, params, matched_rules, None

    m = re.search(r"\+(\d+(?:\.\d+)?)\s*chips\s*for each remaining discard", low)
    if m:
        params["chips_add_per_discard"] = float(m.group(1))
        matched_rules.append("+N chips for each remaining discard")
        return "banner_discards_to_chips", 0.99, params, matched_rules, None

    m = re.search(r"first played face card\s+gives\s*x\s*(\d+(?:\.\d+)?)\s*mult", low)
    if m:
        params["mult_scale"] = float(m.group(1))
        matched_rules.append("first played face card gives xN mult")
        return "photograph_first_face_xmult", 0.99, params, matched_rules, None

    m = re.search(r"each king held in hand\s+gives\s*x\s*(\d+(?:\.\d+)?)\s*mult", low)
    if m:
        params["rank"] = "K"
        params["mult_scale_per_card"] = float(m.group(1))
        matched_rules.append("each king held in hand gives xN mult")
        return "baron_held_kings_xmult", 0.99, params, matched_rules, None

    # New low-risk deterministic families
    if "if played hand contains" in low:
        hand_type = _normalize_hand_type_from_text(low)
        if hand_type in HAND_TYPE_ALIASES:
            m_mult = re.search(r"\+(\d+(?:\.\d+)?)\s*mult\s*if\s*played hand contains", low)
            if m_mult and "this joker gains" not in low:
                params["hand_type"] = hand_type
                params["mult_add"] = float(m_mult.group(1))
                matched_rules.append("+N mult if played hand contains <hand_type>")
                return "hand_contains_mult_add", 0.97, params, matched_rules, None

            m_chips = re.search(r"\+(\d+(?:\.\d+)?)\s*chips\s*if\s*played hand contains", low)
            if m_chips and "this joker gains" not in low and "gains +" not in low:
                params["hand_type"] = hand_type
                params["chips_add"] = float(m_chips.group(1))
                matched_rules.append("+N chips if played hand contains <hand_type>")
                return "hand_contains_chips_add", 0.97, params, matched_rules, None

            m_x = re.search(r"x\s*(\d+(?:\.\d+)?)\s*mult\s*if\s*played hand contains", low)
            if m_x:
                params["hand_type"] = hand_type
                params["mult_scale"] = float(m_x.group(1))
                matched_rules.append("xN mult if played hand contains <hand_type>")
                return "hand_contains_xmult", 0.97, params, matched_rules, None

    if "played hand contains 3 or fewer cards" in low:
        m = re.search(r"\+(\d+(?:\.\d+)?)\s*mult", low)
        if m:
            params["max_cards"] = 3
            params["mult_add"] = float(m.group(1))
            matched_rules.append("+N mult if played hand has <=3 cards")
            return "hand_size_lte_mult_add", 0.97, params, matched_rules, None

    if "played aces give" in low and "chips" in low and "mult" in low:
        m = re.search(r"\+(\d+(?:\.\d+)?)\s*chips\s*and\s*\+(\d+(?:\.\d+)?)\s*mult", low)
        if m:
            params["ranks"] = ["A"]
            params["chips_add_per_card"] = float(m.group(1))
            params["mult_add_per_card"] = float(m.group(2))
            matched_rules.append("played aces give +chips and +mult")
            return "rank_set_chips_mult_per_scoring_card", 0.98, params, matched_rules, None

    if "played 10 or 4 gives" in low and "chips" in low and "mult" in low:
        m = re.search(r"\+(\d+(?:\.\d+)?)\s*chips\s*and\s*\+(\d+(?:\.\d+)?)\s*mult", low)
        if m:
            params["ranks"] = ["10", "4"]
            params["chips_add_per_card"] = float(m.group(1))
            params["mult_add_per_card"] = float(m.group(2))
            matched_rules.append("played 10 or 4 gives +chips and +mult")
            return "rank_set_chips_mult_per_scoring_card", 0.98, params, matched_rules, None

    if "each queen held in hand gives" in low:
        m = re.search(r"\+(\d+(?:\.\d+)?)\s*mult", low)
        if m:
            params["rank"] = "Q"
            params["mult_add_per_card"] = float(m.group(1))
            matched_rules.append("each queen held in hand gives +N mult")
            return "held_rank_mult_add", 0.98, params, matched_rules, None

    if "adds double the rank of lowest ranked card held in hand to mult" in low:
        params["scale"] = 2.0
        matched_rules.append("double lowest held rank to mult")
        return "held_lowest_rank_mult_add", 0.97, params, matched_rules, None

    if "all cards held in hand are spades or clubs" in low:
        m = re.search(r"x\s*(\d+(?:\.\d+)?)\s*mult", low)
        if m:
            params["allowed_suits"] = ["S", "C"]
            params["mult_scale"] = float(m.group(1))
            matched_rules.append("xN if all held cards in S/C")
            return "all_held_suits_xmult", 0.98, params, matched_rules, None

    if "scoring club card and a scoring card of any other suit" in low:
        m = re.search(r"x\s*(\d+(?:\.\d+)?)\s*mult", low)
        if m:
            params["required_suit"] = "C"
            params["mult_scale"] = float(m.group(1))
            matched_rules.append("xN if scoring includes club and other suit")
            return "scoring_club_and_other_xmult", 0.98, params, matched_rules, None

    if "poker hand contains a diamond card, club card, heart card, and spade card" in low:
        m = re.search(r"x\s*(\d+(?:\.\d+)?)\s*mult", low)
        if m:
            params["required_suits"] = ["D", "C", "H", "S"]
            params["mult_scale"] = float(m.group(1))
            matched_rules.append("xN if scoring hand has D/C/H/S")
            return "scoring_all_suits_xmult", 0.98, params, matched_rules, None

    if "0 discards remaining" in low:
        m = re.search(r"\+(\d+(?:\.\d+)?)\s*mult", low)
        if m:
            params["discards_left"] = 0
            params["mult_add"] = float(m.group(1))
            matched_rules.append("+N mult when discards_left==0")
            return "discards_zero_mult_add", 0.97, params, matched_rules, None

    if "final hand of round" in low:
        m = re.search(r"x\s*(\d+(?:\.\d+)?)\s*mult", low)
        if m:
            params["hands_left"] = 1
            params["mult_scale"] = float(m.group(1))
            matched_rules.append("xN mult on final hand")
            return "final_hand_xmult", 0.97, params, matched_rules, None

    if "played kings and queens each give" in low and "x" in low:
        m = re.search(r"x\s*(\d+(?:\.\d+)?)\s*mult", low)
        if m:
            params["ranks"] = ["K", "Q"]
            params["mult_scale_per_card"] = float(m.group(1))
            matched_rules.append("K/Q each give xN mult")
            return "rank_set_xmult_per_scoring_card", 0.98, params, matched_rules, None
    if name in P5_NOOP_NAME_OVERRIDES:
        params["mode"] = "discard_only"
        matched_rules.append("p5_name_override_observed_noop")
        return "observed_noop", 0.86, params, matched_rules, None

    unsupported_reason = _unsupported_reason(low, trigger, key_mech)
    if unsupported_reason in P6_NOOP_REASON_OVERRIDES:
        params["mode"] = "discard_only"
        params["promoted_reason"] = unsupported_reason
        matched_rules.append(f"p6_reason_override_observed_noop:{unsupported_reason}")
        return "observed_noop", 0.72, params, matched_rules, None

    return None, 0.0, {}, matched_rules, unsupported_reason


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


def write_outputs(
    project_root: Path,
    mechanics_root: Path,
    mapping: list[dict[str, Any]],
    unsupported: list[dict[str, Any]],
    out_derived: Path | None = None,
    export_supported_targets: Path | None = None,
) -> dict[str, Any]:
    derived_dir = out_derived if out_derived is not None else (mechanics_root / "derived")
    derived_dir.mkdir(parents=True, exist_ok=True)

    map_path = derived_dir / "joker_template_map.json"
    unsupported_path = derived_dir / "joker_template_unsupported.json"
    map_path.write_text(json.dumps(mapping, ensure_ascii=False, indent=2), encoding="utf-8")
    unsupported_path.write_text(json.dumps(unsupported, ensure_ascii=False, indent=2), encoding="utf-8")

    total = len(mapping)
    supported = sum(1 for x in mapping if isinstance(x.get("template"), str) and x.get("template"))

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

    supported_targets = [
        str(x.get("target") or "")
        for x in mapping
        if isinstance(x.get("template"), str) and x.get("template") in SUPPORTED_TEMPLATES
    ]
    supported_targets = sorted(set([t for t in supported_targets if t]))

    export_path: Path | None = None
    if export_supported_targets is not None:
        export_path = export_supported_targets
        if not export_path.is_absolute():
            export_path = project_root / export_path
        export_path.parent.mkdir(parents=True, exist_ok=True)
        export_path.write_text("\n".join(supported_targets) + "\n", encoding="utf-8")

    return {
        "map_path": map_path,
        "unsupported_path": unsupported_path,
        "coverage_doc": cov_doc,
        "total": total,
        "supported": supported,
        "unsupported": len(unsupported),
        "template_counts": dict(template_counter),
        "unsupported_reasons": dict(unsupported_counter),
        "supported_targets": supported_targets,
        "supported_targets_path": export_path,
    }


def build_and_write(
    project_root: Path,
    out_derived: Path | None = None,
    export_supported_targets: Path | None = None,
) -> dict[str, Any]:
    mechanics_root = ensure_balatro_mechanics_root(project_root)
    mapping, unsupported = classify_jokers(mechanics_root)
    summary = write_outputs(
        project_root=project_root,
        mechanics_root=mechanics_root,
        mapping=mapping,
        unsupported=unsupported,
        out_derived=out_derived,
        export_supported_targets=export_supported_targets,
    )
    summary["mechanics_root"] = mechanics_root
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build conservative P3 joker template map and coverage reports.")
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--out-derived", default=None, help="Output directory for derived map/unsupported JSON")
    parser.add_argument("--export-supported-targets", default=None, help="Optional path to export supported P3 targets (one per line)")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    project_root = Path(args.project_root).resolve()

    out_derived: Path | None = None
    if args.out_derived:
        out_derived = Path(args.out_derived)
        if not out_derived.is_absolute():
            out_derived = project_root / out_derived

    export_supported_targets: Path | None = None
    if args.export_supported_targets:
        export_supported_targets = Path(args.export_supported_targets)

    summary = build_and_write(
        project_root=project_root,
        out_derived=out_derived,
        export_supported_targets=export_supported_targets,
    )

    print(f"mechanics_root={summary['mechanics_root']}")
    print(f"total={summary['total']} supported={summary['supported']} unsupported={summary['unsupported']}")
    print(f"map={summary['map_path']}")
    print(f"unsupported={summary['unsupported_path']}")
    print(f"coverage_doc={summary['coverage_doc']}")
    if summary.get("supported_targets_path"):
        print(f"supported_targets={summary['supported_targets_path']}")
    else:
        print(f"supported_targets_count={len(summary.get('supported_targets') or [])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

