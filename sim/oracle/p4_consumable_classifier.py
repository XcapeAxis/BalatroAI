from __future__ import annotations

if __package__ is None or __package__ == "":
    import sys
    from pathlib import Path as _Path

    sys.path.append(str(_Path(__file__).resolve().parent.parent.parent))

import argparse
import csv
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sim.oracle.p3_joker_classifier import ensure_balatro_mechanics_root

SUPPORTED_TEMPLATES = {
    "planet_hand_level_up",
    "tarot_enhance_single",
    "tarot_enhance_double",
    "tarot_rank_up_to2",
    "tarot_convert_suit_upto3",
    "tarot_destroy_upto2",
    "tarot_copy_left_to_right",
    "spectral_seal_single",
    "spectral_black_hole_all_hands_level_up",
    "spectral_copy_selected_card_twice",
    "spectral_all_hand_random_suit",
    "spectral_all_hand_random_rank_minus_hand_size",
}


HAND_ALIASES = {
    "FOUR OF A KIND": "FOUR_OF_A_KIND",
    "THREE OF A KIND": "THREE_OF_A_KIND",
    "TWO PAIR": "TWO_PAIR",
    "STRAIGHT FLUSH": "STRAIGHT_FLUSH",
    "FULL HOUSE": "FULL_HOUSE",
    "HIGH CARD": "HIGH_CARD",
    "FIVE OF A KIND": "FIVE_OF_A_KIND",
    "FLUSH HOUSE": "FLUSH_HOUSE",
    "FLUSH FIVE": "FLUSH_FIVE",
}


def _slug(text: str) -> str:
    t = re.sub(r"\([^)]*\)", "", str(text or ""))
    t = t.strip().lower()
    t = re.sub(r"[^a-z0-9]+", "_", t)
    t = re.sub(r"_+", "_", t).strip("_")
    return t


def _normalize_hand_name(text: str) -> str:
    t = str(text or "").strip().upper().replace("_", " ")
    t = re.sub(r"\s+", " ", t)
    return HAND_ALIASES.get(t, t.replace(" ", "_"))


def _key_candidates(card_name: str) -> list[str]:
    full_slug = _slug(card_name)
    short_slug = full_slug
    if short_slug.startswith("the_"):
        short_slug = short_slug[4:]

    cands: list[str] = []
    for slug in (short_slug, full_slug):
        if slug:
            cands.append(f"c_{slug}")

    if "hierophant" in full_slug:
        cands.append("c_heirophant")
        cands.append("c_hierophant")

    dedup: list[str] = []
    seen: set[str] = set()
    for c in cands:
        if c not in seen:
            seen.add(c)
            dedup.append(c)
    return dedup


def _target_name(set_type: str, key_guess: str) -> str:
    suffix = str(key_guess or "").strip().lower()
    if suffix.startswith("c_"):
        suffix = suffix[2:]
    suffix = re.sub(r"[^a-z0-9_]+", "_", suffix)
    suffix = re.sub(r"_+", "_", suffix).strip("_")
    return f"p4_{set_type}_{suffix}"


def _read_csv(path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as fp:
        reader = csv.DictReader(fp)
        for row in reader:
            rows.append({k: str(v or "") for k, v in row.items()})
    return rows


def _classify_planet(row: dict[str, str]) -> tuple[str | None, float, dict[str, Any], list[str], str | None]:
    effect = row.get("effect_summary", "")
    m = re.search(r"\+(\d+)\s*Mult\s+and\s+\+(\d+)\s*Chips\s+to\s+(.+)$", effect, flags=re.IGNORECASE)
    if not m:
        return None, 0.0, {}, [], "unrecognized_planet_effect"

    mult_bonus = float(m.group(1))
    chips_bonus = float(m.group(2))
    hand_type = _normalize_hand_name(m.group(3))
    params = {
        "cards_required": 0,
        "hand_type": hand_type,
        "planet_bonus_mult": mult_bonus,
        "planet_bonus_chips": chips_bonus,
    }
    return "planet_hand_level_up", 0.99, params, ["planet:+Mult/+Chips to hand type"], None


def _classify_tarot(row: dict[str, str]) -> tuple[str | None, float, dict[str, Any], list[str], str | None]:
    name_slug = _slug(row.get("card_name", ""))
    effect = row.get("effect_summary", "").lower()

    allow_random = {"sigil", "ouija"}
    if ("random" in effect or "chance" in effect) and name_slug not in allow_random:
        return None, 0.0, {}, [], "probabilistic_or_random"
    if "$" in effect or "money" in effect or "sell value" in effect:
        return None, 0.0, {}, [], "economy_or_shop_related"

    if name_slug.startswith("death"):
        return "tarot_copy_left_to_right", 0.97, {"cards_required": 2}, ["death:convert left card into right card"], None
    if name_slug.startswith("strength"):
        return "tarot_rank_up_to2", 0.97, {"cards_required": 2}, ["strength:increase rank of up to 2 selected cards"], None
    if name_slug in {"justice", "chariot", "devil", "lovers", "tower"}:
        return "tarot_enhance_single", 0.97, {"cards_required": 1, "tarot_name": name_slug}, ["single selected card enhancement"], None
    if name_slug in {"empress", "hierophant", "magician", "heirophant"}:
        return "tarot_enhance_double", 0.97, {"cards_required": 2, "tarot_name": name_slug}, ["two selected cards enhancement"], None
    if name_slug in {"moon", "star", "sun", "world"}:
        suit = {"moon": "C", "star": "D", "sun": "H", "world": "S"}[name_slug]
        return "tarot_convert_suit_upto3", 0.97, {"cards_required": 3, "target_suit": suit}, ["convert up to 3 selected cards to fixed suit"], None
    if "hanged_man" in name_slug:
        return "tarot_destroy_upto2", 0.96, {"cards_required": 2}, ["destroy up to 2 selected cards"], None

    return None, 0.0, {}, [], "unsupported_tarot_complex_or_non_deterministic"


def _classify_spectral(row: dict[str, str]) -> tuple[str | None, float, dict[str, Any], list[str], str | None]:
    name_slug = _slug(row.get("card_name", ""))
    effect = row.get("effect_summary", "").lower()

    allow_random = {"sigil", "ouija"}
    if ("random" in effect or "chance" in effect) and name_slug not in allow_random:
        return None, 0.0, {}, [], "probabilistic_or_random"
    if "$" in effect or "money" in effect:
        return None, 0.0, {}, [], "economy_or_shop_related"
    if any(token in effect for token in ("this run", "for each time", "destroys the others", "must have room")):
        if name_slug not in {"black_hole", "cryptid", "deja_vu", "medium", "talisman", "trance"}:
            return None, 0.0, {}, [], "cross_round_or_complex_state"

    if name_slug == "black_hole":
        return (
            "spectral_black_hole_all_hands_level_up",
            0.99,
            {"cards_required": 0},
            ["black_hole:upgrade every poker hand by one level"],
            None,
        )
    if name_slug in {"deja_vu", "medium", "talisman", "trance"}:
        return "spectral_seal_single", 0.98, {"cards_required": 1, "spectral_name": name_slug}, ["single selected card seal"], None
    if name_slug == "cryptid":
        return "spectral_copy_selected_card_twice", 0.97, {"cards_required": 1}, ["copy selected hand card twice"], None
    if name_slug == "sigil":
        return "spectral_all_hand_random_suit", 0.92, {"cards_required": 0}, ["convert all hand cards to a single random suit"], None
    if name_slug == "ouija":
        return "spectral_all_hand_random_rank_minus_hand_size", 0.90, {"cards_required": 0}, ["convert all hand cards to a single random rank and reduce hand size"], None

    return None, 0.0, {}, [], "unsupported_spectral_complex_or_non_deterministic"


def classify_consumables(mechanics_root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    unsupported: list[dict[str, Any]] = []

    spec = [
        ("planet", mechanics_root / "planet_cards.csv", _classify_planet),
        ("tarot", mechanics_root / "tarot_cards.csv", _classify_tarot),
        ("spectral", mechanics_root / "spectral_cards.csv", _classify_spectral),
    ]

    for set_type, csv_path, classifier in spec:
        if not csv_path.exists():
            raise FileNotFoundError(f"missing CSV: {csv_path}")

        for row in _read_csv(csv_path):
            card_name = row.get("card_name", "").strip()
            key_cands = _key_candidates(card_name)
            key_guess = key_cands[0] if key_cands else ""
            target = _target_name(set_type, key_guess)

            template, confidence, params, rules, reason = classifier(row)
            entry = {
                "set_type": set_type,
                "card_name": card_name,
                "consumable_key": key_guess,
                "key_candidates": key_cands,
                "template": template,
                "confidence": confidence,
                "matched_rules": rules,
                "params": params,
                "target": target,
                "raw_fields_used": {
                    "effect_summary": row.get("effect_summary", ""),
                    "primary_impact": row.get("primary_impact", ""),
                    "constraints_or_costs": row.get("constraints_or_costs", ""),
                    "synergies_or_notes": row.get("synergies_or_notes", ""),
                },
            }
            rows.append(entry)

            if template is None:
                unsupported.append(
                    {
                        "set_type": set_type,
                        "card_name": card_name,
                        "consumable_key": key_guess,
                        "target": target,
                        "reason": reason or "unknown",
                        "effect_summary": row.get("effect_summary", ""),
                        "primary_impact": row.get("primary_impact", ""),
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

    map_path = derived_dir / "p4_consumable_template_map.json"
    unsupported_path = derived_dir / "p4_consumable_template_unsupported.json"
    supported_targets_path = export_supported_targets if export_supported_targets is not None else (derived_dir / "p4_supported_targets.txt")

    if not supported_targets_path.is_absolute():
        supported_targets_path = project_root / supported_targets_path
    supported_targets_path.parent.mkdir(parents=True, exist_ok=True)

    map_path.write_text(json.dumps(mapping, ensure_ascii=False, indent=2), encoding="utf-8")
    unsupported_path.write_text(json.dumps(unsupported, ensure_ascii=False, indent=2), encoding="utf-8")

    total = len(mapping)
    supported_entries = [x for x in mapping if isinstance(x.get("template"), str) and x.get("template") in SUPPORTED_TEMPLATES]
    supported = len(supported_entries)

    template_counter: Counter[str] = Counter()
    set_counter: Counter[str] = Counter()
    for x in supported_entries:
        template_counter[str(x.get("template"))] += 1
        set_counter[str(x.get("set_type"))] += 1

    unsupported_counter: Counter[str] = Counter(str(x.get("reason") or "unknown") for x in unsupported)

    supported_targets = sorted(set(str(x.get("target") or "") for x in supported_entries if str(x.get("target") or "")))
    supported_targets_path.write_text("\n".join(supported_targets) + "\n", encoding="utf-8")

    coverage_doc = project_root / "docs" / "COVERAGE_P4_CONSUMABLES.md"
    coverage_doc.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = []
    lines.append("# P4 Consumable Template Coverage")
    lines.append("")
    lines.append(f"- Generated at: `{datetime.now(timezone.utc).isoformat()}`")
    lines.append(f"- Total consumables: **{total}**")
    lines.append(f"- Supported (conservative): **{supported}**")
    lines.append(f"- Unsupported: **{len(unsupported)}**")
    lines.append("")
    lines.append("## Supported By Set")
    for key in ("planet", "tarot", "spectral"):
        lines.append(f"- `{key}`: {set_counter.get(key, 0)}")
    lines.append("")
    lines.append("## Supported Template Counts")
    for tpl in sorted(SUPPORTED_TEMPLATES):
        lines.append(f"- `{tpl}`: {template_counter.get(tpl, 0)}")
    lines.append("")
    lines.append("## Unsupported Top Reasons")
    for reason, count in unsupported_counter.most_common(20):
        lines.append(f"- `{reason}`: {count}")

    coverage_doc.write_text("\n".join(lines) + "\n", encoding="utf-8")

    status_doc = project_root / "docs" / "COVERAGE_P4_STATUS.md"
    status_doc.write_text(
        "\n".join(
            [
                "# P4 Consumable Coverage Status",
                "",
                f"- total: **{total}**",
                f"- supported: **{supported}**",
                f"- unsupported: **{len(unsupported)}**",
                f"- supported_targets_file: `{supported_targets_path}`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    return {
        "total": total,
        "supported": supported,
        "unsupported": len(unsupported),
        "map_path": map_path,
        "unsupported_path": unsupported_path,
        "supported_targets_path": supported_targets_path,
        "coverage_doc": coverage_doc,
        "status_doc": status_doc,
        "template_counts": dict(template_counter),
        "unsupported_reasons": dict(unsupported_counter),
    }


def build_and_write(
    project_root: Path,
    out_derived: Path | None = None,
    export_supported_targets: Path | None = None,
) -> dict[str, Any]:
    mechanics_root = ensure_balatro_mechanics_root(project_root)
    mapping, unsupported = classify_consumables(mechanics_root)
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
    parser = argparse.ArgumentParser(description="Build conservative P4 consumable template map and coverage docs.")
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--out-derived", default="balatro_mechanics/derived")
    parser.add_argument("--export-supported-targets", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    project_root = Path(args.project_root).resolve()

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
    print(f"supported_targets={summary['supported_targets_path']}")
    print(f"coverage_doc={summary['coverage_doc']}")
    print(f"status_doc={summary['status_doc']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
