from __future__ import annotations

import csv
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

from sim.core.rank_chips import sum_rank_chips
from sim.core.score_basic import evaluate_selected_breakdown


def _normalize_text(value: Any) -> str:
    text = str(value or "").strip().upper()
    out: list[str] = []
    last_underscore = False
    for ch in text:
        if ch.isalnum():
            out.append(ch)
            last_underscore = False
            continue
        if not last_underscore:
            out.append("_")
            last_underscore = True
    return "".join(out).strip("_")


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent


def _candidate_csv_path(filename: str) -> tuple[Path | None, str | None]:
    root = _project_root()
    candidates = [
        root / "balatro_mechanics" / filename,
        root / "Balatro_Mechanics_CSV_UPDATED_20260219" / "final" / filename,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate, str(candidate)
    return None, None


HAND_ALIASES = {
    "FOUR_KIND": "FOUR_OF_A_KIND",
    "FOUR_OF_KIND": "FOUR_OF_A_KIND",
    "HIGHCARD": "HIGH_CARD",
    "THREE_KIND": "THREE_OF_A_KIND",
    "THREE_OF_KIND": "THREE_OF_A_KIND",
    "STRAIGHTFLUSH": "STRAIGHT_FLUSH",
    "FULLHOUSE": "FULL_HOUSE",
}


def _normalize_hand_type(value: Any) -> str:
    key = _normalize_text(value)
    return HAND_ALIASES.get(key, key)


@lru_cache(maxsize=1)
def _load_poker_hands_table() -> tuple[dict[str, tuple[float, float]], str | None]:
    table: dict[str, tuple[float, float]] = {}
    csv_path, source_path = _candidate_csv_path("poker_hands.csv")
    if csv_path is None:
        return table, None

    with csv_path.open("r", encoding="utf-8-sig", newline="") as fp:
        reader = csv.DictReader(fp)
        for row in reader:
            hand_name = _normalize_hand_type(row.get("hand_name"))
            if not hand_name:
                continue
            try:
                base_chips = float(row.get("base_chips") or 0.0)
                base_mult = float(row.get("base_mult") or 1.0)
            except Exception:
                continue
            table[hand_name] = (base_chips, base_mult)

    return table, source_path


@lru_cache(maxsize=1)
def _load_planet_table() -> tuple[dict[str, tuple[float, float]], dict[str, tuple[float, float]], str | None]:
    by_hand: dict[str, tuple[float, float]] = {}
    by_planet_key: dict[str, tuple[float, float]] = {}

    csv_path, source_path = _candidate_csv_path("planet_cards.csv")
    if csv_path is None:
        return by_hand, by_planet_key, None

    pattern = re.compile(r"\+(\d+)\s*Mult\s+and\s+\+(\d+)\s*Chips\s+to\s+(.+)$", re.IGNORECASE)

    with csv_path.open("r", encoding="utf-8-sig", newline="") as fp:
        reader = csv.DictReader(fp)
        for row in reader:
            card_name = str(row.get("card_name") or "").strip()
            effect = str(row.get("effect_summary") or "").strip()
            m = pattern.search(effect)
            if not m:
                continue
            mult_bonus = float(m.group(1))
            chips_bonus = float(m.group(2))
            hand_name = _normalize_hand_type(m.group(3))
            if not hand_name:
                continue
            by_hand[hand_name] = (chips_bonus, mult_bonus)
            if card_name:
                planet_key = "c_" + _normalize_text(card_name).lower()
                by_planet_key[planet_key] = (chips_bonus, mult_bonus)

    return by_hand, by_planet_key, source_path


@lru_cache(maxsize=1)
def _load_modifier_sources() -> dict[str, str | None]:
    out: dict[str, str | None] = {
        "enhancements": None,
        "editions": None,
        "seals": None,
    }
    for filename, key in [
        ("card_modifiers_enhancements.csv", "enhancements"),
        ("card_modifiers_editions.csv", "editions"),
        ("card_modifiers_seals.csv", "seals"),
    ]:
        _, src = _candidate_csv_path(filename)
        out[key] = src
    return out


def _rank_suit_from_card(card: dict[str, Any]) -> tuple[str, str]:
    value = card.get("value") if isinstance(card.get("value"), dict) else {}
    rank = str(card.get("rank") or value.get("rank") or "").strip().upper()
    suit = str(card.get("suit") or value.get("suit") or "").strip().upper()

    if (not rank or not suit) and card.get("key"):
        key = str(card.get("key") or "")
        if "_" in key:
            s, r = key.split("_", 1)
            if not suit:
                suit = s.strip().upper()
            if not rank:
                rank = r.strip().upper()

    if rank == "T":
        rank = "10"
    if suit:
        suit = suit[:1]
    return rank, suit


def _extract_modifier_map(card: dict[str, Any]) -> dict[str, str]:
    raw_modifier = card.get("modifier")
    out: dict[str, str] = {}

    if isinstance(raw_modifier, dict):
        for k, v in raw_modifier.items():
            key = _normalize_text(k)
            val = _normalize_text(v)
            if key and val:
                out[key] = val
    elif isinstance(raw_modifier, list):
        for item in raw_modifier:
            val = _normalize_text(item)
            if val:
                out[val] = val

    label = _normalize_text(card.get("label"))
    if label and "BONUS" in label and "ENHANCEMENT" not in out:
        out["ENHANCEMENT"] = "BONUS"

    return out


def _extract_hand_cards(state: dict[str, Any]) -> list[dict[str, Any]]:
    hand_cards = (state.get("hand") or {}).get("cards") or []
    out: list[dict[str, Any]] = []
    for card in hand_cards:
        if not isinstance(card, dict):
            continue
        rank, suit = _rank_suit_from_card(card)
        out.append(
            {
                "rank": rank,
                "suit": suit,
                "key": str(card.get("key") or "").strip().upper(),
                "modifier_map": _extract_modifier_map(card),
                "raw": card,
            }
        )
    return out


def _action_indices(action: dict[str, Any]) -> list[int]:
    params = action.get("params") if isinstance(action.get("params"), dict) else {}
    try:
        index_base = int(params.get("index_base", 0))
    except Exception:
        index_base = 0

    indices = [int(i) for i in (action.get("indices") or [])]
    if index_base == 1:
        return [i - 1 for i in indices]
    return indices


def _planet_context(action: dict[str, Any], pre_state: dict[str, Any], hand_type: str) -> tuple[int, str | None]:
    ctx = action.get("expected_context") if isinstance(action.get("expected_context"), dict) else {}
    planet = ctx.get("planet") if isinstance(ctx.get("planet"), dict) else {}

    level_gain = 0
    planet_key = None

    if planet:
        try:
            level_gain = int(planet.get("levels") or 0)
        except Exception:
            level_gain = 0
        planet_key = str(planet.get("card_key") or "").strip().lower() or None

    if level_gain <= 0:
        hands = pre_state.get("hands") if isinstance(pre_state.get("hands"), dict) else {}
        for name, info in hands.items():
            if _normalize_hand_type(name) != _normalize_hand_type(hand_type):
                continue
            if isinstance(info, dict):
                try:
                    level = int(info.get("level") or 1)
                except Exception:
                    level = 1
                level_gain = max(0, level - 1)
            break

    return level_gain, planet_key


def _modifier_delta(scoring_cards: list[dict[str, Any]]) -> tuple[float, float, float, list[str]]:
    chips_add = 0.0
    mult_add = 0.0
    mult_scale = 1.0
    partial_reasons: list[str] = []

    for card in scoring_cards:
        mod = card.get("modifier_map") if isinstance(card.get("modifier_map"), dict) else {}
        enh = _normalize_text(mod.get("ENHANCEMENT"))
        edi = _normalize_text(mod.get("EDITION"))
        seal = _normalize_text(mod.get("SEAL"))

        if enh == "BONUS":
            chips_add += 30.0
        elif enh == "MULT":
            mult_add += 4.0
        elif enh == "GLASS":
            mult_scale *= 2.0
        elif enh in {"STONE", "STEEL", "LUCKY", "WILD", "GOLD"}:
            partial_reasons.append(f"enhancement_{enh.lower()}_partial")

        if edi == "FOIL":
            chips_add += 50.0
        elif edi == "HOLO":
            mult_add += 10.0
        elif edi == "POLYCHROME":
            mult_scale *= 1.5
        elif edi == "NEGATIVE":
            partial_reasons.append("edition_negative_not_scoring")

        if seal in {"RED", "GOLD", "BLUE", "PURPLE"}:
            partial_reasons.append(f"seal_{seal.lower()}_partial")

    # Keep order stable / deduplicated
    seen = set()
    compact: list[str] = []
    for reason in partial_reasons:
        if reason not in seen:
            seen.add(reason)
            compact.append(reason)

    return chips_add, mult_add, mult_scale, compact


def compute_expected_for_action(pre_state: dict[str, Any], action: dict[str, Any]) -> dict[str, Any]:
    action_type = str(action.get("action_type") or "").strip().upper()
    if action_type != "PLAY":
        return {
            "available": False,
            "partial": False,
            "action_type": action_type,
            "reason": "not_play_action",
        }

    hand_cards = _extract_hand_cards(pre_state)
    indices = _action_indices(action)
    if not indices:
        return {
            "available": False,
            "partial": False,
            "action_type": action_type,
            "reason": "empty_indices",
        }

    if any(i < 0 or i >= len(hand_cards) for i in indices):
        return {
            "available": False,
            "partial": False,
            "action_type": action_type,
            "reason": "indices_out_of_range",
            "indices": indices,
            "hand_size": len(hand_cards),
        }

    selected = [hand_cards[i] for i in indices]
    score_breakdown = evaluate_selected_breakdown(selected)
    hand_type = _normalize_hand_type(score_breakdown.get("hand_type"))

    poker_table, poker_src = _load_poker_hands_table()
    base_row = poker_table.get(hand_type)
    if base_row is None:
        return {
            "available": False,
            "partial": False,
            "action_type": action_type,
            "hand_type": hand_type,
            "reason": "csv_row_missing" if poker_src else "csv_missing",
            "source_csv": poker_src,
        }

    base_chips, base_mult = base_row
    scoring_cards = score_breakdown.get("scoring_cards") if isinstance(score_breakdown.get("scoring_cards"), list) else selected
    rank_chips = float(sum_rank_chips(scoring_cards))

    core_score = (float(base_chips) + rank_chips) * float(base_mult)

    planet_by_hand, planet_by_key, planet_src = _load_planet_table()
    planet_levels, planet_key = _planet_context(action, pre_state, hand_type)
    planet_bonus_chips = 0.0
    planet_bonus_mult = 0.0
    if planet_levels > 0:
        if planet_key and planet_key in planet_by_key:
            chips_bonus, mult_bonus = planet_by_key[planet_key]
            planet_bonus_chips = float(chips_bonus) * planet_levels
            planet_bonus_mult = float(mult_bonus) * planet_levels
        elif hand_type in planet_by_hand:
            chips_bonus, mult_bonus = planet_by_hand[hand_type]
            planet_bonus_chips = float(chips_bonus) * planet_levels
            planet_bonus_mult = float(mult_bonus) * planet_levels

    mod_chips, mod_mult_add, mod_mult_scale, partial_reasons = _modifier_delta(scoring_cards)

    total_chips_term = float(base_chips) + rank_chips + planet_bonus_chips + mod_chips
    total_mult_term = float(base_mult) + planet_bonus_mult + mod_mult_add
    total_score = total_chips_term * total_mult_term * mod_mult_scale

    partial = bool(partial_reasons)
    source_csvs = [src for src in [poker_src, planet_src, _load_modifier_sources().get("enhancements"), _load_modifier_sources().get("editions"), _load_modifier_sources().get("seals")] if src]

    return {
        "available": True,
        "partial": partial,
        "partial_reasons": partial_reasons,
        "action_type": action_type,
        "hand_type": hand_type,
        "base_chips": float(base_chips),
        "base_mult": float(base_mult),
        "rank_chips": float(rank_chips),
        "planet_levels": int(planet_levels),
        "planet_card_key": planet_key,
        "planet_bonus_chips": float(planet_bonus_chips),
        "planet_bonus_mult": float(planet_bonus_mult),
        "modifier_bonus_chips": float(mod_chips),
        "modifier_bonus_mult_add": float(mod_mult_add),
        "modifier_bonus_mult_scale": float(mod_mult_scale),
        "score_core": float(core_score),
        "score": float(total_score),
        "source_csv": source_csvs,
    }
