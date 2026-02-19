from __future__ import annotations

import csv
from functools import lru_cache
from pathlib import Path
from typing import Any

from sim.core.score_basic import evaluate_selected


def _normalize_text(value: Any) -> str:
    text = str(value or "").strip().upper()
    out = []
    last_underscore = False
    for ch in text:
        if ch.isalnum():
            out.append(ch)
            last_underscore = False
            continue
        if not last_underscore:
            out.append("_")
            last_underscore = True
    normalized = "".join(out).strip("_")
    return normalized


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent


def _candidate_csv_paths() -> list[Path]:
    root = _project_root()
    return [
        root / "balatro_mechanics" / "poker_hands.csv",
        root / "Balatro_Mechanics_CSV_UPDATED_20260219" / "final" / "poker_hands.csv",
    ]


@lru_cache(maxsize=1)
def _load_poker_hands_table() -> tuple[dict[str, tuple[float, float]], str | None]:
    table: dict[str, tuple[float, float]] = {}
    source_path: str | None = None

    csv_path: Path | None = None
    for candidate in _candidate_csv_paths():
        if candidate.exists():
            csv_path = candidate
            break

    if csv_path is None:
        return table, None

    source_path = str(csv_path)
    with csv_path.open("r", encoding="utf-8-sig", newline="") as fp:
        reader = csv.DictReader(fp)
        for row in reader:
            hand_name = _normalize_text(row.get("hand_name"))
            if not hand_name:
                continue
            try:
                base_chips = float(row.get("base_chips") or 0.0)
                base_mult = float(row.get("base_mult") or 1.0)
            except Exception:
                continue
            table[hand_name] = (base_chips, base_mult)

    return table, source_path


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


def _extract_hand_cards(state: dict[str, Any]) -> list[dict[str, Any]]:
    hand_cards = (state.get("hand") or {}).get("cards") or []
    out: list[dict[str, Any]] = []
    for card in hand_cards:
        if not isinstance(card, dict):
            continue
        rank, suit = _rank_suit_from_card(card)
        out.append({"rank": rank, "suit": suit})
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


def _resolve_base_row(hand_type: str, table: dict[str, tuple[float, float]]) -> tuple[float, float] | None:
    aliases = {
        "FOUR_KIND": "FOUR_OF_A_KIND",
        "FOUR_OF_KIND": "FOUR_OF_A_KIND",
        "HIGHCARD": "HIGH_CARD",
        "TWO_PAIR": "TWO_PAIR",
        "THREE_KIND": "THREE_OF_A_KIND",
        "THREE_OF_KIND": "THREE_OF_A_KIND",
        "STRAIGHTFLUSH": "STRAIGHT_FLUSH",
        "FULLHOUSE": "FULL_HOUSE",
    }
    key = _normalize_text(hand_type)
    key = aliases.get(key, key)
    return table.get(key)


def compute_expected_for_action(pre_state: dict[str, Any], action: dict[str, Any]) -> dict[str, Any]:
    action_type = str(action.get("action_type") or "").strip().upper()
    if action_type != "PLAY":
        return {
            "available": False,
            "action_type": action_type,
            "reason": "not_play_action",
        }

    hand_cards = _extract_hand_cards(pre_state)
    indices = _action_indices(action)
    if not indices:
        return {
            "available": False,
            "action_type": action_type,
            "reason": "empty_indices",
        }

    if any(i < 0 or i >= len(hand_cards) for i in indices):
        return {
            "available": False,
            "action_type": action_type,
            "reason": "indices_out_of_range",
            "indices": indices,
            "hand_size": len(hand_cards),
        }

    selected = [hand_cards[i] for i in indices]
    hand_type, _, _ = evaluate_selected(selected)

    table, source_csv = _load_poker_hands_table()
    row = _resolve_base_row(hand_type, table)
    if row is None:
        return {
            "available": False,
            "action_type": action_type,
            "hand_type": hand_type,
            "reason": "csv_row_missing" if source_csv else "csv_missing",
            "source_csv": source_csv,
        }

    base_chips, base_mult = row
    return {
        "available": True,
        "action_type": action_type,
        "hand_type": hand_type,
        "base_chips": float(base_chips),
        "base_mult": float(base_mult),
        "score": float(base_chips * base_mult),
        "source_csv": source_csv,
    }
