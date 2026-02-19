from __future__ import annotations

from typing import Any

from sim.core.rank_chips import normalize_rank, rank_chip, rank_from_card

ODD_RANKS = {"A", "9", "7", "5", "3"}
EVEN_RANKS = {"10", "8", "6", "4", "2"}
FACE_RANKS = {"J", "Q", "K"}


def _rank_from_card(card: dict[str, Any]) -> str:
    rank = normalize_rank(card.get("rank"))
    if rank:
        return rank
    return rank_from_card(card)


def _suit_from_card(card: dict[str, Any]) -> str:
    suit = str(card.get("suit") or "").strip().upper()
    if suit:
        return suit[:1]
    key = str(card.get("key") or "").strip().upper()
    if "_" in key:
        s, _ = key.split("_", 1)
        return s[:1]
    return ""


def _normalize_joker_key(raw: Any) -> str:
    return str(raw or "").strip().lower()


def _default_joker_spec_from_key(key: str) -> dict[str, Any] | None:
    table: dict[str, dict[str, Any]] = {
        "j_joker": {"kind": "flat_mult", "mult_add": 4.0},
        "j_greedy_joker": {"kind": "suit_scoring_mult", "suit": "D", "mult_add_per_card": 3.0},
        "j_lusty_joker": {"kind": "suit_scoring_mult", "suit": "H", "mult_add_per_card": 3.0},
        "j_wrathful_joker": {"kind": "suit_scoring_mult", "suit": "S", "mult_add_per_card": 3.0},
        "j_gluttenous_joker": {"kind": "suit_scoring_mult", "suit": "C", "mult_add_per_card": 3.0},
        "j_banner": {"kind": "remaining_discards_chips", "chips_add_per_discard": 30.0},
        "j_odd_todd": {"kind": "odd_scoring_chips", "chips_add_per_card": 31.0},
        "j_even_steven": {"kind": "even_scoring_mult", "mult_add_per_card": 4.0},
        "j_fibonacci": {
            "kind": "rank_set_scoring_mult",
            "ranks": ["A", "2", "3", "5", "8"],
            "mult_add_per_card": 8.0,
        },
        "j_scary_face": {"kind": "face_scoring_chips", "chips_add_per_card": 30.0},
        "j_photograph": {"kind": "first_face_xmult", "mult_scale": 2.0},
        "j_baron": {"kind": "held_rank_xmult", "rank": "K", "mult_scale_per_card": 1.5},
    }
    raw = table.get(key)
    if raw is None:
        return None
    out = dict(raw)
    out["key"] = key
    return out


def _state_jokers(pre_state: dict[str, Any]) -> list[dict[str, Any]]:
    raw = pre_state.get("jokers")
    cards: list[dict[str, Any]] = []
    if isinstance(raw, dict):
        src = raw.get("cards")
        if isinstance(src, list):
            cards = [x for x in src if isinstance(x, dict)]
    elif isinstance(raw, list):
        cards = [x for x in raw if isinstance(x, dict)]
    out: list[dict[str, Any]] = []
    for card in cards:
        key = _normalize_joker_key(card.get("key"))
        if not key:
            continue
        spec = _default_joker_spec_from_key(key)
        if spec is None:
            continue
        out.append(spec)
    return out


def _context_jokers(action: dict[str, Any]) -> list[dict[str, Any]]:
    ctx = action.get("expected_context") if isinstance(action.get("expected_context"), dict) else {}
    jokers = ctx.get("jokers") if isinstance(ctx.get("jokers"), list) else []
    out: list[dict[str, Any]] = []
    for item in jokers:
        if not isinstance(item, dict):
            continue
        key = _normalize_joker_key(item.get("key") or item.get("joker_key"))
        kind = str(item.get("kind") or "").strip()
        if key and not kind:
            spec = _default_joker_spec_from_key(key)
            if spec is not None:
                merged = dict(spec)
                merged.update(item)
                item = merged
        elif key:
            item = dict(item)
            item["key"] = key
        out.append(dict(item))
    return out


def _active_jokers(action: dict[str, Any], pre_state: dict[str, Any]) -> list[dict[str, Any]]:
    ctx_jokers = _context_jokers(action)
    if ctx_jokers:
        return ctx_jokers
    return _state_jokers(pre_state)


def compute_joker_delta(
    *,
    scoring_cards: list[dict[str, Any]],
    held_cards: list[dict[str, Any]],
    pre_state: dict[str, Any],
    action: dict[str, Any],
    hand_type: str,
) -> tuple[float, float, float, list[dict[str, Any]], list[str]]:
    chips_add = 0.0
    mult_add = 0.0
    mult_scale = 1.0
    breakdown: list[dict[str, Any]] = []
    partial_reasons: list[str] = []

    scoring_ranks = [_rank_from_card(c) for c in scoring_cards]
    scoring_suits = [_suit_from_card(c) for c in scoring_cards]
    held_ranks = [_rank_from_card(c) for c in held_cards]

    for joker in _active_jokers(action, pre_state):
        key = _normalize_joker_key(joker.get("key"))
        kind = str(joker.get("kind") or "").strip().lower()

        entry = {
            "key": key,
            "kind": kind,
            "chips_delta": 0.0,
            "mult_add": 0.0,
            "mult_scale": 1.0,
        }

        if kind == "flat_mult":
            inc = float(joker.get("mult_add") or 0.0)
            mult_add += inc
            entry["mult_add"] = inc

        elif kind == "flat_chips":
            inc = float(joker.get("chips_add") or 0.0)
            chips_add += inc
            entry["chips_delta"] = inc

        elif kind == "suit_scoring_mult":
            target_suit = str(joker.get("suit") or "").strip().upper()[:1]
            per = float(joker.get("mult_add_per_card") or 0.0)
            cnt = sum(1 for s in scoring_suits if s == target_suit)
            inc = float(cnt) * per
            mult_add += inc
            entry["count"] = cnt
            entry["mult_add"] = inc
            entry["suit"] = target_suit

        elif kind == "suit_scoring_chips":
            target_suit = str(joker.get("suit") or "").strip().upper()[:1]
            per = float(joker.get("chips_add_per_card") or 0.0)
            cnt = sum(1 for s in scoring_suits if s == target_suit)
            inc = float(cnt) * per
            chips_add += inc
            entry["count"] = cnt
            entry["chips_delta"] = inc
            entry["suit"] = target_suit

        elif kind == "remaining_discards_chips":
            round_info = pre_state.get("round") if isinstance(pre_state.get("round"), dict) else {}
            discards_left = int(round_info.get("discards_left") or 0)
            per = float(joker.get("chips_add_per_discard") or 0.0)
            inc = float(discards_left) * per
            chips_add += inc
            entry["count"] = discards_left
            entry["chips_delta"] = inc

        elif kind == "odd_scoring_chips":
            per = float(joker.get("chips_add_per_card") or 0.0)
            cnt = sum(1 for r in scoring_ranks if r in ODD_RANKS)
            inc = float(cnt) * per
            chips_add += inc
            entry["count"] = cnt
            entry["chips_delta"] = inc

        elif kind == "even_scoring_mult":
            per = float(joker.get("mult_add_per_card") or 0.0)
            cnt = sum(1 for r in scoring_ranks if r in EVEN_RANKS)
            inc = float(cnt) * per
            mult_add += inc
            entry["count"] = cnt
            entry["mult_add"] = inc

        elif kind == "rank_set_scoring_mult":
            ranks = {str(x).strip().upper() for x in (joker.get("ranks") or [])}
            per = float(joker.get("mult_add_per_card") or 0.0)
            cnt = sum(1 for r in scoring_ranks if r in ranks)
            inc = float(cnt) * per
            mult_add += inc
            entry["count"] = cnt
            entry["mult_add"] = inc
            entry["ranks"] = sorted(ranks)

        elif kind == "face_scoring_chips":
            per = float(joker.get("chips_add_per_card") or 0.0)
            cnt = sum(1 for r in scoring_ranks if r in FACE_RANKS)
            inc = float(cnt) * per
            chips_add += inc
            entry["count"] = cnt
            entry["chips_delta"] = inc

        elif kind == "face_scoring_mult":
            per = float(joker.get("mult_add_per_card") or 0.0)
            cnt = sum(1 for r in scoring_ranks if r in FACE_RANKS)
            inc = float(cnt) * per
            mult_add += inc
            entry["count"] = cnt
            entry["mult_add"] = inc

        elif kind == "first_face_xmult":
            scale = float(joker.get("mult_scale") or 1.0)
            has_face = any(r in FACE_RANKS for r in scoring_ranks)
            if has_face and scale > 0:
                mult_scale *= scale
                entry["mult_scale"] = scale
                entry["count"] = 1

        elif kind == "held_rank_xmult":
            target_rank = str(joker.get("rank") or "").strip().upper()
            per = float(joker.get("mult_scale_per_card") or 1.0)
            cnt = sum(1 for r in held_ranks if r == target_rank)
            if cnt > 0 and per > 0:
                scale = per ** cnt
                mult_scale *= scale
                entry["mult_scale"] = scale
                entry["count"] = cnt
                entry["rank"] = target_rank

        elif kind == "hand_type_mult_add":
            target = str(joker.get("hand_type") or "").strip().upper()
            if target and target == str(hand_type or "").strip().upper():
                inc = float(joker.get("mult_add") or 0.0)
                mult_add += inc
                entry["mult_add"] = inc

        elif kind == "hand_type_chips_add":
            target = str(joker.get("hand_type") or "").strip().upper()
            if target and target == str(hand_type or "").strip().upper():
                inc = float(joker.get("chips_add") or 0.0)
                chips_add += inc
                entry["chips_delta"] = inc

        elif kind == "hand_type_xmult":
            target = str(joker.get("hand_type") or "").strip().upper()
            scale = float(joker.get("mult_scale") or 1.0)
            if target and target == str(hand_type or "").strip().upper() and scale > 0:
                mult_scale *= scale
                entry["mult_scale"] = scale

        elif kind == "rank_set_scoring_chips_mult":
            ranks = {str(x).strip().upper() for x in (joker.get("ranks") or [])}
            chips_per = float(joker.get("chips_add_per_card") or 0.0)
            mult_per = float(joker.get("mult_add_per_card") or 0.0)
            cnt = sum(1 for r in scoring_ranks if r in ranks)
            chips_inc = float(cnt) * chips_per
            mult_inc = float(cnt) * mult_per
            chips_add += chips_inc
            mult_add += mult_inc
            entry["count"] = cnt
            entry["chips_delta"] = chips_inc
            entry["mult_add"] = mult_inc
            entry["ranks"] = sorted(ranks)

        elif kind == "held_rank_mult_add":
            target_rank = str(joker.get("rank") or "").strip().upper()
            per = float(joker.get("mult_add_per_card") or 0.0)
            cnt = sum(1 for r in held_ranks if r == target_rank)
            inc = float(cnt) * per
            mult_add += inc
            entry["count"] = cnt
            entry["mult_add"] = inc
            entry["rank"] = target_rank

        elif kind == "held_lowest_rank_mult_add":
            scale = float(joker.get("scale") or 2.0)
            if held_cards:
                lowest = min(rank_chip(_rank_from_card(card)) for card in held_cards)
                inc = float(lowest) * scale
                mult_add += inc
                entry["mult_add"] = inc
                entry["lowest_rank_chip"] = lowest
                entry["scale"] = scale

        elif kind == "all_held_suits_xmult":
            allowed = {str(x).strip().upper()[:1] for x in (joker.get("allowed_suits") or []) if str(x).strip()}
            scale = float(joker.get("mult_scale") or 1.0)
            held_suits = [_suit_from_card(c) for c in held_cards]
            if held_suits and allowed and all(s in allowed for s in held_suits) and scale > 0:
                mult_scale *= scale
                entry["mult_scale"] = scale
                entry["held_suits"] = held_suits
                entry["allowed_suits"] = sorted(allowed)

        elif kind == "scoring_has_suit_and_other_xmult":
            req = str(joker.get("required_suit") or "").strip().upper()[:1]
            scale = float(joker.get("mult_scale") or 1.0)
            has_req = any(s == req for s in scoring_suits)
            has_other = any(s != req for s in scoring_suits if s)
            if has_req and has_other and scale > 0:
                mult_scale *= scale
                entry["mult_scale"] = scale
                entry["required_suit"] = req

        elif kind == "scoring_has_all_suits_xmult":
            required = {str(x).strip().upper()[:1] for x in (joker.get("required_suits") or []) if str(x).strip()}
            scale = float(joker.get("mult_scale") or 1.0)
            if required and all(s in scoring_suits for s in required) and scale > 0:
                mult_scale *= scale
                entry["mult_scale"] = scale
                entry["required_suits"] = sorted(required)

        elif kind == "discards_left_eq_mult_add":
            target_left = int(joker.get("discards_left") or 0)
            round_info = pre_state.get("round") if isinstance(pre_state.get("round"), dict) else {}
            cur_left = int(round_info.get("discards_left") or 0)
            if cur_left == target_left:
                inc = float(joker.get("mult_add") or 0.0)
                mult_add += inc
                entry["mult_add"] = inc
            entry["discards_left"] = cur_left
            entry["target_discards_left"] = target_left

        elif kind == "hands_left_eq_xmult":
            target_left = int(joker.get("hands_left") or 1)
            round_info = pre_state.get("round") if isinstance(pre_state.get("round"), dict) else {}
            cur_left = int(round_info.get("hands_left") or 0)
            scale = float(joker.get("mult_scale") or 1.0)
            if cur_left == target_left and scale > 0:
                mult_scale *= scale
                entry["mult_scale"] = scale
            entry["hands_left"] = cur_left
            entry["target_hands_left"] = target_left

        elif kind == "hand_size_lte_mult_add":
            max_cards = int(joker.get("max_cards") or 3)
            played_n = len(scoring_cards)
            if played_n <= max_cards:
                inc = float(joker.get("mult_add") or 0.0)
                mult_add += inc
                entry["mult_add"] = inc
            entry["played_cards"] = played_n
            entry["max_cards"] = max_cards

        elif kind == "rank_set_scoring_xmult":
            ranks = {str(x).strip().upper() for x in (joker.get("ranks") or [])}
            per = float(joker.get("mult_scale_per_card") or 1.0)
            cnt = sum(1 for r in scoring_ranks if r in ranks)
            if cnt > 0 and per > 0:
                scale = per ** cnt
                mult_scale *= scale
                entry["mult_scale"] = scale
            entry["count"] = cnt
            entry["ranks"] = sorted(ranks)

        else:
            partial_reasons.append(f"unsupported_joker_kind:{kind or 'unknown'}")

        breakdown.append(entry)

    return chips_add, mult_add, mult_scale, breakdown, partial_reasons
