if __package__ is None or __package__ == "":
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

import argparse
import itertools
import json
import re
import time
from pathlib import Path
from typing import Any

from sim.core.score_basic import evaluate_selected
from sim.oracle.canonicalize_real import canonicalize_real_state
from trainer.env_client import ConnectionError, RPCError, _call_method, get_state, health

TARGETS = [
    "p0_01_straight",
    "p0_02_flush",
    "p0_03_full_house",
    "p0_04_four_kind",
    "p0_05_straight_flush",
    "p0_06_order",
    "p0_07_discard_resource",
    "p0_08_round_eval",
]

ACTIONABLE_PHASES = {"BLIND_SELECT", "SELECTING_HAND", "ROUND_EVAL", "SHOP", "MENU", "GAME_OVER"}

TARGET_TO_HAND_TYPE = {
    "p0_01_straight": "STRAIGHT",
    "p0_02_flush": "FLUSH",
    "p0_03_full_house": "FULL_HOUSE",
    "p0_04_four_kind": "FOUR_OF_A_KIND",
    "p0_05_straight_flush": "STRAIGHT_FLUSH",
}

TARGET_INJECT_KEYS = {
    "p0_01_straight": ["H_T", "S_J", "D_Q", "C_K", "H_A"],
    "p0_02_flush": ["H_2", "H_5", "H_8", "H_J", "H_K"],
    "p0_03_full_house": ["H_9", "S_9", "D_9", "H_K", "S_K"],
    "p0_04_four_kind": ["H_7", "S_7", "D_7", "C_7", "H_A"],
    "p0_05_straight_flush": ["S_6", "S_7", "S_8", "S_9", "S_T"],
}


def normalize_hand_type(value: Any) -> str:
    text = str(value or "").strip().upper()
    text = re.sub(r"[^A-Z0-9]+", "_", text).strip("_")
    aliases = {
        "FOUR_KIND": "FOUR_OF_A_KIND",
        "FOUR_OF_KIND": "FOUR_OF_A_KIND",
        "FOUR_OF_A_KIND": "FOUR_OF_A_KIND",
        "STRAIGHTFLUSH": "STRAIGHT_FLUSH",
        "FULLHOUSE": "FULL_HOUSE",
    }
    return aliases.get(text, text)


def state_phase(state: dict[str, Any]) -> str:
    return str(state.get("state") or "UNKNOWN").upper()


def rank_suit_from_card(card: dict[str, Any]) -> tuple[str, str]:
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
        suit = suit[0]
    return rank, suit


def extract_hand_cards(state: dict[str, Any]) -> list[dict[str, Any]]:
    hand = (state.get("hand") or {}).get("cards") or []
    out: list[dict[str, Any]] = []
    for idx, raw in enumerate(hand):
        if not isinstance(raw, dict):
            continue
        rank, suit = rank_suit_from_card(raw)
        out.append(
            {
                "idx": idx,
                "rank": rank,
                "suit": suit,
                "raw": raw,
            }
        )
    return out


def simplify_for_eval(card: dict[str, Any]) -> dict[str, Any]:
    return {"rank": card.get("rank"), "suit": card.get("suit")}


def enumerate_index_combos(n: int, max_k: int = 5) -> list[list[int]]:
    out: list[list[int]] = []
    for k in range(1, min(max_k, n) + 1):
        for combo in itertools.combinations(range(n), k):
            out.append(list(combo))
    return out


def score_combo(hand_cards: list[dict[str, Any]], indices: list[int]) -> tuple[str, float]:
    cards = [simplify_for_eval(hand_cards[i]) for i in indices]
    hand_type, base_chips, base_mult = evaluate_selected(cards)
    return normalize_hand_type(hand_type), float(base_chips * base_mult)


def best_combo(hand_cards: list[dict[str, Any]]) -> list[int]:
    n = len(hand_cards)
    if n <= 0:
        return []
    best: list[int] = [0]
    best_score = -1.0
    for indices in enumerate_index_combos(n, max_k=5):
        _, score = score_combo(hand_cards, indices)
        if score > best_score:
            best = indices
            best_score = score
    return best


def find_target_combo(hand_cards: list[dict[str, Any]], target_hand_type: str) -> list[int] | None:
    n = len(hand_cards)
    if n <= 0:
        return None
    best: list[int] | None = None
    best_score = -1.0
    for indices in enumerate_index_combos(n, max_k=5):
        hand_type, score = score_combo(hand_cards, indices)
        if hand_type == target_hand_type:
            if score > best_score:
                best = indices
                best_score = score
    return best


def choose_discard_indices(hand_cards: list[dict[str, Any]], target: str) -> list[int]:
    n = len(hand_cards)
    if n <= 0:
        return []

    if target in {"p0_02_flush", "p0_05_straight_flush"}:
        suit_counts: dict[str, int] = {}
        for c in hand_cards:
            s = str(c.get("suit") or "")
            suit_counts[s] = suit_counts.get(s, 0) + 1
        keep_suit = max(suit_counts, key=suit_counts.get)
        idx = [int(c["idx"]) for c in hand_cards if c.get("suit") != keep_suit]
        if idx:
            return idx[: min(3, len(idx))]

    if target in {"p0_03_full_house", "p0_04_four_kind"}:
        rank_counts: dict[str, int] = {}
        for c in hand_cards:
            r = str(c.get("rank") or "")
            rank_counts[r] = rank_counts.get(r, 0) + 1
        keep_ranks = sorted(rank_counts.keys(), key=lambda r: rank_counts[r], reverse=True)[:2]
        idx = [int(c["idx"]) for c in hand_cards if c.get("rank") not in keep_ranks]
        if idx:
            return idx[: min(3, len(idx))]

    if target in {"p0_01_straight", "p0_05_straight_flush"}:
        rank_order = {"A": 14, "K": 13, "Q": 12, "J": 11, "10": 10, "9": 9, "8": 8, "7": 7, "6": 6, "5": 5, "4": 4, "3": 3, "2": 2}
        pairs = [(int(c["idx"]), rank_order.get(str(c.get("rank") or ""), 0)) for c in hand_cards]
        pairs.sort(key=lambda x: x[1])
        idx = [p[0] for p in pairs[:1] + pairs[-1:]]
        if idx:
            return idx[: min(2, len(idx))]

    return [int(hand_cards[0]["idx"])]


def canonical_snapshot(state: dict[str, Any], seed: str) -> dict[str, Any]:
    return canonicalize_real_state(state, seed=seed, rng_events=[], rng_cursor=0)


def current_last_hand_type(state: dict[str, Any]) -> str:
    score = state.get("score") or {}
    round_info = state.get("round") or {}
    return normalize_hand_type(
        score.get("last_hand_type")
        or round_info.get("last_hand_type")
        or score.get("hand_type")
        or round_info.get("hand_type")
        or ""
    )


def _classify_connection_failure_text(text: str) -> str:
    low = str(text or "").lower()
    if any(token in low for token in ("10061", "connection refused", "actively refused", "refused")):
        return "connection refused"
    if any(token in low for token in ("timed out", "timeout", "read timed out", "connect timeout")):
        return "timeout"
    return "health check failed"


def _probe_connection_failure(base_url: str, timeout_sec: float) -> str | None:
    if health(base_url):
        return None

    timeout = max(1.0, min(float(timeout_sec), 3.0))
    reasons: list[str] = []

    for method in ("health", "gamestate"):
        try:
            _call_method(base_url, method, timeout=timeout)
            return None
        except Exception as exc:
            reasons.append(str(exc))

    merged = " | ".join(reasons)
    return _classify_connection_failure_text(merged)


def _hand_brief(state: dict[str, Any], max_cards: int = 8) -> str:
    cards = extract_hand_cards(state)
    brief = []
    for c in cards[:max_cards]:
        rank = str(c.get("rank") or "?")
        suit = str(c.get("suit") or "?")
        brief.append(f"{rank}-{suit}")
    return "[" + ",".join(brief) + "]"


def _state_context_summary(state: dict[str, Any]) -> str:
    phase = state_phase(state)
    round_info = state.get("round") or {}
    hands_left = int(round_info.get("hands_left") or 0)
    discards_left = int(round_info.get("discards_left") or 0)
    hand_text = _hand_brief(state)
    return (
        f"final_phase={phase}; hands_left={hands_left}; "
        f"discards_left={discards_left}; last_hand={hand_text}"
    )


def _augment_failure_reason(base_reason: str | None, state: dict[str, Any]) -> str:
    prefix = str(base_reason or "target_not_hit").strip()
    context = _state_context_summary(state)
    if "final_phase=" in prefix:
        return prefix
    return f"{prefix}; {context}"


def wait_until_actionable(base_url: str, timeout_sec: float, wait_sleep: float) -> dict[str, Any]:
    deadline = time.time() + max(0.5, float(timeout_sec))
    last_state: dict[str, Any] = {}
    while True:
        last_state = get_state(base_url, timeout=max(1.0, float(timeout_sec)))
        if state_phase(last_state) in ACTIONABLE_PHASES:
            return last_state
        if time.time() >= deadline:
            return last_state
        time.sleep(max(0.01, float(wait_sleep)))


class TraceBuilder:
    def __init__(self, base_url: str, state: dict[str, Any], seed: str, index_base: int, timeout_sec: float, wait_sleep: float, max_steps: int, start_state_save_path: str | None = None):
        self.base_url = base_url
        self.state = state
        self.seed = seed
        self.index_base = 1 if index_base == 1 else 0
        self.timeout_sec = timeout_sec
        self.wait_sleep = wait_sleep
        self.max_steps = max_steps
        self.action_trace: list[dict[str, Any]] = []
        self.start_snapshot_override: dict[str, Any] | None = None
        self.start_state_save_path = start_state_save_path

    @property
    def steps_used(self) -> int:
        return len(self.action_trace)

    def _ensure_budget(self) -> None:
        if self.steps_used >= self.max_steps:
            raise RuntimeError("max_steps_exceeded")

    def _action_obj(self, action_type: str, **kwargs: Any) -> dict[str, Any]:
        action = {
            "schema_version": "action_v1",
            "phase": state_phase(self.state),
            "action_type": action_type,
        }
        action.update(kwargs)
        return action

    def _play_or_discard(self, method: str, indices: list[int], allow_error: bool = False) -> tuple[dict[str, Any], str | None]:
        rpc_indices = [int(i) + self.index_base for i in indices]
        try:
            _call_method(self.base_url, method, {"cards": rpc_indices}, timeout=self.timeout_sec)
            self.state = wait_until_actionable(self.base_url, self.timeout_sec, self.wait_sleep)
            return self.state, None
        except (RPCError, ConnectionError) as exc:
            if not allow_error:
                raise
            err = str(exc)
            try:
                self.state = get_state(self.base_url, timeout=self.timeout_sec)
            except Exception:
                pass
            return self.state, err

    def step(self, action_type: str, *, indices: list[int] | None = None, index: int | None = None, allow_error: bool = False, sleep: float | None = None) -> tuple[dict[str, Any], str | None]:
        self._ensure_budget()

        action_type = action_type.upper()
        action: dict[str, Any]
        error_text: str | None = None

        if action_type in {"PLAY", "DISCARD"}:
            local_indices = [int(i) for i in (indices or [])]
            action = self._action_obj(action_type, indices=local_indices, params={"index_base": self.index_base})
            self.action_trace.append(action)
            method = "play" if action_type == "PLAY" else "discard"
            self.state, error_text = self._play_or_discard(method, local_indices, allow_error=allow_error)
            return self.state, error_text

        action = self._action_obj(action_type)
        if action_type == "SELECT":
            action["index"] = int(index or 0)
        if action_type == "START":
            action["seed"] = self.seed
        if action_type == "WAIT":
            action["sleep"] = float(self.wait_sleep if sleep is None else sleep)

        self.action_trace.append(action)

        try:
            if action_type == "SELECT":
                _call_method(self.base_url, "select", {"index": int(action["index"])}, timeout=self.timeout_sec)
            elif action_type == "CASH_OUT":
                _call_method(self.base_url, "cash_out", {}, timeout=self.timeout_sec)
            elif action_type == "NEXT_ROUND":
                _call_method(self.base_url, "next_round", {}, timeout=self.timeout_sec)
            elif action_type == "START":
                _call_method(self.base_url, "start", {"deck": "RED", "stake": "WHITE", "seed": self.seed}, timeout=self.timeout_sec)
            elif action_type == "MENU":
                _call_method(self.base_url, "menu", {}, timeout=self.timeout_sec)
            elif action_type == "SKIP":
                _call_method(self.base_url, "skip", {}, timeout=self.timeout_sec)
            elif action_type == "REROLL":
                _call_method(self.base_url, "reroll", {}, timeout=self.timeout_sec)
            elif action_type == "WAIT":
                time.sleep(float(action["sleep"]))
            else:
                raise ValueError(f"unsupported action_type: {action_type}")

            self.state = wait_until_actionable(self.base_url, self.timeout_sec, self.wait_sleep)
            return self.state, None
        except (RPCError, ConnectionError, ValueError) as exc:
            if not allow_error:
                raise
            error_text = str(exc)
            try:
                self.state = get_state(self.base_url, timeout=self.timeout_sec)
            except Exception:
                pass
            return self.state, error_text


def prepare_selecting_hand(base_url: str, seed: str, timeout_sec: float, wait_sleep: float, max_loops: int = 60) -> dict[str, Any]:
    state = get_state(base_url, timeout=timeout_sec)
    for _ in range(max_loops):
        phase = state_phase(state)
        hand = (state.get("hand") or {}).get("cards") or []

        if phase == "SELECTING_HAND" and hand:
            return state

        if phase in {"MENU", "GAME_OVER"}:
            _call_method(base_url, "start", {"deck": "RED", "stake": "WHITE", "seed": seed}, timeout=timeout_sec)
        elif phase == "BLIND_SELECT":
            _call_method(base_url, "select", {"index": 0}, timeout=timeout_sec)
        elif phase == "ROUND_EVAL":
            _call_method(base_url, "cash_out", {}, timeout=timeout_sec)
        elif phase == "SHOP":
            _call_method(base_url, "next_round", {}, timeout=timeout_sec)
        else:
            time.sleep(wait_sleep)

        state = get_state(base_url, timeout=timeout_sec)

    raise RuntimeError("failed_to_reach_selecting_hand")


def hard_reset_fixture(base_url: str, seed: str, timeout_sec: float, wait_sleep: float) -> dict[str, Any]:
    try:
        _call_method(base_url, "menu", {}, timeout=timeout_sec)
    except Exception:
        pass

    _call_method(base_url, "start", {"deck": "RED", "stake": "WHITE", "seed": seed}, timeout=timeout_sec)
    state = wait_until_actionable(base_url, timeout_sec, wait_sleep)
    if state_phase(state) == "BLIND_SELECT":
        _call_method(base_url, "select", {"index": 0}, timeout=timeout_sec)
        state = wait_until_actionable(base_url, timeout_sec, wait_sleep)
    return state


def detect_index_base(base_url: str, seed: str, timeout_sec: float, wait_sleep: float) -> tuple[int, dict[str, Any]]:
    probe_meta: dict[str, Any] = {
        "method": None,
        "note": "",
    }

    try:
        state = prepare_selecting_hand(base_url, seed, timeout_sec, wait_sleep)
    except Exception as exc:
        probe_meta["note"] = f"prepare_failed:{exc}"
        return 0, probe_meta

    hand = (state.get("hand") or {}).get("cards") or []
    if not hand:
        probe_meta["note"] = "empty_hand_fallback_0"
        return 0, probe_meta

    probe_method = "discard" if int((state.get("round") or {}).get("discards_left") or 0) > 0 else "play"
    probe_meta["method"] = probe_method

    def _try_api_index(idx: int) -> tuple[bool, str]:
        try:
            _call_method(base_url, probe_method, {"cards": [idx]}, timeout=timeout_sec)
            return True, ""
        except (RPCError, ConnectionError) as exc:
            return False, str(exc)

    ok0, err0 = _try_api_index(0)
    if ok0:
        prepare_selecting_hand(base_url, seed, timeout_sec, wait_sleep)
        probe_meta["note"] = "probe_0_ok"
        return 0, probe_meta

    low_err0 = err0.lower()
    maybe_index_issue = any(token in low_err0 for token in ("index", "range", "out of", "cards"))
    if maybe_index_issue:
        ok1, err1 = _try_api_index(1)
        prepare_selecting_hand(base_url, seed, timeout_sec, wait_sleep)
        if ok1:
            probe_meta["note"] = "probe_1_ok_after_probe_0_fail"
            return 1, probe_meta
        probe_meta["note"] = f"probe_0_fail:{err0}; probe_1_fail:{err1}"
        return 0, probe_meta

    prepare_selecting_hand(base_url, seed, timeout_sec, wait_sleep)
    probe_meta["note"] = f"probe_0_non_index_error:{err0}"
    return 0, probe_meta


def restart_to_selecting(builder: TraceBuilder) -> None:
    phase = state_phase(builder.state)
    if phase not in {"MENU", "GAME_OVER"}:
        _, err = builder.step("MENU", allow_error=True)
        if err:
            pass
    builder.step("START")
    if state_phase(builder.state) == "BLIND_SELECT":
        builder.step("SELECT", index=0)


def advance_phase(builder: TraceBuilder) -> None:
    phase = state_phase(builder.state)
    if phase == "BLIND_SELECT":
        builder.step("SELECT", index=0)
    elif phase == "ROUND_EVAL":
        builder.step("CASH_OUT")
    elif phase == "SHOP":
        builder.step("NEXT_ROUND")
    elif phase in {"MENU", "GAME_OVER"}:
        builder.step("START")
        if state_phase(builder.state) == "BLIND_SELECT":
            builder.step("SELECT", index=0)
    else:
        builder.step("WAIT")


def _key_to_rank_suit(key: str) -> tuple[str, str]:
    raw = str(key or "").strip().upper()
    if "_" not in raw:
        return "", ""
    suit, rank = raw.split("_", 1)
    if rank == "T":
        rank = "10"
    return rank, suit[:1]


def _find_indices_for_target_keys(hand_cards: list[dict[str, Any]], keys: list[str]) -> list[int] | None:
    used: set[int] = set()
    out: list[int] = []
    for key in keys:
        target_rank, target_suit = _key_to_rank_suit(key)
        found: int | None = None
        for card in hand_cards:
            idx = int(card.get("idx") or 0)
            if idx in used:
                continue
            if str(card.get("rank") or "") == target_rank and str(card.get("suit") or "") == target_suit:
                found = idx
                break
        if found is None:
            return None
        used.add(found)
        out.append(found)
    return out


def _search_stats_suffix(rounds_seen: set[int], resets_tried: int, total_discards: int, total_plays: int) -> str:
    return (
        f"rounds_tried={max(1, len(rounds_seen))}; "
        f"resets_tried={max(0, resets_tried)}; "
        f"total_discards={max(0, total_discards)}; "
        f"total_plays={max(0, total_plays)}"
    )


def _try_synthesize_target_hand(builder: TraceBuilder, target: str, target_hand_type: str) -> tuple[bool, dict[str, Any], str | None]:
    target_keys = TARGET_INJECT_KEYS.get(target)
    if not target_keys:
        return False, {}, "synthesis_keys_missing"

    for _ in range(20):
        if state_phase(builder.state) == "SELECTING_HAND":
            break
        advance_phase(builder)
    if state_phase(builder.state) != "SELECTING_HAND":
        return False, {}, "synthesis_prepare_failed"

    try:
        for key in target_keys:
            _call_method(builder.base_url, "add", {"key": key}, timeout=builder.timeout_sec)
        builder.state = get_state(builder.base_url, timeout=builder.timeout_sec)
    except (RPCError, ConnectionError) as exc:
        return False, {}, f"synthesis_add_failed:{exc}"

    hand_cards = extract_hand_cards(builder.state)
    indices = _find_indices_for_target_keys(hand_cards, target_keys)
    if indices is None:
        return False, {}, "synthesis_indices_missing"

    builder.start_snapshot_override = json.loads(json.dumps(builder.state, ensure_ascii=False))
    if builder.start_state_save_path:
        try:
            _call_method(builder.base_url, "save", {"path": builder.start_state_save_path}, timeout=builder.timeout_sec)
        except Exception:
            pass

    builder.step("PLAY", indices=indices)
    observed = current_last_hand_type(builder.state)
    hit = (observed == target_hand_type) or (observed == "")
    info = {
        "target_hand_type": target_hand_type,
        "observed": observed,
        "played_indices": indices,
        "synthesis": "add",
        "synthesis_keys": target_keys,
    }
    if hit:
        return True, info, None
    return False, info, f"synthesis_play_miss:observed={observed}"


def generate_hand_target(builder: TraceBuilder, target: str, target_hand_type: str) -> tuple[bool, dict[str, Any], str | None]:
    syn_ok, syn_info, syn_reason = _try_synthesize_target_hand(builder, target, target_hand_type)
    if syn_ok:
        return True, syn_info, None

    rounds_seen: set[int] = set()
    resets_tried = 0
    total_discards = 0
    total_plays = 0

    while builder.steps_used < builder.max_steps:
        round_num = int((builder.state.get("round") or {}).get("round") or 0)
        if round_num <= 0:
            round_num = int((builder.state.get("round") or {}).get("round_num") or 0)
        rounds_seen.add(round_num)

        phase = state_phase(builder.state)
        if phase != "SELECTING_HAND":
            if phase in {"MENU", "GAME_OVER"}:
                restart_to_selecting(builder)
                resets_tried += 1
            else:
                advance_phase(builder)
            continue

        hand_cards = extract_hand_cards(builder.state)
        if not hand_cards:
            builder.step("WAIT")
            continue

        match = find_target_combo(hand_cards, target_hand_type)
        if match is not None:
            builder.step("PLAY", indices=match)
            total_plays += 1
            observed = current_last_hand_type(builder.state)
            return True, {"target_hand_type": target_hand_type, "observed": observed, "played_indices": match}, None

        round_info = builder.state.get("round") or {}
        discards_left = int(round_info.get("discards_left") or 0)
        hands_left = int(round_info.get("hands_left") or 0)

        if discards_left > 0:
            discard_indices = choose_discard_indices(hand_cards, target)
            if discard_indices:
                builder.step("DISCARD", indices=discard_indices)
                total_discards += len(discard_indices)
                continue

        play_indices = best_combo(hand_cards)
        if play_indices:
            builder.step("PLAY", indices=play_indices)
            total_plays += 1
        else:
            builder.step("WAIT")

        if hands_left <= 0 and discards_left <= 0:
            if state_phase(builder.state) in {"SHOP", "ROUND_EVAL"}:
                advance_phase(builder)
            elif state_phase(builder.state) in {"MENU", "GAME_OVER"}:
                restart_to_selecting(builder)
                resets_tried += 1

    suffix = _search_stats_suffix(rounds_seen, resets_tried, total_discards, total_plays)
    if syn_reason:
        return False, syn_info, f"{syn_reason}; {suffix}"
    return False, {}, f"max_steps_exceeded; {suffix}"


def generate_order_target(builder: TraceBuilder) -> tuple[bool, dict[str, Any], str | None]:
    details: dict[str, Any] = {}

    if state_phase(builder.state) != "SELECTING_HAND":
        advance_phase(builder)

    hand_cards = extract_hand_cards(builder.state)
    if len(hand_cards) < 3:
        return False, details, "insufficient_hand_cards_for_order_test"

    base = best_combo(hand_cards)
    if len(base) < 3:
        base = list(range(min(3, len(hand_cards))))
    base = base[:3]
    asc = sorted(base)
    desc = list(reversed(asc))
    if desc == asc:
        desc = asc[::-1]

    builder.step("PLAY", indices=desc)
    details["first_play_indices"] = desc

    restart_to_selecting(builder)
    if state_phase(builder.state) != "SELECTING_HAND":
        return False, details, "failed_to_reach_selecting_after_restart"

    hand_cards = extract_hand_cards(builder.state)
    if len(hand_cards) < 3:
        return False, details, "insufficient_hand_cards_second_order_test"

    base2 = best_combo(hand_cards)
    if len(base2) < 3:
        base2 = list(range(min(3, len(hand_cards))))
    asc2 = sorted(base2[:3])
    builder.step("PLAY", indices=asc2)
    details["second_play_indices"] = asc2
    return True, details, None


def generate_discard_resource_target(builder: TraceBuilder) -> tuple[bool, dict[str, Any], str | None]:
    while builder.steps_used < builder.max_steps:
        phase = state_phase(builder.state)
        if phase != "SELECTING_HAND":
            advance_phase(builder)
            continue

        hand_cards = extract_hand_cards(builder.state)
        if not hand_cards:
            builder.step("WAIT")
            continue

        discards_left = int((builder.state.get("round") or {}).get("discards_left") or 0)
        if discards_left > 0:
            builder.step("DISCARD", indices=[0])
            continue

        before = json.dumps(builder.state, ensure_ascii=False, sort_keys=True)
        _, extra_err = builder.step("DISCARD", indices=[0], allow_error=True)
        after = json.dumps(builder.state, ensure_ascii=False, sort_keys=True)
        return True, {
            "extra_discard_error": extra_err,
            "state_changed_after_extra_discard": before != after,
        }, None

    return False, {}, "max_steps_exceeded"


def generate_round_eval_target(builder: TraceBuilder) -> tuple[bool, dict[str, Any], str | None]:
    while builder.steps_used < builder.max_steps:
        phase = state_phase(builder.state)
        if phase in {"ROUND_EVAL", "SHOP"}:
            return True, {"final_phase": phase}, None

        if phase == "SELECTING_HAND":
            hand_cards = extract_hand_cards(builder.state)
            if not hand_cards:
                builder.step("WAIT")
                continue
            play_indices = best_combo(hand_cards)
            if not play_indices:
                play_indices = [0]
            builder.step("PLAY", indices=play_indices)
            continue

        advance_phase(builder)

    return False, {}, "max_steps_exceeded"


def generate_trace(
    *,
    base_url: str,
    target: str,
    max_steps: int,
    seed: str,
    timeout_sec: float,
    wait_sleep: float,
    out_dir: Path | None = None,
) -> dict[str, Any]:
    if target not in TARGETS:
        raise ValueError(f"unsupported target: {target}")

    connection_reason = _probe_connection_failure(base_url, timeout_sec)
    if connection_reason is not None:
        return {
            "success": False,
            "target": target,
            "steps_used": 0,
            "final_phase": "UNKNOWN",
            "hit_info": {},
            "index_base": 0,
            "index_probe": {"method": None, "note": connection_reason},
            "failure_reason": connection_reason,
            "start_snapshot": None,
            "action_trace": [],
        }

    index_base = 0
    index_probe: dict[str, Any] = {"method": None, "note": ""}
    state: dict[str, Any] | None = None
    init_error: Exception | None = None
    for attempt in range(2):
        try:
            hard_reset_fixture(base_url, seed, timeout_sec, wait_sleep)
            index_base, index_probe = detect_index_base(base_url, seed, timeout_sec, wait_sleep)
            state = prepare_selecting_hand(base_url, seed, timeout_sec, wait_sleep)
            init_error = None
            break
        except (RPCError, ConnectionError, RuntimeError) as exc:
            init_error = exc
            time.sleep(max(0.05, wait_sleep))

    if init_error is not None or state is None:
        reason = _classify_connection_failure_text(str(init_error)) if init_error else "health check failed"
        return {
            "success": False,
            "target": target,
            "steps_used": 0,
            "final_phase": "UNKNOWN",
            "hit_info": {},
            "index_base": 0,
            "index_probe": {"method": None, "note": str(init_error) if init_error else "init_failed"},
            "failure_reason": reason,
            "start_snapshot": None,
            "action_trace": [],
        }

    oracle_start_state_path: str | None = None
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        oracle_start_state_path = str(out_dir / f"oracle_start_state_{target}.jkr")
        try:
            _call_method(base_url, "save", {"path": oracle_start_state_path}, timeout=timeout_sec)
        except Exception:
            pass

    builder = TraceBuilder(
        base_url=base_url,
        state=state,
        seed=seed,
        index_base=index_base,
        timeout_sec=timeout_sec,
        wait_sleep=wait_sleep,
        max_steps=max_steps,
        start_state_save_path=oracle_start_state_path,
    )

    start_snapshot = canonical_snapshot(builder.state, seed=seed)
    start_meta = start_snapshot.setdefault("_meta", {})
    if isinstance(start_meta, dict):
        start_meta["index_base_detected"] = int(index_base)
        start_meta["index_probe"] = dict(index_probe)

    success = False
    hit_info: dict[str, Any] = {}
    failure_reason: str | None = None

    try:
        if target in TARGET_TO_HAND_TYPE:
            success, hit_info, failure_reason = generate_hand_target(builder, target, TARGET_TO_HAND_TYPE[target])
        elif target == "p0_06_order":
            success, hit_info, failure_reason = generate_order_target(builder)
        elif target == "p0_07_discard_resource":
            success, hit_info, failure_reason = generate_discard_resource_target(builder)
        elif target == "p0_08_round_eval":
            success, hit_info, failure_reason = generate_round_eval_target(builder)
        else:
            failure_reason = "unsupported_target"
    except (RPCError, ConnectionError) as exc:
        cls = _classify_connection_failure_text(str(exc))
        if cls in {"connection refused", "timeout"}:
            failure_reason = cls
        else:
            failure_reason = f"rpc_error:{exc}"
    except RuntimeError as exc:
        failure_reason = str(exc)

    if isinstance(builder.start_snapshot_override, dict):
        start_snapshot = canonical_snapshot(builder.start_snapshot_override, seed=seed)
        start_meta = start_snapshot.setdefault("_meta", {})
        if isinstance(start_meta, dict):
            start_meta["index_base_detected"] = int(index_base)
            start_meta["index_probe"] = dict(index_probe)

    if not success:
        failure_reason = _augment_failure_reason(failure_reason, builder.state)

    result = {
        "success": bool(success),
        "target": target,
        "steps_used": builder.steps_used,
        "final_phase": state_phase(builder.state),
        "hit_info": hit_info,
        "index_base": int(index_base),
        "index_probe": dict(index_probe),
        "failure_reason": failure_reason,
        "start_snapshot": start_snapshot,
        "action_trace": builder.action_trace,
        "oracle_start_state_path": oracle_start_state_path,
    }

    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        snap_path = out_dir / f"oracle_start_snapshot_{target}.json"
        action_path = out_dir / f"action_trace_{target}.jsonl"
        snap_path.write_text(json.dumps(start_snapshot, ensure_ascii=False, indent=2), encoding="utf-8")
        with action_path.open("w", encoding="utf-8-sig", newline="\n") as fp:
            for action in builder.action_trace:
                fp.write(json.dumps(action, ensure_ascii=False) + "\n")
        result["artifact_paths"] = {
            "oracle_start_snapshot": str(snap_path),
            "action_trace": str(action_path),
            "oracle_start_state": str(out_dir / f"oracle_start_state_{target}.jkr") if oracle_start_state_path else None,
        }

    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate adaptive oracle action trace for one P0 target.")
    parser.add_argument("--base-url", default="http://127.0.0.1:12346")
    parser.add_argument("--target", required=True, choices=TARGETS)
    parser.add_argument("--max-steps", type=int, default=80)
    parser.add_argument("--seed", default="AAAAAAA")
    parser.add_argument("--timeout-sec", type=float, default=8.0)
    parser.add_argument("--wait-sleep", type=float, default=0.05)
    parser.add_argument("--out-dir")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    out_dir = Path(args.out_dir) if args.out_dir else None
    result = generate_trace(
        base_url=args.base_url,
        target=args.target,
        max_steps=int(args.max_steps),
        seed=args.seed,
        timeout_sec=float(args.timeout_sec),
        wait_sleep=float(args.wait_sleep),
        out_dir=out_dir,
    )

    print(
        "target={target} success={success} steps_used={steps_used} final_phase={final_phase} index_base={index_base} hit_info={hit_info}".format(
            target=result.get("target"),
            success=result.get("success"),
            steps_used=result.get("steps_used"),
            final_phase=result.get("final_phase"),
            index_base=result.get("index_base"),
            hit_info=json.dumps(result.get("hit_info") or {}, ensure_ascii=False),
        )
    )

    if out_dir is None:
        print(json.dumps(result, ensure_ascii=False))

    return 0 if result.get("success") else 1


if __name__ == "__main__":
    raise SystemExit(main())

