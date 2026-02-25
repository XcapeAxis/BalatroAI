import copy
import json
import os
import random
import re
from pathlib import Path
from typing import Any

from sim.core.score_basic import evaluate_selected_breakdown
from sim.score.expected_basic import compute_expected_for_action

SUITS = ["C", "D", "H", "S"]
RANKS = ["A", "K", "Q", "J", "10", "9", "8", "7", "6", "5", "4", "3", "2"]
BLIND_ORDER = ["small", "big", "boss"]
STAKE_ORDER = ["white", "red", "green", "black", "blue", "purple", "orange", "gold"]
_STAKE_RULES_CACHE: dict[str, Any] | None = None


class SimEnv:
    def __init__(self, seed: str = "AAAAAAA", max_hand: int = 8, fail_fast: bool | None = None):
        self.seed = seed
        self.max_hand = max_hand
        raw_flag = os.getenv("SIM_FAIL_FAST")
        if fail_fast is None:
            # Default to strict mode unless explicitly disabled.
            if raw_flag is None or str(raw_flag).strip() == "":
                self.fail_fast = True
            else:
                self.fail_fast = str(raw_flag).strip().lower() in {"1", "true", "yes", "on"}
        else:
            self.fail_fast = bool(fail_fast)
        self._stake_name = "white"
        self._rng = random.Random(seed)
        self._state: dict[str, Any] = {}
        self.reset(seed)

    @staticmethod
    def _sanitize_stake_name(raw: Any) -> str:
        text = str(raw or "").strip().lower()
        return text if text in STAKE_ORDER else "white"

    def _load_stake_rules_payload(self) -> dict[str, Any]:
        global _STAKE_RULES_CACHE
        if _STAKE_RULES_CACHE is not None:
            return dict(_STAKE_RULES_CACHE)

        path = Path(__file__).resolve().parent.parent.parent / "balatro_mechanics" / "derived" / "stakes.json"
        payload: dict[str, Any] = {"stakes": []}
        if path.exists():
            try:
                raw = json.loads(path.read_text(encoding="utf-8-sig"))
                if isinstance(raw, dict):
                    payload = raw
            except Exception:
                if self.fail_fast:
                    raise RuntimeError(f"failed to parse stake rules file: {path}")
                payload = {"stakes": []}
        _STAKE_RULES_CACHE = dict(payload)
        return dict(payload)

    def _stake_entry(self, stake_name: str) -> dict[str, Any] | None:
        payload = self._load_stake_rules_payload()
        for item in (payload.get("stakes") or []):
            if not isinstance(item, dict):
                continue
            if self._sanitize_stake_name(item.get("stake_name")) == stake_name:
                return item
        return None

    @staticmethod
    def _parse_first_number(text: str) -> float | None:
        m = re.search(r"[-+]?\d+(?:\.\d+)?", str(text or ""))
        if not m:
            return None
        try:
            return float(m.group(0))
        except Exception:
            return None

    def _apply_stake_modifiers(self, stake_name: str) -> None:
        entry = self._stake_entry(stake_name)
        modifiers = entry.get("modifiers") if isinstance(entry, dict) else []
        if not isinstance(modifiers, list):
            modifiers = []

        applied_keys: list[str] = []
        degraded = bool(not entry or bool(entry.get("missing")))

        for mod in modifiers:
            if not isinstance(mod, dict):
                continue
            field = str(mod.get("field") or "").strip().lower()
            value_text = str(mod.get("value") or "").strip()
            if not field:
                continue
            applied_keys.append(field)
            val = self._parse_first_number(value_text)
            if val is None:
                continue

            # Conservative application: only simple direct resource deltas.
            if "money" in field:
                self._state["money"] = float(self._state.get("money") or 0.0) + float(val)
            elif "discard" in field and "delta" in field:
                self._state["round"]["discards_left"] = int(self._state["round"]["discards_left"]) + int(val)
            elif "hand" in field and "delta" in field:
                self._state["round"]["hands_left"] = int(self._state["round"]["hands_left"]) + int(val)
            elif "reroll" in field and "cost" in field:
                self._state["round"]["reroll_cost"] = max(0, int(self._state["round"]["reroll_cost"]) + int(val))

        self._state["rules"] = {
            "stake": stake_name,
            "applied_modifiers": sorted(set(applied_keys)),
            "degraded": bool(degraded),
        }

    def _mk_card(self, rank: str, suit: str, serial: int) -> dict[str, Any]:
        key_rank = "T" if rank == "10" else rank
        return {
            "card_id": f"{suit}{rank}-{serial}",
            "rank": rank,
            "suit": suit,
            "effect_text": "",
            "modifier_tags": [],
            "state_tags": [],
            "key": f"{suit}_{key_rank}",
            "value": {
                "rank": rank,
                "suit": suit,
                "effect": "",
            },
            "modifier": [],
            "state": [],
        }

    def _new_deck(self) -> list[dict[str, Any]]:
        cards = []
        serial = 0
        for suit in SUITS:
            for rank in RANKS:
                cards.append(self._mk_card(rank, suit, serial))
                serial += 1
        self._rng.shuffle(cards)
        return cards

    def _target_scores(self, ante: int) -> dict[str, int]:
        base = 300 * max(1, ante)
        return {
            "small": base,
            "big": int(base * 1.5),
            "boss": int(base * 2.0),
        }

    def _default_hands(self) -> dict[str, Any]:
        return {
            "levels": {
                "HIGH_CARD": {"level": 1, "chips": 5.0, "mult": 1.0},
                "PAIR": {"level": 1, "chips": 10.0, "mult": 2.0},
                "TWO_PAIR": {"level": 1, "chips": 20.0, "mult": 2.0},
                "THREE_OF_A_KIND": {"level": 1, "chips": 30.0, "mult": 3.0},
                "STRAIGHT": {"level": 1, "chips": 30.0, "mult": 4.0},
                "FLUSH": {"level": 1, "chips": 35.0, "mult": 4.0},
                "FULL_HOUSE": {"level": 1, "chips": 40.0, "mult": 4.0},
                "FOUR_OF_A_KIND": {"level": 1, "chips": 60.0, "mult": 7.0},
                "STRAIGHT_FLUSH": {"level": 1, "chips": 100.0, "mult": 8.0},
                "FIVE_OF_A_KIND": {"level": 1, "chips": 120.0, "mult": 12.0},
                "FLUSH_HOUSE": {"level": 1, "chips": 140.0, "mult": 14.0},
                "FLUSH_FIVE": {"level": 1, "chips": 160.0, "mult": 16.0},
            }
        }

    def _restore_hands(self, raw_hands: Any) -> dict[str, Any]:
        default = self._default_hands()
        if not isinstance(raw_hands, dict):
            return default

        source = raw_hands.get("levels") if isinstance(raw_hands.get("levels"), dict) else raw_hands
        levels: dict[str, dict[str, float]] = {}
        for raw_name, raw_info in source.items():
            hand_name = str(raw_name or "").strip().upper()
            if not hand_name:
                continue
            if isinstance(raw_info, dict):
                level = int(raw_info.get("level") or 1)
                chips = float(raw_info.get("chips") or 0.0)
                mult = float(raw_info.get("mult") or 1.0)
            else:
                try:
                    level = int(raw_info)
                except Exception:
                    if self.fail_fast:
                        raise ValueError(f"invalid hand level value for {hand_name}: {raw_info!r}")
                    level = 1
                chips = 0.0
                mult = 1.0
            levels[hand_name] = {"level": level, "chips": chips, "mult": mult}

        if not levels:
            return default
        return {"levels": levels}

    def _restore_consumables(self, raw_consumables: Any) -> dict[str, Any]:
        if not isinstance(raw_consumables, dict):
            return {"count": 0, "limit": 2, "highlighted_limit": 1, "cards": []}

        cards_raw = raw_consumables.get("cards")
        cards = cards_raw if isinstance(cards_raw, list) else []
        out_cards: list[dict[str, Any]] = []
        for card in cards:
            if not isinstance(card, dict):
                continue
            out_cards.append(
                {
                    "key": str(card.get("key") or "").strip().lower(),
                    "label": str(card.get("label") or "").strip(),
                    "set": str(card.get("set") or "").strip().upper(),
                }
            )

        return {
            "count": int(raw_consumables.get("count") or len(out_cards)),
            "limit": int(raw_consumables.get("limit") or 2),
            "highlighted_limit": int(raw_consumables.get("highlighted_limit") or 1),
            "cards": out_cards,
        }


    def _restore_market(self, raw_market: Any) -> dict[str, Any]:
        if not isinstance(raw_market, dict):
            return {"count": 0, "limit": 0, "highlighted_limit": 0, "cards": []}

        cards_raw = raw_market.get("cards")
        cards = cards_raw if isinstance(cards_raw, list) else []
        out_cards: list[dict[str, Any]] = []
        for idx, card in enumerate(cards):
            if not isinstance(card, dict):
                continue
            cost_obj = card.get("cost")
            buy_cost = 0.0
            if isinstance(cost_obj, dict):
                buy_cost = float(cost_obj.get("buy") or 0.0)
            else:
                try:
                    buy_cost = float(cost_obj or 0.0)
                except Exception:
                    if self.fail_fast:
                        raise ValueError(f"invalid market buy cost: {cost_obj!r}")
                    buy_cost = 0.0
            out_cards.append(
                {
                    "key": str(card.get("key") or "").strip().lower(),
                    "label": str(card.get("label") or "").strip(),
                    "set": str(card.get("set") or "").strip().upper(),
                    "cost": {"buy": buy_cost},
                    "slot_index": int(card.get("slot_index") if isinstance(card.get("slot_index"), int) else idx),
                }
            )

        out_cards.sort(key=lambda c: (str(c.get("slot_index") or 0), str(c.get("key") or ""), str(c.get("set") or "")))
        return {
            "count": int(raw_market.get("count") or len(out_cards)),
            "limit": int(raw_market.get("limit") or 0),
            "highlighted_limit": int(raw_market.get("highlighted_limit") or 0),
            "cards": out_cards,
        }

    def _restore_used_vouchers(self, raw_used: Any) -> list[str]:
        out: list[str] = []
        if isinstance(raw_used, dict):
            out.extend(str(k).strip().lower() for k in raw_used.keys() if str(k).strip())
        elif isinstance(raw_used, list):
            for item in raw_used:
                if isinstance(item, dict):
                    key = str(item.get("key") or item.get("id") or "").strip().lower()
                    if key:
                        out.append(key)
                else:
                    key = str(item).strip().lower()
                    if key:
                        out.append(key)
        elif isinstance(raw_used, str):
            key = raw_used.strip().lower()
            if key:
                out.append(key)
        return sorted(set(out))


    def _apply_expected_shop_context(self, action: dict[str, Any], info: dict[str, Any]) -> None:
        expected_context = action.get("expected_context") if isinstance(action.get("expected_context"), dict) else {}
        shop_market = expected_context.get("shop_market") if isinstance(expected_context.get("shop_market"), dict) else {}
        if not shop_market:
            return

        if "shop" in shop_market:
            self._state["shop"] = self._restore_market(shop_market.get("shop"))
        if "vouchers" in shop_market:
            self._state["vouchers"] = self._restore_market(shop_market.get("vouchers"))
        if "packs" in shop_market:
            self._state["packs"] = self._restore_market(shop_market.get("packs"))
        if "consumables" in shop_market:
            self._state["consumables"] = self._restore_consumables(shop_market.get("consumables"))
        if "used_vouchers" in shop_market:
            self._state["used_vouchers"] = self._restore_used_vouchers(shop_market.get("used_vouchers"))

        economy = shop_market.get("economy") if isinstance(shop_market.get("economy"), dict) else {}
        if economy and "money" in economy:
            try:
                self._state["money"] = float(economy.get("money") or 0.0)
            except Exception:
                if self.fail_fast:
                    raise ValueError(f"invalid economy.money in expected shop context: {economy.get('money')!r}")

        info["shop_context_applied"] = True

    def _apply_rng_market_items(self, field: str, items: list[dict[str, Any]]) -> None:
        market_cards: list[dict[str, Any]] = []
        for idx, item in enumerate(items):
            if not isinstance(item, dict):
                continue
            key = str(item.get("key") or "").strip().lower()
            kind = str(item.get("kind") or item.get("set") or "").strip().upper()
            slot = int(item.get("slot") if isinstance(item.get("slot"), int) else item.get("slot_index") if isinstance(item.get("slot_index"), int) else idx)
            try:
                cost = float(item.get("cost") or 0.0)
            except Exception:
                if self.fail_fast:
                    raise ValueError(f"invalid RNG market item cost: {item.get('cost')!r}")
                cost = 0.0
            market_cards.append(
                {
                    "key": key,
                    "set": kind,
                    "label": key.upper(),
                    "cost": {"buy": cost},
                    "slot_index": slot,
                }
            )
        market_cards.sort(key=lambda c: (int(c.get("slot_index") or 0), str(c.get("key") or ""), str(c.get("set") or "")))
        self._state[field] = {
            "count": len(market_cards),
            "limit": max(len(market_cards), int((self._state.get(field) or {}).get("limit") or 0)),
            "highlighted_limit": int((self._state.get(field) or {}).get("highlighted_limit") or 0),
            "cards": market_cards,
        }

    def _apply_rng_replay_outcomes(self, outcomes: list[Any], info: dict[str, Any]) -> None:
        applied = 0
        for outcome in outcomes:
            if not isinstance(outcome, dict):
                continue
            typ = str(outcome.get("type") or "").strip().lower()
            if typ == "shop_offers":
                items = outcome.get("items") if isinstance(outcome.get("items"), list) else []
                self._apply_rng_market_items("shop", items)
                applied += 1
            elif typ == "voucher_offers":
                items = outcome.get("items") if isinstance(outcome.get("items"), list) else []
                self._apply_rng_market_items("vouchers", items)
                applied += 1
            elif typ == "pack_offers":
                items = outcome.get("items") if isinstance(outcome.get("items"), list) else []
                self._apply_rng_market_items("packs", items)
                applied += 1
            elif typ == "pack_choices":
                raw_choices = outcome.get("choices") if isinstance(outcome.get("choices"), list) else []
                choices: list[dict[str, Any]] = []
                for idx, key in enumerate(raw_choices):
                    key_text = str(key or "").strip().lower()
                    if not key_text:
                        continue
                    choices.append({"key": key_text, "slot_index": idx})
                self._state["pack_choices"] = choices
                applied += 1
            elif typ == "blind_signal":
                signal = outcome.get("signal") if isinstance(outcome.get("signal"), dict) else {}
                blind = str(signal.get("selected") or signal.get("blind") or "").strip().lower()
                if blind in BLIND_ORDER:
                    self._state["round"]["blind"] = blind
                applied += 1
            elif typ == "tags_signal":
                tags = outcome.get("tags") if isinstance(outcome.get("tags"), list) else []
                self._state["tags"] = [str(x).strip().lower() for x in tags if str(x).strip()]
                applied += 1
            elif typ in {"money_change", "econ_delta"}:
                raw_delta = outcome.get("delta") if "delta" in outcome else outcome.get("value")
                try:
                    delta = float(raw_delta or 0.0)
                except Exception:
                    raise ValueError(f"invalid {typ} outcome delta/value: {raw_delta}")
                self._state["money"] = float(self._state.get("money") or 0.0) + delta
                applied += 1
            elif typ == "prob_trigger":
                key = str(outcome.get("key") or "").strip().lower()
                value = bool(outcome.get("value"))
                trig = self._state.get("_prob_triggers") if isinstance(self._state.get("_prob_triggers"), dict) else {}
                trig[key] = value
                self._state["_prob_triggers"] = trig
                applied += 1
            elif typ == "shop_roll":
                raw = outcome.get("value")
                offers = raw if isinstance(raw, list) else []
                cards: list[dict[str, Any]] = []
                for idx, key in enumerate(offers):
                    key_s = str(key or "").strip().lower()
                    if not key_s:
                        continue
                    cards.append(
                        {
                            "key": key_s,
                            "set": "JOKER",
                            "label": key_s.upper(),
                            "cost": {"buy": 0.0},
                            "slot_index": idx,
                        }
                    )
                if cards:
                    self._state["shop"] = {
                        "count": len(cards),
                        "limit": max(len(cards), int((self._state.get("shop") or {}).get("limit") or 0)),
                        "highlighted_limit": int((self._state.get("shop") or {}).get("highlighted_limit") or 0),
                        "cards": cards,
                    }
                applied += 1
            elif typ in {"phase_transition", "hand_cards"}:
                # Observable oracle tokens used for replay determinism diagnostics.
                applied += 1

        if applied:
            info["rng_replay_applied"] = applied

    def _consume_rng_replay(self, action: dict[str, Any], info: dict[str, Any]) -> None:
        replay = action.get("rng_replay") if isinstance(action.get("rng_replay"), dict) else None
        if not replay:
            return
        enabled = bool(replay.get("enabled") or False)
        outcomes = replay.get("outcomes") if isinstance(replay.get("outcomes"), list) else []
        if enabled and replay.get("outcomes") is not None and not isinstance(replay.get("outcomes"), list):
            raise ValueError("rng_replay.outcomes must be list")
        if enabled and len(outcomes) == 0:
            raise ValueError("rng_replay enabled but outcomes are empty")
        self._state["_rng_replay_last"] = list(outcomes)
        info["rng_replay_enabled"] = enabled
        info["rng_replay_consumed"] = len(outcomes)
        if enabled:
            self._apply_rng_replay_outcomes(outcomes, info)

    def _make_blinds(self, ante: int, selected: str, selecting: bool) -> dict[str, dict[str, Any]]:
        scores = self._target_scores(ante)
        out: dict[str, dict[str, Any]] = {}
        for key in BLIND_ORDER:
            if selecting:
                status = "SELECT" if key == selected else "WAIT"
            else:
                status = "CURRENT" if key == selected else "UPCOMING"
            out[key] = {"score": int(scores[key]), "status": status}
        return out

    def _refresh_done_flags(self) -> None:
        phase = str(self._state.get("state") or "")
        if phase == "GAME_OVER":
            self._state["done"] = True
            self._state["won"] = False
        else:
            self._state["done"] = False
            self._state["won"] = False

    @staticmethod
    def _normalize_rank(raw_rank: Any) -> str:
        rank = str(raw_rank or "").strip().upper()
        if rank == "T":
            return "10"
        if rank in {"A", "K", "Q", "J", "10", "9", "8", "7", "6", "5", "4", "3", "2"}:
            return rank
        if rank.isdigit() and rank in {"10", "9", "8", "7", "6", "5", "4", "3", "2"}:
            return rank
        return rank

    @staticmethod
    def _normalize_suit(raw_suit: Any) -> str:
        suit = str(raw_suit or "").strip().upper()
        if suit:
            return suit[0]
        return suit

    @staticmethod
    def _normalize_tags(raw_tags: Any) -> list[str]:
        if isinstance(raw_tags, list):
            tags: list[str] = []
            for item in raw_tags:
                if isinstance(item, (str, int, float, bool)):
                    text = str(item).strip()
                    if text:
                        tags.append(text)
        elif raw_tags is None:
            tags = []
        elif isinstance(raw_tags, (str, int, float, bool)):
            text = str(raw_tags).strip()
            tags = [text] if text else []
        else:
            tags = []
        return sorted(set(tags))

    def _canonical_card_to_internal(self, card: dict[str, Any], fallback_uid: str) -> dict[str, Any]:
        value = card.get("value") if isinstance(card.get("value"), dict) else {}

        rank = self._normalize_rank(card.get("rank") or value.get("rank"))
        suit = self._normalize_suit(card.get("suit") or value.get("suit"))

        key = str(card.get("key") or "")
        if not key and rank and suit:
            key_rank = "T" if rank == "10" else rank
            key = f"{suit}_{key_rank}"

        uid = str(card.get("uid") or card.get("card_id") or card.get("id") or key or fallback_uid)

        modifier = self._normalize_tags(card.get("modifier"))
        if not modifier:
            modifier = self._normalize_tags(card.get("modifier_tags"))
        state_tags = self._normalize_tags(card.get("state"))
        if not state_tags:
            state_tags = self._normalize_tags(card.get("state_tags"))

        effect_text = str(card.get("effect_text") or value.get("effect") or "")
        return {
            "card_id": uid,
            "rank": rank,
            "suit": suit,
            "effect_text": effect_text,
            "modifier_tags": list(modifier),
            "state_tags": list(state_tags),
            "key": key,
            "value": {
                "rank": rank,
                "suit": suit,
                "effect": effect_text,
            },
            "modifier": list(modifier),
            "state": list(state_tags),
        }

    def _refill_deck_if_needed(self) -> None:
        deck = self._state["deck"]["cards"]
        discard = self._state["discard"]["cards"]
        if deck:
            return
        if not discard:
            return
        deck.extend(discard)
        discard.clear()
        self._rng.shuffle(deck)

    def _draw_one(self) -> dict[str, Any] | None:
        self._refill_deck_if_needed()
        deck = self._state["deck"]["cards"]
        if not deck:
            return None
        return deck.pop(0)

    def _draw_to_hand(self) -> None:
        hand = self._state["hand"]["cards"]
        while len(hand) < self.max_hand:
            card = self._draw_one()
            if card is None:
                break
            hand.append(card)

    def _begin_round(self, next_round: bool = False) -> None:
        if next_round:
            self._state["round_num"] = int(self._state.get("round_num") or 0) + 1
        round_num = int(self._state.get("round_num") or 1)
        ante = max(1, ((round_num - 1) // 3) + 1)
        self._state["ante_num"] = ante

        all_cards = []
        for zone in ("deck", "discard", "hand", "played"):
            all_cards.extend(self._state[zone]["cards"])
            self._state[zone]["cards"] = []
        if not all_cards:
            all_cards = self._new_deck()
        self._rng.shuffle(all_cards)
        self._state["deck"]["cards"] = all_cards

        self._state["round"] = {
            "hands_left": 4,
            "discards_left": 4,
            "chips": 0,
            "reroll_cost": 5,
            "blind": "small",
        }
        self._state["score"] = {
            "chips": 0.0,
            "mult": 1.0,
            "target_chips": float(self._target_scores(ante)["small"]),
            "last_hand_type": "",
            "last_base_chips": 0.0,
            "last_base_mult": 1.0,
        }
        self._state["state"] = "BLIND_SELECT"
        self._state["blinds"] = self._make_blinds(ante, selected="small", selecting=True)
        self._refresh_done_flags()

    def load_snapshot(self, canonical_state: dict[str, Any]) -> dict[str, Any]:
        required = {"zones", "round", "score", "economy", "jokers", "rng", "flags", "phase"}
        missing = [k for k in sorted(required) if k not in canonical_state]
        if missing:
            raise ValueError(f"snapshot missing required keys: {missing}")

        zones = canonical_state.get("zones") or {}

        def _restore_zone(zone_name: str) -> list[dict[str, Any]]:
            raw_cards = zones.get(zone_name) or []
            if not isinstance(raw_cards, list):
                return []
            restored: list[dict[str, Any]] = []
            for idx, raw in enumerate(raw_cards):
                if isinstance(raw, dict):
                    restored.append(self._canonical_card_to_internal(raw, f"{zone_name}-{idx}"))
            return restored

        round_info = canonical_state.get("round") or {}
        score_info = canonical_state.get("score") or {}
        economy_info = canonical_state.get("economy") or {}
        flags = canonical_state.get("flags") or {}
        rng_info = canonical_state.get("rng") or {}
        rules_info = canonical_state.get("rules") if isinstance(canonical_state.get("rules"), dict) else {}
        stake_name = self._sanitize_stake_name(rules_info.get("stake") or canonical_state.get("stake"))
        applied_modifiers = rules_info.get("applied_modifiers") if isinstance(rules_info.get("applied_modifiers"), list) else []

        rng_seed = rng_info.get("seed")
        if isinstance(rng_seed, str) and rng_seed:
            self.seed = rng_seed
        self._rng = random.Random(self.seed)
        rng_cursor = max(0, int(rng_info.get("cursor") or 0))
        for _ in range(rng_cursor):
            self._rng.random()

        ante_num = max(1, int(round_info.get("ante") or 1))
        blind = str(round_info.get("blind") or "small")
        if blind not in BLIND_ORDER:
            blind = "small"
        phase = str(canonical_state.get("phase") or "UNKNOWN")
        selecting = phase == "BLIND_SELECT"

        hand_cards = _restore_zone("hand")
        # Balatro observable hand size cap is 8; snapshots may temporarily contain >8 cards (debug/add),
        # but draw-to-hand should still target the game cap instead of snapshot length.
        self.max_hand = 8

        chips = float(score_info.get("chips") or 0.0)
        self._state = {
            "state": phase,
            "deck": {"cards": _restore_zone("deck")},
            "discard": {"cards": _restore_zone("discard")},
            "hand": {"cards": hand_cards},
            "played": {"cards": _restore_zone("played")},
            "round": {
                "hands_left": int(round_info.get("hands_left") or 0),
                "discards_left": int(round_info.get("discards_left") or 0),
                "chips": chips,
                "reroll_cost": 5,
                "blind": blind,
            },
            "score": {
                "chips": chips,
                "mult": float(score_info.get("mult") or 1.0),
                "target_chips": float(score_info.get("target_chips") or 0.0),
                "last_hand_type": str(score_info.get("last_hand_type") or ""),
                "last_base_chips": float(score_info.get("last_base_chips") or 0.0),
                "last_base_mult": float(score_info.get("last_base_mult") or 1.0),
            },
            "blinds": self._make_blinds(ante_num, selected=blind, selecting=selecting),
            "money": float(economy_info.get("money") or 0.0),
            "jokers": list(canonical_state.get("jokers") or []),
            "consumables": self._restore_consumables(canonical_state.get("consumables")),
            "shop": self._restore_market(canonical_state.get("shop")),
            "vouchers": self._restore_market(canonical_state.get("vouchers")),
            "packs": self._restore_market(canonical_state.get("packs")),
            "used_vouchers": self._restore_used_vouchers(canonical_state.get("used_vouchers")),
            "hands": self._restore_hands(canonical_state.get("hands")),
            "tags": self._normalize_tags(canonical_state.get("tags")),
            "rules": {
                "stake": stake_name,
                "applied_modifiers": sorted(set(str(x) for x in applied_modifiers if str(x))),
                "degraded": bool(rules_info.get("degraded") or False),
            },
            "ante_num": ante_num,
            "round_num": int(round_info.get("round_num") or 1),
            "done": bool(flags.get("done") or False),
            "won": bool(flags.get("won") or False),
            "_rng_events": list(rng_info.get("events") or []),
        }
        self._stake_name = stake_name
        return copy.deepcopy(self._state)

    def reset(
        self,
        seed: str | None = None,
        from_snapshot: dict[str, Any] | None = None,
        stake: str | None = None,
    ) -> dict[str, Any]:
        if from_snapshot is not None:
            return self.load_snapshot(from_snapshot)

        if seed is not None:
            self.seed = seed
        if stake is not None:
            self._stake_name = self._sanitize_stake_name(stake)
        self._rng = random.Random(self.seed)

        self._state = {
            "state": "MENU",
            "deck": {"cards": self._new_deck()},
            "discard": {"cards": []},
            "hand": {"cards": []},
            "played": {"cards": []},
            "round": {
                "hands_left": 4,
                "discards_left": 4,
                "chips": 0,
                "reroll_cost": 5,
                "blind": "small",
            },
            "score": {
                "chips": 0.0,
                "mult": 1.0,
                "target_chips": 300.0,
                "last_hand_type": "",
                "last_base_chips": 0.0,
                "last_base_mult": 1.0,
            },
            "blinds": self._make_blinds(1, selected="small", selecting=True),
            "money": 4,
            "jokers": [],
            "consumables": {"count": 0, "limit": 2, "highlighted_limit": 1, "cards": []},
            "shop": {"count": 0, "limit": 0, "highlighted_limit": 0, "cards": []},
            "vouchers": {"count": 0, "limit": 0, "highlighted_limit": 0, "cards": []},
            "packs": {"count": 0, "limit": 0, "highlighted_limit": 0, "cards": []},
            "used_vouchers": [],
            "hands": self._default_hands(),
            "tags": [],
            "rules": {"stake": self._stake_name, "applied_modifiers": [], "degraded": False},
            "ante_num": 1,
            "round_num": 1,
            "done": False,
            "won": False,
        }
        self._begin_round(next_round=False)
        self._apply_stake_modifiers(self._stake_name)
        return copy.deepcopy(self._state)

    def get_state(self) -> dict[str, Any]:
        return copy.deepcopy(self._state)

    def _phase_default_action(self) -> dict[str, Any]:
        phase = str(self._state.get("state") or "UNKNOWN")
        if phase == "BLIND_SELECT":
            return {"action_type": "SELECT", "index": 0}
        if phase == "SELECTING_HAND":
            hand = self._state["hand"]["cards"]
            if hand:
                return {"action_type": "PLAY", "indices": [0]}
            return {"action_type": "WAIT"}
        if phase == "ROUND_EVAL":
            return {"action_type": "CASH_OUT"}
        if phase == "SHOP":
            return {"action_type": "NEXT_ROUND"}
        if phase in {"MENU", "GAME_OVER"}:
            return {"action_type": "START", "seed": self.seed}
        return {"action_type": "WAIT"}

    def _pop_cards_by_indices(self, indices: list[int]) -> list[dict[str, Any]]:
        hand = self._state["hand"]["cards"]
        if len(indices) > 5:
            raise ValueError("indices length must be <= 5")
        if len(indices) != len(set(indices)):
            raise ValueError("indices must be unique")
        if any(i < 0 or i >= len(hand) for i in indices):
            raise ValueError("indices out of range")
        selected = [hand[i] for i in indices]
        for i in sorted(indices, reverse=True):
            hand.pop(i)
        return selected

    def step(self, action: dict[str, Any]) -> tuple[dict[str, Any], float, bool, dict[str, Any]]:
        if not isinstance(action, dict):
            raise ValueError("action must be a dict")

        prev_chips = float((self._state.get("round") or {}).get("chips") or 0.0)
        info: dict[str, Any] = {"backend": "sim", "overridden": False}

        self._consume_rng_replay(action, info)

        action_type = str(action.get("action_type") or "WAIT").upper()
        if action_type == "AUTO":
            action = self._phase_default_action()
            action_type = str(action.get("action_type") or "WAIT").upper()
            info["overridden"] = True

        phase = str(self._state.get("state") or "UNKNOWN")

        if action_type == "START":
            seed = action.get("seed")
            raw_stake = action.get("stake")
            stake_name = self._sanitize_stake_name(raw_stake) if raw_stake is not None else self._stake_name
            st = self.reset(seed=seed if isinstance(seed, str) else self.seed, stake=stake_name)
            reward = float((st.get("round") or {}).get("chips") or 0.0) - prev_chips
            return st, reward, bool(st.get("done") or False), info

        if action_type == "MENU":
            self._state["state"] = "MENU"
            if isinstance(self._state.get("round"), dict):
                self._state["round"]["hands_left"] = 0
                self._state["round"]["discards_left"] = 0
                self._state["round"]["chips"] = 0.0
            if isinstance(self._state.get("score"), dict):
                self._state["score"]["chips"] = 0.0
            self._state["hand"]["cards"] = []
            self._state["played"]["cards"] = []
            self._state["discard"]["cards"] = []
            self._refresh_done_flags()
            now = copy.deepcopy(self._state)
            reward = float((now.get("round") or {}).get("chips") or 0.0) - prev_chips
            done = bool(now.get("done") or False)
            return now, reward, done, info

        if phase == "BLIND_SELECT":
            if action_type not in {"SELECT", "SKIP", "WAIT"}:
                if self.fail_fast:
                    raise ValueError(f"invalid action_type={action_type} for phase={phase}")
                fallback = self._phase_default_action()
                action_type = fallback["action_type"]
                action = fallback
                info["overridden"] = True
            idx = int(action.get("index", 0))
            idx = max(0, min(idx, len(BLIND_ORDER) - 1))
            blind = BLIND_ORDER[idx]
            self._state["round"]["blind"] = blind
            ante = int(self._state.get("ante_num") or 1)
            self._state["score"]["target_chips"] = float(self._target_scores(ante)[blind])
            self._state["state"] = "SELECTING_HAND"
            self._state["blinds"] = self._make_blinds(ante, selected=blind, selecting=False)
            self._draw_to_hand()

        elif phase == "SELECTING_HAND":
            if action_type == "DISCARD":
                if int(self._state["round"]["discards_left"]) <= 0:
                    raise ValueError("no_discards_left")
                else:
                    indices = [int(i) for i in (action.get("indices") or [])]
                    selected = self._pop_cards_by_indices(indices)
                    self._state["discard"]["cards"].extend(selected)
                    self._state["round"]["discards_left"] = int(self._state["round"]["discards_left"]) - 1
                    self._draw_to_hand()
                    if int(self._state["round"]["hands_left"]) <= 0 and int(self._state["round"]["discards_left"]) <= 0:
                        self._state["state"] = "GAME_OVER"
                    else:
                        self._state["state"] = "SELECTING_HAND"

            elif action_type == "PLAY":
                if int(self._state["round"]["hands_left"]) <= 0:
                    raise ValueError("no_hands_left")
                else:
                    indices = [int(i) for i in (action.get("indices") or [])]
                    if not indices:
                        raise ValueError("PLAY requires at least one index")
                    pre_state_for_expected = copy.deepcopy(self._state)
                    hand_before = copy.deepcopy((pre_state_for_expected.get("hand") or {}).get("cards") or [])
                    selected = self._pop_cards_by_indices(indices)
                    self._state["played"]["cards"] = selected
                    self._state["discard"]["cards"].extend(selected)

                    score_info = evaluate_selected_breakdown(selected)
                    hand_type = str(score_info.get("hand_type") or "")
                    base_chips = float(score_info.get("base_chips") or 0.0)
                    base_mult = float(score_info.get("base_mult") or 1.0)
                    gain = float(score_info.get("total_delta") or 0.0)

                    expected_context = action.get("expected_context") if isinstance(action.get("expected_context"), dict) else {}
                    planet_context = expected_context.get("planet") if isinstance(expected_context.get("planet"), dict) else {}
                    modifier_context = expected_context.get("modifier") if isinstance(expected_context.get("modifier"), dict) else {}
                    jokers_context = expected_context.get("jokers") if isinstance(expected_context.get("jokers"), list) else []
                    use_expected = (
                        (planet_context and bool(planet_context.get("applied", True)))
                        or (modifier_context and bool(modifier_context.get("applied", True)))
                        or bool(jokers_context)
                    )
                    if use_expected:
                        try:
                            expected = compute_expected_for_action(pre_state_for_expected, action)
                            if bool(expected.get("available")):
                                gain = float(expected.get("score") or gain)
                                bonus_mult = (
                                    float(expected.get("planet_bonus_mult") or 0.0)
                                    + float(expected.get("modifier_bonus_mult_add") or 0.0)
                                    + float(expected.get("joker_bonus_mult_add") or 0.0)
                                )
                                bonus_scale = (
                                    float(expected.get("modifier_bonus_mult_scale") or 1.0)
                                    * float(expected.get("joker_bonus_mult_scale") or 1.0)
                                )
                                base_mult = float((base_mult + bonus_mult) * bonus_scale)
                        except Exception:
                            if self.fail_fast:
                                raise

                    self._state["round"]["chips"] = float(self._state["round"]["chips"]) + gain
                    self._state["score"]["chips"] = float(self._state["round"]["chips"])
                    self._state["score"]["mult"] = float(base_mult)
                    self._state["score"]["last_hand_type"] = hand_type
                    self._state["score"]["last_base_chips"] = float(base_chips)
                    self._state["score"]["last_base_mult"] = float(base_mult)

                    self._state["round"]["hands_left"] = int(self._state["round"]["hands_left"]) - 1
                    self._draw_to_hand()

                    if float(self._state["round"]["chips"]) >= float(self._state["score"]["target_chips"]):
                        self._state["state"] = "ROUND_EVAL"
                    elif int(self._state["round"]["hands_left"]) <= 0 and int(self._state["round"]["discards_left"]) <= 0:
                        self._state["state"] = "GAME_OVER"
                    else:
                        self._state["state"] = "SELECTING_HAND"
            elif action_type == "WAIT":
                pass
            else:
                if self.fail_fast:
                    raise ValueError(f"invalid action_type={action_type} for phase={phase}")
                fallback = self._phase_default_action()
                info["overridden"] = True
                return self.step(fallback)

        elif phase == "ROUND_EVAL":
            if action_type in {"CASH_OUT", "WAIT"}:
                if action_type == "CASH_OUT":
                    payout = max(1, int(float(self._state["round"]["chips"]) // 100))
                    self._state["money"] = int(self._state.get("money") or 0) + payout
                self._state["state"] = "SHOP"
            else:
                if self.fail_fast:
                    raise ValueError(f"invalid action_type={action_type} for phase={phase}")
                info["overridden"] = True
                return self.step({"action_type": "CASH_OUT"})

        elif phase == "SHOP":
            if action_type == "NEXT_ROUND":
                self._begin_round(next_round=True)
            elif action_type == "REROLL":
                self._state["money"] = max(0, int(self._state.get("money") or 0) - int((self._state.get("round") or {}).get("reroll_cost") or 0))
            elif action_type == "BUY":
                params = action.get("params") if isinstance(action.get("params"), dict) else {}
                if "pack" in params:
                    self._state["state"] = "SMODS_BOOSTER_OPENED"
            elif action_type in {"PACK", "SELL", "USE", "WAIT"}:
                pass
            else:
                if self.fail_fast:
                    raise ValueError(f"invalid action_type={action_type} for phase={phase}")
                info["overridden"] = True
                return self.step({"action_type": "NEXT_ROUND"})
            self._apply_expected_shop_context(action, info)

        elif phase == "SMODS_BOOSTER_OPENED":
            if action_type in {"PACK", "SKIP", "WAIT"}:
                self._state["state"] = "SHOP"
            elif action_type == "NEXT_ROUND":
                self._begin_round(next_round=True)
            else:
                if self.fail_fast:
                    raise ValueError(f"invalid action_type={action_type} for phase={phase}")
                info["overridden"] = True
                self._state["state"] = "SHOP"
            self._apply_expected_shop_context(action, info)

        elif phase in {"MENU", "GAME_OVER"}:
            if self.fail_fast:
                raise ValueError(f"action on terminal/menu phase requires explicit START, got {action_type}")
            info["overridden"] = True
            return self.step({"action_type": "START", "seed": self.seed})

        else:
            if action_type == "WAIT":
                pass
            else:
                if self.fail_fast:
                    raise ValueError(f"unknown phase={phase} with action_type={action_type}")
                info["overridden"] = True
                return self.step(self._phase_default_action())

        self._refresh_done_flags()
        now = copy.deepcopy(self._state)
        reward = float((now.get("round") or {}).get("chips") or 0.0) - prev_chips
        done = bool(now.get("done") or False)
        return now, reward, done, info
