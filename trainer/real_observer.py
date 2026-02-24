from __future__ import annotations

if __package__ is None or __package__ == "":
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).resolve().parent.parent))

import argparse
import json
import time
from pathlib import Path
from typing import Any

from trainer.env_client import get_state
from trainer.utils import setup_logger, warn_if_unstable_python


def get_gamestate(base_url: str, timeout_sec: float = 8.0) -> dict[str, Any]:
    return get_state(base_url, timeout=timeout_sec)


def _card_key(card: dict[str, Any]) -> str:
    return str(card.get("key") or "")


def _card_rank(card: dict[str, Any]) -> str:
    value = card.get("value") if isinstance(card.get("value"), dict) else {}
    rank = card.get("rank") or value.get("rank")
    return str(rank or "")


def _card_suit(card: dict[str, Any]) -> str:
    value = card.get("value") if isinstance(card.get("value"), dict) else {}
    suit = card.get("suit") or value.get("suit")
    return str(suit or "")


def _card_effect(card: dict[str, Any]) -> str:
    value = card.get("value") if isinstance(card.get("value"), dict) else {}
    effect = value.get("effect")
    return str(effect or "")


def _normalize_tags(value: Any) -> list[str]:
    if isinstance(value, list):
        out = [str(x).strip() for x in value if str(x).strip()]
        return sorted(set(out))
    if isinstance(value, dict):
        out = [str(k).strip() for k, v in value.items() if v]
        return sorted(set(x for x in out if x))
    if value:
        return [str(value)]
    return []


def parse_hand(gamestate: dict[str, Any]) -> dict[str, Any]:
    hand = gamestate.get("hand") if isinstance(gamestate.get("hand"), dict) else {}
    cards_raw = hand.get("cards") if isinstance(hand.get("cards"), list) else []
    cards: list[dict[str, Any]] = []
    for idx, card in enumerate(cards_raw):
        if not isinstance(card, dict):
            continue
        cards.append(
            {
                "index": idx,
                "key": _card_key(card),
                "rank": _card_rank(card),
                "suit": _card_suit(card),
                "effect": _card_effect(card),
                "modifier": _normalize_tags(card.get("modifier")),
                "state": _normalize_tags(card.get("state")),
            }
        )
    return {"hand_size": len(cards), "cards": cards}


def _parse_market_block(block: Any) -> dict[str, Any]:
    if not isinstance(block, dict):
        return {"count": 0, "cards": []}
    cards_raw = block.get("cards") if isinstance(block.get("cards"), list) else []
    cards: list[dict[str, Any]] = []
    for idx, card in enumerate(cards_raw):
        if not isinstance(card, dict):
            continue
        cards.append(
            {
                "index": idx,
                "key": str(card.get("key") or ""),
                "label": str(card.get("label") or ""),
                "cost": float(card.get("cost") or 0.0),
                "set": str(card.get("set") or ""),
            }
        )
    count = int(block.get("count") or len(cards))
    return {"count": count, "cards": cards}


def parse_shop(gamestate: dict[str, Any]) -> dict[str, Any]:
    return {
        "phase": str(gamestate.get("state") or "UNKNOWN"),
        "shop": _parse_market_block(gamestate.get("shop")),
        "vouchers": _parse_market_block(gamestate.get("vouchers")),
        "packs": _parse_market_block(gamestate.get("packs")),
        "consumables": _parse_market_block(gamestate.get("consumables")),
    }


def parse_resources(gamestate: dict[str, Any]) -> dict[str, Any]:
    round_info = gamestate.get("round") if isinstance(gamestate.get("round"), dict) else {}
    score_info = gamestate.get("score") if isinstance(gamestate.get("score"), dict) else {}
    economy = gamestate.get("economy") if isinstance(gamestate.get("economy"), dict) else {}
    return {
        "phase": str(gamestate.get("state") or "UNKNOWN"),
        "hands_left": int(round_info.get("hands_left") or 0),
        "discards_left": int(round_info.get("discards_left") or 0),
        "ante": int(round_info.get("ante") or 0),
        "round_num": int(round_info.get("round_num") or 0),
        "blind": str(round_info.get("blind") or ""),
        "round_chips": float(round_info.get("chips") or 0.0),
        "score_chips": float(score_info.get("chips") or 0.0),
        "score_mult": float(score_info.get("mult") or 0.0),
        "target_chips": float(score_info.get("target_chips") or 0.0),
        "money": float(economy.get("money") or 0.0),
    }


def build_observation(gamestate: dict[str, Any]) -> dict[str, Any]:
    return {
        "phase": str(gamestate.get("state") or "UNKNOWN"),
        "resources": parse_resources(gamestate),
        "hand": parse_hand(gamestate),
        "shop": parse_shop(gamestate),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Read-only observer for real balatrobot gamestate.")
    parser.add_argument("--base-url", default="http://127.0.0.1:12346")
    parser.add_argument("--timeout-sec", type=float, default=8.0)
    parser.add_argument("--once", action="store_true", help="Read one snapshot and exit.")
    parser.add_argument("--loop", action="store_true", help="Keep polling snapshots.")
    parser.add_argument("--interval", type=float, default=1.0)
    parser.add_argument("--out", default="", help="Optional JSON output path.")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    logger = setup_logger("trainer.real_observer")
    warn_if_unstable_python(logger)

    loop = bool(args.loop and not args.once)
    outputs: list[dict[str, Any]] = []

    while True:
        try:
            state = get_gamestate(args.base_url, timeout_sec=args.timeout_sec)
        except Exception as exc:
            logger.error("failed to fetch gamestate from %s: %s", args.base_url, exc)
            return 2

        obs = build_observation(state)
        outputs.append(obs)
        logger.info(
            "phase=%s hand=%d hands_left=%d discards_left=%d money=%.2f",
            obs["phase"],
            int(obs["hand"]["hand_size"]),
            int(obs["resources"]["hands_left"]),
            int(obs["resources"]["discards_left"]),
            float(obs["resources"]["money"]),
        )
        print(json.dumps(obs, ensure_ascii=False))

        if not loop:
            break
        time.sleep(max(0.05, float(args.interval)))

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload: Any = outputs[0] if len(outputs) == 1 else outputs
        out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info("wrote output: %s", out_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
