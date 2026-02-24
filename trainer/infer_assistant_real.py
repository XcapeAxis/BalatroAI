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

from sim.core.score_basic import evaluate_selected_breakdown
from trainer import action_space, action_space_shop
from trainer.env_client import create_backend
from trainer.features import extract_features
from trainer.infer_assistant import (
    _forward_hand,
    _load_model,
    _predict_topk,
    _require_torch,
    _state_to_hand_batch,
    _state_to_shop_batch,
)
from trainer.real_observer import build_observation
from trainer.utils import setup_logger, warn_if_unstable_python


def _latest_model() -> Path | None:
    root = Path("trainer_runs")
    if not root.exists():
        return None
    bests = sorted(root.rglob("best.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
    if bests:
        return bests[0]
    lasts = sorted(root.rglob("last.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
    if lasts:
        return lasts[0]
    return None


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Real backend inference assistant (read-only by default).")
    parser.add_argument("--base-url", default="http://127.0.0.1:12346")
    parser.add_argument("--model", default="", help="Optional model path. Auto-detects latest best.pt if omitted.")
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument("--interval", type=float, default=0.5)
    parser.add_argument("--timeout-sec", type=float, default=8.0)
    parser.add_argument("--seed", default="AAAAAAA")
    parser.add_argument("--execute", action="store_true", help="Enable action execution on real backend.")
    parser.add_argument("--once", action="store_true", help="Run one observation/suggestion cycle.")
    parser.add_argument("--loop", action="store_true", help="Run continuously.")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--out", default="", help="Optional output path (txt/json).")
    return parser.parse_args()


def _safe_cards(state: dict[str, Any]) -> list[dict[str, Any]]:
    hand = state.get("hand") if isinstance(state.get("hand"), dict) else {}
    cards = hand.get("cards") if isinstance(hand.get("cards"), list) else []
    return [c for c in cards if isinstance(c, dict)]


def _heuristic_hand_rankings(state: dict[str, Any], topk: int) -> list[dict[str, Any]]:
    cards = _safe_cards(state)
    hand_size = min(len(cards), action_space.MAX_HAND)
    if hand_size <= 0:
        return []

    legal_ids = action_space.legal_action_ids(hand_size)
    ranked: list[dict[str, Any]] = []
    discards_left = int(((state.get("round") or {}).get("discards_left") or 0))
    for aid in legal_ids:
        atype, mask_int = action_space.decode(hand_size, aid)
        indices = action_space.mask_to_indices(mask_int, hand_size)
        if atype == "PLAY":
            selected = [cards[i] for i in indices if 0 <= i < len(cards)]
            breakdown = evaluate_selected_breakdown(selected)
            proxy = float(breakdown.get("total_delta") or 0.0)
            explain = {
                "hand_type": str(breakdown.get("hand_type") or ""),
                "expected_delta": proxy,
                "base_chips": float(breakdown.get("base_chips") or 0.0),
                "base_mult": float(breakdown.get("base_mult") or 0.0),
            }
        else:
            keep_bonus = max(0, discards_left) * 0.1
            proxy = keep_bonus - (0.01 * len(indices))
            explain = {"hand_type": "DISCARD", "expected_delta": 0.0, "discards_left": discards_left}

        ranked.append(
            {
                "action_id": int(aid),
                "action_type": atype,
                "indices": indices,
                "score": float(proxy),
                "explain": explain,
            }
        )

    ranked.sort(key=lambda x: float(x["score"]), reverse=True)
    return ranked[: max(1, int(topk))]


def _heuristic_shop_rankings(state: dict[str, Any], topk: int) -> list[dict[str, Any]]:
    legal_ids = action_space_shop.legal_action_ids(state)
    economy = state.get("economy") if isinstance(state.get("economy"), dict) else {}
    money = float(economy.get("money") or 0.0)
    ranked: list[dict[str, Any]] = []

    for aid in legal_ids:
        action = action_space_shop.action_from_id(state, aid)
        at = str(action.get("action_type") or "")
        score = 0.0
        if at == "NEXT_ROUND":
            score = 0.9
        elif at == "BUY":
            score = 0.7 if money >= 4 else 0.2
        elif at == "PACK":
            score = 0.65
        elif at == "REROLL":
            score = 0.5 if money >= 5 else 0.1
        elif at == "SELL":
            score = 0.3
        elif at == "USE":
            score = 0.45
        else:
            score = 0.05

        ranked.append(
            {
                "action_id": int(aid),
                "action": action,
                "score": float(score),
                "explain": {"money": money, "phase": str(state.get("state") or "")},
            }
        )
    ranked.sort(key=lambda x: float(x["score"]), reverse=True)
    return ranked[: max(1, int(topk))]


def main() -> int:
    args = _parse_args()
    logger = setup_logger("trainer.infer_real")
    warn_if_unstable_python(logger)

    mode = "EXECUTE" if args.execute else "READONLY"
    logger.info("MODE=%s", mode)

    model_path: Path | None = None
    if args.model:
        candidate = Path(args.model)
        if candidate.exists():
            model_path = candidate
    if model_path is None:
        model_path = _latest_model()
    if model_path is not None:
        logger.info("Using model: %s", model_path)
    else:
        logger.warning("No model found; fallback to heuristic ranking.")

    torch = None
    nn = None
    device = None
    model = None
    is_multi = False
    if model_path is not None:
        try:
            torch, nn = _require_torch()
            if args.device == "auto":
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            else:
                device = torch.device(args.device)
            model, is_multi = _load_model(
                argparse.Namespace(
                    model=str(model_path),
                    max_actions=action_space.max_actions(),
                    max_shop_actions=action_space_shop.max_actions(),
                ),
                torch,
                nn,
                device,
            )
            model.eval()
            logger.info("Loaded model successfully (multi_head=%s).", is_multi)
        except Exception as exc:
            logger.warning("Model load failed, using heuristic only: %s", exc)
            model = None
            is_multi = False

    backend = create_backend("real", base_url=args.base_url, timeout_sec=args.timeout_sec, seed=args.seed, logger=logger)

    loop = bool(args.loop and not args.once)
    output_rows: list[dict[str, Any]] = []

    try:
        while True:
            try:
                state = backend.get_state()
            except Exception as exc:
                logger.error("Failed to fetch real gamestate: %s", exc)
                return 2

            obs = build_observation(state)
            phase = str(state.get("state") or "UNKNOWN")
            logger.info(
                "phase=%s hand=%d hands_left=%d discards_left=%d money=%.2f",
                phase,
                int(obs["hand"]["hand_size"]),
                int(obs["resources"]["hands_left"]),
                int(obs["resources"]["discards_left"]),
                float(obs["resources"]["money"]),
            )

            row: dict[str, Any] = {"mode": mode, "phase": phase, "observation": obs}

            if phase == "SELECTING_HAND":
                ranked: list[dict[str, Any]]
                if model is not None and torch is not None and device is not None:
                    cards = _safe_cards(state)
                    hand_size = min(len(cards), action_space.MAX_HAND)
                    if hand_size <= 0:
                        ranked = []
                    else:
                        legal_ids = action_space.legal_action_ids(hand_size)
                        batch = _state_to_hand_batch(state, torch)
                        batch = {k: v.to(device) for k, v in batch.items()}
                        with torch.no_grad():
                            logits = _forward_hand(model, batch, is_multi)
                        topk_pairs = _predict_topk(
                            logits,
                            legal_ids,
                            action_space.max_actions(),
                            args.topk,
                            torch,
                            device,
                        )
                        ranked = []
                        for aid, score in topk_pairs:
                            at, mask_int = action_space.decode(hand_size, aid)
                            idxs = action_space.mask_to_indices(mask_int, hand_size)
                            selected = [cards[i] for i in idxs if 0 <= i < len(cards)]
                            breakdown = evaluate_selected_breakdown(selected) if at == "PLAY" else {}
                            ranked.append(
                                {
                                    "action_id": int(aid),
                                    "action_type": at,
                                    "indices": idxs,
                                    "score": float(score),
                                    "explain": {
                                        "hand_type": str(breakdown.get("hand_type") or at),
                                        "expected_delta": float(breakdown.get("total_delta") or 0.0),
                                        "base_chips": float(breakdown.get("base_chips") or 0.0),
                                        "base_mult": float(breakdown.get("base_mult") or 0.0),
                                    },
                                }
                            )
                else:
                    ranked = _heuristic_hand_rankings(state, args.topk)

                logger.info("Top-%d hand suggestions:", len(ranked))
                for idx, item in enumerate(ranked, start=1):
                    logger.info(
                        "  #%d %s indices=%s score=%.4f explain=%s",
                        idx,
                        item.get("action_type"),
                        item.get("indices"),
                        float(item.get("score") or 0.0),
                        item.get("explain"),
                    )
                row["topk"] = ranked

                if args.execute and ranked:
                    top = ranked[0]
                    action = {"action_type": str(top["action_type"]), "indices": list(top.get("indices") or [])}
                    after, reward, done, info = backend.step(action)
                    logger.info("Executed top-1 hand action=%s reward=%.4f done=%s info=%s", action, reward, done, info)
                    row["executed"] = {"action": action, "reward": float(reward), "done": bool(done)}
                    row["after_phase"] = str(after.get("state") or "UNKNOWN")

            elif phase in action_space_shop.SHOP_PHASES:
                ranked_shop: list[dict[str, Any]]
                if model is not None and torch is not None and device is not None and is_multi:
                    legal_ids = action_space_shop.legal_action_ids(state)
                    batch = _state_to_shop_batch(state, torch)
                    batch = {k: v.to(device) for k, v in batch.items()}
                    with torch.no_grad():
                        logits = model.forward_shop(batch)
                    topk_pairs = _predict_topk(
                        logits,
                        legal_ids,
                        action_space_shop.max_actions(),
                        args.topk,
                        torch,
                        device,
                    )
                    ranked_shop = []
                    for aid, score in topk_pairs:
                        action = action_space_shop.action_from_id(state, aid)
                        ranked_shop.append(
                            {
                                "action_id": int(aid),
                                "action": action,
                                "score": float(score),
                                "explain": {"phase": phase},
                            }
                        )
                else:
                    ranked_shop = _heuristic_shop_rankings(state, args.topk)

                logger.info("Top-%d shop suggestions:", len(ranked_shop))
                for idx, item in enumerate(ranked_shop, start=1):
                    logger.info(
                        "  #%d action=%s score=%.4f explain=%s",
                        idx,
                        item.get("action"),
                        float(item.get("score") or 0.0),
                        item.get("explain"),
                    )
                row["topk"] = ranked_shop

                if args.execute and ranked_shop:
                    action = dict(ranked_shop[0].get("action") or {"action_type": "WAIT"})
                    after, reward, done, info = backend.step(action)
                    logger.info("Executed top-1 shop action=%s reward=%.4f done=%s info=%s", action, reward, done, info)
                    row["executed"] = {"action": action, "reward": float(reward), "done": bool(done)}
                    row["after_phase"] = str(after.get("state") or "UNKNOWN")
            else:
                logger.info("No decision head for phase=%s (observation only).", phase)

            output_rows.append(row)
            if not loop:
                break
            time.sleep(max(0.05, float(args.interval)))
    finally:
        backend.close()

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload: Any = output_rows[0] if len(output_rows) == 1 else output_rows
        out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info("wrote output: %s", out_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
