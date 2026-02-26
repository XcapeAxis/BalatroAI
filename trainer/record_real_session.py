from __future__ import annotations

if __package__ is None or __package__ == "":
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).resolve().parent.parent))

import argparse
import hashlib
import json
import secrets
import time
from collections import Counter
from pathlib import Path
from typing import Any

from trainer import action_space_shop
from trainer.env_client import create_backend
from trainer.infer_assistant_real import _heuristic_hand_rankings, _heuristic_shop_rankings
from trainer.real_observer import build_observation
from trainer.utils import setup_logger, timestamp, warn_if_unstable_python

CONFIRM_TEXT = "I_UNDERSTAND"
ARM_TOKEN_STATE = Path("docs/artifacts/p14/arm_tokens/latest.json")


def _write_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _stable_hash(value: Any) -> str:
    text = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _new_arm_token() -> str:
    token = secrets.token_urlsafe(18)
    ARM_TOKEN_STATE.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "issued_at": timestamp(),
        "used": False,
        "token_sha256": hashlib.sha256(token.encode("utf-8")).hexdigest(),
    }
    ARM_TOKEN_STATE.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return token


def _consume_arm_token(token: str) -> tuple[bool, str]:
    if not ARM_TOKEN_STATE.exists():
        return False, "arm_token_not_initialized"
    try:
        payload = json.loads(ARM_TOKEN_STATE.read_text(encoding="utf-8"))
    except Exception:
        return False, "arm_token_state_corrupt"
    if not isinstance(payload, dict):
        return False, "arm_token_state_corrupt"
    if bool(payload.get("used")):
        return False, "arm_token_already_used"
    expected = str(payload.get("token_sha256") or "")
    got = hashlib.sha256(token.encode("utf-8")).hexdigest()
    if not expected or expected != got:
        return False, "arm_token_mismatch"
    payload["used"] = True
    payload["used_at"] = timestamp()
    ARM_TOKEN_STATE.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return True, ""


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Real session recorder (shadow by default, controlled execute optional).")
    parser.add_argument("--base-url", default="http://127.0.0.1:12346")
    parser.add_argument("--model", default="", help="Reserved for compatibility; current recorder uses heuristic suggestions.")
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--interval", type=float, default=1.0)
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument("--mode", choices=["shadow", "execute"], default="", help=argparse.SUPPRESS)
    parser.add_argument("--execute", action="store_true", help="Enable controlled execute mode.")
    parser.add_argument("--print-arm-token", action="store_true", help="Print one-time arm token and exit.")
    parser.add_argument("--arm-token", default="", help="One-time arm token required for --execute.")
    parser.add_argument("--confirm", default="", help='Must equal "I_UNDERSTAND" when --execute is used.')
    parser.add_argument("--max-actions", type=int, default=12)
    parser.add_argument("--rate-limit-sec", type=float, default=2.0)
    parser.add_argument(
        "--strict-errors",
        dest="strict_errors",
        action="store_true",
        default=True,
        help="Exit immediately when fetch/execute errors occur (default: true).",
    )
    parser.add_argument(
        "--no-strict-errors",
        dest="strict_errors",
        action="store_false",
        help="Continue recording after errors when possible.",
    )
    parser.add_argument("--include-raw", action="store_true", help="Attach raw gamestate per row (needed for dagger reconstruction).")
    parser.add_argument("--out", default="", help="Output session jsonl path.")
    return parser.parse_args()


def _suggestions(state: dict[str, Any], topk: int) -> list[dict[str, Any]]:
    phase = str(state.get("state") or "UNKNOWN")
    if phase == "SELECTING_HAND":
        return _heuristic_hand_rankings(state, topk=topk)
    if phase in action_space_shop.SHOP_PHASES:
        return _heuristic_shop_rankings(state, topk=topk)
    return []


def _choose_safe_action(state: dict[str, Any], suggestions: list[dict[str, Any]]) -> dict[str, Any] | None:
    phase = str(state.get("state") or "UNKNOWN")
    if phase == "SELECTING_HAND":
        round_info = state.get("round") if isinstance(state.get("round"), dict) else {}
        discards_left = int(round_info.get("discards_left") or 0)
        hand_cards = (state.get("hand") or {}).get("cards") or []
        if discards_left > 0 and hand_cards:
            return {"action_type": "DISCARD", "indices": [0]}
        if hand_cards:
            return {"action_type": "PLAY", "indices": [0]}
        return None

    if phase in action_space_shop.SHOP_PHASES:
        return {"action_type": "NEXT_ROUND"}
    return None


def main() -> int:
    args = _parse_args()
    logger = setup_logger("trainer.record_real_session")
    warn_if_unstable_python(logger)

    if args.print_arm_token:
        print(_new_arm_token())
        return 0

    execute_enabled = bool(args.execute or str(args.mode).lower() == "execute")
    mode = "execute" if execute_enabled else "shadow"
    if execute_enabled:
        if not args.arm_token:
            logger.error("execute requested but --arm-token missing")
            return 2
        if args.confirm != CONFIRM_TEXT:
            logger.error('execute requested but --confirm is invalid (expected "%s")', CONFIRM_TEXT)
            return 2
        ok, reason = _consume_arm_token(str(args.arm_token))
        if not ok:
            logger.error("execute token rejected: %s", reason)
            return 2
        logger.info("MODE=EXECUTE (controlled)")
    else:
        logger.info("MODE=SHADOW")

    if not args.out:
        logger.error("--out is required unless --print-arm-token is used")
        return 2
    out_path = Path(args.out)
    if out_path.exists():
        out_path.unlink()
    backend = create_backend("real", base_url=args.base_url, timeout_sec=8.0, seed="AAAAAAA", logger=logger)

    phase_counter: Counter[str] = Counter()
    written = 0
    actions_count = 0
    changed_steps = 0
    last_exec_ts = 0.0
    hard_error = False

    try:
        for step_idx in range(max(1, int(args.steps))):
            try:
                before_state = backend.get_state()
            except Exception as exc:
                row = {
                    "ts": timestamp(),
                    "step_idx": step_idx,
                    "base_url": args.base_url,
                    "mode": mode,
                    "errors": [f"fetch_failed:{exc}"],
                }
                _write_jsonl(out_path, row)
                written += 1
                if args.strict_errors:
                    logger.error("step=%d fetch failed (strict): %s", step_idx, exc)
                    hard_error = True
                    break
                logger.warning("step=%d fetch failed: %s", step_idx, exc)
                time.sleep(max(0.05, float(args.interval)))
                continue

            obs = build_observation(before_state)
            phase = str(obs.get("phase") or "UNKNOWN")
            phase_counter[phase] += 1

            hand_keys = [str(c.get("key") or "") for c in (obs.get("hand", {}).get("cards") or [])]
            shop_cards = obs.get("shop", {}).get("shop", {}).get("cards") or []
            shop_offers = [
                {
                    "index": int(c.get("index") or 0),
                    "key": str(c.get("key") or ""),
                    "cost": float(c.get("cost") or 0.0),
                    "kind": "shop",
                }
                for c in shop_cards
                if isinstance(c, dict)
            ]
            suggestions = _suggestions(before_state, topk=max(1, int(args.topk)))
            before_hash = _stable_hash(obs)
            row_errors: list[str] = []
            action_sent: dict[str, Any] | None = None
            action_result: dict[str, Any] | None = None
            after_obs = obs
            after_state = before_state

            if execute_enabled and actions_count < max(0, int(args.max_actions)):
                now = time.time()
                if (now - last_exec_ts) >= float(args.rate_limit_sec):
                    safe_action = _choose_safe_action(before_state, suggestions)
                    if safe_action is not None:
                        try:
                            action_sent = dict(safe_action)
                            next_state, reward, done, info = backend.step(action_sent)
                            after_state = next_state
                            after_obs = build_observation(after_state)
                            action_result = {
                                "reward": float(reward),
                                "done": bool(done),
                                "info": info,
                            }
                            actions_count += 1
                            last_exec_ts = now
                        except Exception as exc:
                            row_errors.append(f"execute_failed:{exc}")
                            if args.strict_errors:
                                logger.error("step=%d execute failed (strict): %s", step_idx, exc)
                                hard_error = True
                            else:
                                execute_enabled = False
                                mode = "shadow"
                                logger.error("step=%d execute failed, downgrade to shadow: %s", step_idx, exc)
                    else:
                        row_errors.append("no_safe_action")
                else:
                    row_errors.append("rate_limited")

            after_hash = _stable_hash(after_obs)
            if after_hash != before_hash:
                changed_steps += 1

            row = {
                "ts": timestamp(),
                "step_idx": step_idx,
                "base_url": args.base_url,
                "mode": mode,
                "phase": phase,
                "gamestate_min": obs,
                "gamestate_min_before": obs,
                "gamestate_min_after": after_obs,
                "hand_cards": hand_keys,
                "shop_offers": shop_offers,
                "model_suggestions_topk": suggestions,
                "action_sent": action_sent,
                "action_result": action_result,
                "state_hash_before": before_hash,
                "state_hash_after": after_hash,
                "state_changed": bool(after_hash != before_hash),
                "errors": row_errors,
            }
            if args.include_raw or args.execute:
                row["gamestate_raw_before"] = before_state
                row["gamestate_raw_after"] = after_state

            _write_jsonl(out_path, row)
            written += 1
            logger.info(
                "step=%d phase=%s hand=%d shop=%d action=%s changed=%s",
                step_idx,
                phase,
                len(hand_keys),
                len(shop_offers),
                json.dumps(action_sent, ensure_ascii=False) if action_sent else "-",
                bool(after_hash != before_hash),
            )
            if hard_error:
                break
            if step_idx + 1 < int(args.steps):
                time.sleep(max(0.05, float(args.interval)))
    finally:
        backend.close()

    summary = {
        "out": str(out_path),
        "steps_written": written,
        "actions_count": actions_count,
        "state_changed_steps": changed_steps,
        "phase_distribution": dict(phase_counter),
    }
    summary_path = out_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("session recorded: steps=%d out=%s", written, out_path)
    logger.info("summary: %s", summary_path)
    if hard_error:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
