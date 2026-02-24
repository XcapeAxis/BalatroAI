from __future__ import annotations

if __package__ is None or __package__ == "":
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).resolve().parent.parent))

import argparse
import json
import secrets
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from trainer import action_space_shop
from trainer.env_client import create_backend
from trainer.infer_assistant_real import _heuristic_hand_rankings, _heuristic_shop_rankings
from trainer.real_observer import build_observation
from trainer.utils import setup_logger, timestamp, warn_if_unstable_python

CONFIRM_TEXT = "I_UNDERSTAND"


@dataclass
class ExecutorState:
    mode: str
    execute_enabled: bool
    last_exec_ts: float


def _projection(state: dict[str, Any]) -> dict[str, Any]:
    obs = build_observation(state)
    return {
        "phase": obs.get("phase"),
        "resources": obs.get("resources"),
        "hand_keys": [str(c.get("key") or "") for c in (obs.get("hand", {}).get("cards") or [])],
        "shop_keys": [str(c.get("key") or "") for c in (obs.get("shop", {}).get("shop", {}).get("cards") or [])],
        "voucher_keys": [str(c.get("key") or "") for c in (obs.get("shop", {}).get("vouchers", {}).get("cards") or [])],
        "pack_keys": [str(c.get("key") or "") for c in (obs.get("shop", {}).get("packs", {}).get("cards") or [])],
        "consumable_keys": [str(c.get("key") or "") for c in (obs.get("shop", {}).get("consumables", {}).get("cards") or [])],
    }


def _select_top_action(state: dict[str, Any], topk: int) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    phase = str(state.get("state") or "UNKNOWN")
    if phase == "SELECTING_HAND":
        ranked = _heuristic_hand_rankings(state, topk=topk)
        if not ranked:
            return [], None
        action = {"action_type": str(ranked[0]["action_type"]), "indices": list(ranked[0]["indices"])}
        return ranked, action
    if phase in action_space_shop.SHOP_PHASES:
        ranked_shop = _heuristic_shop_rankings(state, topk=topk)
        if not ranked_shop:
            return [], None
        action = dict(ranked_shop[0]["action"])
        return ranked_shop, action
    return [], None


def _write_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Safe real executor with explicit arm token and readonly-by-default mode.")
    parser.add_argument("--base-url", default="http://127.0.0.1:12346")
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--loop", action="store_true")
    parser.add_argument("--interval", type=float, default=1.0)
    parser.add_argument("--execute", action="store_true", help="Enable execute mode (still requires arm token + confirm).")
    parser.add_argument("--arm-token", default="")
    parser.add_argument("--confirm", default="")
    parser.add_argument("--print-arm-token", action="store_true", help="Print a random one-time arm token and exit.")
    parser.add_argument("--min-action-interval", type=float, default=2.0, help="Rate limit: minimum seconds between actions.")
    parser.add_argument("--out-dir", default="docs/artifacts/p13/exec_logs")
    return parser.parse_args()


def _resolve_mode(args: argparse.Namespace, logger) -> ExecutorState:
    execute_requested = bool(args.execute)
    if not execute_requested:
        return ExecutorState(mode="READONLY", execute_enabled=False, last_exec_ts=0.0)
    if not args.arm_token:
        logger.warning("execute requested but --arm-token missing; forcing READONLY")
        return ExecutorState(mode="READONLY", execute_enabled=False, last_exec_ts=0.0)
    if args.confirm != CONFIRM_TEXT:
        logger.warning("execute requested but --confirm invalid; forcing READONLY")
        return ExecutorState(mode="READONLY", execute_enabled=False, last_exec_ts=0.0)
    return ExecutorState(mode="EXECUTE", execute_enabled=True, last_exec_ts=0.0)


def main() -> int:
    args = _parse_args()
    logger = setup_logger("trainer.real_executor")
    warn_if_unstable_python(logger)

    if args.print_arm_token:
        print(secrets.token_urlsafe(16))
        return 0

    ex = _resolve_mode(args, logger)
    logger.info("MODE=%s", ex.mode)

    stamp = time.strftime("%Y%m%d-%H%M%S")
    out_dir = Path(args.out_dir) / stamp
    log_path = out_dir / "exec.jsonl"
    summary_path = out_dir / "summary.json"

    backend = create_backend("real", base_url=args.base_url, timeout_sec=8.0, seed="AAAAAAA", logger=logger)
    loop = bool(args.loop and not args.once)
    rows = 0

    try:
        while True:
            try:
                before = backend.get_state()
            except Exception as exc:
                logger.error("failed to fetch pre-state: %s", exc)
                return 2

            ranked, top_action = _select_top_action(before, topk=max(1, int(args.topk)))
            row: dict[str, Any] = {
                "timestamp": timestamp(),
                "mode": ex.mode,
                "base_url": args.base_url,
                "phase": str(before.get("state") or "UNKNOWN"),
                "before": _projection(before),
                "topk": ranked,
            }

            if ex.execute_enabled and top_action is not None:
                now = time.time()
                if (now - ex.last_exec_ts) < float(args.min_action_interval):
                    row["executed"] = False
                    row["failure_reason"] = "rate_limited"
                else:
                    try:
                        after, reward, done, info = backend.step(top_action)
                        row["executed"] = True
                        row["action"] = top_action
                        row["after"] = _projection(after)
                        row["reward"] = float(reward)
                        row["done"] = bool(done)
                        row["info"] = info
                        ex.last_exec_ts = now
                    except Exception as exc:
                        row["executed"] = False
                        row["failure_reason"] = f"execute_failed:{exc}"
                        ex.mode = "READONLY"
                        ex.execute_enabled = False
                        logger.error("execute failed, downgrade to READONLY: %s", exc)
            else:
                row["executed"] = False
                if top_action is None:
                    row["failure_reason"] = "no_action_for_phase"

            _write_jsonl(log_path, row)
            rows += 1
            if not loop:
                break
            time.sleep(max(0.05, float(args.interval)))
    finally:
        backend.close()

    summary = {"timestamp": stamp, "mode": ex.mode, "rows": rows, "log_path": str(log_path)}
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("wrote %d rows: %s", rows, log_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
