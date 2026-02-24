from __future__ import annotations

if __package__ is None or __package__ == "":
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).resolve().parent.parent))

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from trainer import action_space_shop
from trainer.env_client import create_backend
from trainer.infer_assistant_real import _heuristic_hand_rankings, _heuristic_shop_rankings
from trainer.real_observer import build_observation
from trainer.utils import setup_logger


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Terminal UI assistant for real backend.")
    parser.add_argument("--base-url", default="http://127.0.0.1:12346")
    parser.add_argument("--model", default="", help="Reserved for future model inference in TUI.")
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--interval", type=float, default=0.5)
    return parser.parse_args()


def _log_path() -> Path:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    path = Path("docs/artifacts/p12/ui_logs") / stamp / "tui.log"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _append_log(path: Path, payload: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _render_once(backend, topk: int, enable_execute: bool, logger, log_path: Path) -> int:
    state = backend.get_state()
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
    print(json.dumps({"phase": phase, "resources": obs["resources"]}, ensure_ascii=False))
    print("cards:", [c.get("key") for c in obs["hand"]["cards"]])

    top_action = None
    if phase == "SELECTING_HAND":
        ranked = _heuristic_hand_rankings(state, topk=topk)
        for idx, item in enumerate(ranked, start=1):
            print(f"#{idx} {item['action_type']} indices={item['indices']} score={item['score']:.4f} explain={item['explain']}")
        if ranked:
            top_action = {"action_type": str(ranked[0]["action_type"]), "indices": list(ranked[0]["indices"])}
    elif phase in action_space_shop.SHOP_PHASES:
        ranked = _heuristic_shop_rankings(state, topk=topk)
        for idx, item in enumerate(ranked, start=1):
            print(f"#{idx} action={item['action']} score={item['score']:.4f} explain={item['explain']}")
        if ranked:
            top_action = dict(ranked[0]["action"])
    else:
        print(f"phase={phase} has no actionable suggestions.")

    _append_log(
        log_path,
        {
            "event": "refresh",
            "phase": phase,
            "top_action": top_action,
            "execute_enabled": enable_execute,
        },
    )

    if enable_execute and top_action:
        after, reward, done, info = backend.step(top_action)
        print(f"executed={top_action} reward={reward:.4f} done={done} after_phase={after.get('state')}")
        _append_log(
            log_path,
            {
                "event": "execute",
                "action": top_action,
                "reward": reward,
                "done": done,
                "after_phase": str(after.get("state") or "UNKNOWN"),
                "info": info,
            },
        )
    return 0


def main() -> int:
    args = _parse_args()
    logger = setup_logger("ui.assistant_tui")
    mode = "EXECUTE" if args.execute else "READONLY"
    logger.info("MODE=%s", mode)
    log_path = _log_path()
    logger.info("ui_log=%s", log_path)

    backend = create_backend("real", base_url=args.base_url, timeout_sec=8.0, seed="AAAAAAA", logger=logger)
    try:
        if args.once:
            return _render_once(backend, args.topk, args.execute, logger, log_path)

        print("Commands: r=refresh, e=toggle execute, x=execute top1 now, q=quit")
        execute_enabled = bool(args.execute)
        while True:
            cmd = input("> ").strip().lower()
            if cmd == "q":
                return 0
            if cmd == "e":
                execute_enabled = not execute_enabled
                print(f"execute_enabled={execute_enabled}")
                continue
            if cmd in {"r", "x", ""}:
                _render_once(backend, args.topk, execute_enabled if cmd != "x" else True, logger, log_path)
                time.sleep(max(0.05, float(args.interval)))
                continue
            print("unknown command, use r/e/x/q")
    finally:
        backend.close()


if __name__ == "__main__":
    raise SystemExit(main())
