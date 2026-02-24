from __future__ import annotations

if __package__ is None or __package__ == "":
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).resolve().parent.parent))

import argparse
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any

from trainer import action_space_shop
from trainer.env_client import create_backend
from trainer.infer_assistant_real import _heuristic_hand_rankings, _heuristic_shop_rankings
from trainer.real_observer import build_observation
from trainer.utils import setup_logger, timestamp, warn_if_unstable_python


def _write_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Shadow-mode real session recorder with model suggestions.")
    parser.add_argument("--base-url", default="http://127.0.0.1:12346")
    parser.add_argument("--model", default="", help="Reserved for compatibility; current recorder uses heuristic suggestions.")
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--interval", type=float, default=1.0)
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument("--mode", choices=["shadow", "execute"], default="shadow")
    parser.add_argument("--include-raw", action="store_true", help="Attach raw gamestate per row (needed for dagger reconstruction).")
    parser.add_argument("--out", required=True, help="Output session jsonl path.")
    return parser.parse_args()


def _suggestions(state: dict[str, Any], topk: int) -> list[dict[str, Any]]:
    phase = str(state.get("state") or "UNKNOWN")
    if phase == "SELECTING_HAND":
        return _heuristic_hand_rankings(state, topk=topk)
    if phase in action_space_shop.SHOP_PHASES:
        return _heuristic_shop_rankings(state, topk=topk)
    return []


def main() -> int:
    args = _parse_args()
    logger = setup_logger("trainer.record_real_session")
    warn_if_unstable_python(logger)

    out_path = Path(args.out)
    backend = create_backend("real", base_url=args.base_url, timeout_sec=8.0, seed="AAAAAAA", logger=logger)

    phase_counter: Counter[str] = Counter()
    written = 0

    try:
        for step_idx in range(max(1, int(args.steps))):
            try:
                state = backend.get_state()
            except Exception as exc:
                row = {
                    "ts": timestamp(),
                    "step_idx": step_idx,
                    "base_url": args.base_url,
                    "mode": args.mode,
                    "errors": [f"fetch_failed:{exc}"],
                }
                _write_jsonl(out_path, row)
                written += 1
                logger.warning("step=%d fetch failed: %s", step_idx, exc)
                time.sleep(max(0.05, float(args.interval)))
                continue

            obs = build_observation(state)
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
            suggestions = _suggestions(state, topk=max(1, int(args.topk)))
            row = {
                "ts": timestamp(),
                "step_idx": step_idx,
                "base_url": args.base_url,
                "mode": args.mode,
                "phase": phase,
                "gamestate_min": obs,
                "hand_cards": hand_keys,
                "shop_offers": shop_offers,
                "model_suggestions_topk": suggestions,
                "errors": [],
            }
            if args.include_raw:
                row["gamestate_raw"] = state

            _write_jsonl(out_path, row)
            written += 1
            logger.info("step=%d phase=%s hand=%d shop=%d", step_idx, phase, len(hand_keys), len(shop_offers))
            if step_idx + 1 < int(args.steps):
                time.sleep(max(0.05, float(args.interval)))
    finally:
        backend.close()

    summary = {
        "out": str(out_path),
        "steps_written": written,
        "phase_distribution": dict(phase_counter),
    }
    summary_path = out_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("session recorded: steps=%d out=%s", written, out_path)
    logger.info("summary: %s", summary_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
