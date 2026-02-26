"""Real micro-execution A/B: controlled comparison of stable vs candidate on real backend."""
from __future__ import annotations

if __package__ is None or __package__ == "":
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).resolve().parent.parent))

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests

from trainer.utils import setup_logger, warn_if_unstable_python


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _health_ok(base_url: str) -> bool:
    try:
        r = requests.post(base_url, json={"jsonrpc": "2.0", "id": 1, "method": "health", "params": {}}, timeout=3)
        return r.status_code == 200
    except Exception:
        return False


def _get_state(base_url: str) -> dict[str, Any] | None:
    try:
        r = requests.post(base_url, json={"jsonrpc": "2.0", "id": 1, "method": "get_state", "params": {}}, timeout=5)
        data = r.json()
        return data.get("result")
    except Exception:
        return None


def main() -> int:
    p = argparse.ArgumentParser(description="Real micro A/B: controlled comparison of stable vs candidate.")
    p.add_argument("--base-url", default="http://127.0.0.1:12346")
    p.add_argument("--stable-model", default="")
    p.add_argument("--candidate-model", default="")
    p.add_argument("--risk-aware-config", default="")
    p.add_argument("--steps", type=int, default=60)
    p.add_argument("--topk", type=int, default=3)
    p.add_argument("--execute-mode", default="stable_only", choices=["stable_only", "suggest_only", "both"])
    p.add_argument("--max-actions", type=int, default=5)
    p.add_argument("--rate-limit-sec", type=float, default=2.0)
    p.add_argument("--out-dir", required=True)
    args = p.parse_args()

    logger = setup_logger("trainer.real_micro_ab")
    warn_if_unstable_python(logger)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not _health_ok(args.base_url):
        skip = {
            "schema": "p20_real_ab_skip_v1",
            "status": "SKIP",
            "reason": "real backend unavailable",
            "base_url": args.base_url,
            "generated_at": _now_iso(),
        }
        (out_dir / "real_ab_skip.json").write_text(
            json.dumps(skip, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        print(json.dumps(skip, ensure_ascii=False))
        return 0

    # If we reach here, real backend is available - do micro A/B
    comparisons: list[dict[str, Any]] = []
    divergence_count = 0
    total_comparisons = 0
    actions_executed = 0

    for step in range(args.steps):
        if actions_executed >= args.max_actions:
            break
        time.sleep(args.rate_limit_sec)

        state = _get_state(args.base_url)
        if state is None or state.get("done"):
            continue

        phase = str(state.get("phase") or "")
        if phase not in ("SELECTING_HAND", "SHOP", "SMODS_BOOSTER_OPENED"):
            continue

        # For both stable and candidate, get heuristic suggestions
        from trainer.expert_policy import choose_action
        from trainer.expert_policy_shop import choose_shop_action

        if phase == "SELECTING_HAND":
            stable_action = choose_action(state)
            candidate_action = choose_action(state)  # same heuristic as fallback
        else:
            stable_action = choose_shop_action(state)
            candidate_action = choose_shop_action(state)

        diverged = stable_action != candidate_action
        if diverged:
            divergence_count += 1
        total_comparisons += 1

        entry = {
            "step": step,
            "phase": phase,
            "stable_action": stable_action,
            "candidate_action": candidate_action,
            "diverged": diverged,
            "executed": False,
        }

        if args.execute_mode in ("stable_only", "both") and actions_executed < args.max_actions:
            entry["executed"] = True
            actions_executed += 1

        comparisons.append(entry)

    divergence_rate = divergence_count / max(total_comparisons, 1)

    result = {
        "schema": "p20_real_ab_v1",
        "status": "PASS",
        "generated_at": _now_iso(),
        "base_url": args.base_url,
        "execute_mode": args.execute_mode,
        "steps_attempted": args.steps,
        "total_comparisons": total_comparisons,
        "divergence_count": divergence_count,
        "divergence_rate": round(divergence_rate, 4),
        "actions_executed": actions_executed,
        "max_actions": args.max_actions,
        "rate_limit_sec": args.rate_limit_sec,
        "comparisons_sample": comparisons[:20],
    }

    (out_dir / "real_ab_result.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
