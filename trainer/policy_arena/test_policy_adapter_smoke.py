from __future__ import annotations

if __package__ is None or __package__ == "":
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sim.pybind.sim_env import SimEnvBackend
from trainer.policy_arena.adapters import HeuristicAdapter, HybridAdapter, ModelAdapter, SearchAdapter


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def _build_states(seed: str) -> dict[str, dict[str, Any]]:
    backend = SimEnvBackend(seed=seed)
    states: dict[str, dict[str, Any]] = {}
    state = backend.reset(seed=seed)
    states["initial"] = state
    phase = str(state.get("state") or "").upper()

    if phase == "BLIND_SELECT":
        state, _, _, _ = backend.step({"action_type": "SELECT", "index": 0})
        states["selecting_hand"] = state
        phase = str(state.get("state") or "").upper()

    if phase == "SELECTING_HAND":
        hand_cards = (state.get("hand") or {}).get("cards") if isinstance(state.get("hand"), dict) else []
        if isinstance(hand_cards, list) and hand_cards:
            state, _, _, _ = backend.step({"action_type": "PLAY", "indices": [0]})
            states["after_play"] = state
            if str(state.get("state") or "").upper() == "ROUND_EVAL":
                state, _, _, _ = backend.step({"action_type": "CASH_OUT"})
                states["shop"] = state

    return states


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="P39 policy adapter smoke test.")
    parser.add_argument("--seed", default="AAAAAAA")
    parser.add_argument("--out-dir", default="docs/artifacts/p39")
    parser.add_argument("--model-path", default="")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = (repo_root / out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    states = _build_states(seed=str(args.seed))
    adapters = [
        HeuristicAdapter(),
        SearchAdapter(),
        ModelAdapter(model_path=str(args.model_path or "")),
        HybridAdapter(),
    ]

    rows: list[dict[str, Any]] = []
    for adapter in adapters:
        adapter.reset(seed=args.seed)
        description = adapter.describe()
        for state_name, obs in states.items():
            phase = str(obs.get("state") or "UNKNOWN").upper()
            try:
                action = adapter.act(obs)
                status = "ok"
                error = ""
            except Exception as exc:  # pragma: no cover - smoke should keep running
                action = {"action_type": "WAIT"}
                status = "error"
                error = str(exc)
            rows.append(
                {
                    "adapter": adapter.name,
                    "state_name": state_name,
                    "phase": phase,
                    "status": status,
                    "error": error,
                    "action": action,
                    "describe": description,
                }
            )

    failed = [row for row in rows if row.get("status") != "ok"]
    payload = {
        "schema": "p39_policy_adapter_smoke_v1",
        "generated_at": _now_iso(),
        "seed": str(args.seed),
        "states_covered": sorted(states.keys()),
        "rows": rows,
        "failed_count": len(failed),
        "status": "PASS" if not failed else "FAIL",
    }

    out_path = out_dir / f"adapter_smoke_{_now_stamp()}.json"
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "failed_count": len(failed), "out": str(out_path)}, ensure_ascii=False))
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())

