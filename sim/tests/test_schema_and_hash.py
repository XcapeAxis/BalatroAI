if __package__ is None or __package__ == "":
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from sim.core.hashing import state_hash_full
from sim.core.validate import validate_action, validate_state, validate_trace_line


def main() -> int:
    state = {
        "schema_version": "state_v1",
        "phase": "SELECTING_HAND",
        "zones": {"deck": [], "discard": [], "hand": [], "played": []},
        "round": {"hands_left": 4, "discards_left": 4, "ante": 1, "round_num": 1, "blind": "small"},
        "score": {
            "chips": 0,
            "mult": 1,
            "target_chips": 300,
            "last_hand_type": "",
            "last_base_chips": 0,
            "last_base_mult": 1,
        },
        "economy": {"money": 4},
        "jokers": [],
        "rng": {"mode": "native", "seed": "AAAAAAA", "cursor": 0, "events": []},
        "flags": {"done": False, "won": False},
    }
    action = {"schema_version": "action_v1", "phase": "SELECTING_HAND", "action_type": "PLAY", "indices": [0]}
    trace_line = {
        "schema_version": "trace_v1",
        "step_id": 0,
        "phase": "SELECTING_HAND",
        "action": action,
        "state_hash_full": state_hash_full(state),
        "state_hash_hand_core": state_hash_full(state),
        "reward": 0.0,
        "done": False,
        "info": {"source": "self_test"},
    }

    validate_state(state)
    validate_action(action)
    validate_trace_line(trace_line)

    h1 = state_hash_full(state)
    h2 = state_hash_full(state)
    assert h1 == h2, "state hash must be deterministic"

    print("schema/hash self-check passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
