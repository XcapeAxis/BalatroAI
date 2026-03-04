from __future__ import annotations

if __package__ is None or __package__ == "":
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

import argparse
import json
from pathlib import Path
from typing import Any

from trainer.closed_loop.replay_manifest import now_iso, now_stamp, to_abs_path, write_json
from trainer.common.slices import compute_slice_labels


def _mock_samples() -> list[dict[str, Any]]:
    return [
        {
            "state": {
                "ante_num": 1,
                "money": 18,
                "round": {"hands_left": 4, "discards_left": 2},
                "jokers": [{"key": "j_blueprint"}],
            },
            "phase": "SELECTING_HAND",
            "action_type": "PLAY",
        },
        {
            "state": {
                "ante_num": 4,
                "money": 2,
                "chips_gap": 0.84,
                "round": {"hands_left": 1, "discards_left": 0},
                "jokers": [{"key": "j_egg", "counter": 3}],
            },
            "phase": "SHOP",
            "action_type": "SHOP_BUY",
        },
        {
            "state": {
                "round_num": 14,
                "money": 6,
                "round": {"hands_left": 2, "discards_left": 1},
                "tags": ["position_sensitive"],
            },
            "phase": "SELECTING_HAND",
            "action_type": "MOVE_HAND_CARD",
        },
        {
            "state": {},
            "phase": "",
            "action_type": "",
        },
    ]


def run_slice_smoke(
    *,
    out_dir: str | Path = "docs/artifacts/p41",
    out_path: str | Path | None = None,
) -> dict[str, Any]:
    repo_root = Path(__file__).resolve().parents[2]
    samples = _mock_samples()
    rows: list[dict[str, Any]] = []
    for idx, sample in enumerate(samples, start=1):
        rows.append(
            {
                "sample_id": f"mock_{idx:02d}",
                "input": sample,
                "slice_labels": compute_slice_labels(sample),
            }
        )

    if out_path:
        target_path = to_abs_path(repo_root, out_path)
    else:
        target_path = to_abs_path(repo_root, out_dir) / f"slice_smoke_{now_stamp()}.json"

    payload = {
        "schema": "p41_slice_smoke_v1",
        "generated_at": now_iso(),
        "sample_count": len(rows),
        "rows": rows,
    }
    write_json(target_path, payload)
    return {
        "status": "ok",
        "slice_smoke_json": str(target_path),
        "sample_count": len(rows),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a small P41 slice-label smoke artifact.")
    parser.add_argument("--out-dir", default="docs/artifacts/p41")
    parser.add_argument("--out-path", default="")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = run_slice_smoke(
        out_dir=args.out_dir,
        out_path=(args.out_path if str(args.out_path).strip() else None),
    )
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

