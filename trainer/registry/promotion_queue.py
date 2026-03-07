from __future__ import annotations

if __package__ is None or __package__ == "":
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

import argparse
import json
from pathlib import Path
from typing import Any

from trainer.registry.checkpoint_query import promoted_by_family, promotion_review_queue, stale_drafts, waiting_on_arena
from trainer.registry.checkpoint_registry import list_entries


def build_promotion_queue_summary(entries: list[dict[str, Any]]) -> dict[str, Any]:
    promoted = promoted_by_family(entries)
    review = promotion_review_queue(entries)
    arena_wait = waiting_on_arena(entries)
    drafts = stale_drafts(entries, limit=50)
    return {
        "schema": "p51_promotion_queue_v1",
        "counts": {
            "total": len(entries),
            "promotion_review": len(review),
            "waiting_on_arena": len(arena_wait),
            "stale_drafts": len(drafts),
            "promoted_families": len(promoted),
        },
        "promotion_review": review,
        "waiting_on_arena": arena_wait,
        "stale_drafts": drafts,
        "promoted_by_family": promoted,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="P51 promotion queue summary")
    parser.add_argument("--registry", default="")
    parser.add_argument("--out", default="")
    args = parser.parse_args()
    registry_path = Path(args.registry).resolve() if str(args.registry).strip() else None
    payload = build_promotion_queue_summary(list_entries(registry_path))
    if str(args.out).strip():
        out_path = Path(args.out).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
