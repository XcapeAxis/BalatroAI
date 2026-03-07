from __future__ import annotations

if __package__ is None or __package__ == "":
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

import argparse
import json
from pathlib import Path
from typing import Any

from trainer.campaigns.campaign_schema import CAMPAIGN_STAGE_IDS
from trainer.campaigns.resume_state import load_campaign_state


def build_campaign_overview(payload: dict[str, Any]) -> dict[str, Any]:
    stages = [dict(item) for item in (payload.get("stages") or []) if isinstance(item, dict)]
    counts: dict[str, int] = {}
    for stage in stages:
        token = str(stage.get("status") or "pending")
        counts[token] = int(counts.get(token, 0)) + 1
    return {
        "schema": "p51_campaign_overview_v1",
        "campaign_id": str(payload.get("campaign_id") or ""),
        "run_id": str(payload.get("run_id") or ""),
        "experiment_id": str(payload.get("experiment_id") or ""),
        "seed": str(payload.get("seed") or ""),
        "counts": counts,
        "stages": stages,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="P51 nightly campaign overview")
    parser.add_argument("--state-path", required=True)
    parser.add_argument("--campaign-id", default="p51-smoke")
    parser.add_argument("--run-id", default="adhoc")
    parser.add_argument("--experiment-id", default="p51_registry_smoke")
    parser.add_argument("--seed", default="AAAAAAA")
    args = parser.parse_args()
    state = load_campaign_state(
        state_path=args.state_path,
        campaign_id=str(args.campaign_id),
        run_id=str(args.run_id),
        experiment_id=str(args.experiment_id),
        seed=str(args.seed),
        stage_ids=list(CAMPAIGN_STAGE_IDS),
        metadata={},
    )
    print(json.dumps(build_campaign_overview(state), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
