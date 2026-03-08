from __future__ import annotations

if __package__ is None or __package__ == "":
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

import argparse
import json
from pathlib import Path

from trainer.autonomy.decision_policy import classify_action, classify_condition, load_decision_policy


def main() -> int:
    parser = argparse.ArgumentParser(description="P57 decision policy smoke/check")
    parser.add_argument("--policy", default="")
    parser.add_argument("--action", default="")
    parser.add_argument("--condition", default="")
    args = parser.parse_args()
    policy = load_decision_policy(Path(args.policy).resolve() if str(args.policy).strip() else None)
    if args.action:
        payload = {"policy_path": policy.get("source_path"), **classify_action(args.action, policy)}
    elif args.condition:
        payload = {"policy_path": policy.get("source_path"), **classify_condition(args.condition, policy)}
    else:
        payload = policy
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
