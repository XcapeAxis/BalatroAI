from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from trainer.experiments.seed_policy import (
    materialize_nightly_seed_set,
    materialize_seed_set,
    read_seed_policy,
    validate_seed_policy,
    write_json,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="P23 seed governance CLI")
    p.add_argument("--config", default="configs/experiments/seeds_p23.yaml")
    p.add_argument("--validate-policy", action="store_true")
    p.add_argument("--materialize", default="")
    p.add_argument("--materialize-nightly", action="store_true")
    p.add_argument("--write", default="")
    p.add_argument("--git-commit", default="")
    p.add_argument("--date-bucket", default="")
    p.add_argument("--run-id", default="")
    p.add_argument("--nightly-extra-count", type=int, default=0)
    p.add_argument("--allow-single-seed", action="store_true")
    return p.parse_args()


def _print_json(payload: dict) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def main() -> int:
    args = parse_args()
    policy_path = Path(args.config).resolve()
    policy = read_seed_policy(policy_path)

    wrote = False
    final_payload: dict | None = None

    if args.validate_policy:
        validation = validate_seed_policy(policy)
        final_payload = validation
        if not validation.get("ok", False):
            if args.write:
                write_json(Path(args.write).resolve(), validation)
                wrote = True
            _print_json(validation)
            return 2
        if not args.materialize and not args.materialize_nightly:
            if args.write:
                write_json(Path(args.write).resolve(), validation)
                wrote = True
            _print_json(validation)
            return 0

    if args.materialize:
        final_payload = materialize_seed_set(
            policy,
            args.materialize,
            explicit_single_seed_override=bool(args.allow_single_seed),
        )

    if args.materialize_nightly:
        date_bucket = args.date_bucket or datetime.now().strftime("%Y%m%d")
        run_id = args.run_id or datetime.now().strftime("%Y%m%d-%H%M%S")
        final_payload = materialize_nightly_seed_set(
            policy,
            git_commit=args.git_commit or "unknown",
            date_bucket=date_bucket,
            run_id=run_id,
            extra_count_override=(args.nightly_extra_count if args.nightly_extra_count > 0 else None),
            explicit_single_seed_override=bool(args.allow_single_seed),
        )

    if final_payload is None:
        raise SystemExit(
            "nothing to do. choose --validate-policy and/or --materialize/--materialize-nightly"
        )

    if args.write:
        write_json(Path(args.write).resolve(), final_payload)
        wrote = True
    if not wrote:
        _print_json(final_payload)
    else:
        _print_json(final_payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

