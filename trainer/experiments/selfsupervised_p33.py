from __future__ import annotations

if __package__ is None or __package__ == "":
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from trainer.self_supervised.train import run_p33_selfsup_training
from trainer.utils import setup_logger, warn_if_unstable_python


def _now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="P33 self-supervised plumbing experiment runner.")
    p.add_argument("--config", default="configs/experiments/p33_selfsup.yaml")
    p.add_argument("--out-dir", default="")
    p.add_argument("--max-samples", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--quiet", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    logger = setup_logger("trainer.experiments.selfsupervised_p33")
    warn_if_unstable_python(logger)
    repo_root = Path(__file__).resolve().parents[2]
    config_path = (repo_root / args.config).resolve()

    summary: dict[str, Any] = run_p33_selfsup_training(
        config_path=config_path,
        out_dir=(args.out_dir or None),
        seed_override=(int(args.seed) if int(args.seed) > 0 else None),
        max_samples_override=(int(args.max_samples) if int(args.max_samples) > 0 else None),
        quiet=bool(args.quiet),
    )

    artifact_root = (repo_root / "docs/artifacts/p33").resolve()
    artifact_root.mkdir(parents=True, exist_ok=True)
    summary_path = artifact_root / f"selfsup_training_summary_{_now_stamp()}.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("p33 selfsup training complete: %s", summary_path)
    print(json.dumps({"status": "PASS", "summary_path": str(summary_path), "run_dir": summary.get("run_dir")}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

