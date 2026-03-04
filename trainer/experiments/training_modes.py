from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


MODE_CATEGORY_MAINLINE = "mainline"
MODE_CATEGORY_LEGACY_BASELINE = "legacy_baseline"
MODE_CATEGORY_EXPERIMENTAL = "experimental"

STATUS_ACTIVE = "active"
STATUS_LEGACY = "legacy"
STATUS_STUB = "stub"


@dataclass(frozen=True)
class TrainingMode:
    mode_id: str
    category: str
    description: str
    default_enabled: bool
    supports_closed_loop: bool
    supports_p22_quick: bool
    status: str = STATUS_ACTIVE
    required_artifacts: tuple[str, ...] = ()
    dependencies: tuple[str, ...] = ()


_REGISTRY: dict[str, TrainingMode] = {
    "bc_finetune": TrainingMode(
        mode_id="bc_finetune",
        category=MODE_CATEGORY_LEGACY_BASELINE,
        description="Behavior cloning finetune from replay mix labels.",
        default_enabled=False,
        supports_closed_loop=True,
        supports_p22_quick=False,
        status=STATUS_LEGACY,
        required_artifacts=("replay_mix_manifest",),
        dependencies=("trainer/train_bc.py",),
    ),
    "dagger_refresh": TrainingMode(
        mode_id="dagger_refresh",
        category=MODE_CATEGORY_LEGACY_BASELINE,
        description="DAgger refresh collection for baseline/probe data refresh.",
        default_enabled=False,
        supports_closed_loop=False,
        supports_p22_quick=False,
        status=STATUS_LEGACY,
        dependencies=("trainer/dagger_collect.py", "trainer/dagger_collect_v4.py"),
    ),
    "selfsup_warm_bc": TrainingMode(
        mode_id="selfsup_warm_bc",
        category=MODE_CATEGORY_MAINLINE,
        description="Self-supervised warm-start placeholder for candidate initialization.",
        default_enabled=True,
        supports_closed_loop=True,
        supports_p22_quick=True,
        status=STATUS_STUB,
        required_artifacts=("selfsup_encoder_checkpoint",),
        dependencies=("trainer/experiments/selfsup_train.py",),
    ),
    "rl_ppo_lite": TrainingMode(
        mode_id="rl_ppo_lite",
        category=MODE_CATEGORY_MAINLINE,
        description="P42 PPO-lite RL candidate training.",
        default_enabled=True,
        supports_closed_loop=True,
        supports_p22_quick=True,
        status=STATUS_ACTIVE,
        required_artifacts=("rl_config",),
        dependencies=("trainer/rl/ppo_lite.py",),
    ),
    "closed_loop_replay_curriculum": TrainingMode(
        mode_id="closed_loop_replay_curriculum",
        category=MODE_CATEGORY_MAINLINE,
        description="Closed-loop replay + curriculum path for v2/v42 pipelines.",
        default_enabled=True,
        supports_closed_loop=True,
        supports_p22_quick=True,
        status=STATUS_ACTIVE,
        required_artifacts=("replay_mix_manifest",),
        dependencies=("trainer/closed_loop/closed_loop_runner.py", "trainer/closed_loop/candidate_train.py"),
    ),
}


def list_training_modes() -> list[dict[str, Any]]:
    rows = [asdict(mode) for mode in _REGISTRY.values()]
    return sorted(rows, key=lambda row: str(row.get("mode_id") or ""))


def get_training_mode(mode_id: str) -> TrainingMode | None:
    return _REGISTRY.get(str(mode_id).strip().lower())


def is_mainline_mode(mode_id: str) -> bool:
    mode = get_training_mode(mode_id)
    if mode is None:
        return False
    return mode.category == MODE_CATEGORY_MAINLINE


def mode_category(mode_id: str, default: str = MODE_CATEGORY_EXPERIMENTAL) -> str:
    mode = get_training_mode(mode_id)
    if mode is None:
        return default
    return mode.category


def default_enabled(mode_id: str, default: bool = False) -> bool:
    mode = get_training_mode(mode_id)
    if mode is None:
        return default
    return bool(mode.default_enabled)


def _resolve_repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Training mode registry and category lookup.")
    parser.add_argument("--list", action="store_true", help="Print registry as JSON.")
    parser.add_argument("--mode", default="", help="Optional mode id lookup.")
    parser.add_argument("--json-out", default="", help="Optional path to write JSON output.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload: dict[str, Any]
    if str(args.mode or "").strip():
        mode = get_training_mode(str(args.mode).strip().lower())
        payload = {"mode": asdict(mode) if mode else None}
    else:
        payload = {"schema": "training_mode_registry_v1", "modes": list_training_modes()}

    if bool(args.list) or not str(args.mode or "").strip():
        text = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
        print(text, end="")
    else:
        text = json.dumps(payload, ensure_ascii=False)
        print(text)

    json_out = str(args.json_out or "").strip()
    if json_out:
        out_path = Path(json_out)
        if not out_path.is_absolute():
            out_path = _resolve_repo_root() / out_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
