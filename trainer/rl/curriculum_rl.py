from __future__ import annotations

if __package__ is None or __package__ == "":
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

import argparse
import copy
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:  # pragma: no cover - optional dependency
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None


def _safe_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _read_yaml_or_json(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8-sig")
    if path.suffix.lower() in {".yaml", ".yml"}:
        if yaml is not None:
            payload = yaml.safe_load(text)
        else:
            try:
                payload = json.loads(text)
            except Exception:
                sidecar = path.with_suffix(".json")
                if sidecar.exists():
                    payload = json.loads(sidecar.read_text(encoding="utf-8-sig"))
                else:
                    raise RuntimeError(f"PyYAML unavailable and no sidecar JSON for: {path}")
    else:
        payload = json.loads(text)
    if not isinstance(payload, dict):
        raise ValueError(f"config must be mapping: {path}")
    return payload


def _deep_merge(base: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _normalize_stage_defs(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    raw = payload.get("stages") if isinstance(payload.get("stages"), dict) else None
    if raw:
        return {
            str(name): dict(cfg)
            for name, cfg in raw.items()
            if isinstance(name, str) and isinstance(cfg, dict)
        }
    out: dict[str, dict[str, Any]] = {}
    for key in ("stage1_basic", "stage2_midgame", "stage3_highrisk", "stage1", "stage2", "stage3"):
        value = payload.get(key)
        if isinstance(value, dict):
            out[str(key)] = dict(value)
    return out


def _normalize_schedule(payload: dict[str, Any], stage_defs: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    raw = payload.get("schedule") if isinstance(payload.get("schedule"), list) else []
    schedule: list[dict[str, Any]] = []
    for idx, row in enumerate(raw, start=1):
        if not isinstance(row, dict):
            continue
        stage_name = str(row.get("stage") or row.get("name") or "").strip()
        if not stage_name:
            continue
        schedule.append(
            {
                "phase_index": idx,
                "stage": stage_name,
                "start_iteration": max(1, _safe_int(row.get("start_iteration"), idx)),
                "until_iteration": max(0, _safe_int(row.get("until_iteration"), 0)),
            }
        )
    if schedule:
        return schedule

    default_order = [
        "stage1_basic",
        "stage2_midgame",
        "stage3_highrisk",
        "stage1",
        "stage2",
        "stage3",
    ]
    starts = [1, 3, 5]
    untils = [2, 4, 0]
    for idx, name in enumerate(default_order):
        if name not in stage_defs:
            continue
        pos = len(schedule)
        schedule.append(
            {
                "phase_index": pos + 1,
                "stage": name,
                "start_iteration": starts[min(pos, len(starts) - 1)],
                "until_iteration": untils[min(pos, len(untils) - 1)],
            }
        )
    return schedule


@dataclass(frozen=True)
class CurriculumStage:
    stage: str
    phase_index: int
    start_iteration: int
    until_iteration: int
    source_sampling_weights: dict[str, float]
    reward: dict[str, Any]
    rollout: dict[str, Any]
    difficulty: dict[str, Any]
    hard_case_sampling: dict[str, Any]
    self_imitation: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "phase_index": self.phase_index,
            "start_iteration": self.start_iteration,
            "until_iteration": self.until_iteration,
            "source_sampling_weights": dict(self.source_sampling_weights),
            "reward": dict(self.reward),
            "rollout": dict(self.rollout),
            "difficulty": dict(self.difficulty),
            "hard_case_sampling": dict(self.hard_case_sampling),
            "self_imitation": dict(self.self_imitation),
        }


class CurriculumScheduler:
    def __init__(self, *, schema: str, stages: list[CurriculumStage], config_path: str = "") -> None:
        self.schema = str(schema or "p44_curriculum_config_v1")
        self.stages = list(stages)
        self.config_path = str(config_path or "")

    @classmethod
    def from_mapping(cls, payload: dict[str, Any], *, config_path: str = "") -> "CurriculumScheduler":
        stage_defs = _normalize_stage_defs(payload)
        schedule = _normalize_schedule(payload, stage_defs)
        stages: list[CurriculumStage] = []
        for row in schedule:
            stage_name = str(row.get("stage") or "").strip()
            if not stage_name:
                continue
            stage_cfg = dict(stage_defs.get(stage_name) or {})
            stages.append(
                CurriculumStage(
                    stage=stage_name,
                    phase_index=max(1, _safe_int(row.get("phase_index"), len(stages) + 1)),
                    start_iteration=max(1, _safe_int(row.get("start_iteration"), len(stages) + 1)),
                    until_iteration=max(0, _safe_int(row.get("until_iteration"), 0)),
                    source_sampling_weights={
                        str(k): float(v)
                        for k, v in (stage_cfg.get("source_sampling_weights") or {}).items()
                    }
                    if isinstance(stage_cfg.get("source_sampling_weights"), dict)
                    else {},
                    reward=dict(stage_cfg.get("reward") or {}) if isinstance(stage_cfg.get("reward"), dict) else {},
                    rollout=dict(stage_cfg.get("rollout") or {}) if isinstance(stage_cfg.get("rollout"), dict) else {},
                    difficulty=dict(stage_cfg.get("difficulty") or {}) if isinstance(stage_cfg.get("difficulty"), dict) else {},
                    hard_case_sampling=dict(stage_cfg.get("hard_case_sampling") or {})
                    if isinstance(stage_cfg.get("hard_case_sampling"), dict)
                    else {},
                    self_imitation=dict(stage_cfg.get("self_imitation") or {})
                    if isinstance(stage_cfg.get("self_imitation"), dict)
                    else {},
                )
            )
        return cls(
            schema=str(payload.get("schema") or "p44_curriculum_config_v1"),
            stages=stages,
            config_path=config_path,
        )

    @classmethod
    def from_path(cls, path: str | Path) -> "CurriculumScheduler":
        cfg_path = Path(path).resolve()
        return cls.from_mapping(_read_yaml_or_json(cfg_path), config_path=str(cfg_path))

    def enabled(self) -> bool:
        return bool(self.stages)

    def resolve_stage(self, training_iteration: int) -> CurriculumStage | None:
        if not self.stages:
            return None
        iteration = max(1, int(training_iteration))
        chosen = self.stages[0]
        for stage in self.stages:
            if iteration < stage.start_iteration:
                break
            chosen = stage
            if stage.until_iteration > 0 and iteration <= stage.until_iteration:
                return stage
        return chosen

    def apply_to_config(self, base_cfg: dict[str, Any], *, training_iteration: int) -> tuple[dict[str, Any], dict[str, Any]]:
        stage = self.resolve_stage(training_iteration)
        if stage is None:
            return copy.deepcopy(base_cfg), {
                "enabled": False,
                "training_iteration": int(training_iteration),
                "stage": "",
            }

        patch: dict[str, Any] = {}
        if stage.reward:
            patch.setdefault("env", {})
            patch["env"].setdefault("reward", {})
            patch["env"]["reward"].update(stage.reward)
        if stage.rollout:
            rollout_patch = dict(stage.rollout)
            max_steps = rollout_patch.pop("max_steps", None)
            if max_steps is not None:
                patch.setdefault("rollout", {})
                patch["rollout"]["max_steps_per_episode"] = max(1, _safe_int(max_steps, 120))
                patch.setdefault("distributed", {})
                patch["distributed"]["max_steps_per_episode"] = max(1, _safe_int(max_steps, 120))
            if rollout_patch:
                patch.setdefault("rollout", {})
                patch["rollout"] = _deep_merge(patch["rollout"], rollout_patch)
        if stage.difficulty:
            patch.setdefault("env", {})
            if stage.difficulty.get("max_ante") is not None:
                patch["env"]["max_ante"] = max(0, _safe_int(stage.difficulty.get("max_ante"), 0))
            for key in ("stake", "timeout_sec", "max_auto_steps", "auto_advance"):
                if stage.difficulty.get(key) is not None:
                    patch["env"][key] = stage.difficulty.get(key)
        if stage.hard_case_sampling:
            patch.setdefault("hard_case_sampling", {})
            patch["hard_case_sampling"] = _deep_merge(patch["hard_case_sampling"], stage.hard_case_sampling)
        if stage.self_imitation:
            patch.setdefault("self_imitation", {})
            patch["self_imitation"] = _deep_merge(patch["self_imitation"], stage.self_imitation)
        merged = _deep_merge(base_cfg, patch)
        return merged, {
            "enabled": True,
            "training_iteration": int(training_iteration),
            "stage": stage.stage,
            "phase_index": int(stage.phase_index),
            "start_iteration": int(stage.start_iteration),
            "until_iteration": int(stage.until_iteration),
            "source_sampling_weights": dict(stage.source_sampling_weights),
            "reward": dict(stage.reward),
            "rollout": dict(stage.rollout),
            "difficulty": dict(stage.difficulty),
            "hard_case_sampling": dict(stage.hard_case_sampling),
            "self_imitation": dict(stage.self_imitation),
            "config_path": self.config_path,
        }

    def plan_payload(self, *, seeds: list[str] | None = None) -> dict[str, Any]:
        seed_list = [str(seed).strip() for seed in (seeds or []) if str(seed).strip()]
        return {
            "schema": "p44_curriculum_plan_v1",
            "enabled": self.enabled(),
            "config_path": self.config_path,
            "phase_count": len(self.stages),
            "seeds": seed_list,
            "phases": [stage.to_dict() for stage in self.stages],
        }


def load_curriculum_scheduler(
    *,
    config_path: str | Path | None = None,
    config: dict[str, Any] | None = None,
) -> CurriculumScheduler:
    if isinstance(config, dict):
        return CurriculumScheduler.from_mapping(config, config_path="")
    if config_path is None or not str(config_path).strip():
        return CurriculumScheduler(schema="p44_curriculum_config_v1", stages=[], config_path="")
    return CurriculumScheduler.from_path(config_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="P44 curriculum RL scheduler.")
    parser.add_argument("--config", default="configs/experiments/p44_curriculum.yaml")
    parser.add_argument("--iteration", type=int, default=1)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    scheduler = load_curriculum_scheduler(config_path=args.config)
    stage = scheduler.resolve_stage(int(args.iteration))
    payload = {
        "schema": "p44_curriculum_cli_v1",
        "config": str(args.config),
        "iteration": int(args.iteration),
        "stage": stage.to_dict() if stage is not None else None,
        "plan": scheduler.plan_payload(),
    }
    print(json.dumps(payload, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
