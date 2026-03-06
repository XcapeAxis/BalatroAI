from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from trainer.closed_loop.replay_manifest import stable_hash_text


IMAGINED_SOURCE_TYPE = "imagined_world_model"
IMAGINATION_LINEAGE_VERSION = "p46_imagination_v1"
IMAGINATION_RECORD_SCHEMA = "p46_imagined_record_v1"


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _safe_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return bool(default)
    token = str(value).strip().lower()
    if token in {"1", "true", "yes", "y"}:
        return True
    if token in {"0", "false", "no", "n"}:
        return False
    return bool(default)


def make_root_sample_id(record: dict[str, Any], *, source_path: str | Path = "") -> str:
    source = str(source_path or record.get("source_path") or record.get("path") or "")
    episode_id = str(record.get("episode_id") or "")
    step_id = str(record.get("step_id") or "")
    phase = str(record.get("phase") or "")
    expert_action_id = str(record.get("expert_action_id") if record.get("expert_action_id") is not None else "")
    shop_expert_action_id = str(
        record.get("shop_expert_action_id") if record.get("shop_expert_action_id") is not None else ""
    )
    packed = "|".join(
        [
            source,
            episode_id,
            step_id,
            phase,
            expert_action_id,
            shop_expert_action_id,
        ]
    )
    return stable_hash_text(packed)[:24]


@dataclass(frozen=True)
class ImaginedLineageFields:
    source_type: str = IMAGINED_SOURCE_TYPE
    world_model_checkpoint: str = ""
    imagination_horizon: int = 1
    uncertainty_score: float = 0.0
    uncertainty_gate_passed: bool = False
    root_sample_id: str = ""
    imagined_step_idx: int = 1
    teacher_seed: str = ""
    valid_for_training: bool = False
    lineage_version: str = IMAGINATION_LINEAGE_VERSION

    @classmethod
    def from_mapping(cls, raw: dict[str, Any] | None) -> "ImaginedLineageFields":
        payload = raw if isinstance(raw, dict) else {}
        return cls(
            source_type=str(payload.get("source_type") or IMAGINED_SOURCE_TYPE),
            world_model_checkpoint=str(payload.get("world_model_checkpoint") or ""),
            imagination_horizon=max(1, _safe_int(payload.get("imagination_horizon"), 1)),
            uncertainty_score=max(0.0, _safe_float(payload.get("uncertainty_score"), 0.0)),
            uncertainty_gate_passed=_safe_bool(payload.get("uncertainty_gate_passed"), False),
            root_sample_id=str(payload.get("root_sample_id") or ""),
            imagined_step_idx=max(1, _safe_int(payload.get("imagined_step_idx"), 1)),
            teacher_seed=str(payload.get("teacher_seed") or ""),
            valid_for_training=_safe_bool(payload.get("valid_for_training"), False),
            lineage_version=str(payload.get("lineage_version") or IMAGINATION_LINEAGE_VERSION),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def imagined_record_fields(row: dict[str, Any] | None) -> ImaginedLineageFields:
    payload = row if isinstance(row, dict) else {}
    return ImaginedLineageFields.from_mapping(payload)


def apply_imagination_metadata(
    root_record: dict[str, Any],
    *,
    world_model_checkpoint: str,
    imagination_horizon: int,
    uncertainty_score: float,
    uncertainty_gate_passed: bool,
    root_sample_id: str,
    imagined_step_idx: int,
    teacher_seed: str,
    valid_for_training: bool,
    predicted_reward: float,
    predicted_score_delta: float,
    predicted_resource_delta: list[float] | None = None,
    predicted_done: bool = False,
    predicted_latent: list[float] | None = None,
    source_path: str = "",
    source_run_id: str = "",
) -> dict[str, Any]:
    row = copy.deepcopy(root_record)
    metadata = ImaginedLineageFields(
        world_model_checkpoint=str(world_model_checkpoint or ""),
        imagination_horizon=max(1, int(imagination_horizon)),
        uncertainty_score=max(0.0, float(uncertainty_score)),
        uncertainty_gate_passed=bool(uncertainty_gate_passed),
        root_sample_id=str(root_sample_id or ""),
        imagined_step_idx=max(1, int(imagined_step_idx)),
        teacher_seed=str(teacher_seed or ""),
        valid_for_training=bool(valid_for_training),
    )
    row.update(
        {
            "schema": IMAGINATION_RECORD_SCHEMA,
            "source_type": IMAGINED_SOURCE_TYPE,
            "source_run_id": str(source_run_id or ""),
            "source_path": str(source_path or ""),
            "world_model_checkpoint": metadata.world_model_checkpoint,
            "imagination_horizon": int(metadata.imagination_horizon),
            "uncertainty_score": float(metadata.uncertainty_score),
            "uncertainty_gate_passed": bool(metadata.uncertainty_gate_passed),
            "root_sample_id": str(metadata.root_sample_id),
            "imagined_step_idx": int(metadata.imagined_step_idx),
            "teacher_seed": str(metadata.teacher_seed),
            "valid_for_training": bool(metadata.valid_for_training),
            "lineage_version": metadata.lineage_version,
            "predicted_reward": float(predicted_reward),
            "predicted_score_delta": float(predicted_score_delta),
            "predicted_resource_delta": list(predicted_resource_delta or []),
            "predicted_done": bool(predicted_done),
            "predicted_latent": list(predicted_latent or []),
        }
    )
    row.setdefault("metadata", {})
    if isinstance(row.get("metadata"), dict):
        row["metadata"]["imagination"] = metadata.to_dict()
    return row


def is_imagined_record(row: dict[str, Any] | None) -> bool:
    if not isinstance(row, dict):
        return False
    return str(row.get("source_type") or "").strip().lower() == IMAGINED_SOURCE_TYPE


def imagination_schema_markdown() -> list[str]:
    return [
        "# P46 Imagined Sample Schema",
        "",
        "Synthetic records remain root-conditioned and keep lineage separate from real replay.",
        "",
        "## Required lineage fields",
        "",
        f"- `source_type`: `{IMAGINED_SOURCE_TYPE}`",
        "- `world_model_checkpoint`",
        "- `imagination_horizon`",
        "- `uncertainty_score`",
        "- `uncertainty_gate_passed`",
        "- `root_sample_id`",
        "- `imagined_step_idx`",
        "- `teacher_seed`",
        "- `valid_for_training`",
        f"- `lineage_version`: `{IMAGINATION_LINEAGE_VERSION}`",
        "",
        "## Notes",
        "",
        "- step 1 rows can be BC-compatible and trainable when uncertainty gate passes",
        "- step >1 rows may be logged for diagnostics but can remain `valid_for_training=false` when no decoded raw state exists",
        "- synthetic samples never overwrite the original root lineage",
    ]
