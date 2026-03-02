from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:  # pragma: no cover
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None


def load_mapping(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    if yaml is not None:
        payload = yaml.safe_load(text)
    else:
        payload = json.loads(text)
    if not isinstance(payload, dict):
        raise ValueError(f"campaign file must be mapping: {path}")
    return payload


@dataclass
class StageSpec:
    stage_id: str
    purpose: str
    matrix_ref: str
    mode: str = "quick"
    include: list[str] = field(default_factory=list)
    exclude: list[str] = field(default_factory=list)
    seed_set_name: str = ""
    seed_limit: int = 0
    failure_policy: str = "continue"
    depends_on: list[str] = field(default_factory=list)
    max_experiments: int = 0

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "StageSpec":
        stage_id = str(payload.get("stage_id") or "").strip()
        if not stage_id:
            raise ValueError("stage.stage_id is required")
        matrix_ref = str(payload.get("matrix_ref") or "").strip()
        if not matrix_ref:
            raise ValueError(f"{stage_id}: matrix_ref is required")
        purpose = str(payload.get("purpose") or "validation").strip()
        mode = str(payload.get("mode") or "quick").strip().lower()
        failure_policy = str(payload.get("failure_policy") or "continue").strip().lower()
        if failure_policy not in {"continue", "fail_fast"}:
            raise ValueError(f"{stage_id}: failure_policy must be continue/fail_fast")
        include = [str(x) for x in (payload.get("include") or [])]
        exclude = [str(x) for x in (payload.get("exclude") or [])]
        depends_on = [str(x) for x in (payload.get("depends_on") or [])]
        return cls(
            stage_id=stage_id,
            purpose=purpose,
            matrix_ref=matrix_ref,
            mode=mode,
            include=include,
            exclude=exclude,
            seed_set_name=str(payload.get("seed_set_name") or ""),
            seed_limit=int(payload.get("seed_limit") or 0),
            failure_policy=failure_policy,
            depends_on=depends_on,
            max_experiments=int(payload.get("max_experiments") or 0),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage_id": self.stage_id,
            "purpose": self.purpose,
            "matrix_ref": self.matrix_ref,
            "mode": self.mode,
            "include": self.include,
            "exclude": self.exclude,
            "seed_set_name": self.seed_set_name,
            "seed_limit": self.seed_limit,
            "failure_policy": self.failure_policy,
            "depends_on": self.depends_on,
            "max_experiments": self.max_experiments,
        }


@dataclass
class CampaignBudget:
    max_wall_time_minutes: int = 120
    max_parallel: int = 1
    max_experiments: int = 12
    max_retries: int = 1

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "CampaignBudget":
        return cls(
            max_wall_time_minutes=max(1, int(payload.get("max_wall_time_minutes") or 120)),
            max_parallel=max(1, int(payload.get("max_parallel") or 1)),
            max_experiments=max(1, int(payload.get("max_experiments") or 12)),
            max_retries=max(1, int(payload.get("max_retries") or 1)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_wall_time_minutes": self.max_wall_time_minutes,
            "max_parallel": self.max_parallel,
            "max_experiments": self.max_experiments,
            "max_retries": self.max_retries,
        }


@dataclass
class CampaignSpec:
    campaign_id: str
    mode: str
    budget: CampaignBudget
    stages: list[StageSpec]
    post_actions: dict[str, bool]
    seed_policy_config: str

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "CampaignSpec":
        campaign_id = str(payload.get("campaign_id") or "").strip()
        if not campaign_id:
            raise ValueError("campaign_id is required")
        mode = str(payload.get("mode") or "quick").strip().lower()
        if mode not in {"quick", "nightly"}:
            raise ValueError("mode must be quick/nightly")
        budget = CampaignBudget.from_dict(payload.get("budget") or {})
        raw_stages = payload.get("stages")
        if not isinstance(raw_stages, list) or not raw_stages:
            raise ValueError("stages must be non-empty list")
        stages = [StageSpec.from_dict(x) for x in raw_stages]
        post_actions = payload.get("post_actions")
        if not isinstance(post_actions, dict):
            post_actions = {}
        normalized_post = {
            "coverage": bool(post_actions.get("coverage", True)),
            "flake": bool(post_actions.get("flake", False)),
            "ranking": bool(post_actions.get("ranking", True)),
            "champion_update": bool(post_actions.get("champion_update", True)),
            "cleanup": bool(post_actions.get("cleanup", True)),
            "triage": bool(post_actions.get("triage", True)),
            "dashboard": bool(post_actions.get("dashboard", True)),
            "bisect": bool(post_actions.get("bisect", True)),
        }
        seed_policy_config = str(
            payload.get("seed_policy_config") or "configs/experiments/seeds_p23.yaml"
        ).strip()
        spec = cls(
            campaign_id=campaign_id,
            mode=mode,
            budget=budget,
            stages=stages,
            post_actions=normalized_post,
            seed_policy_config=seed_policy_config,
        )
        spec.validate_dependencies()
        return spec

    def validate_dependencies(self) -> None:
        known = {s.stage_id for s in self.stages}
        for stage in self.stages:
            for dep in stage.depends_on:
                if dep not in known:
                    raise ValueError(f"stage {stage.stage_id} depends_on unknown stage {dep}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "campaign_id": self.campaign_id,
            "mode": self.mode,
            "budget": self.budget.to_dict(),
            "seed_policy_config": self.seed_policy_config,
            "stages": [s.to_dict() for s in self.stages],
            "post_actions": self.post_actions,
        }


def load_campaign(path: Path) -> CampaignSpec:
    return CampaignSpec.from_dict(load_mapping(path))

