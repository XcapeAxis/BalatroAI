from __future__ import annotations

if __package__ is None or __package__ == "":
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from trainer.closed_loop.replay_manifest import write_json


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def discover_world_model_checkpoint(repo_root: Path | None = None) -> str:
    root = (repo_root or _repo_root()) / "docs/artifacts/p45/wm_train"
    if not root.exists():
        return ""
    candidates = sorted(root.glob("**/best.pt"), key=lambda path: str(path))
    return str(candidates[-1].resolve()) if candidates else ""


def discover_policy_model_path(repo_root: Path | None = None) -> str:
    repo = repo_root or _repo_root()
    candidates: list[Path] = []
    for root in (
        repo / "docs/artifacts/p46/imagination_pipeline",
        repo / "docs/artifacts/p41/candidate_train",
        repo / "docs/artifacts/p22/runs",
    ):
        if not root.exists():
            continue
        candidates.extend(root.glob("**/best.pt"))
    ordered = sorted({path.resolve() for path in candidates}, key=lambda path: str(path))
    return str(ordered[-1]) if ordered else ""


@dataclass
class ControllerCapability:
    controller_id: str
    controller_type: str
    status: str
    supports_action_types: list[str]
    supports_position_sensitive: bool
    supports_stateful_jokers: bool
    requires_world_model: bool
    estimated_inference_cost: str
    recommended_use_cases: list[str]
    known_failure_modes: list[str]
    entry_file: str
    dependency_paths: list[str]
    notes: str = ""


def _controller_rows(*, world_model_checkpoint: str, model_path: str) -> list[ControllerCapability]:
    wm_available = bool(world_model_checkpoint) and Path(world_model_checkpoint).exists()
    policy_available = bool(model_path) and Path(model_path).exists()
    return [
        ControllerCapability(
            controller_id="policy_baseline",
            controller_type="policy",
            status="active" if policy_available else "stub",
            supports_action_types=["play", "discard", "shop", "consumable", "transition"],
            supports_position_sensitive=False,
            supports_stateful_jokers=False,
            requires_world_model=False,
            estimated_inference_cost="low",
            recommended_use_cases=["low-latency baseline inference", "stable regression compare"],
            known_failure_modes=["checkpoint_missing", "confidence_collapse_in_unseen_slices"],
            entry_file="trainer/policy_arena/adapters/model_adapter.py",
            dependency_paths=[model_path] if policy_available else [],
            notes="Uses the latest candidate checkpoint when available; otherwise remains a stub.",
        ),
        ControllerCapability(
            controller_id="heuristic_baseline",
            controller_type="heuristic",
            status="active",
            supports_action_types=["play", "discard", "shop", "consumable", "transition"],
            supports_position_sensitive=False,
            supports_stateful_jokers=False,
            requires_world_model=False,
            estimated_inference_cost="low",
            recommended_use_cases=["fallback path", "no-model environments", "quick smoke coverage"],
            known_failure_modes=["limited adaptation to unusual joker states", "lower peak strength"],
            entry_file="trainer/policy_arena/adapters/heuristic_adapter.py",
            dependency_paths=["trainer/expert_policy.py", "trainer/expert_policy_shop.py"],
            notes="Always available and serves as the final fallback controller.",
        ),
        ControllerCapability(
            controller_id="search_baseline",
            controller_type="search",
            status="active",
            supports_action_types=["play", "discard", "shop", "transition"],
            supports_position_sensitive=False,
            supports_stateful_jokers=False,
            requires_world_model=False,
            estimated_inference_cost="medium",
            recommended_use_cases=["low-confidence hand selection", "high-risk late-game slices"],
            known_failure_modes=["budget pressure", "shop/booster phases fall back to heuristic"],
            entry_file="trainer/policy_arena/adapters/search_adapter.py",
            dependency_paths=["trainer/search_expert.py", "trainer/expert_policy.py", "trainer/expert_policy_shop.py"],
            notes="Best suited for SELECTING_HAND states with enough search budget.",
        ),
        ControllerCapability(
            controller_id="policy_plus_wm_rerank",
            controller_type="wm_rerank",
            status="active" if wm_available else "stub",
            supports_action_types=["play", "discard", "shop", "transition"],
            supports_position_sensitive=False,
            supports_stateful_jokers=False,
            requires_world_model=True,
            estimated_inference_cost="medium",
            recommended_use_cases=["confident policy states", "short-horizon world-model reranking"],
            known_failure_modes=["world_model_bias", "uncertainty_underestimation", "checkpoint_missing"],
            entry_file="trainer/policy_arena/adapters/wm_rerank_adapter.py",
            dependency_paths=([world_model_checkpoint] if wm_available else []) + ([model_path] if policy_available else []),
            notes="Wraps the policy candidate set with short-horizon world-model reranking.",
        ),
        ControllerCapability(
            controller_id="hybrid_controller_v1",
            controller_type="hybrid",
            status="experimental",
            supports_action_types=["play", "discard", "shop", "consumable", "transition"],
            supports_position_sensitive=False,
            supports_stateful_jokers=False,
            requires_world_model=wm_available,
            estimated_inference_cost="adaptive",
            recommended_use_cases=["state-aware controller routing", "budget-aware evaluation", "uncertainty-aware fallback"],
            known_failure_modes=["routing_rules_need_recalibration", "budget_search_tradeoff", "wm_overtrust_if_thresholds_are_loose"],
            entry_file="trainer/hybrid/hybrid_controller.py",
            dependency_paths=[
                "trainer/hybrid/router.py",
                "trainer/hybrid/routing_features.py",
                "trainer/hybrid/controller_registry.py",
            ]
            + ([world_model_checkpoint] if wm_available else []),
            notes="Rule-based, explainable router over policy/search/heuristic/world-model-assisted controllers.",
        ),
    ]


def build_controller_registry(
    *,
    repo_root: str | Path | None = None,
    world_model_checkpoint: str = "",
    model_path: str = "",
) -> dict[str, Any]:
    repo = Path(repo_root).resolve() if repo_root else _repo_root()
    resolved_wm = str(world_model_checkpoint or discover_world_model_checkpoint(repo))
    resolved_model = str(model_path or discover_policy_model_path(repo))
    rows = _controller_rows(world_model_checkpoint=resolved_wm, model_path=resolved_model)
    return {
        "schema": "p48_controller_registry_v1",
        "generated_at": _now_iso(),
        "repo_root": str(repo),
        "world_model_checkpoint": resolved_wm,
        "policy_model_path": resolved_model,
        "controllers": [asdict(row) for row in rows],
    }


def registry_table_lines(payload: dict[str, Any]) -> list[str]:
    rows = payload.get("controllers") if isinstance(payload.get("controllers"), list) else []
    lines = [
        "controller_id | type | status | cost | requires_wm | entry_file",
        "--- | --- | --- | --- | --- | ---",
    ]
    for row in rows:
        if not isinstance(row, dict):
            continue
        lines.append(
            "{id} | {ctype} | {status} | {cost} | {wm} | {entry}".format(
                id=row.get("controller_id"),
                ctype=row.get("controller_type"),
                status=row.get("status"),
                cost=row.get("estimated_inference_cost"),
                wm=str(bool(row.get("requires_world_model"))).lower(),
                entry=row.get("entry_file"),
            )
        )
    return lines


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="P48 controller capability registry.")
    parser.add_argument("--list", action="store_true", help="print the registry to stdout")
    parser.add_argument("--json", action="store_true", help="print JSON instead of a table")
    parser.add_argument("--world-model-checkpoint", default="")
    parser.add_argument("--model-path", default="")
    parser.add_argument("--out-dir", default="docs/artifacts/p48")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = _repo_root()
    payload = build_controller_registry(
        repo_root=repo_root,
        world_model_checkpoint=str(args.world_model_checkpoint or ""),
        model_path=str(args.model_path or ""),
    )
    out_root = Path(args.out_dir)
    if not out_root.is_absolute():
        out_root = (repo_root / out_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    out_path = out_root / f"controller_registry_{_now_stamp()}.json"
    write_json(out_path, payload)
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print("\n".join(registry_table_lines(payload)))
    print(str(out_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
