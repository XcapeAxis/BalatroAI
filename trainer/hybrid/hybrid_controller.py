from __future__ import annotations

if __package__ is None or __package__ == "":
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

import argparse
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any

try:  # pragma: no cover
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None

from trainer.closed_loop.regression_triage import run_regression_triage
from trainer.closed_loop.replay_manifest import build_seeds_payload, now_iso, now_stamp, to_abs_path, write_json, write_markdown
from trainer.hybrid.controller_registry import build_controller_registry, discover_policy_model_path, discover_world_model_checkpoint
from trainer.hybrid.router import RuleBasedHybridRouter, run_router_smoke, summarize_routing_trace
from trainer.hybrid.routing_features import collect_sample_states, extract_routing_features, run_routing_feature_smoke
from trainer.policy_arena.adapters.heuristic_adapter import HeuristicAdapter
from trainer.policy_arena.adapters.model_adapter import ModelAdapter
from trainer.policy_arena.adapters.search_adapter import SearchAdapter
from trainer.policy_arena.adapters.wm_rerank_adapter import WMRerankAdapter
from trainer.policy_arena.policy_adapter import AdapterDescriptor, BasePolicyAdapter, normalize_action, phase_from_obs
from trainer.world_model.lookahead_planner import WorldModelLookaheadPlanner


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


def _read_yaml_or_json(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8-sig")
    if path.suffix.lower() in {".yaml", ".yml"}:
        if yaml is not None:
            payload = yaml.safe_load(text)
        else:
            sidecar = path.with_suffix(".json")
            if not sidecar.exists():
                raise RuntimeError(f"PyYAML unavailable and no sidecar JSON for: {path}")
            payload = json.loads(sidecar.read_text(encoding="utf-8-sig"))
    else:
        payload = json.loads(text)
    if not isinstance(payload, dict):
        raise ValueError(f"config must be mapping: {path}")
    return payload


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as fp:
        fp.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def _read_json_list(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
        return [row for row in payload if isinstance(row, dict)] if isinstance(payload, list) else []
    except Exception:
        return []


def _run_process(command: list[str], *, cwd: Path, timeout_sec: int) -> dict[str, Any]:
    proc = subprocess.run(command, cwd=str(cwd), text=True, capture_output=True, timeout=max(60, int(timeout_sec)))
    return {
        "returncode": int(proc.returncode),
        "stdout": str(proc.stdout or ""),
        "stderr": str(proc.stderr or ""),
        "command": command,
    }


def _resolve_seeds(cfg: dict[str, Any], seeds_override: list[str] | None, *, quick: bool) -> list[str]:
    if seeds_override:
        seeds = [str(seed).strip() for seed in seeds_override if str(seed).strip()]
    else:
        raw = cfg.get("seeds")
        seeds = [str(seed).strip() for seed in raw if str(seed).strip()] if isinstance(raw, list) else ["AAAAAAA", "BBBBBBB"]
    return seeds[:2] if quick and len(seeds) > 2 else seeds


def _pick_summary_row(summary_rows: list[dict[str, Any]], policy_id: str) -> dict[str, Any]:
    token = str(policy_id or "").strip()
    for row in summary_rows:
        if str(row.get("policy_id") or "") == token:
            return row
    return {}


def _selection_distribution(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    counter: Counter[str] = Counter()
    for row in rows:
        if not isinstance(row, dict):
            continue
        counter[str(row.get("selected_controller") or "unknown")] += 1
    total = sum(counter.values())
    return [
        {"controller_id": key, "count": int(value), "ratio": float(value) / max(1, total)}
        for key, value in sorted(counter.items(), key=lambda kv: (-kv[1], kv[0]))
    ]


class AdaptiveHybridController(BasePolicyAdapter):
    def __init__(
        self,
        *,
        name: str = "hybrid_controller_v1",
        model_path: str = "",
        world_model_checkpoint: str = "",
        top_k: int = 4,
        router_config: dict[str, Any] | None = None,
        search_max_branch: int = 80,
        search_max_depth: int = 2,
        search_time_budget_ms: float = 15.0,
        wm_horizon: int = 1,
        wm_uncertainty_penalty: float = 0.5,
        trace_path: str = "",
        trace_context: dict[str, Any] | None = None,
    ) -> None:
        self.model_path = str(model_path or "")
        self.world_model_checkpoint = str(world_model_checkpoint or "")
        self.top_k = max(1, int(top_k))
        self.search_max_branch = int(search_max_branch)
        self.search_max_depth = int(search_max_depth)
        self.search_time_budget_ms = float(search_time_budget_ms)
        self.wm_horizon = max(1, int(wm_horizon))
        self.wm_uncertainty_penalty = float(wm_uncertainty_penalty)
        self.trace_path = str(trace_path or "")
        self.trace_context = dict(trace_context or {})
        self.trace_index = 0
        self.last_trace: dict[str, Any] = {}

        self.registry = build_controller_registry(world_model_checkpoint=self.world_model_checkpoint, model_path=self.model_path)
        self.controller_map = {
            str(row.get("controller_id") or ""): row
            for row in (self.registry.get("controllers") if isinstance(self.registry.get("controllers"), list) else [])
            if isinstance(row, dict) and str(row.get("controller_id") or "")
        }
        self.router = RuleBasedHybridRouter(router_config)
        self.planner = WorldModelLookaheadPlanner(
            checkpoint_path=self.world_model_checkpoint,
            horizon=self.wm_horizon,
            uncertainty_penalty=self.wm_uncertainty_penalty,
            reward_weight=1.0,
            score_weight=0.5,
            value_weight=0.15,
            terminal_bonus=0.0,
        )
        self.controllers: dict[str, BasePolicyAdapter] = {
            "policy_baseline": ModelAdapter(name=f"{name}_policy", model_path=self.model_path, strategy="bc"),
            "heuristic_baseline": HeuristicAdapter(name=f"{name}_heuristic"),
            "search_baseline": SearchAdapter(
                name=f"{name}_search",
                max_branch=self.search_max_branch,
                max_depth=self.search_max_depth,
                time_budget_ms=self.search_time_budget_ms,
            ),
            "policy_plus_wm_rerank": WMRerankAdapter(
                name=f"{name}_wm_rerank",
                base_policy="policy_baseline",
                candidate_source="policy_topk",
                model_path=self.model_path,
                world_model_checkpoint=self.world_model_checkpoint,
                top_k=self.top_k,
                horizon=self.wm_horizon,
                uncertainty_penalty=self.wm_uncertainty_penalty,
                search_max_branch=self.search_max_branch,
                search_max_depth=self.search_max_depth,
                search_time_budget_ms=self.search_time_budget_ms,
            ),
        }
        super().__init__(
            descriptor=AdapterDescriptor(
                name=name,
                family="hybrid",
                status="experimental",
                supports_batch=False,
                supports_shop=True,
                supports_consumables=True,
                supports_position_actions=False,
                notes="Rule-based adaptive hybrid controller over policy/search/world-model rerank.",
            )
        )

    def describe(self) -> dict[str, Any]:
        payload = super().describe()
        payload["adapter"]["assist_mode"] = "hybrid_router"
        payload["adapter"]["wm_assist_enabled"] = bool(self.planner.available)
        payload["adapter"]["world_model_checkpoint"] = self.world_model_checkpoint
        payload["adapter"]["model_path"] = self.model_path
        payload["adapter"]["top_k"] = int(self.top_k)
        payload["adapter"]["search_max_depth"] = int(self.search_max_depth)
        payload["adapter"]["search_time_budget_ms"] = float(self.search_time_budget_ms)
        payload["adapter"]["wm_uncertainty_penalty"] = float(self.wm_uncertainty_penalty)
        payload["adapter"]["router"] = self.router.describe()
        payload["adapter"]["controller_registry"] = self.registry
        payload["adapter"]["trace_path"] = self.trace_path
        return payload

    def reset(self, seed: str | int | None = None) -> None:
        super().reset(seed)
        for controller in self.controllers.values():
            controller.reset(seed)

    def _available_controllers(self) -> dict[str, dict[str, Any]]:
        rows: dict[str, dict[str, Any]] = {}
        for controller_id, controller in self.controllers.items():
            capability = dict(self.controller_map.get(controller_id) or {})
            status = str((controller.describe().get("adapter") or {}).get("status") or capability.get("status") or "unknown")
            if status != "stub":
                capability["status"] = status
                rows[controller_id] = capability
        if "heuristic_baseline" not in rows and "heuristic_baseline" in self.controller_map:
            rows["heuristic_baseline"] = dict(self.controller_map["heuristic_baseline"])
        return rows

    def _fallback_order(self, selected: str) -> list[str]:
        ordered = [selected, "policy_baseline", "heuristic_baseline", "search_baseline"]
        seen: set[str] = set()
        rows: list[str] = []
        for token in ordered:
            if token in seen:
                continue
            seen.add(token)
            if token in self.controllers:
                rows.append(token)
        return rows

    def act(self, obs: dict[str, Any], legal_actions: list[dict[str, Any]] | None = None) -> dict[str, Any]:
        features = extract_routing_features(
            obs=obs,
            registry_payload=self.registry,
            planner=self.planner,
            legal_actions=legal_actions,
            seed=self._seed,
            top_k=self.top_k,
            model_path=self.model_path,
            search_max_branch=self.search_max_branch,
            search_max_depth=self.search_max_depth,
            search_time_budget_ms=self.search_time_budget_ms,
        )
        decision = self.router.route(features=features, available_controllers=self._available_controllers())
        phase = phase_from_obs(obs)
        initial_selected = str(decision.get("selected_controller") or "heuristic_baseline")
        final_selected = initial_selected
        final_action: dict[str, Any] = {}
        tried: list[str] = []
        fallback_reason = ""

        for controller_id in self._fallback_order(initial_selected):
            controller = self.controllers.get(controller_id)
            if controller is None:
                continue
            tried.append(controller_id)
            try:
                final_action = normalize_action(controller.act(obs, legal_actions=legal_actions), phase=phase)
                final_selected = controller_id
                if controller_id != initial_selected:
                    fallback_reason = f"fallback_from_{initial_selected}_to_{controller_id}"
                break
            except Exception as exc:
                fallback_reason = f"{controller_id}_failed:{exc}"

        if not final_action:
            final_selected = "heuristic_baseline"
            final_action = normalize_action(self.controllers["heuristic_baseline"].act(obs, legal_actions=legal_actions), phase=phase)
            if not fallback_reason:
                fallback_reason = f"fallback_from_{initial_selected}_to_heuristic_baseline"

        self.trace_index += 1
        trace = {
            "schema": "p48_hybrid_trace_v1",
            "generated_at": now_iso(),
            "trace_index": int(self.trace_index),
            "seed": str(self._seed),
            "phase": phase,
            "selected_controller": final_selected,
            "initial_selected_controller": initial_selected,
            "routing_reason": str(decision.get("routing_reason") or ""),
            "routing_score_breakdown": decision.get("routing_score_breakdown") if isinstance(decision.get("routing_score_breakdown"), dict) else {},
            "rejected_controllers": decision.get("rejected_controllers") if isinstance(decision.get("rejected_controllers"), list) else [],
            "key_feature_values": decision.get("key_feature_values") if isinstance(decision.get("key_feature_values"), dict) else {},
            "features": features,
            "final_action": final_action,
            "fallback_used": bool(final_selected != initial_selected),
            "fallback_reason": fallback_reason,
            "controllers_tried": tried,
            "trace_context": self.trace_context,
        }
        self.last_trace = trace
        if self.trace_path:
            _append_jsonl(Path(self.trace_path), trace)
        return final_action

    def candidate_actions(self, obs: dict[str, Any], legal_actions: list[dict[str, Any]] | None = None, *, top_k: int = 4) -> list[dict[str, Any]]:
        features = extract_routing_features(
            obs=obs,
            registry_payload=self.registry,
            planner=self.planner,
            legal_actions=legal_actions,
            seed=self._seed,
            top_k=max(1, int(top_k)),
            model_path=self.model_path,
            search_max_branch=self.search_max_branch,
            search_max_depth=self.search_max_depth,
            search_time_budget_ms=self.search_time_budget_ms,
        )
        selected = str(self.router.route(features=features, available_controllers=self._available_controllers()).get("selected_controller") or "heuristic_baseline")
        controller = self.controllers.get(selected) or self.controllers["heuristic_baseline"]
        try:
            return controller.candidate_actions(obs, legal_actions=legal_actions, top_k=max(1, int(top_k)))
        except Exception:
            return self.controllers["heuristic_baseline"].candidate_actions(obs, legal_actions=legal_actions, top_k=max(1, int(top_k)))

    def close(self) -> None:
        for controller in self.controllers.values():
            controller.close()
        super().close()


def _routing_summary_from_trace(trace_path: Path) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    if trace_path.exists():
        for line in trace_path.read_text(encoding="utf-8-sig").splitlines():
            text = line.strip()
            if not text:
                continue
            try:
                payload = json.loads(text)
            except Exception:
                continue
            if isinstance(payload, dict):
                rows.append(payload)
    summary = summarize_routing_trace(rows)
    summary["routing_decision_impact"] = {
        "selection_distribution": _selection_distribution(rows),
        "fallback_rate": float(sum(1 for row in rows if bool((row or {}).get("fallback_used")))) / max(1, len(rows)),
    }
    return summary


def _routing_summary_markdown(summary: dict[str, Any]) -> list[str]:
    lines = [
        "# P48 Routing Summary",
        "",
        f"- decision_count: {int(summary.get('decision_count') or 0)}",
        f"- fallback_rate: {float(((summary.get('routing_decision_impact') or {}).get('fallback_rate') or 0.0)):.3f}",
        "",
        "## Controller Selection Distribution",
    ]
    rows = summary.get("controller_selection_distribution") if isinstance(summary.get("controller_selection_distribution"), list) else []
    if rows:
        for row in rows:
            if not isinstance(row, dict):
                continue
            lines.append(
                "- {controller}: count={count} ratio={ratio:.3f}".format(
                    controller=row.get("controller_id"),
                    count=int(row.get("count") or 0),
                    ratio=float(row.get("ratio") or 0.0),
                )
            )
    else:
        lines.append("- none")
    return lines


def run_hybrid_controller_smoke(
    *,
    out_dir: str | Path | None = None,
    seed: str = "AAAAAAA",
    max_states: int = 8,
    top_k: int = 4,
    model_path: str = "",
    world_model_checkpoint: str = "",
    router_config: dict[str, Any] | None = None,
    search_max_branch: int = 80,
    search_max_depth: int = 2,
    search_time_budget_ms: float = 15.0,
    wm_horizon: int = 1,
    wm_uncertainty_penalty: float = 0.5,
) -> dict[str, Any]:
    repo_root = Path(__file__).resolve().parents[2]
    output_root = (repo_root / "docs/artifacts/p48").resolve() if out_dir is None else (Path(out_dir) if Path(out_dir).is_absolute() else (repo_root / out_dir).resolve())
    output_root.mkdir(parents=True, exist_ok=True)
    run_id = now_stamp()
    trace_path = (output_root / "router_traces" / run_id / "routing_trace.jsonl").resolve()
    controller = AdaptiveHybridController(
        name="hybrid_controller_v1",
        model_path=str(model_path or discover_policy_model_path(repo_root)),
        world_model_checkpoint=str(world_model_checkpoint or discover_world_model_checkpoint(repo_root)),
        top_k=top_k,
        router_config=router_config,
        search_max_branch=search_max_branch,
        search_max_depth=search_max_depth,
        search_time_budget_ms=search_time_budget_ms,
        wm_horizon=wm_horizon,
        wm_uncertainty_penalty=wm_uncertainty_penalty,
        trace_path=str(trace_path),
        trace_context={"mode": "smoke", "run_id": run_id},
    )
    controller.reset(seed)
    rows: list[dict[str, Any]] = []
    try:
        for sample in collect_sample_states(seed=seed, max_states=max_states):
            state = sample.get("state") if isinstance(sample.get("state"), dict) else {}
            action = controller.act(state)
            rows.append(
                {
                    "sample_id": str(sample.get("sample_id") or ""),
                    "step_idx": int(sample.get("step_idx") or 0),
                    "phase": str(sample.get("phase") or phase_from_obs(state)),
                    "selected_controller": str((controller.last_trace or {}).get("selected_controller") or ""),
                    "routing_reason": str((controller.last_trace or {}).get("routing_reason") or ""),
                    "final_action": action,
                }
            )
    finally:
        controller.close()
    summary = _routing_summary_from_trace(trace_path)
    write_json(trace_path.parent / "routing_summary.json", summary)
    payload = {
        "schema": "p48_hybrid_controller_smoke_v1",
        "generated_at": now_iso(),
        "seed": seed,
        "sample_count": len(rows),
        "rows": rows,
        "trace_path": str(trace_path),
        "routing_summary_json": str((trace_path.parent / "routing_summary.json").resolve()),
    }
    out_path = output_root / f"hybrid_controller_smoke_{run_id}.json"
    write_json(out_path, payload)
    return {"status": "ok", "out": str(out_path), "payload": payload}


def run_hybrid_controller_pipeline(
    *,
    config_path: str | Path,
    out_dir: str | Path | None = None,
    run_id: str = "",
    quick: bool = False,
    dry_run: bool = False,
    seeds_override: list[str] | None = None,
) -> dict[str, Any]:
    repo_root = Path(__file__).resolve().parents[2]
    cfg = _read_yaml_or_json(to_abs_path(repo_root, config_path))
    output_cfg = cfg.get("output") if isinstance(cfg.get("output"), dict) else {}
    routing_cfg = cfg.get("routing") if isinstance(cfg.get("routing"), dict) else {}
    router_cfg = routing_cfg.get("router") if isinstance(routing_cfg.get("router"), dict) else {}
    controller_cfg = cfg.get("controllers") if isinstance(cfg.get("controllers"), dict) else {}
    search_cfg = controller_cfg.get("search") if isinstance(controller_cfg.get("search"), dict) else {}
    wm_cfg = controller_cfg.get("wm_rerank") if isinstance(controller_cfg.get("wm_rerank"), dict) else {}
    arena_cfg = cfg.get("arena_compare") if isinstance(cfg.get("arena_compare"), dict) else {}

    run_name = str(run_id or output_cfg.get("run_id") or now_stamp())
    run_root = (repo_root / str(output_cfg.get("artifacts_root") or "docs/artifacts/p48/arena_ablation")).resolve() if out_dir is None else (Path(out_dir) if Path(out_dir).is_absolute() else (repo_root / out_dir).resolve())
    run_dir = run_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    seeds = _resolve_seeds(cfg, seeds_override, quick=quick)
    write_json(run_dir / "seeds_used.json", build_seeds_payload(seeds, seed_policy_version="p48.hybrid_controller"))
    model_path = str(controller_cfg.get("policy_model_path") or discover_policy_model_path(repo_root))
    wm_checkpoint = str(controller_cfg.get("world_model_checkpoint") or discover_world_model_checkpoint(repo_root))

    write_json(run_dir / "controller_registry.json", build_controller_registry(repo_root=repo_root, world_model_checkpoint=wm_checkpoint, model_path=model_path))
    feature_smoke = run_routing_feature_smoke(out_dir=run_dir, seed=seeds[0], max_states=min(6, _safe_int(routing_cfg.get("feature_smoke_states"), 6)), top_k=max(2, _safe_int(wm_cfg.get("top_k"), 4)), model_path=model_path, world_model_checkpoint=wm_checkpoint, search_max_branch=_safe_int(search_cfg.get("max_branch"), 80), search_max_depth=_safe_int(search_cfg.get("max_depth"), 2), search_time_budget_ms=_safe_float(search_cfg.get("time_budget_ms"), 15.0))
    router_smoke = run_router_smoke(out_dir=run_dir, seed=seeds[0], max_states=min(6, _safe_int(routing_cfg.get("router_smoke_states"), 6)), model_path=model_path, world_model_checkpoint=wm_checkpoint, router_config=router_cfg)
    hybrid_smoke = run_hybrid_controller_smoke(out_dir=run_dir, seed=seeds[0], max_states=min(10, _safe_int(routing_cfg.get("hybrid_smoke_states"), 8)), top_k=max(2, _safe_int(wm_cfg.get("top_k"), 4)), model_path=model_path, world_model_checkpoint=wm_checkpoint, router_config=router_cfg, search_max_branch=_safe_int(search_cfg.get("max_branch"), 80), search_max_depth=_safe_int(search_cfg.get("max_depth"), 2), search_time_budget_ms=_safe_float(search_cfg.get("time_budget_ms"), 15.0), wm_horizon=_safe_int(wm_cfg.get("horizon"), 1), wm_uncertainty_penalty=_safe_float(wm_cfg.get("uncertainty_penalty"), 0.5))

    policy_assist_map = {
        "policy_plus_wm_rerank": {
            "base_policy": "policy_baseline",
            "candidate_source": "policy_topk",
            "model_path": model_path,
            "world_model_checkpoint": wm_checkpoint,
            "top_k": max(2, _safe_int(wm_cfg.get("top_k"), 4)),
            "horizon": _safe_int(wm_cfg.get("horizon"), 1),
            "uncertainty_penalty": _safe_float(wm_cfg.get("uncertainty_penalty"), 0.5),
            "search_max_branch": _safe_int(search_cfg.get("max_branch"), 80),
            "search_max_depth": _safe_int(search_cfg.get("max_depth"), 2),
            "search_time_budget_ms": _safe_float(search_cfg.get("time_budget_ms"), 15.0),
        },
        "hybrid_controller_v1": {
            "model_path": model_path,
            "world_model_checkpoint": wm_checkpoint,
            "top_k": max(2, _safe_int(wm_cfg.get("top_k"), 4)),
            "router_config": router_cfg,
            "search_max_branch": _safe_int(search_cfg.get("max_branch"), 80),
            "search_max_depth": _safe_int(search_cfg.get("max_depth"), 2),
            "search_time_budget_ms": _safe_float(search_cfg.get("time_budget_ms"), 15.0),
            "wm_horizon": _safe_int(wm_cfg.get("horizon"), 1),
            "wm_uncertainty_penalty": _safe_float(wm_cfg.get("uncertainty_penalty"), 0.5),
            "trace_path": str((run_dir / "router_traces" / run_name / "routing_trace.jsonl").resolve()),
            "trace_context": {"pipeline": "p48", "run_id": run_name},
        },
    }
    write_json(run_dir / "policy_assist_map.json", policy_assist_map)
    write_json(run_dir / "policy_model_map.json", {"policy_baseline": model_path, "policy_plus_wm_rerank": model_path})

    policies = [str(arena_cfg.get("policy_baseline") or "policy_baseline"), str(arena_cfg.get("wm_rerank_baseline") or "policy_plus_wm_rerank"), str(arena_cfg.get("hybrid_policy") or "hybrid_controller_v1"), str(arena_cfg.get("search_baseline") or "search_baseline"), str(arena_cfg.get("heuristic_baseline") or "heuristic_baseline")]
    unique_policies: list[str] = []
    for policy_id in policies:
        if policy_id and policy_id not in unique_policies:
            unique_policies.append(policy_id)
    arena_cmd = [
        sys.executable,
        "-B",
        "-m",
        "trainer.policy_arena.arena_runner",
        "--out-dir",
        str(run_root),
        "--run-id",
        run_name,
        "--backend",
        str(arena_cfg.get("backend") or "sim"),
        "--mode",
        str(arena_cfg.get("mode") or "long_episode"),
        "--policies",
        ",".join(unique_policies),
        "--policy-assist-map-json",
        str((run_dir / "policy_assist_map.json").resolve()),
        "--policy-model-map-json",
        str((run_dir / "policy_model_map.json").resolve()),
        "--world-model-checkpoint",
        wm_checkpoint,
        "--seeds",
        ",".join(seeds),
        "--episodes-per-seed",
        str(max(1, _safe_int(arena_cfg.get("episodes_per_seed"), 1 if quick else 2))),
        "--max-steps",
        str(max(1, _safe_int(arena_cfg.get("max_steps"), 120 if quick else 180))),
        "--skip-unavailable",
    ]
    if quick:
        arena_cmd.append("--quick")
    arena_result = {"returncode": 0, "stdout": "", "stderr": "", "command": arena_cmd} if dry_run else _run_process(arena_cmd, cwd=repo_root, timeout_sec=_safe_int(arena_cfg.get("timeout_sec"), 3600))

    summary_rows = _read_json_list(run_dir / "summary_table.json")
    routing_summary = _routing_summary_from_trace(run_dir / "router_traces" / run_name / "routing_trace.jsonl")
    write_json(run_dir / "routing_summary.json", routing_summary)
    write_markdown(run_dir / "routing_summary.md", _routing_summary_markdown(routing_summary))
    run_manifest = _read_json(run_dir / "run_manifest.json")
    if run_manifest:
        run_manifest["hybrid_controller"] = {
            "world_model_checkpoint": wm_checkpoint,
            "policy_model_path": model_path,
            "routing_summary_json": str((run_dir / "routing_summary.json").resolve()),
            "controller_registry_json": str((run_dir / "controller_registry.json").resolve()),
            "wm_assist_enabled": bool(wm_checkpoint),
        }
        write_json(run_dir / "run_manifest.json", run_manifest)

    champion_policy = str(arena_cfg.get("champion_policy") or arena_cfg.get("policy_baseline") or "policy_baseline")
    candidate_policy = str(arena_cfg.get("candidate_policy") or arena_cfg.get("hybrid_policy") or "hybrid_controller_v1")
    promotion_payload = {
        "schema": "p48_promotion_decision_v1",
        "generated_at": now_iso(),
        "decision": "observe",
        "recommend_promotion": False,
        "candidate_policy_id": candidate_policy,
        "champion_policy_id": champion_policy,
        "reasons": ["champion_rules_unavailable_or_dry_run"],
    }
    if (run_dir / "summary_table.json").exists() and bool(arena_cfg.get("enable_champion_rules", True)) and not dry_run:
        champion_cmd = [sys.executable, "-B", "-m", "trainer.policy_arena.champion_rules", "--summary-json", str(run_dir / "summary_table.json"), "--out-dir", str(run_dir / "champion_eval"), "--candidate-policy", candidate_policy, "--champion-policy", champion_policy, "--episode-records-jsonl", str(run_dir / "episode_records.jsonl"), "--bucket-metrics-json", str(run_dir / "bucket_metrics.json"), "--champion-json", str(arena_cfg.get("champion_json") or "docs/artifacts/p22/champion.json")]
        champion_result = _run_process(champion_cmd, cwd=repo_root, timeout_sec=600)
        for line in str(champion_result.get("stdout") or "").splitlines():
            try:
                payload = json.loads(line.strip())
            except Exception:
                continue
            if isinstance(payload, dict) and Path(str(payload.get("json") or "")).exists():
                promotion_payload = _read_json(Path(str(payload.get("json"))))
                break
    write_json(run_dir / "promotion_decision.json", promotion_payload)

    triage_dir = (repo_root / str((cfg.get("triage") or {}).get("output_artifacts_root") or "docs/artifacts/p48/triage")).resolve() / run_name
    triage_summary = {"status": "skipped"} if dry_run or not (run_dir / "summary_table.json").exists() else run_regression_triage(current_run_dir=run_dir, out_dir=triage_dir)

    baseline_row = _pick_summary_row(summary_rows, champion_policy)
    hybrid_row = _pick_summary_row(summary_rows, candidate_policy)
    wm_row = _pick_summary_row(summary_rows, str(arena_cfg.get("wm_rerank_baseline") or "policy_plus_wm_rerank"))
    search_row = _pick_summary_row(summary_rows, str(arena_cfg.get("search_baseline") or "search_baseline"))
    summary = {
        "schema": "p48_hybrid_controller_pipeline_v1",
        "generated_at": now_iso(),
        "run_id": run_name,
        "status": ("ok" if int(arena_result.get("returncode") or 0) == 0 else "failed"),
        "dry_run": bool(dry_run),
        "world_model_checkpoint": wm_checkpoint,
        "policy_model_path": model_path,
        "registry_json": str((run_dir / "controller_registry.json").resolve()),
        "feature_smoke_out": str(feature_smoke.get("out") or ""),
        "router_smoke_md": str(router_smoke.get("routing_summary_md") or ""),
        "hybrid_smoke_out": str(hybrid_smoke.get("out") or ""),
        "routing_summary_json": str((run_dir / "routing_summary.json").resolve()),
        "promotion_decision_json": str((run_dir / "promotion_decision.json").resolve()),
        "triage_report_json": str(triage_summary.get("triage_report_json") or ""),
        "baseline_policy": champion_policy,
        "candidate_policy": candidate_policy,
        "baseline_score": _safe_float(baseline_row.get("mean_total_score"), 0.0),
        "hybrid_score": _safe_float(hybrid_row.get("mean_total_score"), 0.0),
        "hybrid_delta_vs_baseline": _safe_float(hybrid_row.get("mean_total_score"), 0.0) - _safe_float(baseline_row.get("mean_total_score"), 0.0),
        "wm_rerank_score": _safe_float(wm_row.get("mean_total_score"), 0.0),
        "search_score": _safe_float(search_row.get("mean_total_score"), 0.0),
        "controller_selection_distribution": routing_summary.get("controller_selection_distribution") if isinstance(routing_summary.get("controller_selection_distribution"), list) else [],
    }
    write_json(run_dir / "pipeline_summary.json", summary)
    write_markdown(
        run_dir / "pipeline_summary.md",
        [
            "# P48 Hybrid Controller Pipeline",
            "",
            f"- run_id: `{run_name}`",
            f"- baseline_policy: `{champion_policy}` score={float(summary.get('baseline_score') or 0.0):.6f}",
            f"- candidate_policy: `{candidate_policy}` score={float(summary.get('hybrid_score') or 0.0):.6f}",
            f"- hybrid_delta_vs_baseline: {float(summary.get('hybrid_delta_vs_baseline') or 0.0):.6f}",
            f"- wm_rerank_score: {float(summary.get('wm_rerank_score') or 0.0):.6f}",
            f"- search_score: {float(summary.get('search_score') or 0.0):.6f}",
        ],
    )

    summary["pipeline_summary_json"] = str((run_dir / "pipeline_summary.json").resolve())
    write_json(run_dir / "pipeline_summary.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="P48 adaptive hybrid controller smoke and pipeline.")
    parser.add_argument("--config", default="configs/experiments/p48_hybrid_controller_smoke.yaml")
    parser.add_argument("--out-dir", default="")
    parser.add_argument("--run-id", default="")
    parser.add_argument("--seed", default="AAAAAAA")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-states", type=int, default=8)
    parser.add_argument("--list-registry", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    if args.list_registry:
        print(json.dumps(build_controller_registry(repo_root=repo_root, world_model_checkpoint=discover_world_model_checkpoint(repo_root), model_path=discover_policy_model_path(repo_root)), ensure_ascii=False, indent=2))
        return 0
    if str(args.config or "").strip():
        summary = run_hybrid_controller_pipeline(config_path=args.config, out_dir=(args.out_dir if str(args.out_dir).strip() else None), run_id=str(args.run_id or ""), quick=bool(args.quick), dry_run=bool(args.dry_run))
    else:
        summary = run_hybrid_controller_smoke(out_dir=(args.out_dir if str(args.out_dir).strip() else None), seed=str(args.seed or "AAAAAAA"), max_states=max(1, int(args.max_states)))
    print(json.dumps(summary, ensure_ascii=False))
    return 0 if str(summary.get("status") or "ok") not in {"failed", "error"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
