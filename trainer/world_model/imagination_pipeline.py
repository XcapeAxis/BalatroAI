from __future__ import annotations

if __package__ is None or __package__ == "":
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

try:  # pragma: no cover - optional dependency
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None

from trainer.closed_loop.closed_loop_runner import run_closed_loop
from trainer.closed_loop.regression_triage import run_regression_triage
from trainer.closed_loop.replay_manifest import build_seeds_payload, now_iso, now_stamp, to_abs_path, write_json, write_markdown
from trainer.world_model.imagination_rollout import run_imagination_rollout
from trainer.world_model.train import run_world_model_train


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
    proc = subprocess.run(
        command,
        cwd=str(cwd),
        text=True,
        capture_output=True,
        timeout=max(60, int(timeout_sec)),
    )
    return {
        "returncode": int(proc.returncode),
        "stdout": str(proc.stdout or ""),
        "stderr": str(proc.stderr or ""),
        "command": command,
    }


def _resolve_world_model_checkpoint(repo_root: Path, cfg: dict[str, Any], *, quick: bool) -> tuple[str, dict[str, Any]]:
    world_model_cfg = cfg.get("world_model") if isinstance(cfg.get("world_model"), dict) else {}
    checkpoint_raw = str(world_model_cfg.get("checkpoint") or "").strip()
    if checkpoint_raw:
        checkpoint = to_abs_path(repo_root, checkpoint_raw)
        if checkpoint.exists():
            return str(checkpoint), {}
    latest_candidates = sorted((repo_root / "docs/artifacts/p45/wm_train").glob("**/best.pt"), key=lambda path: str(path))
    if latest_candidates:
        return str(latest_candidates[-1].resolve()), {}
    if not bool(world_model_cfg.get("auto_train_if_missing", True)):
        raise FileNotFoundError("P46 requires a P45 world model checkpoint")
    train_cfg_path = to_abs_path(repo_root, str(world_model_cfg.get("config") or "configs/experiments/p45_world_model_smoke.yaml"))
    summary = run_world_model_train(
        config_path=train_cfg_path,
        out_dir=(repo_root / "docs/artifacts/p46/world_model_bootstrap").resolve(),
        run_id=("p46-bootstrap-" + now_stamp()),
        quick=bool(quick),
    )
    checkpoint = str(summary.get("best_checkpoint") or "")
    if not checkpoint:
        raise RuntimeError("world model bootstrap training did not produce best checkpoint")
    return checkpoint, summary


def _resolve_seeds(cfg: dict[str, Any], seeds_override: list[str] | None, *, quick: bool) -> list[str]:
    if seeds_override:
        seeds = [str(seed).strip() for seed in seeds_override if str(seed).strip()]
        return seeds[:2] if quick and len(seeds) > 2 else seeds
    raw = cfg.get("seeds")
    if isinstance(raw, list):
        seeds = [str(seed).strip() for seed in raw if str(seed).strip()]
    else:
        seeds = ["AAAAAAA", "BBBBBBB"]
    return seeds[:2] if quick and len(seeds) > 2 else seeds


def _recipe_alias(recipe_id: str) -> str:
    return f"candidate_{str(recipe_id).strip().lower()}"


def _recipe_candidate_cfg(recipe_cfg: dict[str, Any], recipe_id: str) -> dict[str, Any]:
    candidate_cfg = recipe_cfg.get("candidate") if isinstance(recipe_cfg.get("candidate"), dict) else {}
    replay_cfg = recipe_cfg.get("replay") if isinstance(recipe_cfg.get("replay"), dict) else {}
    default_filter = "uncertainty_gate" if recipe_id.endswith("filtered") else ("disabled" if recipe_id == "real_only" else "none")
    filter_mode = str(candidate_cfg.get("filter_mode") or default_filter)
    require_gate = replay_cfg.get("require_uncertainty_gate_passed")
    if require_gate is None:
        require_gate = recipe_id.endswith("filtered")
    return {
        "filter_mode": filter_mode,
        "require_uncertainty_gate_passed": bool(require_gate),
    }


def _recipe_checkpoint(recipe_run: dict[str, Any]) -> str:
    run_dir = Path(str(recipe_run.get("run_dir") or ""))
    candidate_ref = _read_json(run_dir / "candidate_train_ref.json")
    summary = candidate_ref.get("summary") if isinstance(candidate_ref.get("summary"), dict) else {}
    return str(summary.get("best_checkpoint") or "")


def _merge_imagined_source(base_sources: list[dict[str, Any]], imagined_manifest_json: str, recipe_cfg: dict[str, Any]) -> list[dict[str, Any]]:
    sources = [dict(source) for source in base_sources if isinstance(source, dict) and str(source.get("type") or "") != "imagined_world_model"]
    replay_cfg = recipe_cfg.get("replay") if isinstance(recipe_cfg.get("replay"), dict) else {}
    sources.append(
        {
            "id": "imagined_world_model",
            "type": "imagined_world_model",
            "weight": float(replay_cfg.get("imagined_weight") or 0.10),
            "enabled": bool(replay_cfg.get("enabled", recipe_cfg.get("id") != "real_only")),
            "imagination_manifest": imagined_manifest_json,
            "max_samples": int(replay_cfg.get("max_samples") or 128),
            "require_uncertainty_gate_passed": bool(replay_cfg.get("require_uncertainty_gate_passed", True)),
            "max_imagined_fraction": float(replay_cfg.get("max_imagined_fraction") or 0.10),
            "max_imagination_horizon": int(replay_cfg.get("max_imagination_horizon") or 1),
        }
    )
    return sources


def _pick_summary_row(summary_path: Path, policy_id: str) -> dict[str, Any]:
    rows = _read_json_list(summary_path)
    for row in rows:
        if str(row.get("policy_id") or "") == str(policy_id):
            return row
    return {}


def _run_combined_arena_compare(
    *,
    repo_root: Path,
    cfg: dict[str, Any],
    run_dir: Path,
    seeds: list[str],
    recipe_runs: dict[str, dict[str, Any]],
    quick: bool,
) -> dict[str, Any]:
    arena_cfg = cfg.get("arena_compare") if isinstance(cfg.get("arena_compare"), dict) else {}
    if not bool(arena_cfg.get("enabled", True)):
        return {"status": "skipped", "reason": "arena_compare_disabled"}

    champion_policy = str(arena_cfg.get("champion_policy") or "heuristic_baseline")
    include_unfiltered = bool(arena_cfg.get("include_unfiltered", False))
    policy_model_map: dict[str, str] = {}
    policies = [champion_policy]
    for recipe_id in ("real_only", "real_plus_imagined_filtered"):
        summary = recipe_runs.get(recipe_id) if isinstance(recipe_runs.get(recipe_id), dict) else {}
        checkpoint = _recipe_checkpoint(summary)
        if checkpoint:
            alias = _recipe_alias(recipe_id)
            policies.append(alias)
            policy_model_map[alias] = checkpoint
    if include_unfiltered:
        summary = recipe_runs.get("real_plus_imagined") if isinstance(recipe_runs.get("real_plus_imagined"), dict) else {}
        checkpoint = _recipe_checkpoint(summary)
        if checkpoint:
            alias = _recipe_alias("real_plus_imagined")
            policies.append(alias)
            policy_model_map[alias] = checkpoint

    if len(policy_model_map) < 2 and not include_unfiltered:
        return {"status": "stub", "reason": "insufficient_candidate_checkpoints"}

    arena_root = (repo_root / str(arena_cfg.get("output_artifacts_root") or "docs/artifacts/p46/arena_compare")).resolve()
    arena_root.mkdir(parents=True, exist_ok=True)
    arena_run_id = str(arena_cfg.get("run_id") or run_dir.name)
    policy_map_path = run_dir / "policy_model_map.json"
    write_json(policy_map_path, policy_model_map)

    episodes_per_seed = int(arena_cfg.get("episodes_per_seed") or (1 if quick else 2))
    max_steps = int(arena_cfg.get("max_steps") or (120 if quick else 180))
    timeout_sec = int(arena_cfg.get("timeout_sec") or 3600)
    cmd = [
        sys.executable,
        "-B",
        "-m",
        "trainer.policy_arena.arena_runner",
        "--out-dir",
        str(arena_root),
        "--run-id",
        arena_run_id,
        "--backend",
        "sim",
        "--mode",
        str(arena_cfg.get("mode") or "long_episode"),
        "--policies",
        ",".join(policies),
        "--policy-model-map-json",
        str(policy_map_path),
        "--seeds",
        ",".join(seeds),
        "--episodes-per-seed",
        str(max(1, episodes_per_seed)),
        "--max-steps",
        str(max(1, max_steps)),
        "--skip-unavailable",
    ]
    if quick:
        cmd.append("--quick")
    result = _run_process(cmd, cwd=repo_root, timeout_sec=timeout_sec)
    arena_run_dir = arena_root / arena_run_id
    summary_json = arena_run_dir / "summary_table.json"
    bucket_json = arena_run_dir / "bucket_metrics.json"
    episode_records = arena_run_dir / "episode_records.jsonl"

    decision_payload: dict[str, Any] = {}
    champion_rules_cfg = arena_cfg.get("champion_rules") if isinstance(arena_cfg.get("champion_rules"), dict) else {}
    filtered_alias = _recipe_alias("real_plus_imagined_filtered")
    if summary_json.exists() and bool(champion_rules_cfg.get("enabled", True)) and filtered_alias in policy_model_map:
        champion_out = arena_run_dir / "champion_eval"
        champion_cmd = [
            sys.executable,
            "-B",
            "-m",
            "trainer.policy_arena.champion_rules",
            "--summary-json",
            str(summary_json),
            "--out-dir",
            str(champion_out),
            "--candidate-policy",
            filtered_alias,
            "--champion-policy",
            champion_policy,
            "--episode-records-jsonl",
            str(episode_records),
            "--bucket-metrics-json",
            str(bucket_json),
            "--champion-json",
            str(champion_rules_cfg.get("champion_json") or "docs/artifacts/p22/champion.json"),
        ]
        champion_result = _run_process(champion_cmd, cwd=repo_root, timeout_sec=600)
        for line in str(champion_result.get("stdout") or "").splitlines():
            text = line.strip()
            if not text:
                continue
            try:
                payload = json.loads(text)
            except Exception:
                continue
            if not isinstance(payload, dict):
                continue
            decision_path = Path(str(payload.get("json") or ""))
            if decision_path.exists():
                decision_payload = _read_json(decision_path)
                break
        if not decision_payload:
            decision_payload = {
                "schema": "p46_promotion_decision_v1",
                "generated_at": now_iso(),
                "decision": "observe",
                "recommend_promotion": False,
                "candidate_policy_id": filtered_alias,
                "champion_policy_id": champion_policy,
                "reasons": ["champion_rules_unavailable"],
            }
        write_json(arena_run_dir / "promotion_decision.json", decision_payload)
        write_markdown(
            arena_run_dir / "promotion_decision.md",
            [
                "# P46 Promotion Decision",
                "",
                f"- decision: `{decision_payload.get('decision')}`",
                f"- recommend_promotion: `{decision_payload.get('recommend_promotion')}`",
                f"- candidate_policy_id: `{decision_payload.get('candidate_policy_id')}`",
                f"- champion_policy_id: `{decision_payload.get('champion_policy_id')}`",
            ],
        )
    return {
        "status": "ok" if int(result.get("returncode") or 0) == 0 and summary_json.exists() else "failed",
        "arena_run_dir": str(arena_run_dir),
        "summary_table_json": str(summary_json),
        "bucket_metrics_json": str(bucket_json),
        "episode_records_jsonl": str(episode_records),
        "policy_model_map_json": str(policy_map_path),
        "promotion_decision_json": str(arena_run_dir / "promotion_decision.json"),
    }


def run_imagination_pipeline(
    *,
    config_path: str | Path,
    out_dir: str | Path | None = None,
    run_id: str = "",
    quick: bool = False,
    dry_run: bool = False,
    seeds_override: list[str] | None = None,
) -> dict[str, Any]:
    repo_root = Path(__file__).resolve().parents[2]
    cfg_path = to_abs_path(repo_root, config_path)
    cfg = _read_yaml_or_json(cfg_path)
    output_cfg = cfg.get("output") if isinstance(cfg.get("output"), dict) else {}
    chosen_run_id = str(run_id or output_cfg.get("run_id") or now_stamp())
    run_root = (
        (repo_root / str(output_cfg.get("artifacts_root") or "docs/artifacts/p46/imagination_pipeline")).resolve()
        if out_dir is None
        else (Path(out_dir) if Path(out_dir).is_absolute() else (repo_root / out_dir).resolve())
    )
    run_dir = run_root / chosen_run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    seeds = _resolve_seeds(cfg, seeds_override, quick=quick)
    write_json(run_dir / "seeds_used.json", build_seeds_payload(seeds, seed_policy_version="p46.imagination.pipeline"))

    checkpoint_path, bootstrap_summary = _resolve_world_model_checkpoint(repo_root, cfg, quick=quick)

    imagination_cfg = dict(cfg)
    imagination_cfg.setdefault("world_model", {})
    if isinstance(imagination_cfg.get("world_model"), dict):
        imagination_cfg["world_model"]["checkpoint"] = checkpoint_path
    generated_imagination_cfg = run_dir / "generated_imagination_config.json"
    write_json(generated_imagination_cfg, imagination_cfg)
    imagination_summary = run_imagination_rollout(
        config_path=generated_imagination_cfg,
        out_dir=(repo_root / "docs/artifacts/p46/imagination_rollouts").resolve(),
        run_id=chosen_run_id,
        quick=quick,
        dry_run=dry_run,
        seeds_override=seeds,
    )

    closed_loop_cfg = cfg.get("closed_loop") if isinstance(cfg.get("closed_loop"), dict) else {}
    base_closed_loop = _read_yaml_or_json(
        to_abs_path(repo_root, str(closed_loop_cfg.get("base_config") or "configs/experiments/p41_closed_loop_v2_smoke.yaml"))
    )
    base_replay = _read_yaml_or_json(
        to_abs_path(repo_root, str(closed_loop_cfg.get("base_replay_config") or "configs/experiments/p41_replay_mix_smoke.yaml"))
    )
    base_candidate = _read_yaml_or_json(
        to_abs_path(repo_root, str(closed_loop_cfg.get("base_candidate_config") or "configs/experiments/p41_candidate_smoke.yaml"))
    )

    recipe_runs: dict[str, dict[str, Any]] = {}
    recipes = cfg.get("recipes") if isinstance(cfg.get("recipes"), list) else []
    for recipe_cfg in recipes:
        if not isinstance(recipe_cfg, dict):
            continue
        recipe_id = str(recipe_cfg.get("id") or "").strip().lower()
        if not recipe_id or not bool(recipe_cfg.get("enabled", True)):
            continue

        replay_cfg = dict(base_replay)
        replay_cfg["seeds"] = list(seeds)
        replay_cfg["sources"] = _merge_imagined_source(
            base_replay.get("sources") if isinstance(base_replay.get("sources"), list) else [],
            str(imagination_summary.get("imagined_manifest_json") or ""),
            recipe_cfg,
        )
        generated_replay_cfg = run_dir / "generated_configs" / f"replay_{recipe_id}.json"
        write_json(generated_replay_cfg, replay_cfg)

        candidate_cfg = dict(base_candidate)
        candidate_cfg["candidate_modes"] = ["bc_finetune"]
        candidate_cfg["allow_legacy_fallback"] = False
        candidate_cfg["seeds"] = list(seeds)
        candidate_cfg["training"] = dict(candidate_cfg.get("training") if isinstance(candidate_cfg.get("training"), dict) else {})
        prefer_sources = candidate_cfg["training"].get("prefer_source_types")
        if not isinstance(prefer_sources, list) or not prefer_sources:
            prefer_sources = ["p13_dagger_or_real", "arena_failures", "selfsup_replay"]
        if recipe_id != "real_only" and "imagined_world_model" not in prefer_sources:
            prefer_sources = ["imagined_world_model", *[str(token) for token in prefer_sources]]
        candidate_cfg["training"]["prefer_source_types"] = prefer_sources
        recipe_candidate_cfg = _recipe_candidate_cfg(recipe_cfg, recipe_id)
        candidate_cfg["imagination"] = {
            "recipe": recipe_id,
            "enabled": recipe_id != "real_only",
            "filter_mode": str(recipe_candidate_cfg.get("filter_mode") or "none"),
            "require_uncertainty_gate_passed": bool(recipe_candidate_cfg.get("require_uncertainty_gate_passed")),
            "max_imagination_horizon": int(
                (((recipe_cfg.get("replay") or {}).get("max_imagination_horizon")) if isinstance(recipe_cfg.get("replay"), dict) else 1) or 1
            ),
            "max_imagined_fraction": float(
                (((recipe_cfg.get("replay") or {}).get("max_imagined_fraction")) if isinstance(recipe_cfg.get("replay"), dict) else 0.0) or 0.0
            ),
        }
        generated_candidate_cfg = run_dir / "generated_configs" / f"candidate_{recipe_id}.json"
        write_json(generated_candidate_cfg, candidate_cfg)

        closed_loop_recipe_cfg = dict(base_closed_loop)
        closed_loop_recipe_cfg["seeds"] = list(seeds)
        closed_loop_recipe_cfg["replay_mixer"] = {"config": str(generated_replay_cfg), "quick": bool(quick)}
        closed_loop_recipe_cfg["candidate_train"] = {"config": str(generated_candidate_cfg), "quick": bool(quick)}
        closed_loop_recipe_cfg["failure_mining"] = {"enabled": bool(recipe_cfg.get("failure_mining_enabled", False))}
        closed_loop_recipe_cfg["regression_triage"] = {"enabled": False}
        arena_enabled = bool(recipe_cfg.get("run_closed_loop", True))
        closed_loop_recipe_cfg["arena_eval"] = dict(
            closed_loop_recipe_cfg.get("arena_eval") if isinstance(closed_loop_recipe_cfg.get("arena_eval"), dict) else {}
        )
        closed_loop_recipe_cfg["arena_eval"]["enabled"] = arena_enabled
        generated_closed_loop_cfg = run_dir / "generated_configs" / f"closed_loop_{recipe_id}.json"
        write_json(generated_closed_loop_cfg, closed_loop_recipe_cfg)

        recipe_out_dir = run_dir / "recipe_runs" / recipe_id
        recipe_summary = run_closed_loop(
            config_path=generated_closed_loop_cfg,
            out_dir=recipe_out_dir,
            run_id=f"{chosen_run_id}-{recipe_id}",
            quick=quick,
            dry_run=dry_run,
            seeds_override=seeds,
        )
        recipe_runs[recipe_id] = recipe_summary

    arena_compare_summary = _run_combined_arena_compare(
        repo_root=repo_root,
        cfg=cfg,
        run_dir=run_dir,
        seeds=seeds,
        recipe_runs=recipe_runs,
        quick=quick,
    )
    triage_cfg = cfg.get("triage") if isinstance(cfg.get("triage"), dict) else {}
    triage_dir = (repo_root / str(triage_cfg.get("output_artifacts_root") or "docs/artifacts/p46/triage") / chosen_run_id).resolve()
    triage_dir.mkdir(parents=True, exist_ok=True)
    baseline_run_dir = Path(str((recipe_runs.get("real_only") or {}).get("run_dir") or "")).resolve() if recipe_runs.get("real_only") else None
    filtered_run_dir = Path(str((recipe_runs.get("real_plus_imagined_filtered") or {}).get("run_dir") or "")).resolve() if recipe_runs.get("real_plus_imagined_filtered") else None
    triage_summary = (
        run_regression_triage(
            current_run_dir=filtered_run_dir,
            out_dir=triage_dir,
            baseline_run_dir=baseline_run_dir,
        )
        if filtered_run_dir is not None and filtered_run_dir.exists() and baseline_run_dir is not None and baseline_run_dir.exists()
        else {"status": "skipped", "reason": "baseline_or_filtered_run_missing"}
    )

    combined_summary_json = Path(str(arena_compare_summary.get("summary_table_json") or ""))
    filtered_alias = _recipe_alias("real_plus_imagined_filtered")
    real_alias = _recipe_alias("real_only")
    filtered_row = _pick_summary_row(combined_summary_json, filtered_alias) if combined_summary_json.exists() else {}
    real_row = _pick_summary_row(combined_summary_json, real_alias) if combined_summary_json.exists() else {}
    filtered_score = float(filtered_row.get("mean_total_score") or 0.0)
    real_score = float(real_row.get("mean_total_score") or 0.0)
    filtered_win = float(filtered_row.get("win_rate") or 0.0)
    filtered_invalid = float(filtered_row.get("invalid_action_rate") or 0.0)
    imagination_stats = _read_json(Path(str(imagination_summary.get("imagined_stats_json") or "")))
    acceptance_rate = float(imagination_stats.get("acceptance_rate") or 0.0)

    summary_payload = {
        "schema": "p46_imagination_pipeline_summary_v1",
        "generated_at": now_iso(),
        "run_id": chosen_run_id,
        "status": "ok",
        "config_path": str(cfg_path),
        "run_dir": str(run_dir),
        "world_model_checkpoint": checkpoint_path,
        "world_model_bootstrap_summary": bootstrap_summary,
        "imagination_summary": imagination_summary,
        "recipe_runs": recipe_runs,
        "arena_compare_summary": arena_compare_summary,
        "triage_summary": triage_summary,
        "metrics": {
            "score": filtered_score if filtered_score > 0.0 else real_score,
            "avg_reward": filtered_score if filtered_score > 0.0 else real_score,
            "win_rate": filtered_win,
            "illegal_action_rate": filtered_invalid,
            "p46_real_only_score": real_score,
            "p46_filtered_score": filtered_score,
            "p46_filtered_delta_vs_real_only": filtered_score - real_score,
            "p46_imagined_acceptance_rate": acceptance_rate,
            "p46_imagined_sample_count": int(imagination_summary.get("total_imagined_samples") or 0),
        },
    }
    write_json(run_dir / "pipeline_summary.json", summary_payload)
    write_markdown(
        run_dir / "pipeline_summary.md",
        [
            f"# P46 Imagination Pipeline ({chosen_run_id})",
            "",
            f"- world_model_checkpoint: `{checkpoint_path}`",
            f"- imagined_samples: {int(imagination_summary.get('total_imagined_samples') or 0)}",
            f"- acceptance_rate: {acceptance_rate:.4f}",
            f"- real_only_score: {real_score:.6f}",
            f"- filtered_score: {filtered_score:.6f}",
            f"- filtered_delta_vs_real_only: {filtered_score - real_score:.6f}",
            f"- arena_compare_summary_json: `{arena_compare_summary.get('summary_table_json')}`",
            f"- triage_report_json: `{triage_summary.get('triage_report_json')}`",
        ],
    )
    filtered_checkpoint = _recipe_checkpoint(recipe_runs.get("real_plus_imagined_filtered") or {})
    return {
        "status": "ok",
        "run_id": chosen_run_id,
        "run_dir": str(run_dir),
        "pipeline_summary_json": str(run_dir / "pipeline_summary.json"),
        "pipeline_summary_md": str(run_dir / "pipeline_summary.md"),
        "seeds_used_json": str(run_dir / "seeds_used.json"),
        "arena_compare_summary_json": str(arena_compare_summary.get("summary_table_json") or ""),
        "promotion_decision_json": str(arena_compare_summary.get("promotion_decision_json") or ""),
        "triage_report_json": str(triage_summary.get("triage_report_json") or ""),
        "best_checkpoint": filtered_checkpoint,
        "metrics": summary_payload["metrics"],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the P46 imagination augmentation pipeline.")
    parser.add_argument("--config", default="configs/experiments/p46_imagination_smoke.yaml")
    parser.add_argument("--out-dir", default="")
    parser.add_argument("--run-id", default="")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--seeds", default="")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    seeds = [token.strip() for token in str(args.seeds or "").split(",") if token.strip()]
    summary = run_imagination_pipeline(
        config_path=args.config,
        out_dir=(args.out_dir if str(args.out_dir).strip() else None),
        run_id=str(args.run_id or ""),
        quick=bool(args.quick),
        dry_run=bool(args.dry_run),
        seeds_override=(seeds or None),
    )
    print(json.dumps(summary, ensure_ascii=False))
    return 0 if str(summary.get("status")) in {"ok", "stub"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
