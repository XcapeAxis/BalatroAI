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

from trainer.closed_loop.regression_triage import run_regression_triage
from trainer.closed_loop.replay_manifest import build_seeds_payload, now_iso, now_stamp, to_abs_path, write_json, write_markdown
from trainer.world_model.lookahead_planner import run_planner_smoke


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


def _resolve_world_model_checkpoint(repo_root: Path, cfg: dict[str, Any]) -> str:
    wm_cfg = cfg.get("world_model") if isinstance(cfg.get("world_model"), dict) else {}
    raw = str(wm_cfg.get("checkpoint") or "").strip()
    if raw:
        checkpoint = to_abs_path(repo_root, raw)
        if checkpoint.exists():
            return str(checkpoint)
    candidates = sorted((repo_root / "docs/artifacts/p45/wm_train").glob("**/best.pt"), key=lambda path: str(path))
    if candidates:
        return str(candidates[-1].resolve())
    raise FileNotFoundError("P47 requires a P45 world model checkpoint")


def _resolve_world_model_eval_ref(repo_root: Path, cfg: dict[str, Any]) -> str:
    wm_cfg = cfg.get("world_model") if isinstance(cfg.get("world_model"), dict) else {}
    raw = str(wm_cfg.get("eval_ref") or "").strip()
    if raw:
        path = to_abs_path(repo_root, raw)
        if path.exists():
            return str(path)
    candidates = sorted((repo_root / "docs/artifacts/p45/wm_eval").glob("**/eval_metrics.json"), key=lambda path: str(path))
    return str(candidates[-1].resolve()) if candidates else ""


def _resolve_seeds(cfg: dict[str, Any], seeds_override: list[str] | None, *, quick: bool) -> list[str]:
    if seeds_override:
        seeds = [str(seed).strip() for seed in seeds_override if str(seed).strip()]
    else:
        raw = cfg.get("seeds")
        seeds = [str(seed).strip() for seed in raw if str(seed).strip()] if isinstance(raw, list) else ["AAAAAAA", "BBBBBBB"]
    return seeds[:2] if quick and len(seeds) > 2 else seeds


def _pick_summary_row(summary_path: Path, policy_id: str) -> dict[str, Any]:
    rows = _read_json_list(summary_path)
    for row in rows:
        if str(row.get("policy_id") or "") == str(policy_id):
            return row
    return {}


def _copy_json(src_path: str | Path, dst_path: Path) -> str:
    src = Path(str(src_path))
    if not src.exists():
        return ""
    payload = _read_json(src)
    write_json(dst_path, payload)
    return str(dst_path)


def _adapter_compare_markdown(planner_summary: dict[str, Any], variants: list[dict[str, Any]]) -> list[str]:
    lines = [
        "# P47 Adapter Compare",
        "",
        f"- candidate_source: `{planner_summary.get('candidate_source')}`",
        f"- world_model_checkpoint: `{planner_summary.get('world_model_checkpoint')}`",
        f"- top_action_before: `{planner_summary.get('top_action_before')}`",
        f"- top_action_after: `{planner_summary.get('top_action_after')}`",
        "",
        "## Variants",
    ]
    for variant in variants:
        lines.append(
            "- {id}: base={base} source={source} top_k={top_k} horizon={horizon} unc_penalty={unc:.3f}".format(
                id=variant.get("id"),
                base=variant.get("base_policy"),
                source=variant.get("candidate_source"),
                top_k=_safe_int(variant.get("top_k"), 0),
                horizon=_safe_int(variant.get("horizon"), 0),
                unc=_safe_float(variant.get("uncertainty_penalty"), 0.0),
            )
        )
    return lines


def _write_adapter_compare(repo_root: Path, planner_summary: dict[str, Any], variants: list[dict[str, Any]]) -> str:
    path = (repo_root / "docs/artifacts/p47" / f"adapter_compare_{now_stamp()}.md").resolve()
    write_markdown(path, _adapter_compare_markdown(planner_summary, variants))
    return str(path)


def run_model_based_search_pipeline(
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
    planner_cfg = cfg.get("planner") if isinstance(cfg.get("planner"), dict) else {}
    arena_cfg = cfg.get("arena_compare") if isinstance(cfg.get("arena_compare"), dict) else {}
    run_name = str(run_id or output_cfg.get("run_id") or now_stamp())
    run_root = (
        (repo_root / str(output_cfg.get("artifacts_root") or "docs/artifacts/p47/arena_ablation")).resolve()
        if out_dir is None
        else (Path(out_dir) if Path(out_dir).is_absolute() else (repo_root / out_dir).resolve())
    )
    run_dir = run_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    seeds = _resolve_seeds(cfg, seeds_override, quick=quick)
    write_json(run_dir / "seeds_used.json", build_seeds_payload(seeds, seed_policy_version="p47.model_based_search"))
    wm_checkpoint = _resolve_world_model_checkpoint(repo_root, cfg)
    wm_eval_ref = _resolve_world_model_eval_ref(repo_root, cfg)

    planning_source = str(((cfg.get("candidate_actions") or {}).get("source")) if isinstance(cfg.get("candidate_actions"), dict) else "" or "heuristic_candidates")
    planner_summary = run_planner_smoke(
        config_path=cfg_path,
        out_dir=(repo_root / "docs/artifacts/p47/lookahead").resolve(),
        run_id=run_name,
        quick=quick,
        checkpoint_override=wm_checkpoint,
        source_override=planning_source,
        model_path_override=str(((cfg.get("candidate_actions") or {}).get("model_path")) if isinstance(cfg.get("candidate_actions"), dict) else "" or ""),
    )

    variants = arena_cfg.get("variants") if isinstance(arena_cfg.get("variants"), list) else []
    normalized_variants: list[dict[str, Any]] = []
    for variant in variants:
        if not isinstance(variant, dict):
            continue
        row = dict(variant)
        row["id"] = str(row.get("id") or "")
        row["base_policy"] = str(row.get("base_policy") or "heuristic_baseline")
        row["candidate_source"] = str(row.get("candidate_source") or row.get("base_policy") or "heuristic_candidates")
        row["world_model_checkpoint"] = wm_checkpoint
        row["assist_mode"] = "rerank"
        row["top_k"] = _safe_int(row.get("top_k"), _safe_int(((cfg.get("candidate_actions") or {}).get("top_k")), 4))
        row["horizon"] = _safe_int(row.get("horizon"), _safe_int(planner_cfg.get("horizon"), 1))
        row["uncertainty_penalty"] = _safe_float(row.get("uncertainty_penalty"), _safe_float(planner_cfg.get("uncertainty_penalty"), 0.5))
        row["gamma"] = _safe_float(row.get("gamma"), _safe_float(planner_cfg.get("gamma"), 0.95))
        row["reward_weight"] = _safe_float(row.get("reward_weight"), _safe_float(planner_cfg.get("reward_weight"), 1.0))
        row["score_weight"] = _safe_float(row.get("score_weight"), _safe_float(planner_cfg.get("score_weight"), 0.5))
        row["value_weight"] = _safe_float(row.get("value_weight"), _safe_float(planner_cfg.get("value_weight"), 0.15))
        row["terminal_bonus"] = _safe_float(row.get("terminal_bonus"), _safe_float(planner_cfg.get("terminal_bonus"), 0.0))
        row["search_max_branch"] = _safe_int(row.get("search_max_branch"), _safe_int(((cfg.get("candidate_actions") or {}).get("search_max_branch")), 80))
        row["search_max_depth"] = _safe_int(row.get("search_max_depth"), _safe_int(((cfg.get("candidate_actions") or {}).get("search_max_depth")), 2))
        row["search_time_budget_ms"] = _safe_float(row.get("search_time_budget_ms"), _safe_float(((cfg.get("candidate_actions") or {}).get("search_time_budget_ms")), 15.0))
        if row["id"]:
            normalized_variants.append(row)
    if not normalized_variants:
        normalized_variants = [
            {
                "id": "heuristic_wm_rerank_h1",
                "base_policy": "heuristic_baseline",
                "candidate_source": "heuristic_candidates",
                "world_model_checkpoint": wm_checkpoint,
                "assist_mode": "rerank",
                "top_k": 4,
                "horizon": 1,
                "uncertainty_penalty": 0.5,
                "gamma": 0.95,
                "reward_weight": 1.0,
                "score_weight": 0.5,
                "value_weight": 0.15,
                "terminal_bonus": 0.0,
                "search_max_branch": 80,
                "search_max_depth": 2,
                "search_time_budget_ms": 15.0,
            }
        ]

    adapter_compare_md = _write_adapter_compare(repo_root, planner_summary, normalized_variants)
    write_json(run_dir / "policy_assist_map.json", {row["id"]: row for row in normalized_variants})
    model_paths = arena_cfg.get("policy_model_map") if isinstance(arena_cfg.get("policy_model_map"), dict) else {}
    write_json(run_dir / "policy_model_map.json", model_paths)

    baseline_policy = str(arena_cfg.get("baseline_policy") or "heuristic_baseline")
    policies = [baseline_policy] + [str(row["id"]) for row in normalized_variants]
    if bool(arena_cfg.get("include_search_baseline", False)) and "search_expert" not in policies:
        policies.insert(1, "search_expert")

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
        ",".join(policies),
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
    arena_result = _run_process(arena_cmd, cwd=repo_root, timeout_sec=_safe_int(arena_cfg.get("timeout_sec"), 3600))

    summary_json = run_dir / "summary_table.json"
    bucket_json = run_dir / "bucket_metrics.json"
    episode_jsonl = run_dir / "episode_records.jsonl"
    warnings_log = run_dir / "warnings.log"
    run_manifest_path = run_dir / "run_manifest.json"

    arena_manifest = _read_json(run_manifest_path)
    champion_policy = str(arena_cfg.get("champion_policy") or baseline_policy)
    candidate_policy = str(arena_cfg.get("candidate_policy") or normalized_variants[0]["id"])
    champion_result = {"returncode": 0, "stdout": "", "stderr": "", "command": []}
    promotion_payload: dict[str, Any] = {}
    if summary_json.exists() and bool(arena_cfg.get("enable_champion_rules", True)):
        champion_out = run_dir / "champion_eval"
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
            candidate_policy,
            "--champion-policy",
            champion_policy,
            "--episode-records-jsonl",
            str(episode_jsonl),
            "--bucket-metrics-json",
            str(bucket_json),
            "--champion-json",
            str(arena_cfg.get("champion_json") or "docs/artifacts/p22/champion.json"),
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
                promotion_payload = _read_json(decision_path)
                break
        if not promotion_payload:
            promotion_payload = {
                "schema": "p47_promotion_decision_v1",
                "generated_at": now_iso(),
                "decision": "observe",
                "recommend_promotion": False,
                "candidate_policy_id": candidate_policy,
                "champion_policy_id": champion_policy,
                "reasons": ["champion_rules_unavailable"],
            }
        write_json(run_dir / "promotion_decision.json", promotion_payload)
        write_markdown(
            run_dir / "promotion_decision.md",
            [
                "# P47 Promotion Decision",
                "",
                f"- decision: `{promotion_payload.get('decision')}`",
                f"- recommend_promotion: `{promotion_payload.get('recommend_promotion')}`",
                f"- candidate_policy_id: `{promotion_payload.get('candidate_policy_id')}`",
                f"- champion_policy_id: `{promotion_payload.get('champion_policy_id')}`",
            ],
        )

    slice_eval_ref = str(promotion_payload.get("slice_decision_breakdown_json") or "")
    slice_eval_json = _copy_json(slice_eval_ref, run_dir / "slice_eval.json") if slice_eval_ref else ""
    write_json(
        run_dir / "run_manifest.json",
        {
            **arena_manifest,
            "schema": "p47_model_based_search_manifest_v1",
            "generated_at": now_iso(),
            "run_id": run_name,
            "run_dir": str(run_dir),
            "config_path": str(cfg_path),
            "world_model_rerank": {
                "wm_checkpoint": wm_checkpoint,
                "wm_eval_ref": wm_eval_ref,
                "wm_assist_enabled": True,
                "assist_mode": "rerank",
                "baseline_policy": baseline_policy,
                "candidate_policy": candidate_policy,
                "champion_policy": champion_policy,
                "policy_assist_map_json": str((run_dir / "policy_assist_map.json").resolve()),
                "policy_model_map_json": str((run_dir / "policy_model_map.json").resolve()),
                "planner_summary_json": str(Path(str(planner_summary.get("planner_stats_json") or "")).resolve()),
                "adapter_compare_md": adapter_compare_md,
                "slice_eval_json": slice_eval_json,
            },
            "paths": {
                "run_dir": str(run_dir),
                "summary_table_json": str(summary_json),
                "bucket_metrics_json": str(bucket_json),
                "episode_records_jsonl": str(episode_jsonl),
                "warnings_log": str(warnings_log),
                "promotion_decision_json": str(run_dir / "promotion_decision.json"),
                "slice_eval_json": slice_eval_json,
                "planner_summary_json": str(Path(str(planner_summary.get("planner_stats_json") or "")).resolve()),
            },
        },
    )

    triage_root = (repo_root / str((cfg.get("triage") or {}).get("output_artifacts_root") if isinstance(cfg.get("triage"), dict) else "docs/artifacts/p47/triage")).resolve()
    triage_dir = triage_root / run_name
    triage_dir.mkdir(parents=True, exist_ok=True)
    triage_summary = run_regression_triage(current_run_dir=run_dir, out_dir=triage_dir)

    summary_rows = _read_json_list(summary_json)
    baseline_row = _pick_summary_row(summary_json, baseline_policy) if summary_json.exists() else {}
    candidate_row = _pick_summary_row(summary_json, candidate_policy) if summary_json.exists() else {}
    best_variant = max(
        [row for row in summary_rows if str(row.get("policy_id") or "") != baseline_policy],
        key=lambda row: float(row.get("mean_total_score") or 0.0),
        default={},
    )
    baseline_score = _safe_float(baseline_row.get("mean_total_score"), 0.0)
    candidate_score = _safe_float(candidate_row.get("mean_total_score"), 0.0)
    best_score = _safe_float(best_variant.get("mean_total_score"), candidate_score)

    summary_payload = {
        "schema": "p47_model_based_search_summary_v1",
        "generated_at": now_iso(),
        "run_id": run_name,
        "status": "ok" if int(arena_result.get("returncode") or 0) == 0 and summary_json.exists() else "failed",
        "config_path": str(cfg_path),
        "run_dir": str(run_dir),
        "world_model_checkpoint": wm_checkpoint,
        "world_model_eval_ref": wm_eval_ref,
        "planner_summary": planner_summary,
        "arena_returncode": int(arena_result.get("returncode") or 0),
        "champion_rules_returncode": int(champion_result.get("returncode") or 0),
        "summary_table_json": str(summary_json),
        "slice_eval_json": slice_eval_json,
        "promotion_decision_json": str(run_dir / "promotion_decision.json"),
        "triage_report_json": str(triage_summary.get("triage_report_json") or ""),
        "policy_variants": normalized_variants,
        "metrics": {
            "score": candidate_score if candidate_score > 0.0 else best_score,
            "avg_reward": candidate_score if candidate_score > 0.0 else best_score,
            "win_rate": _safe_float(candidate_row.get("win_rate"), 0.0),
            "illegal_action_rate": _safe_float(candidate_row.get("invalid_action_rate"), 0.0),
            "p47_baseline_score": baseline_score,
            "p47_candidate_score": candidate_score,
            "p47_best_variant_score": best_score,
            "p47_candidate_delta_vs_baseline": candidate_score - baseline_score,
            "p47_best_variant_delta_vs_baseline": best_score - baseline_score,
        },
    }
    write_json(run_dir / "pipeline_summary.json", summary_payload)
    write_markdown(
        run_dir / "pipeline_summary.md",
        [
            f"# P47 Model-Based Search ({run_name})",
            "",
            f"- world_model_checkpoint: `{wm_checkpoint}`",
            f"- baseline_policy: `{baseline_policy}`",
            f"- candidate_policy: `{candidate_policy}`",
            f"- baseline_score: {baseline_score:.6f}",
            f"- candidate_score: {candidate_score:.6f}",
            f"- candidate_delta_vs_baseline: {candidate_score - baseline_score:.6f}",
            f"- best_variant: `{best_variant.get('policy_id')}`",
            f"- best_variant_score: {best_score:.6f}",
            f"- summary_table_json: `{summary_json}`",
            f"- triage_report_json: `{triage_summary.get('triage_report_json')}`",
            f"- adapter_compare_md: `{adapter_compare_md}`",
        ],
    )
    return {
        "status": str(summary_payload.get("status") or "failed"),
        "run_id": run_name,
        "run_dir": str(run_dir),
        "pipeline_summary_json": str(run_dir / "pipeline_summary.json"),
        "pipeline_summary_md": str(run_dir / "pipeline_summary.md"),
        "planner_summary_json": str(planner_summary.get("planner_stats_json") or ""),
        "summary_table_json": str(summary_json),
        "promotion_decision_json": str(run_dir / "promotion_decision.json"),
        "triage_report_json": str(triage_summary.get("triage_report_json") or ""),
        "adapter_compare_md": adapter_compare_md,
        "slice_eval_json": slice_eval_json,
        "seeds_used_json": str(run_dir / "seeds_used.json"),
        "metrics": summary_payload["metrics"],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the P47 uncertainty-aware model-based search pipeline.")
    parser.add_argument("--config", default="configs/experiments/p47_wm_search_smoke.yaml")
    parser.add_argument("--out-dir", default="")
    parser.add_argument("--run-id", default="")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--seeds", default="")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    seeds = [token.strip() for token in str(args.seeds or "").split(",") if token.strip()]
    summary = run_model_based_search_pipeline(
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
