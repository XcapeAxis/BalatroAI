from __future__ import annotations

if __package__ is None or __package__ == "":
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

import argparse
import hashlib
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

try:  # pragma: no cover - optional dependency
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None

from trainer.closed_loop.replay_manifest import (
    append_jsonl,
    build_seeds_payload,
    now_iso,
    now_stamp,
    read_json,
    to_abs_path,
    write_json,
)
from trainer.closed_loop.curriculum_scheduler import build_curriculum_plan, build_phase_allocations
from trainer.experiments.training_modes import (
    MODE_CATEGORY_EXPERIMENTAL,
    MODE_CATEGORY_LEGACY_BASELINE,
    mode_category,
)
from trainer.rl.ppo_lite import run_ppo_lite_training


def _read_yaml_or_json(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8-sig")
    if path.suffix.lower() in {".yml", ".yaml"}:
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


def _seed_to_int(seed: str) -> int:
    digest = hashlib.sha256(seed.encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def _run_process(command: list[str], *, cwd: Path, timeout_sec: int) -> dict[str, Any]:
    start = time.time()
    try:
        proc = subprocess.run(
            command,
            cwd=str(cwd),
            text=True,
            capture_output=True,
            timeout=max(1, timeout_sec),
        )
        return {
            "returncode": int(proc.returncode),
            "stdout": str(proc.stdout or ""),
            "stderr": str(proc.stderr or ""),
            "elapsed_sec": time.time() - start,
            "timed_out": False,
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "returncode": 124,
            "stdout": str(exc.stdout or ""),
            "stderr": str(exc.stderr or ""),
            "elapsed_sec": time.time() - start,
            "timed_out": True,
        }


def _pick_python_exe(repo_root: Path) -> str:
    venv_py = repo_root / ".venv_trainer" / "Scripts" / "python.exe"
    if venv_py.exists():
        return str(venv_py)
    return str(sys.executable)


def _pick_replay_manifest_path(repo_root: Path, cfg: dict[str, Any]) -> Path:
    replay_raw = str(cfg.get("replay_mix_manifest") or "").strip()
    if replay_raw:
        return to_abs_path(repo_root, replay_raw)
    latest_root = to_abs_path(repo_root, "docs/artifacts/p40/replay_mixer")
    if latest_root.exists():
        runs = sorted([p for p in latest_root.iterdir() if p.is_dir()], key=lambda p: p.name)
        if runs:
            manifest = runs[-1] / "replay_mix_manifest.json"
            if manifest.exists():
                return manifest
    raise FileNotFoundError("replay mix manifest not provided and no latest manifest found")


def _normalize_mode_ids(values: Any) -> list[str]:
    if isinstance(values, str):
        token = values.strip().lower()
        return [token] if token else []
    if not isinstance(values, list):
        return []
    out: list[str] = []
    for value in values:
        token = str(value).strip().lower()
        if token:
            out.append(token)
    return out


def _resolve_requested_modes(cfg: dict[str, Any]) -> list[str]:
    explicit_modes = _normalize_mode_ids(cfg.get("candidate_modes"))
    if explicit_modes:
        return explicit_modes
    mode = str(cfg.get("mode") or "").strip().lower()
    if mode:
        return [mode]
    # P43 default mainline order.
    return ["rl_ppo_lite", "selfsup_warm_bc"]


def _resolve_training_mode(cfg: dict[str, Any]) -> dict[str, Any]:
    requested_modes = _resolve_requested_modes(cfg)
    allow_legacy_fallback = bool(cfg.get("allow_legacy_fallback") or cfg.get("fallback_to_legacy") or False)
    legacy_fallback_modes = _normalize_mode_ids(cfg.get("legacy_fallback_modes")) or ["bc_finetune"]

    known_modes = {"rl_ppo_lite", "selfsup_warm_bc", "bc_finetune", "dagger_refresh"}
    selected_mode = ""
    unsupported_modes: list[str] = []
    for mode in requested_modes:
        if mode in known_modes:
            selected_mode = mode
            break
        unsupported_modes.append(mode)

    fallback_used = False
    fallback_reason = ""
    if not selected_mode and allow_legacy_fallback:
        for mode in legacy_fallback_modes:
            if mode in known_modes:
                selected_mode = mode
                fallback_used = True
                fallback_reason = (
                    "mainline_modes_unavailable:" + ",".join(requested_modes)
                    if requested_modes
                    else "mainline_modes_unavailable"
                )
                break

    if not selected_mode:
        selected_mode = requested_modes[0] if requested_modes else "rl_ppo_lite"
        if unsupported_modes:
            fallback_reason = "unsupported_requested_modes:" + ",".join(unsupported_modes)

    selected_category = mode_category(selected_mode, default=MODE_CATEGORY_EXPERIMENTAL)
    legacy_paths_used: list[str] = []
    if selected_category == MODE_CATEGORY_LEGACY_BASELINE:
        legacy_paths_used = ["trainer/train_bc.py", "trainer/dagger_collect.py", "trainer/dagger_collect_v4.py"]

    return {
        "requested_modes": requested_modes,
        "selected_mode": selected_mode,
        "selected_category": selected_category,
        "allow_legacy_fallback": allow_legacy_fallback,
        "legacy_fallback_modes": legacy_fallback_modes,
        "fallback_used": fallback_used,
        "fallback_reason": fallback_reason,
        "legacy_paths_used": legacy_paths_used,
    }


def _pick_bc_entries(
    replay_manifest: dict[str, Any],
    *,
    prefer_source_types: list[str],
) -> list[dict[str, Any]]:
    selected = replay_manifest.get("selected_entries")
    if not isinstance(selected, list):
        return []
    rows = [row for row in selected if isinstance(row, dict)]
    if not rows:
        return []

    bc_rows = [r for r in rows if str(r.get("format_hint") or "").strip().lower() == "bc_record_v1"]
    if bc_rows:
        return bc_rows

    for source_type in prefer_source_types:
        subset = [r for r in rows if str(r.get("source_type") or "").strip().lower() == source_type.lower()]
        if subset:
            return subset
    return rows


def _entry_source_type(entry: dict[str, Any]) -> str:
    return str(entry.get("source_type") or "").strip().lower()


def _is_imagined_entry(entry: dict[str, Any]) -> bool:
    return _entry_source_type(entry) == "imagined_world_model"


def _normalize_imagination_config(cfg: dict[str, Any], replay_manifest: dict[str, Any]) -> dict[str, Any]:
    raw = cfg.get("imagination") if isinstance(cfg.get("imagination"), dict) else {}
    recipe = str(raw.get("recipe") or cfg.get("training_recipe") or "real_only").strip().lower() or "real_only"
    filter_mode = str(raw.get("filter_mode") or "").strip().lower()
    if not filter_mode:
        if recipe == "real_only":
            filter_mode = "disabled"
        elif recipe.endswith("filtered"):
            filter_mode = "uncertainty_gate"
        else:
            filter_mode = "none"
    require_gate = bool(raw.get("require_uncertainty_gate_passed", filter_mode in {"uncertainty_gate", "filtered"}))
    max_horizon = max(1, int(raw.get("max_imagination_horizon") or 1))
    manifest_sources = replay_manifest.get("sources") if isinstance(replay_manifest.get("sources"), list) else []
    imagined_source_rows = [
        row
        for row in manifest_sources
        if isinstance(row, dict) and str(row.get("source_type") or "").strip().lower() == "imagined_world_model"
    ]
    imagined_selected = sum(int(row.get("selected_samples") or 0) for row in imagined_source_rows)
    total_selected = int(((replay_manifest.get("totals") or {}).get("selected_samples") if isinstance(replay_manifest.get("totals"), dict) else 0) or 0)
    inferred_fraction = float(imagined_selected) / max(1, total_selected)
    requested_fraction = float(raw.get("imagined_fraction") or raw.get("max_imagined_fraction") or inferred_fraction)
    return {
        "recipe": recipe,
        "enabled": recipe != "real_only",
        "filter_mode": filter_mode,
        "require_uncertainty_gate_passed": require_gate,
        "max_imagination_horizon": max_horizon,
        "imagined_fraction": float(requested_fraction),
        "imagined_selected_samples": int(imagined_selected),
        "source_count": len(imagined_source_rows),
    }


def _is_bc_labeled_row(obj: dict[str, Any]) -> bool:
    phase = str(obj.get("phase") or "")
    has_hand_label = obj.get("expert_action_id") is not None and phase == "SELECTING_HAND"
    has_shop_label = obj.get("shop_expert_action_id") is not None and phase in {"SHOP", "SMODS_BOOSTER_OPENED"}
    return bool(has_hand_label or has_shop_label)


def _row_allowed_for_recipe(obj: dict[str, Any], *, entry: dict[str, Any], imagination_cfg: dict[str, Any]) -> tuple[bool, str]:
    if not _is_bc_labeled_row(obj):
        return False, "not_bc_labeled"
    if not _is_imagined_entry(entry):
        return True, "real"
    recipe = str(imagination_cfg.get("recipe") or "real_only")
    if recipe == "real_only":
        return False, "imagined_disabled"
    if not bool(obj.get("valid_for_training", True)):
        return False, "imagined_invalid_for_training"
    if int(obj.get("imagined_step_idx") or 1) > int(imagination_cfg.get("max_imagination_horizon") or 1):
        return False, "imagined_horizon_filtered"
    filter_mode = str(imagination_cfg.get("filter_mode") or "none")
    if filter_mode in {"uncertainty_gate", "filtered"} and not bool(obj.get("uncertainty_gate_passed", False)):
        return False, "imagined_uncertainty_gate"
    return True, "imagined"


def _extract_replay_line_limit(entry: dict[str, Any]) -> int:
    span = entry.get("sample_span") if isinstance(entry.get("sample_span"), dict) else {}
    line_end = int(span.get("line_end") or entry.get("sample_count") or 0)
    return max(0, line_end)


def _build_subset_dataset(
    *,
    entries: list[dict[str, Any]],
    out_path: Path,
    max_rows: int,
    imagination_cfg: dict[str, Any],
) -> dict[str, Any]:
    kept = 0
    scanned = 0
    source_files: list[str] = []
    source_breakdown: dict[str, int] = {}
    imagined_filtered_breakdown: dict[str, int] = {}
    uncertainty_values: list[float] = []
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="\n") as out_fp:
        for entry in entries:
            src_raw = str(entry.get("path") or "").strip()
            if not src_raw:
                continue
            src = Path(src_raw)
            if not src.exists():
                continue
            source_files.append(str(src))
            local_cap = _extract_replay_line_limit(entry)
            with src.open("r", encoding="utf-8", errors="replace") as in_fp:
                for line_idx, line in enumerate(in_fp, start=1):
                    if local_cap > 0 and line_idx > local_cap:
                        break
                    text = line.strip()
                    if not text:
                        continue
                    scanned += 1
                    try:
                        obj = json.loads(text)
                    except Exception:
                        continue
                    if not isinstance(obj, dict):
                        continue
                    allowed, reason = _row_allowed_for_recipe(obj, entry=entry, imagination_cfg=imagination_cfg)
                    if not allowed:
                        imagined_filtered_breakdown[reason] = int(imagined_filtered_breakdown.get(reason, 0)) + 1
                        continue
                    out_fp.write(json.dumps(obj, ensure_ascii=False) + "\n")
                    kept += 1
                    source_key = _entry_source_type(entry) or "unknown"
                    source_breakdown[source_key] = int(source_breakdown.get(source_key, 0)) + 1
                    if source_key == "imagined_world_model":
                        uncertainty_values.append(float(obj.get("uncertainty_score") or 0.0))
                    if max_rows > 0 and kept >= max_rows:
                        return {
                            "kept_rows": kept,
                            "scanned_rows": scanned,
                            "source_files": source_files,
                            "source_breakdown": source_breakdown,
                            "imagined_filtered_breakdown": imagined_filtered_breakdown,
                            "imagined_mean_uncertainty": (sum(uncertainty_values) / max(1, len(uncertainty_values))),
                        }
    return {
        "kept_rows": kept,
        "scanned_rows": scanned,
        "source_files": source_files,
        "source_breakdown": source_breakdown,
        "imagined_filtered_breakdown": imagined_filtered_breakdown,
        "imagined_mean_uncertainty": (sum(uncertainty_values) / max(1, len(uncertainty_values))),
    }


def _build_subset_dataset_from_allocations(
    *,
    allocations: list[dict[str, Any]],
    out_path: Path,
    max_rows: int,
    imagination_cfg: dict[str, Any],
) -> dict[str, Any]:
    kept = 0
    scanned = 0
    source_files: list[str] = []
    source_breakdown: dict[str, int] = {}
    imagined_filtered_breakdown: dict[str, int] = {}
    uncertainty_values: list[float] = []
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="\n") as out_fp:
        for alloc in allocations:
            src_raw = str(alloc.get("path") or "").strip()
            if not src_raw:
                continue
            src = Path(src_raw)
            if not src.exists():
                continue
            source_files.append(str(src))
            take_rows = max(0, int(alloc.get("take_rows") or 0))
            if take_rows <= 0:
                continue
            written_for_entry = 0
            with src.open("r", encoding="utf-8", errors="replace") as in_fp:
                for line in in_fp:
                    text = line.strip()
                    if not text:
                        continue
                    scanned += 1
                    try:
                        obj = json.loads(text)
                    except Exception:
                        continue
                    if not isinstance(obj, dict):
                        continue
                    entry_like = {"source_type": str(alloc.get("source_type") or "")}
                    allowed, reason = _row_allowed_for_recipe(obj, entry=entry_like, imagination_cfg=imagination_cfg)
                    if not allowed:
                        imagined_filtered_breakdown[reason] = int(imagined_filtered_breakdown.get(reason, 0)) + 1
                        continue
                    out_fp.write(json.dumps(obj, ensure_ascii=False) + "\n")
                    kept += 1
                    written_for_entry += 1
                    source_key = _entry_source_type(entry_like) or "unknown"
                    source_breakdown[source_key] = int(source_breakdown.get(source_key, 0)) + 1
                    if source_key == "imagined_world_model":
                        uncertainty_values.append(float(obj.get("uncertainty_score") or 0.0))
                    if written_for_entry >= take_rows:
                        break
                    if max_rows > 0 and kept >= max_rows:
                        return {
                            "kept_rows": kept,
                            "scanned_rows": scanned,
                            "source_files": source_files,
                            "source_breakdown": source_breakdown,
                            "imagined_filtered_breakdown": imagined_filtered_breakdown,
                            "imagined_mean_uncertainty": (sum(uncertainty_values) / max(1, len(uncertainty_values))),
                        }
    return {
        "kept_rows": kept,
        "scanned_rows": scanned,
        "source_files": source_files,
        "source_breakdown": source_breakdown,
        "imagined_filtered_breakdown": imagined_filtered_breakdown,
        "imagined_mean_uncertainty": (sum(uncertainty_values) / max(1, len(uncertainty_values))),
    }


def _read_train_metrics(path: Path) -> dict[str, Any]:
    payload = read_json(path)
    if not isinstance(payload, dict):
        return {}
    history = payload.get("history") if isinstance(payload.get("history"), list) else []
    final_row = history[-1] if history and isinstance(history[-1], dict) else {}
    val_hand = final_row.get("val_hand") if isinstance(final_row.get("val_hand"), dict) else {}
    val_shop = final_row.get("val_shop") if isinstance(final_row.get("val_shop"), dict) else {}
    return {
        "val_hand_acc1": float(val_hand.get("acc1") or 0.0),
        "val_hand_acc3": float(val_hand.get("acc3") or 0.0),
        "val_hand_loss": float(val_hand.get("loss") or 0.0),
        "val_shop_acc1": float(val_shop.get("acc1") or 0.0),
        "val_shop_illegal_rate": float(val_shop.get("illegal_rate") or 0.0),
        "val_shop_loss": float(val_shop.get("loss") or 0.0),
    }


def run_candidate_training(
    *,
    config_path: str | Path,
    out_dir: str | Path | None = None,
    run_id: str = "",
    quick: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    repo_root = Path(__file__).resolve().parents[2]
    cfg_path = to_abs_path(repo_root, config_path)
    cfg = _read_yaml_or_json(cfg_path)

    output_cfg = cfg.get("output") if isinstance(cfg.get("output"), dict) else {}
    artifacts_root = str(output_cfg.get("artifacts_root") or "docs/artifacts/p40/candidate_train")
    chosen_run_id = str(run_id or output_cfg.get("run_id") or now_stamp())
    if out_dir:
        run_dir = to_abs_path(repo_root, out_dir)
    else:
        run_dir = to_abs_path(repo_root, artifacts_root) / chosen_run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    mode_resolution = _resolve_training_mode(cfg)
    mode = str(mode_resolution.get("selected_mode") or "rl_ppo_lite").strip().lower()
    training_mode_category = str(mode_resolution.get("selected_category") or MODE_CATEGORY_EXPERIMENTAL)
    fallback_used = bool(mode_resolution.get("fallback_used"))
    fallback_reason = str(mode_resolution.get("fallback_reason") or "")
    legacy_paths_used = [str(x) for x in (mode_resolution.get("legacy_paths_used") or [])]
    requested_modes = [str(x) for x in (mode_resolution.get("requested_modes") or [])]
    legacy_fallback_modes = [str(x) for x in (mode_resolution.get("legacy_fallback_modes") or [])]
    write_json(run_dir / "mode_resolution.json", mode_resolution)
    seeds_cfg = cfg.get("seeds")
    if isinstance(seeds_cfg, list):
        seeds = [str(s).strip() for s in seeds_cfg if str(s).strip()]
    else:
        seeds = ["AAAAAAA", "BBBBBBB"]
    if quick and len(seeds) > 2:
        seeds = seeds[:2]
    seeds_payload = build_seeds_payload(seeds, seed_policy_version="p40.candidate_train")
    write_json(run_dir / "seeds_used.json", seeds_payload)

    if mode == "rl_ppo_lite":
        rl_config_path = str(cfg.get("rl_config") or "").strip()
        rl_config_payload: dict[str, Any] | None = None
        if rl_config_path:
            resolved_rl = to_abs_path(repo_root, rl_config_path)
            if resolved_rl.exists():
                rl_config_payload = _read_yaml_or_json(resolved_rl)
        training_cfg = cfg.get("training") if isinstance(cfg.get("training"), dict) else {}
        timeout_sec = int(training_cfg.get("timeout_sec") or 3600)

        rl_summary = run_ppo_lite_training(
            config_path=(to_abs_path(repo_root, rl_config_path) if rl_config_path else None),
            config=rl_config_payload,
            out_dir=run_dir / "rl_train",
            run_id=f"{chosen_run_id}-rl",
            quick=bool(quick),
            dry_run=bool(dry_run),
            seeds_override=seeds,
        )
        metrics_payload = read_json(Path(str(rl_summary.get("metrics") or "")))
        if not isinstance(metrics_payload, dict):
            metrics_payload = {
                "schema": "p44_candidate_train_metrics_v1",
                "generated_at": now_iso(),
                "run_id": chosen_run_id,
                "status": str(rl_summary.get("status") or "stub"),
                "mode": mode,
                "training_mode": mode,
                "training_mode_category": training_mode_category,
                "seed_count": len(seeds),
                "ok_seed_count": 0,
                "mean_reward": 0.0,
                "mean_score": 0.0,
                "invalid_action_rate": 0.0,
                "policy_loss": 0.0,
                "value_loss": 0.0,
                "entropy": 0.0,
                "kl_divergence": 0.0,
                "candidate_checkpoint": str(rl_summary.get("best_checkpoint") or ""),
            }
        metrics_payload["training_mode"] = mode
        metrics_payload["training_mode_category"] = training_mode_category
        metrics_payload["fallback_used"] = fallback_used
        metrics_payload["fallback_reason"] = fallback_reason
        metrics_payload["legacy_paths_used"] = legacy_paths_used
        status = str(rl_summary.get("status") or "stub")
        best_checkpoint = Path(str(rl_summary.get("best_checkpoint") or "")) if str(rl_summary.get("best_checkpoint") or "").strip() else None
        best_checkpoint_txt = run_dir / "best_checkpoint.txt"
        best_checkpoint_txt.write_text((str(best_checkpoint) if best_checkpoint else "") + "\n", encoding="utf-8")
        loaded_curriculum = read_json(Path(str(rl_summary.get("curriculum_plan") or "")))
        curriculum_plan = loaded_curriculum if isinstance(loaded_curriculum, dict) else {
            "enabled": False,
            "phase_count": 0,
            "phases": [],
            "seeds": list(seeds),
            "reason": "curriculum_plan_missing",
        }
        write_json(run_dir / "curriculum_plan.json", curriculum_plan)
        curriculum_applied_path = Path(str(rl_summary.get("curriculum_applied") or run_dir / "curriculum_applied.jsonl"))
        if not curriculum_applied_path.exists():
            curriculum_applied_path.parent.mkdir(parents=True, exist_ok=True)
            curriculum_applied_path.write_text("", encoding="utf-8")

        manifest = {
            "schema": "p44_candidate_train_manifest_v1",
            "generated_at": now_iso(),
            "run_id": chosen_run_id,
            "status": status,
            "mode": mode,
            "training_mode": mode,
            "training_mode_category": training_mode_category,
            "candidate_modes": requested_modes,
            "fallback_used": fallback_used,
            "fallback_reason": fallback_reason,
            "legacy_fallback_modes": legacy_fallback_modes,
            "legacy_paths_used": legacy_paths_used,
            "config_path": str(cfg_path),
            "run_dir": str(run_dir),
            "quick": bool(quick),
            "dry_run": bool(dry_run),
            "replay_mix_manifest": str(cfg.get("replay_mix_manifest") or ""),
            "replay_selected_entries": 0,
            "curriculum_plan": curriculum_plan,
            "curriculum_config_path": "",
            "curriculum_applied": str(curriculum_applied_path),
            "rl_train_summary": rl_summary,
            "rl_config_path": str(to_abs_path(repo_root, rl_config_path)) if rl_config_path else "",
            "candidate_checkpoint": str(best_checkpoint) if best_checkpoint else "",
            "metrics_ref": str(run_dir / "metrics.json"),
            "timeout_sec": int(timeout_sec),
            "multi_seed_eval": str(rl_summary.get("eval_seed_results") or ""),
            "diagnostics_json": str(rl_summary.get("diagnostics_json") or ""),
            "diagnostics_report_md": str(rl_summary.get("diagnostics_report_md") or ""),
        }
        write_json(run_dir / "candidate_train_manifest.json", manifest)
        write_json(run_dir / "metrics.json", metrics_payload)
        return {
            "status": status,
            "run_id": chosen_run_id,
            "run_dir": str(run_dir),
            "candidate_train_manifest": str(run_dir / "candidate_train_manifest.json"),
            "metrics": str(run_dir / "metrics.json"),
            "best_checkpoint": str(best_checkpoint) if best_checkpoint else "",
            "training_mode": mode,
            "training_mode_category": training_mode_category,
            "fallback_used": fallback_used,
            "fallback_reason": fallback_reason,
            "legacy_paths_used": legacy_paths_used,
            "seeds_used": str(run_dir / "seeds_used.json"),
            "curriculum_plan": str(run_dir / "curriculum_plan.json"),
            "curriculum_applied": str(curriculum_applied_path),
            "reward_config": str((run_dir / "rl_train" / "reward_config.json")),
            "warnings_log": str((run_dir / "rl_train" / "warnings.log")),
            "multi_seed_eval": str(rl_summary.get("eval_seed_results") or ""),
            "diagnostics_json": str(rl_summary.get("diagnostics_json") or ""),
            "diagnostics_report_md": str(rl_summary.get("diagnostics_report_md") or ""),
        }

    if mode == "selfsup_warm_bc":
        status = "dry_run" if dry_run else "stub"
        stub_reason = "selfsup_warm_bc_not_implemented_in_p43"
        checkpoint_path = run_dir / "candidate_selfsup_warm_stub_checkpoint.json"
        checkpoint_payload = {
            "schema": "p43_selfsup_warm_stub_checkpoint_v1",
            "generated_at": now_iso(),
            "run_id": chosen_run_id,
            "training_mode": mode,
            "training_mode_category": training_mode_category,
            "reason": stub_reason,
        }
        write_json(checkpoint_path, checkpoint_payload)
        best_checkpoint_txt = run_dir / "best_checkpoint.txt"
        best_checkpoint_txt.write_text(str(checkpoint_path) + "\n", encoding="utf-8")
        curriculum_plan = {
            "enabled": False,
            "phase_count": 0,
            "phases": [],
            "seeds": list(seeds),
            "reason": "not_applicable_for_selfsup_warm_bc_stub",
        }
        write_json(run_dir / "curriculum_plan.json", curriculum_plan)
        curriculum_applied_path = run_dir / "curriculum_applied.jsonl"
        curriculum_applied_path.parent.mkdir(parents=True, exist_ok=True)
        if not curriculum_applied_path.exists():
            curriculum_applied_path.write_text("", encoding="utf-8")

        metrics_payload = {
            "schema": "p43_candidate_train_metrics_v1",
            "generated_at": now_iso(),
            "run_id": chosen_run_id,
            "status": status,
            "mode": mode,
            "training_mode": mode,
            "training_mode_category": training_mode_category,
            "fallback_used": fallback_used,
            "fallback_reason": fallback_reason,
            "legacy_paths_used": legacy_paths_used,
            "seed_count": len(seeds),
            "ok_seed_count": 0,
            "mean_reward": 0.0,
            "invalid_action_rate": 1.0,
            "final_loss": 0.0,
            "candidate_checkpoint": str(checkpoint_path),
            "reason": stub_reason,
        }
        manifest = {
            "schema": "p43_candidate_train_manifest_v1",
            "generated_at": now_iso(),
            "run_id": chosen_run_id,
            "status": status,
            "mode": mode,
            "training_mode": mode,
            "training_mode_category": training_mode_category,
            "candidate_modes": requested_modes,
            "fallback_used": fallback_used,
            "fallback_reason": fallback_reason,
            "legacy_fallback_modes": legacy_fallback_modes,
            "legacy_paths_used": legacy_paths_used,
            "reason": stub_reason,
            "config_path": str(cfg_path),
            "run_dir": str(run_dir),
            "quick": bool(quick),
            "dry_run": bool(dry_run),
            "curriculum_plan": curriculum_plan,
            "curriculum_config_path": "",
            "curriculum_applied": str(curriculum_applied_path),
            "candidate_checkpoint": str(checkpoint_path),
            "metrics_ref": str(run_dir / "metrics.json"),
        }
        write_json(run_dir / "candidate_train_manifest.json", manifest)
        write_json(run_dir / "metrics.json", metrics_payload)
        return {
            "status": status,
            "run_id": chosen_run_id,
            "run_dir": str(run_dir),
            "candidate_train_manifest": str(run_dir / "candidate_train_manifest.json"),
            "metrics": str(run_dir / "metrics.json"),
            "best_checkpoint": str(checkpoint_path),
            "training_mode": mode,
            "training_mode_category": training_mode_category,
            "fallback_used": fallback_used,
            "fallback_reason": fallback_reason,
            "legacy_paths_used": legacy_paths_used,
            "seeds_used": str(run_dir / "seeds_used.json"),
            "curriculum_plan": str(run_dir / "curriculum_plan.json"),
            "curriculum_applied": str(curriculum_applied_path),
        }

    train_cfg = cfg.get("training") if isinstance(cfg.get("training"), dict) else {}
    epochs = int(train_cfg.get("epochs") or 1)
    batch_size = int(train_cfg.get("batch_size") or 64)
    lr = float(train_cfg.get("lr") or 1e-3)
    timeout_sec = int(train_cfg.get("timeout_sec") or 1800)
    max_rows_per_seed = int(train_cfg.get("max_rows_per_seed") or (256 if quick else 2000))

    prefer_sources = train_cfg.get("prefer_source_types")
    if not isinstance(prefer_sources, list) or not prefer_sources:
        prefer_sources = ["p13_dagger_or_real", "arena_failures"]

    replay_manifest_path = _pick_replay_manifest_path(repo_root, cfg)
    replay_manifest = read_json(replay_manifest_path)
    if not isinstance(replay_manifest, dict):
        raise RuntimeError(f"invalid replay manifest: {replay_manifest_path}")
    replay_entries = _pick_bc_entries(replay_manifest, prefer_source_types=[str(x) for x in prefer_sources])
    imagination_cfg = _normalize_imagination_config(cfg, replay_manifest)

    curriculum_cfg = cfg.get("curriculum") if isinstance(cfg.get("curriculum"), dict) else {}
    curriculum_cfg_path = str(cfg.get("curriculum_config") or "").strip()
    if curriculum_cfg_path:
        loaded_curriculum = _read_yaml_or_json(to_abs_path(repo_root, curriculum_cfg_path))
        if isinstance(loaded_curriculum, dict):
            curriculum_cfg = loaded_curriculum
    curriculum_plan = build_curriculum_plan(
        curriculum_cfg=curriculum_cfg,
        default_epochs=max(1, epochs),
        quick=bool(quick),
        seeds=seeds,
    )
    write_json(run_dir / "curriculum_plan.json", curriculum_plan)
    curriculum_applied_path = run_dir / "curriculum_applied.jsonl"

    progress_path = run_dir / "progress.jsonl"
    seed_results: list[dict[str, Any]] = []
    best_checkpoint: Path | None = None
    best_loss: float | None = None
    stub_reason = ""

    for idx, seed in enumerate(seeds, start=1):
        event_base = {
            "schema": "p40_candidate_train_progress_v1",
            "ts": now_iso(),
            "run_id": chosen_run_id,
            "seed": seed,
            "seed_index": idx,
            "seed_total": len(seeds),
            "mode": mode,
        }
        append_jsonl(progress_path, {**event_base, "stage": "seed_start", "status": "running"})

        seed_dir = run_dir / "seed_runs" / f"seed_{idx:03d}_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        subset_path = seed_dir / "train_subset.jsonl"
        per_seed_curriculum: list[dict[str, Any]] = []
        if bool(curriculum_plan.get("enabled")) and isinstance(curriculum_plan.get("phases"), list):
            phase_datasets: list[Path] = []
            for phase in curriculum_plan.get("phases") or []:
                if not isinstance(phase, dict):
                    continue
                phase_name = str(phase.get("name") or f"phase_{len(phase_datasets)+1}")
                phase_idx = int(phase.get("phase_index") or (len(phase_datasets) + 1))
                phase_rows_cap = int(phase.get("target_rows") or 0) or max_rows_per_seed
                allocations = build_phase_allocations(
                    entries=replay_entries,
                    phase=phase,
                    default_target_rows=phase_rows_cap,
                )
                phase_dataset_path = seed_dir / "curriculum_phases" / f"{phase_idx:02d}_{phase_name}" / "dataset.jsonl"
                phase_meta = _build_subset_dataset_from_allocations(
                    allocations=allocations,
                    out_path=phase_dataset_path,
                    max_rows=phase_rows_cap,
                    imagination_cfg=imagination_cfg,
                )
                phase_sources: dict[str, int] = {}
                for alloc in allocations:
                    st = str(alloc.get("source_type") or "unknown")
                    phase_sources[st] = int(phase_sources.get(st, 0)) + int(alloc.get("take_rows") or 0)
                phase_summary = {
                    "schema": "p41_curriculum_applied_v1",
                    "ts": now_iso(),
                    "run_id": chosen_run_id,
                    "seed": seed,
                    "phase_index": phase_idx,
                    "phase_name": phase_name,
                    "target_rows": phase_rows_cap,
                    "actual_rows": int(phase_meta.get("kept_rows") or 0),
                    "source_weights": phase.get("source_weights") if isinstance(phase.get("source_weights"), dict) else {},
                    "slice_weights": phase.get("slice_weights") if isinstance(phase.get("slice_weights"), dict) else {},
                    "source_row_allocation": phase_sources,
                }
                append_jsonl(curriculum_applied_path, phase_summary)
                append_jsonl(
                    progress_path,
                    {
                        **event_base,
                        "stage": "curriculum_phase",
                        "status": "ok",
                        "phase_index": phase_idx,
                        "phase_name": phase_name,
                        "target_rows": phase_rows_cap,
                        "actual_rows": int(phase_meta.get("kept_rows") or 0),
                    },
                )
                per_seed_curriculum.append(phase_summary)
                if int(phase_meta.get("kept_rows") or 0) > 0:
                    phase_datasets.append(phase_dataset_path)

            subset_path.parent.mkdir(parents=True, exist_ok=True)
            with subset_path.open("w", encoding="utf-8", newline="\n") as out_fp:
                for phase_dataset in phase_datasets:
                    if not phase_dataset.exists():
                        continue
                    for line in phase_dataset.read_text(encoding="utf-8", errors="replace").splitlines():
                        if line.strip():
                            out_fp.write(line.strip() + "\n")
            subset_rows = 0
            if subset_path.exists():
                subset_rows = len([line for line in subset_path.read_text(encoding="utf-8", errors="replace").splitlines() if line.strip()])
            aggregate_breakdown: dict[str, int] = {}
            for phase_summary in per_seed_curriculum:
                if not isinstance(phase_summary, dict):
                    continue
                allocation = phase_summary.get("source_row_allocation") if isinstance(phase_summary.get("source_row_allocation"), dict) else {}
                for source_type, count in allocation.items():
                    aggregate_breakdown[str(source_type)] = int(aggregate_breakdown.get(str(source_type), 0)) + int(count or 0)
            subset_meta = {
                "kept_rows": subset_rows,
                "scanned_rows": 0,
                "source_files": [str(p) for p in phase_datasets],
                "source_breakdown": aggregate_breakdown,
                "imagined_filtered_breakdown": {},
                "imagined_mean_uncertainty": 0.0,
            }
        else:
            subset_meta = _build_subset_dataset(
                entries=replay_entries,
                out_path=subset_path,
                max_rows=max_rows_per_seed,
                imagination_cfg=imagination_cfg,
            )

        if dry_run:
            seed_result = {
                "seed": seed,
                "status": "dry_run",
                "subset_rows": int(subset_meta.get("kept_rows") or 0),
                "source_breakdown": subset_meta.get("source_breakdown") if isinstance(subset_meta.get("source_breakdown"), dict) else {},
                "imagined_filtered_breakdown": subset_meta.get("imagined_filtered_breakdown") if isinstance(subset_meta.get("imagined_filtered_breakdown"), dict) else {},
                "curriculum_phases": per_seed_curriculum,
                "metrics": {},
                "run_dir": str(seed_dir),
            }
            seed_results.append(seed_result)
            append_jsonl(progress_path, {**event_base, "stage": "seed_done", "status": "dry_run", "metrics": {}})
            continue

        if mode != "bc_finetune":
            stub_reason = f"unsupported_mode:{mode}"
            seed_result = {
                "seed": seed,
                "status": "stub",
                "subset_rows": int(subset_meta.get("kept_rows") or 0),
                "source_breakdown": subset_meta.get("source_breakdown") if isinstance(subset_meta.get("source_breakdown"), dict) else {},
                "imagined_filtered_breakdown": subset_meta.get("imagined_filtered_breakdown") if isinstance(subset_meta.get("imagined_filtered_breakdown"), dict) else {},
                "curriculum_phases": per_seed_curriculum,
                "metrics": {},
                "run_dir": str(seed_dir),
                "reason": stub_reason,
            }
            seed_results.append(seed_result)
            append_jsonl(progress_path, {**event_base, "stage": "seed_done", "status": "stub", "reason": stub_reason})
            continue

        if int(subset_meta.get("kept_rows") or 0) <= 0:
            stub_reason = "no_bc_compatible_rows_in_replay_mix"
            seed_result = {
                "seed": seed,
                "status": "stub",
                "subset_rows": 0,
                "source_breakdown": subset_meta.get("source_breakdown") if isinstance(subset_meta.get("source_breakdown"), dict) else {},
                "imagined_filtered_breakdown": subset_meta.get("imagined_filtered_breakdown") if isinstance(subset_meta.get("imagined_filtered_breakdown"), dict) else {},
                "curriculum_phases": per_seed_curriculum,
                "metrics": {},
                "run_dir": str(seed_dir),
                "reason": stub_reason,
            }
            seed_results.append(seed_result)
            append_jsonl(progress_path, {**event_base, "stage": "seed_done", "status": "stub", "reason": stub_reason})
            continue

        command = [
            _pick_python_exe(repo_root),
            "-B",
            "trainer/train_bc.py",
            "--train-jsonl",
            str(subset_path),
            "--epochs",
            str(
                max(
                    1,
                    int(
                        sum(int(p.get("epochs") or 1) for p in (curriculum_plan.get("phases") or []) if isinstance(p, dict))
                    )
                    if bool(curriculum_plan.get("enabled"))
                    else epochs
                )
            ),
            "--batch-size",
            str(max(1, batch_size)),
            "--lr",
            str(lr),
            "--seed",
            str(_seed_to_int(seed)),
            "--out-dir",
            str(seed_dir / "bc_run"),
        ]
        append_jsonl(progress_path, {**event_base, "stage": "train_cmd", "status": "running", "command": command})
        result = _run_process(command, cwd=repo_root, timeout_sec=timeout_sec)
        metrics_path = seed_dir / "bc_run" / "train_metrics.json"
        metrics = _read_train_metrics(metrics_path) if result["returncode"] == 0 else {}
        final_loss = float(metrics.get("val_hand_loss", 0.0)) + float(metrics.get("val_shop_loss", 0.0))
        checkpoint = seed_dir / "bc_run" / "best.pt"
        status = "ok" if result["returncode"] == 0 and checkpoint.exists() else "failed"

        seed_result = {
            "seed": seed,
            "status": status,
            "subset_rows": int(subset_meta.get("kept_rows") or 0),
            "source_breakdown": subset_meta.get("source_breakdown") if isinstance(subset_meta.get("source_breakdown"), dict) else {},
            "imagined_filtered_breakdown": subset_meta.get("imagined_filtered_breakdown") if isinstance(subset_meta.get("imagined_filtered_breakdown"), dict) else {},
            "imagined_mean_uncertainty": float(subset_meta.get("imagined_mean_uncertainty") or 0.0),
            "curriculum_phases": per_seed_curriculum,
            "metrics": metrics,
            "final_loss": float(final_loss),
            "checkpoint": str(checkpoint) if checkpoint.exists() else "",
            "stdout_tail": (result.get("stdout") or "")[-800:],
            "stderr_tail": (result.get("stderr") or "")[-800:],
            "elapsed_sec": float(result.get("elapsed_sec") or 0.0),
            "returncode": int(result.get("returncode") or 0),
            "run_dir": str(seed_dir),
        }
        seed_results.append(seed_result)
        append_jsonl(
            progress_path,
            {
                **event_base,
                "stage": "seed_done",
                "status": status,
                "metrics": metrics,
                "final_loss": final_loss,
                "checkpoint": seed_result["checkpoint"],
            },
        )

        if status == "ok":
            if best_loss is None or final_loss < best_loss:
                best_loss = final_loss
                best_checkpoint = checkpoint

    if best_checkpoint is None:
        stub_checkpoint = run_dir / "candidate_stub_checkpoint.json"
        stub_payload = {
            "schema": "p40_stub_checkpoint_v1",
            "generated_at": now_iso(),
            "run_id": chosen_run_id,
            "mode": mode,
            "reason": stub_reason or "no_successful_seed_run",
            "seed_results": seed_results,
        }
        write_json(stub_checkpoint, stub_payload)
        best_checkpoint = stub_checkpoint

    best_checkpoint_txt = run_dir / "best_checkpoint.txt"
    best_checkpoint_txt.write_text(str(best_checkpoint) + "\n", encoding="utf-8")

    ok_rows = [r for r in seed_results if str(r.get("status")) == "ok"]
    metric_rows = [r.get("metrics") for r in ok_rows if isinstance(r.get("metrics"), dict)]
    mean_hand_top1 = (
        statistics.mean([float(m.get("val_hand_acc1") or 0.0) for m in metric_rows]) if metric_rows else 0.0
    )
    mean_hand_top3 = (
        statistics.mean([float(m.get("val_hand_acc3") or 0.0) for m in metric_rows]) if metric_rows else 0.0
    )
    mean_shop_top1 = (
        statistics.mean([float(m.get("val_shop_acc1") or 0.0) for m in metric_rows]) if metric_rows else 0.0
    )
    mean_illegal = (
        statistics.mean([float(m.get("val_shop_illegal_rate") or 0.0) for m in metric_rows]) if metric_rows else 1.0
    )
    mean_loss = (
        statistics.mean(
            [float(m.get("val_hand_loss") or 0.0) + float(m.get("val_shop_loss") or 0.0) for m in metric_rows]
        )
        if metric_rows
        else 0.0
    )
    imagined_rows_total = sum(
        int(((row.get("source_breakdown") or {}).get("imagined_world_model") if isinstance(row.get("source_breakdown"), dict) else 0) or 0)
        for row in seed_results
        if isinstance(row, dict)
    )
    subset_rows_total = sum(int(row.get("subset_rows") or 0) for row in seed_results if isinstance(row, dict))
    imagined_fraction = float(imagined_rows_total) / max(1, subset_rows_total)

    status = "ok" if ok_rows else ("dry_run" if dry_run else "stub")
    metrics_payload = {
        "schema": "p41_candidate_train_metrics_v2",
        "generated_at": now_iso(),
        "run_id": chosen_run_id,
        "status": status,
        "mode": mode,
        "training_mode": mode,
        "training_mode_category": training_mode_category,
        "fallback_used": fallback_used,
        "fallback_reason": fallback_reason,
        "legacy_paths_used": legacy_paths_used,
        "curriculum_enabled": bool(curriculum_plan.get("enabled")),
        "curriculum_phase_count": int(curriculum_plan.get("phase_count") or len(curriculum_plan.get("phases") or [])),
        "seed_count": len(seeds),
        "ok_seed_count": len(ok_rows),
        "hand_top1": mean_hand_top1,
        "hand_top3": mean_hand_top3,
        "shop_top1": mean_shop_top1,
        "illegal_action_rate": mean_illegal,
        "final_loss": mean_loss,
        "imagined_enabled": bool(imagination_cfg.get("enabled")),
        "imagined_filter_mode": str(imagination_cfg.get("filter_mode") or ""),
        "imagined_fraction": float(imagined_fraction),
        "imagination_recipe": str(imagination_cfg.get("recipe") or "real_only"),
        "candidate_checkpoint": str(best_checkpoint),
    }
    manifest = {
        "schema": "p41_candidate_train_manifest_v2",
        "generated_at": now_iso(),
        "run_id": chosen_run_id,
        "status": status,
        "mode": mode,
        "training_mode": mode,
        "training_mode_category": training_mode_category,
        "candidate_modes": requested_modes,
        "fallback_used": fallback_used,
        "fallback_reason": fallback_reason,
        "legacy_fallback_modes": legacy_fallback_modes,
        "legacy_paths_used": legacy_paths_used,
        "config_path": str(cfg_path),
        "run_dir": str(run_dir),
        "quick": bool(quick),
        "dry_run": bool(dry_run),
        "replay_mix_manifest": str(replay_manifest_path),
        "replay_selected_entries": len(replay_entries),
        "imagined_enabled": bool(imagination_cfg.get("enabled")),
        "imagined_filter_mode": str(imagination_cfg.get("filter_mode") or ""),
        "imagined_fraction": float(imagined_fraction),
        "imagined_requested_fraction": float(imagination_cfg.get("imagined_fraction") or 0.0),
        "imagination_recipe": str(imagination_cfg.get("recipe") or "real_only"),
        "curriculum_plan": curriculum_plan,
        "curriculum_config_path": curriculum_cfg_path,
        "curriculum_applied": str(curriculum_applied_path),
        "seed_results": seed_results,
        "candidate_checkpoint": str(best_checkpoint),
        "metrics_ref": str(run_dir / "metrics.json"),
    }

    write_json(run_dir / "candidate_train_manifest.json", manifest)
    write_json(run_dir / "metrics.json", metrics_payload)

    return {
        "status": status,
        "run_id": chosen_run_id,
        "run_dir": str(run_dir),
        "candidate_train_manifest": str(run_dir / "candidate_train_manifest.json"),
        "metrics": str(run_dir / "metrics.json"),
        "best_checkpoint": str(best_checkpoint),
        "training_mode": mode,
        "training_mode_category": training_mode_category,
        "fallback_used": fallback_used,
        "fallback_reason": fallback_reason,
        "legacy_paths_used": legacy_paths_used,
        "seeds_used": str(run_dir / "seeds_used.json"),
        "curriculum_plan": str(run_dir / "curriculum_plan.json"),
        "curriculum_applied": str(curriculum_applied_path),
        "imagination_recipe": str(imagination_cfg.get("recipe") or "real_only"),
        "imagined_fraction": float(imagined_fraction),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="P40/P41/P42 candidate training loop (mainline-first, legacy fallback optional).")
    parser.add_argument("--config", default="configs/experiments/p40_candidate_smoke.yaml")
    parser.add_argument("--out-dir", default="")
    parser.add_argument("--run-id", default="")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = run_candidate_training(
        config_path=args.config,
        out_dir=(args.out_dir if str(args.out_dir).strip() else None),
        run_id=str(args.run_id or ""),
        quick=bool(args.quick),
        dry_run=bool(args.dry_run),
    )
    print(json.dumps(summary, ensure_ascii=False))
    return 0 if str(summary.get("status")) in {"ok", "stub", "dry_run"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
