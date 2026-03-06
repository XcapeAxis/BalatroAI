from __future__ import annotations

if __package__ is None or __package__ == "":
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

try:  # pragma: no cover - optional dependency
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None

from trainer.closed_loop.replay_manifest import now_iso, now_stamp, read_json, read_jsonl, write_json
from trainer.monitoring.progress_schema import append_progress_event, build_progress_event, get_gpu_mem_mb
from trainer.runtime.runtime_profile import load_runtime_profile
from trainer.world_model.dataset import build_world_model_dataset
from trainer.world_model.diagnostics import write_diagnostics
from trainer.world_model.losses import compute_world_model_losses
from trainer.world_model.model import load_world_model_from_checkpoint


def _require_torch():
    try:
        import torch
    except Exception as exc:  # pragma: no cover - runtime guarded
        raise RuntimeError("PyTorch is required for trainer.world_model.eval") from exc
    return torch


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


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as fp:
        for row in rows:
            fp.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_markdown(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


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


def _resolve_dataset_manifest(
    *,
    repo_root: Path,
    config_path: Path,
    explicit_path: str,
    checkpoint_payload: dict[str, Any],
    quick: bool,
) -> str:
    if explicit_path:
        path = Path(explicit_path)
        if not path.is_absolute():
            path = (repo_root / path).resolve()
        return str(path)
    extra = checkpoint_payload.get("extra") if isinstance(checkpoint_payload.get("extra"), dict) else {}
    direct = str(extra.get("dataset_manifest") or "").strip()
    if direct:
        return direct
    dataset_summary = extra.get("dataset_summary") if isinstance(extra.get("dataset_summary"), dict) else {}
    direct = str(dataset_summary.get("dataset_manifest_json") or "").strip()
    if direct:
        return direct
    built = build_world_model_dataset(config_path=config_path, quick=quick)
    return str(built.get("dataset_manifest_json") or "")


def _batch_from_rows(rows: list[dict[str, Any]], *, device: Any) -> dict[str, Any]:
    torch = _require_torch()
    obs_t = torch.tensor([list(row.get("obs_t") or []) for row in rows], dtype=torch.float32, device=device)
    obs_t1 = torch.tensor([list(row.get("obs_t1") or []) for row in rows], dtype=torch.float32, device=device)
    action_ids = torch.tensor([int(row.get("action_id") or 0) for row in rows], dtype=torch.long, device=device)
    reward_t = torch.tensor([_safe_float(row.get("reward_t"), 0.0) for row in rows], dtype=torch.float32, device=device)
    score_delta_t = torch.tensor([_safe_float(row.get("score_delta_t"), 0.0) for row in rows], dtype=torch.float32, device=device)
    resource_delta_t = torch.tensor([list(row.get("resource_delta_t") or [0.0] * 5) for row in rows], dtype=torch.float32, device=device)
    latent_rows = [list(row.get("latent_t1") or []) for row in rows]
    latent_t1 = None
    if latent_rows and all(latent_rows) and len(set(len(item) for item in latent_rows)) == 1:
        latent_t1 = torch.tensor(latent_rows, dtype=torch.float32, device=device)
    return {
        "obs_t": obs_t,
        "obs_t1": obs_t1,
        "action_id": action_ids,
        "reward_t": reward_t,
        "score_delta_t": score_delta_t,
        "resource_delta_t": resource_delta_t,
        "latent_t1": latent_t1,
    }


def _mean(values: list[float]) -> float:
    return float(sum(values) / max(1, len(values)))


def _slice_summary(prediction_rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for row in prediction_rows:
        slices = row.get("slice_labels") if isinstance(row.get("slice_labels"), dict) else {}
        for key in ("slice_stage", "slice_resource_pressure", "slice_action_type"):
            grouped[key][str(slices.get(key, "unknown"))].append(row)

    payload: dict[str, Any] = {}
    for key, labels in grouped.items():
        payload[key] = [
            {
                "label": label,
                "count": len(rows),
                "mean_combined_error": _mean([_safe_float(row.get("combined_error"), 0.0) for row in rows]),
                "mean_reward_abs_error": _mean([_safe_float(row.get("reward_error_abs"), 0.0) for row in rows]),
                "mean_uncertainty": _mean([_safe_float(row.get("uncertainty_pred"), 0.0) for row in rows]),
            }
            for label, rows in sorted(labels.items(), key=lambda item: (-len(item[1]), item[0]))
        ]
    return payload


def run_world_model_eval(
    *,
    config_path: str | Path,
    checkpoint_path: str | Path,
    dataset_manifest_path: str | Path | None = None,
    out_dir: str | Path | None = None,
    run_id: str = "",
    quick: bool = False,
    runtime_profile: dict[str, Any] | None = None,
) -> dict[str, Any]:
    repo_root = Path(__file__).resolve().parents[2]
    cfg_path = Path(config_path)
    if not cfg_path.is_absolute():
        cfg_path = (repo_root / cfg_path).resolve()
    cfg = _read_yaml_or_json(cfg_path)
    eval_cfg = cfg.get("eval") if isinstance(cfg.get("eval"), dict) else {}
    manifest_path = _resolve_dataset_manifest(
        repo_root=repo_root,
        config_path=cfg_path,
        explicit_path=str(dataset_manifest_path or ""),
        checkpoint_payload=load_world_model_from_checkpoint(checkpoint_path, device="cpu")[2],
        quick=quick,
    )
    manifest_obj = read_json(Path(manifest_path))
    if not isinstance(manifest_obj, dict):
        raise ValueError(f"dataset manifest unreadable: {manifest_path}")
    dataset_path = Path(str(manifest_obj.get("dataset_jsonl") or ""))
    if not dataset_path.exists():
        raise ValueError(f"dataset jsonl missing: {dataset_path}")
    rows = [row for row in read_jsonl(dataset_path) if isinstance(row, dict)]
    split = str(eval_cfg.get("split") or "val").strip().lower()
    rows = [row for row in rows if str(row.get("split") or "train").strip().lower() == split]
    if quick and len(rows) > 128:
        rows = rows[:128]
    if not rows:
        raise ValueError(f"no rows available for split={split}")

    chosen_run_id = str(run_id or now_stamp())
    out_root = (
        (repo_root / str(eval_cfg.get("output_artifacts_root") or "docs/artifacts/p45/wm_eval")).resolve()
        if out_dir is None
        else (Path(out_dir) if Path(out_dir).is_absolute() else (repo_root / out_dir).resolve())
    )
    run_dir = out_root / chosen_run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    runtime_profile_payload = (
        dict(runtime_profile)
        if isinstance(runtime_profile, dict)
        else load_runtime_profile(config=cfg, component="p45_world_model_eval").to_dict()
    )
    runtime_resolved = (
        runtime_profile_payload.get("resolved_profile", {}).get("resolved")
        if isinstance(runtime_profile_payload.get("resolved_profile"), dict)
        else {}
    )
    learner_device = str((runtime_resolved or {}).get("learner_device") or "cpu")
    rollout_device = str((runtime_resolved or {}).get("rollout_device") or "cpu")
    runtime_profile_json = run_dir / "runtime_profile.json"
    progress_unified_path = run_dir / "progress.unified.jsonl"
    write_json(runtime_profile_json, runtime_profile_payload)

    torch = _require_torch()
    device = learner_device if str(learner_device or "").strip() else "cpu"
    model, _model_config, checkpoint_payload = load_world_model_from_checkpoint(checkpoint_path, device=device)
    batch_size = max(
        8,
        _safe_int(
            (runtime_resolved or {}).get("batch_size"),
            int(eval_cfg.get("batch_size") or 128),
        ),
    )
    loss_weights = ((cfg.get("train") or {}).get("loss_weights") if isinstance(cfg.get("train"), dict) else {}) or {}

    prediction_rows: list[dict[str, Any]] = []
    totals: dict[str, list[float]] = defaultdict(list)
    model.eval()
    with torch.no_grad():
        for start in range(0, len(rows), batch_size):
            chunk = rows[start : start + batch_size]
            batch = _batch_from_rows(chunk, device="cpu")
            outputs = model(batch["obs_t"], batch["action_id"], batch["obs_t1"])
            losses = compute_world_model_losses(outputs=outputs, batch=batch, loss_weights=loss_weights)

            reward_pred = outputs["reward_pred"].detach().cpu().tolist()
            score_pred = outputs["score_pred"].detach().cpu().tolist()
            uncertainty_pred = outputs["uncertainty_pred"].detach().cpu().tolist()
            latent_error = losses["per_item_latent_error"].detach().cpu().tolist()
            reward_abs = losses["per_item_reward_error_abs"].detach().cpu().tolist()
            score_abs = losses["per_item_score_error_abs"].detach().cpu().tolist()
            combined_error = losses["per_item_combined_error"].detach().cpu().tolist()

            for idx, row in enumerate(chunk):
                pred_row = {
                    "sample_id": str(row.get("sample_id") or ""),
                    "source_type": str(row.get("source_type") or ""),
                    "seed": str(row.get("seed") or ""),
                    "phase_t": str(row.get("phase_t") or ""),
                    "reward_t": _safe_float(row.get("reward_t"), 0.0),
                    "reward_pred": _safe_float(reward_pred[idx], 0.0),
                    "reward_error_abs": _safe_float(reward_abs[idx], 0.0),
                    "score_delta_t": _safe_float(row.get("score_delta_t"), 0.0),
                    "score_pred": _safe_float(score_pred[idx], 0.0),
                    "score_error_abs": _safe_float(score_abs[idx], 0.0),
                    "latent_error": _safe_float(latent_error[idx], 0.0),
                    "uncertainty_pred": _safe_float(uncertainty_pred[idx], 0.0),
                    "combined_error": _safe_float(combined_error[idx], 0.0),
                    "done_t": bool(row.get("done_t", False)),
                    "slice_labels": row.get("slice_labels") if isinstance(row.get("slice_labels"), dict) else {},
                }
                prediction_rows.append(pred_row)

            for key in (
                "total_loss",
                "latent_loss",
                "reward_loss",
                "score_loss",
                "resource_loss",
                "uncertainty_loss",
                "reward_mae",
                "score_mae",
                "resource_mae",
                "combined_error_mean",
                "uncertainty_mean",
            ):
                totals[key].append(_safe_float(losses[key].detach().cpu().item(), 0.0))

    slice_eval = _slice_summary(prediction_rows)
    diagnostics = write_diagnostics(
        out_dir=run_dir,
        prediction_rows=prediction_rows,
        topk=max(4, int(eval_cfg.get("diagnostics_topk") or 12)),
    )
    eval_metrics = {
        "schema": "p45_world_model_eval_metrics_v1",
        "generated_at": now_iso(),
        "run_id": chosen_run_id,
        "checkpoint_path": str(Path(checkpoint_path).resolve()),
        "dataset_manifest_path": str(Path(manifest_path).resolve()),
        "split": split,
        "sample_count": len(prediction_rows),
        "latent_transition_error": _mean([_safe_float(row.get("latent_error"), 0.0) for row in prediction_rows]),
        "reward_prediction_error": _mean([_safe_float(row.get("reward_error_abs"), 0.0) for row in prediction_rows]),
        "score_delta_prediction_error": _mean([_safe_float(row.get("score_error_abs"), 0.0) for row in prediction_rows]),
        "uncertainty_mean": _mean([_safe_float(row.get("uncertainty_pred"), 0.0) for row in prediction_rows]),
        "uncertainty_calibration": diagnostics.get("payload") if isinstance(diagnostics.get("payload"), dict) else {},
        "done_prediction": {
            "status": "skipped_not_implemented",
            "note": "P45 v1 does not add an explicit done head yet.",
        },
        "runtime_profile": runtime_profile_payload,
        "learner_device": learner_device,
        "rollout_device": rollout_device,
        "gpu_mem_mb": get_gpu_mem_mb(device),
        "losses": {key: _mean(values) for key, values in totals.items()},
        "slice_eval_json": str(run_dir / "slice_eval.json"),
        "diagnostics_json": str(run_dir / "diagnostics.json"),
        "prediction_rows_jsonl": str(run_dir / "prediction_rows.jsonl"),
    }

    lines = [
        f"# P45 World Model Eval ({chosen_run_id})",
        "",
        f"- checkpoint_path: `{Path(checkpoint_path).resolve()}`",
        f"- dataset_manifest_path: `{Path(manifest_path).resolve()}`",
        f"- split: `{split}`",
        f"- sample_count: {int(eval_metrics.get('sample_count') or 0)}",
        f"- latent_transition_error: {float(eval_metrics.get('latent_transition_error') or 0.0):.6f}",
        f"- reward_prediction_error: {float(eval_metrics.get('reward_prediction_error') or 0.0):.6f}",
        f"- score_delta_prediction_error: {float(eval_metrics.get('score_delta_prediction_error') or 0.0):.6f}",
        f"- uncertainty_mean: {float(eval_metrics.get('uncertainty_mean') or 0.0):.6f}",
        "",
        "## Done Prediction",
        f"- status: {((eval_metrics.get('done_prediction') or {}).get('status') or 'unknown')}",
        f"- note: {((eval_metrics.get('done_prediction') or {}).get('note') or '')}",
    ]

    write_json(run_dir / "eval_metrics.json", eval_metrics)
    write_json(run_dir / "slice_eval.json", slice_eval)
    _write_jsonl(run_dir / "prediction_rows.jsonl", prediction_rows)
    _write_markdown(run_dir / "eval_metrics.md", lines)
    append_progress_event(
        progress_unified_path,
        build_progress_event(
            run_id=chosen_run_id,
            component="p45_wm_eval",
            phase="eval",
            status="completed",
            step=len(prediction_rows),
            epoch_or_iter=1,
            metrics={
                "latent_transition_error": eval_metrics.get("latent_transition_error"),
                "reward_prediction_error": eval_metrics.get("reward_prediction_error"),
                "score_delta_prediction_error": eval_metrics.get("score_delta_prediction_error"),
                "uncertainty_mean": eval_metrics.get("uncertainty_mean"),
                "sample_count": eval_metrics.get("sample_count"),
            },
            device_profile=runtime_profile_payload,
            learner_device=learner_device,
            rollout_device=rollout_device,
            throughput=float(len(prediction_rows)) / max(1.0, float(len(rows) / max(1, batch_size))),
            gpu_mem_mb=get_gpu_mem_mb(device),
        ),
    )
    return {
        "status": "ok",
        "run_id": chosen_run_id,
        "run_dir": str(run_dir),
        "eval_metrics_json": str(run_dir / "eval_metrics.json"),
        "eval_metrics_md": str(run_dir / "eval_metrics.md"),
        "slice_eval_json": str(run_dir / "slice_eval.json"),
        "diagnostics_json": str(run_dir / "diagnostics.json"),
        "prediction_rows_jsonl": str(run_dir / "prediction_rows.jsonl"),
        "runtime_profile_json": str(runtime_profile_json),
        "progress_unified_jsonl": str(progress_unified_path),
        "metrics": eval_metrics,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a P45 world model checkpoint against replay transitions.")
    parser.add_argument("--config", default="configs/experiments/p45_world_model_smoke.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset-manifest", default="")
    parser.add_argument("--out-dir", default="")
    parser.add_argument("--run-id", default="")
    parser.add_argument("--quick", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = run_world_model_eval(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        dataset_manifest_path=(args.dataset_manifest if str(args.dataset_manifest).strip() else None),
        out_dir=(args.out_dir if str(args.out_dir).strip() else None),
        run_id=str(args.run_id or ""),
        quick=bool(args.quick),
    )
    print(json.dumps(summary, ensure_ascii=False))
    return 0 if str(summary.get("status")) == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
