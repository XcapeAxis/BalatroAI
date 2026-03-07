from __future__ import annotations

if __package__ is None or __package__ == "":
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

import argparse
import json
import random
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:  # pragma: no cover
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None

from trainer.closed_loop.replay_manifest import build_seeds_payload, write_json
from trainer.hybrid.learned_router_model import (
    CHECKPOINT_SCHEMA,
    apply_controller_mask,
    build_checkpoint_payload,
    build_mlp,
    save_router_checkpoint,
    softmax_with_mask,
)
from trainer.hybrid.router_dataset import build_router_dataset
from trainer.hybrid.router_schema import (
    available_controller_mask,
    encode_routing_features,
    normalize_target_scores,
    sample_label_index,
    supported_controller_ids,
)
from trainer.monitoring.progress_schema import append_progress_event, build_progress_event, get_gpu_mem_mb
from trainer.registry.checkpoint_registry import register_checkpoint
from trainer.runtime.runtime_profile import load_runtime_profile


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def _require_torch():
    try:
        import torch
        from torch.utils.data import DataLoader, Dataset
    except Exception as exc:  # pragma: no cover - runtime guarded
        raise RuntimeError("PyTorch is required for learned router training") from exc
    return torch, DataLoader, Dataset


def _read_yaml_or_json(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8-sig")
    if path.suffix.lower() in {".yaml", ".yml"}:
        if yaml is None:
            try:
                payload = json.loads(text)
            except Exception:
                sidecar = path.with_suffix(".json")
                if sidecar.exists():
                    payload = json.loads(sidecar.read_text(encoding="utf-8"))
                else:
                    raise RuntimeError(f"PyYAML unavailable for {path}")
        else:
            payload = yaml.safe_load(text)
    else:
        payload = json.loads(text)
    if not isinstance(payload, dict):
        raise ValueError(f"config must be mapping: {path}")
    return payload


def _read_json(path: Path) -> dict[str, Any] | list[Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception:
        return None


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8", errors="replace") as fp:
        for line in fp:
            text = line.strip()
            if not text:
                continue
            try:
                payload = json.loads(text)
            except Exception:
                continue
            if isinstance(payload, dict):
                rows.append(payload)
    return rows


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _git_commit(repo_root: Path) -> str:
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_root),
            text=True,
            capture_output=True,
            timeout=10,
            check=False,
        )
    except Exception:
        return ""
    return str(proc.stdout or "").strip() if int(proc.returncode) == 0 else ""


def _runtime_profile_name(runtime_profile: dict[str, Any]) -> str:
    resolved_profile = runtime_profile.get("resolved_profile") if isinstance(runtime_profile.get("resolved_profile"), dict) else {}
    resolved = resolved_profile.get("resolved") if isinstance(resolved_profile.get("resolved"), dict) else {}
    return str(resolved.get("profile_name") or runtime_profile.get("profile_name") or "single_gpu_mainline")


def _default_config() -> dict[str, Any]:
    return {
        "schema": "p52_learned_router_train_config_v1",
        "runtime": {"device_profile": "single_gpu_mainline"},
        "dataset": {"manifest": "", "auto_build_quick": True},
        "train": {
            "seed": 5201,
            "epochs": 8,
            "batch_size": 64,
            "lr": 1e-3,
            "weight_decay": 1e-4,
            "dropout": 0.10,
            "hidden_dims": [128, 64],
            "val_fraction": 0.20,
            "top_k": 2,
            "class_weighting": "inverse_freq",
            "temperature": 1.0,
            "require_cuda": True,
            "checkpoint_metric": "val_top1_accuracy",
        },
        "output": {"artifacts_root": "docs/artifacts/p52/router_train"},
    }


def _merged_config(config_path: str | Path | None) -> dict[str, Any]:
    payload = _default_config()
    if not config_path:
        return payload
    cfg_path = Path(config_path)
    if not cfg_path.is_absolute():
        cfg_path = (Path(__file__).resolve().parents[2] / cfg_path).resolve()
    override = _read_yaml_or_json(cfg_path)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(payload.get(key), dict):
            merged = dict(payload.get(key) or {})
            merged.update(value)
            payload[key] = merged
        else:
            payload[key] = value
    return payload


def _dataset_artifacts_root(repo_root: Path, cfg: dict[str, Any]) -> Path:
    dataset_cfg = cfg.get("dataset") if isinstance(cfg.get("dataset"), dict) else {}
    token = str(dataset_cfg.get("artifacts_root") or "docs/artifacts/p52/router_dataset")
    return (repo_root / token).resolve()


def _train_artifacts_root(repo_root: Path, cfg: dict[str, Any], out_dir: str | Path | None) -> Path:
    train_cfg = cfg.get("train") if isinstance(cfg.get("train"), dict) else {}
    output_cfg = cfg.get("output") if isinstance(cfg.get("output"), dict) else {}
    configured_root = str(train_cfg.get("artifacts_root") or output_cfg.get("artifacts_root") or "docs/artifacts/p52/router_train")
    if out_dir is None:
        return (repo_root / configured_root).resolve()
    target = Path(out_dir)
    return target if target.is_absolute() else (repo_root / target).resolve()


def _discover_latest_dataset_manifest(repo_root: Path, cfg: dict[str, Any]) -> Path | None:
    root = _dataset_artifacts_root(repo_root, cfg)
    if not root.exists():
        return None
    candidates = sorted(root.glob("**/router_dataset_manifest.json"), key=lambda item: str(item))
    return candidates[-1].resolve() if candidates else None


class RouterRowsDataset:
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self.rows = list(rows)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return dict(self.rows[index])


def _collate_rows(items: list[dict[str, Any]]) -> dict[str, Any]:
    torch, _DataLoader, _Dataset = _require_torch()
    return {
        "features": torch.tensor([list(item.get("features") or []) for item in items], dtype=torch.float32),
        "available_mask": torch.tensor([list(item.get("available_mask") or []) for item in items], dtype=torch.float32),
        "target_probs": torch.tensor([list(item.get("target_probs") or []) for item in items], dtype=torch.float32),
        "label_idx": torch.tensor([int(item.get("label_idx") or 0) for item in items], dtype=torch.long),
        "sample_weight": torch.tensor([_safe_float(item.get("sample_weight"), 1.0) for item in items], dtype=torch.float32),
        "slice_stage": [str(item.get("slice_stage") or "unknown") for item in items],
        "target_controller_label": [str(item.get("target_controller_label") or "") for item in items],
        "sample_id": [str(item.get("sample_id") or "") for item in items],
    }


def _move_batch(batch: dict[str, Any], device: Any) -> dict[str, Any]:
    torch, _DataLoader, _Dataset = _require_torch()
    moved: dict[str, Any] = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            moved[key] = value.to(device, non_blocking=not str(device).startswith("cpu"))
        else:
            moved[key] = value
    return moved


def _prepare_rows(samples: list[dict[str, Any]], feature_encoder: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    controller_ids = supported_controller_ids()
    for sample in samples:
        encoded = encode_routing_features(sample, feature_encoder)
        label_idx = sample_label_index(sample, controller_ids=controller_ids)
        available_mask = available_controller_mask(sample.get("available_controllers"), controller_ids=controller_ids)
        if label_idx < 0 or sum(available_mask) <= 0.0:
            continue
        rows.append(
            {
                "sample_id": str(sample.get("sample_id") or ""),
                "features": list(encoded.get("vector") or []),
                "available_mask": list(available_mask),
                "target_probs": list(normalize_target_scores(sample, controller_ids=controller_ids)),
                "label_idx": label_idx,
                "sample_weight": max(0.10, _safe_float(sample.get("label_confidence"), 1.0)),
                "slice_stage": str(sample.get("slice_stage") or "unknown"),
                "target_controller_label": str(sample.get("target_controller_label") or ""),
            }
        )
    return rows


def _split_rows(rows: list[dict[str, Any]], *, seed: int, val_fraction: float) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    ordered = list(rows)
    random.Random(seed).shuffle(ordered)
    if len(ordered) <= 1:
        return ordered, ordered
    val_count = max(1, min(len(ordered) - 1, int(round(len(ordered) * max(0.05, min(0.50, val_fraction))))))
    return ordered[val_count:], ordered[:val_count]


def _class_weights(rows: list[dict[str, Any]], *, strategy: str) -> list[float]:
    counts = defaultdict(int)
    for row in rows:
        counts[int(row.get("label_idx") or 0)] += 1
    total = max(1, sum(counts.values()))
    weights = []
    for class_index in range(len(supported_controller_ids())):
        count = max(1, counts.get(class_index, 0))
        if strategy == "inverse_freq":
            weights.append(float(total) / (len(supported_controller_ids()) * count))
        else:
            weights.append(1.0)
    return weights


def _soft_cross_entropy(
    *,
    logits: Any,
    available_mask: Any,
    target_probs: Any,
    sample_weight: Any,
    class_weights: Any,
    label_idx: Any,
) -> Any:
    torch, _DataLoader, _Dataset = _require_torch()
    masked_logits = apply_controller_mask(logits, available_mask)
    log_probs = torch.log_softmax(masked_logits, dim=-1)
    per_sample = -(target_probs * log_probs).sum(dim=-1)
    per_sample = per_sample * sample_weight
    per_sample = per_sample * class_weights[label_idx]
    return per_sample.mean()


def evaluate_router_rows(
    *,
    model: Any,
    rows: list[dict[str, Any]],
    device: Any,
    batch_size: int,
    top_k: int,
    temperature: float,
) -> dict[str, Any]:
    torch, DataLoader, _Dataset = _require_torch()
    loader = DataLoader(RouterRowsDataset(rows), batch_size=max(1, batch_size), shuffle=False, collate_fn=_collate_rows)
    controller_ids = supported_controller_ids()
    confusion = [[0 for _ in controller_ids] for _ in controller_ids]
    slice_totals: dict[str, dict[str, int]] = defaultdict(lambda: {"correct": 0, "count": 0})
    losses: list[float] = []
    nll_values: list[float] = []
    brier_values: list[float] = []
    top1_correct = 0
    topk_correct = 0
    total = 0

    model.eval()
    with torch.no_grad():
        for batch in loader:
            moved = _move_batch(batch, device)
            logits = model(moved["features"])
            probs = softmax_with_mask(logits, moved["available_mask"], temperature=temperature)
            masked_logits = apply_controller_mask(logits, moved["available_mask"])
            log_probs = torch.log_softmax(masked_logits, dim=-1)
            losses.extend((-(moved["target_probs"] * log_probs).sum(dim=-1)).detach().cpu().tolist())
            brier_values.extend((((probs - moved["target_probs"]) ** 2).sum(dim=-1)).detach().cpu().tolist())
            pred = torch.argmax(probs, dim=-1)
            topk = torch.topk(probs, k=max(1, min(int(top_k), probs.shape[-1])), dim=-1).indices
            label_idx = moved["label_idx"]
            top1_correct += int((pred == label_idx).sum().item())
            topk_correct += int((topk == label_idx.unsqueeze(-1)).any(dim=-1).sum().item())
            total += int(label_idx.shape[0])
            picked_log_probs = log_probs.gather(1, label_idx.unsqueeze(1)).squeeze(1)
            nll_values.extend((-picked_log_probs).detach().cpu().tolist())
            pred_rows = pred.detach().cpu().tolist()
            label_rows = label_idx.detach().cpu().tolist()
            for pred_idx, label_idx_value, slice_stage in zip(pred_rows, label_rows, batch["slice_stage"]):
                confusion[int(label_idx_value)][int(pred_idx)] += 1
                slice_totals[str(slice_stage)]["count"] += 1
                if int(pred_idx) == int(label_idx_value):
                    slice_totals[str(slice_stage)]["correct"] += 1

    confusion_payload = {
        "schema": "p52_router_confusion_v1",
        "controller_ids": controller_ids,
        "matrix": confusion,
    }
    slice_eval_rows = [
        {
            "slice_stage": slice_stage,
            "count": int(values["count"]),
            "top1_accuracy": float(values["correct"]) / max(1, int(values["count"])),
        }
        for slice_stage, values in sorted(slice_totals.items(), key=lambda item: (-item[1]["count"], item[0]))
    ]
    metrics = {
        "top1_accuracy": float(top1_correct) / max(1, total),
        "topk_accuracy": float(topk_correct) / max(1, total),
        "loss": (sum(losses) / max(1, len(losses))) if losses else 0.0,
        "nll": (sum(nll_values) / max(1, len(nll_values))) if nll_values else 0.0,
        "brier": (sum(brier_values) / max(1, len(brier_values))) if brier_values else 0.0,
        "count": total,
        "confusion": confusion_payload,
        "slice_eval": slice_eval_rows,
    }
    return metrics


def run_learned_router_train(
    *,
    config_path: str | Path | None = None,
    out_dir: str | Path | None = None,
    run_id: str = "",
    quick: bool = False,
    dataset_manifest_path: str | Path | None = None,
) -> dict[str, Any]:
    repo_root = Path(__file__).resolve().parents[2]
    cfg = _merged_config(config_path)
    dataset_cfg = cfg.get("dataset") if isinstance(cfg.get("dataset"), dict) else {}
    train_cfg = cfg.get("train") if isinstance(cfg.get("train"), dict) else {}
    output_cfg = cfg.get("output") if isinstance(cfg.get("output"), dict) else {}
    registry_cfg = cfg.get("registry") if isinstance(cfg.get("registry"), dict) else {}

    manifest_path = Path(dataset_manifest_path).resolve() if dataset_manifest_path else None
    if manifest_path is None and str(dataset_cfg.get("manifest") or "").strip():
        raw = str(dataset_cfg.get("manifest") or "")
        manifest_path = Path(raw)
        if not manifest_path.is_absolute():
            manifest_path = (repo_root / manifest_path).resolve()
    if manifest_path is None or not manifest_path.exists():
        if bool(dataset_cfg.get("auto_build_quick", True)):
            dataset_summary = build_router_dataset(
                config_path=config_path,
                out_dir=_dataset_artifacts_root(repo_root, cfg),
                quick=quick,
            )
            manifest_path = Path(str(dataset_summary.get("dataset_manifest_json") or "")).resolve()
        else:
            manifest_path = _discover_latest_dataset_manifest(repo_root, cfg)
    if manifest_path is None or not manifest_path.exists():
        raise FileNotFoundError("router dataset manifest not found")

    dataset_manifest = _read_json(manifest_path)
    if not isinstance(dataset_manifest, dict):
        raise ValueError(f"invalid dataset manifest: {manifest_path}")
    dataset_jsonl = Path(str(dataset_manifest.get("dataset_jsonl") or "")).resolve()
    feature_encoder_path = Path(str(dataset_manifest.get("feature_encoder_json") or "")).resolve()
    feature_encoder = _read_json(feature_encoder_path)
    if not isinstance(feature_encoder, dict):
        raise ValueError(f"invalid feature encoder: {feature_encoder_path}")
    samples = [row for row in _read_jsonl(dataset_jsonl) if isinstance(row, dict) and bool(row.get("valid_for_training"))]
    if not samples:
        raise ValueError(f"router training dataset is empty: {dataset_jsonl}")

    train_seed = _safe_int(train_cfg.get("seed"), 5201)
    random.seed(train_seed)
    torch, DataLoader, Dataset = _require_torch()
    torch.manual_seed(train_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(train_seed)

    runtime_profile = load_runtime_profile(config=cfg, component="p52_learned_router").to_dict()
    resolved_profile = (
        runtime_profile.get("resolved_profile", {}).get("resolved")
        if isinstance(runtime_profile.get("resolved_profile"), dict)
        else {}
    )
    learner_device = str((resolved_profile or {}).get("learner_device") or "cpu")
    rollout_device = str((resolved_profile or {}).get("rollout_device") or "cpu")
    require_cuda = bool(train_cfg.get("require_cuda", True))
    if require_cuda and not learner_device.startswith("cuda"):
        raise RuntimeError(f"P52 learned router expected CUDA learner_device, resolved {learner_device}")
    device = torch.device(learner_device)

    rows = _prepare_rows(samples, feature_encoder)
    if not rows:
        raise ValueError("no valid router training rows after encoding")
    train_rows, val_rows = _split_rows(rows, seed=train_seed, val_fraction=_safe_float(train_cfg.get("val_fraction"), 0.20))
    if not train_rows:
        train_rows = list(rows)
    if not val_rows:
        val_rows = list(train_rows[: max(1, min(8, len(train_rows)))])

    chosen_run_id = str(run_id or output_cfg.get("run_id") or _now_stamp())
    output_root = _train_artifacts_root(repo_root, cfg, out_dir)
    run_dir = output_root / chosen_run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    progress_path = run_dir / "progress.jsonl"
    progress_unified_path = run_dir / "progress.unified.jsonl"
    runtime_profile_json = run_dir / "runtime_profile.json"
    write_json(runtime_profile_json, runtime_profile)

    input_dim = int(feature_encoder.get("output_dim") or len(rows[0]["features"]))
    output_dim = len(supported_controller_ids())
    hidden_dims = train_cfg.get("hidden_dims") if isinstance(train_cfg.get("hidden_dims"), list) else [128, 64]
    dropout = _safe_float(train_cfg.get("dropout"), 0.10)
    model = build_mlp(input_dim=input_dim, output_dim=output_dim, hidden_dims=hidden_dims, dropout=dropout).to(device)

    batch_size = max(8, _safe_int((resolved_profile or {}).get("batch_size"), _safe_int(train_cfg.get("batch_size"), 64)))
    if quick:
        batch_size = min(batch_size, 64)
    train_loader = DataLoader(RouterRowsDataset(train_rows), batch_size=batch_size, shuffle=True, collate_fn=_collate_rows)
    val_batch_size = min(max(8, batch_size), max(8, len(val_rows)))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=max(1e-5, _safe_float(train_cfg.get("lr"), 1e-3)),
        weight_decay=max(0.0, _safe_float(train_cfg.get("weight_decay"), 1e-4)),
    )
    epochs = max(1, _safe_int(train_cfg.get("epochs"), 8))
    if quick:
        epochs = min(epochs, 4)
    top_k = max(1, _safe_int(train_cfg.get("top_k"), 2))
    temperature = max(1e-6, _safe_float(train_cfg.get("temperature"), 1.0))
    class_weights_tensor = torch.tensor(
        _class_weights(train_rows, strategy=str(train_cfg.get("class_weighting") or "inverse_freq")),
        dtype=torch.float32,
        device=device,
    )

    model_config = {
        "input_dim": input_dim,
        "output_dim": output_dim,
        "hidden_dims": [int(dim) for dim in hidden_dims],
        "dropout": dropout,
    }
    best_metric_name = str(train_cfg.get("checkpoint_metric") or "val_top1_accuracy")
    best_metric_value = float("-inf")
    best_checkpoint = run_dir / "best.pt"
    last_checkpoint = run_dir / "last.pt"
    history_rows: list[dict[str, Any]] = []

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_losses: list[float] = []
        for batch in train_loader:
            moved = _move_batch(batch, device)
            logits = model(moved["features"])
            loss = _soft_cross_entropy(
                logits=logits,
                available_mask=moved["available_mask"],
                target_probs=moved["target_probs"],
                sample_weight=moved["sample_weight"],
                class_weights=class_weights_tensor,
                label_idx=moved["label_idx"],
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.detach().cpu().item()))

        train_eval = evaluate_router_rows(
            model=model,
            rows=train_rows,
            device=device,
            batch_size=min(batch_size, max(8, len(train_rows))),
            top_k=top_k,
            temperature=temperature,
        )
        val_eval = evaluate_router_rows(
            model=model,
            rows=val_rows,
            device=device,
            batch_size=val_batch_size,
            top_k=top_k,
            temperature=temperature,
        )
        metric_value = _safe_float(val_eval.get(best_metric_name), _safe_float(val_eval.get("top1_accuracy"), 0.0))
        checkpoint_payload = build_checkpoint_payload(
            state_dict=model.state_dict(),
            model_config=model_config,
            feature_encoder=feature_encoder,
            controller_ids=supported_controller_ids(),
            training_summary={
                "epoch": epoch,
                "train_loss": (sum(epoch_losses) / max(1, len(epoch_losses))) if epoch_losses else 0.0,
                "val_top1_accuracy": _safe_float(val_eval.get("top1_accuracy"), 0.0),
                "val_topk_accuracy": _safe_float(val_eval.get("topk_accuracy"), 0.0),
                "val_loss": _safe_float(val_eval.get("loss"), 0.0),
            },
            calibration={"temperature": temperature},
        )
        save_router_checkpoint(last_checkpoint, checkpoint_payload)
        if metric_value >= best_metric_value:
            best_metric_value = metric_value
            save_router_checkpoint(best_checkpoint, checkpoint_payload)

        progress_event = build_progress_event(
            run_id=chosen_run_id,
            component="p52_learned_router",
            phase="train",
            status="running" if epoch < epochs else "completed",
            step=epoch,
            epoch_or_iter=epoch,
            metrics={
                "train_loss": (sum(epoch_losses) / max(1, len(epoch_losses))) if epoch_losses else 0.0,
                "val_top1_accuracy": _safe_float(val_eval.get("top1_accuracy"), 0.0),
                "val_topk_accuracy": _safe_float(val_eval.get("topk_accuracy"), 0.0),
                "val_loss": _safe_float(val_eval.get("loss"), 0.0),
                "val_nll": _safe_float(val_eval.get("nll"), 0.0),
                "val_brier": _safe_float(val_eval.get("brier"), 0.0),
            },
            device_profile=runtime_profile,
            learner_device=learner_device,
            rollout_device=rollout_device,
            gpu_mem_mb=get_gpu_mem_mb(device),
        )
        append_progress_event(progress_path, progress_event)
        append_progress_event(progress_unified_path, progress_event)
        history_rows.append(progress_event)

    best_checkpoint_txt = run_dir / "best_checkpoint.txt"
    best_checkpoint_txt.write_text(str(best_checkpoint.resolve()) + "\n", encoding="utf-8")

    best_payload = torch.load(best_checkpoint, map_location=device)
    model.load_state_dict(best_payload.get("state_dict") if isinstance(best_payload, dict) else {})
    train_eval = evaluate_router_rows(
        model=model,
        rows=train_rows,
        device=device,
        batch_size=min(batch_size, max(8, len(train_rows))),
        top_k=top_k,
        temperature=temperature,
    )
    val_eval = evaluate_router_rows(
        model=model,
        rows=val_rows,
        device=device,
        batch_size=val_batch_size,
        top_k=top_k,
        temperature=temperature,
    )

    confusion_matrix_path = run_dir / "confusion_matrix.json"
    slice_eval_path = run_dir / "slice_eval.json"
    write_json(confusion_matrix_path, val_eval.get("confusion") or {})
    write_json(slice_eval_path, val_eval.get("slice_eval") or [])

    checkpoint_registry_entry = register_checkpoint(
        {
            "family": "learned_router",
            "training_mode": str(registry_cfg.get("training_mode") or "p52_learned_router"),
            "training_mode_category": str(registry_cfg.get("training_mode_category") or "mainline"),
            "source_run_id": chosen_run_id,
            "source_experiment_id": Path(str(config_path)).stem if config_path else "p52_learned_router",
            "seed_or_seed_group": str(train_seed),
            "device_profile": _runtime_profile_name(runtime_profile),
            "training_python": sys.executable,
            "artifact_path": str(best_checkpoint.resolve()),
            "status": "draft",
            "metrics_ref": str((run_dir / "metrics.json").resolve()),
            "lineage_refs": {
                "train_manifest_json": str((run_dir / "train_manifest.json").resolve()),
                "dataset_manifest_json": str(manifest_path.resolve()),
                "feature_encoder_json": str(feature_encoder_path.resolve()),
                "runtime_profile_json": str(runtime_profile_json.resolve()),
                "progress_unified_jsonl": str(progress_unified_path.resolve()),
                "confusion_matrix_json": str(confusion_matrix_path.resolve()),
                "slice_eval_json": str(slice_eval_path.resolve()),
            },
            "git_commit": _git_commit(repo_root),
            "notes": str(registry_cfg.get("notes") or "auto_registered_from_p52_learned_router_train"),
        }
    )

    metrics_payload = {
        "schema": "p52_learned_router_metrics_v1",
        "generated_at": _now_iso(),
        "run_id": chosen_run_id,
        "checkpoint_schema": CHECKPOINT_SCHEMA,
        "train_count": len(train_rows),
        "val_count": len(val_rows),
        "train_top1_accuracy": _safe_float(train_eval.get("top1_accuracy"), 0.0),
        "train_topk_accuracy": _safe_float(train_eval.get("topk_accuracy"), 0.0),
        "train_loss": _safe_float(train_eval.get("loss"), 0.0),
        "val_top1_accuracy": _safe_float(val_eval.get("top1_accuracy"), 0.0),
        "val_topk_accuracy": _safe_float(val_eval.get("topk_accuracy"), 0.0),
        "val_loss": _safe_float(val_eval.get("loss"), 0.0),
        "val_nll": _safe_float(val_eval.get("nll"), 0.0),
        "val_brier": _safe_float(val_eval.get("brier"), 0.0),
        "best_metric_name": best_metric_name,
        "best_metric_value": best_metric_value,
        "checkpoint_id": str(checkpoint_registry_entry.get("checkpoint_id") or ""),
        "learner_device": learner_device,
        "training_python": sys.executable,
    }
    train_manifest = {
        "schema": "p52_learned_router_train_manifest_v1",
        "generated_at": _now_iso(),
        "run_id": chosen_run_id,
        "run_dir": str(run_dir.resolve()),
        "dataset_manifest_json": str(manifest_path.resolve()),
        "feature_encoder_json": str(feature_encoder_path.resolve()),
        "config": cfg,
        "model_config": model_config,
        "runtime_profile_json": str(runtime_profile_json.resolve()),
        "progress_jsonl": str(progress_path.resolve()),
        "progress_unified_jsonl": str(progress_unified_path.resolve()),
        "metrics_json": str((run_dir / "metrics.json").resolve()),
        "best_checkpoint": str(best_checkpoint.resolve()),
        "last_checkpoint": str(last_checkpoint.resolve()),
        "best_checkpoint_txt": str(best_checkpoint_txt.resolve()),
        "checkpoint_id": str(checkpoint_registry_entry.get("checkpoint_id") or ""),
        "controller_ids": supported_controller_ids(),
    }
    write_json(run_dir / "metrics.json", metrics_payload)
    write_json(run_dir / "train_manifest.json", train_manifest)
    write_json(run_dir / "seeds_used.json", build_seeds_payload([str(train_seed)], seed_policy_version="p52.learned_router_train"))

    return {
        "status": "ok",
        "run_id": chosen_run_id,
        "run_dir": str(run_dir),
        "train_manifest_json": str((run_dir / "train_manifest.json").resolve()),
        "metrics_json": str((run_dir / "metrics.json").resolve()),
        "progress_jsonl": str(progress_path.resolve()),
        "progress_unified_jsonl": str(progress_unified_path.resolve()),
        "runtime_profile_json": str(runtime_profile_json.resolve()),
        "best_checkpoint": str(best_checkpoint.resolve()),
        "best_checkpoint_txt": str(best_checkpoint_txt.resolve()),
        "confusion_matrix_json": str(confusion_matrix_path.resolve()),
        "slice_eval_json": str(slice_eval_path.resolve()),
        "checkpoint_id": str(checkpoint_registry_entry.get("checkpoint_id") or ""),
        "learner_device": learner_device,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the P52 learned router.")
    parser.add_argument("--config", default="")
    parser.add_argument("--out-dir", default="")
    parser.add_argument("--run-id", default="")
    parser.add_argument("--dataset-manifest", default="")
    parser.add_argument("--quick", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = run_learned_router_train(
        config_path=(str(args.config).strip() or None),
        out_dir=(str(args.out_dir).strip() or None),
        run_id=str(args.run_id or ""),
        quick=bool(args.quick),
        dataset_manifest_path=(str(args.dataset_manifest).strip() or None),
    )
    print(json.dumps(summary, ensure_ascii=False))
    return 0 if str(summary.get("status") or "") == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
