from __future__ import annotations

if __package__ is None or __package__ == "":
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

import argparse
import json
import random
from pathlib import Path
from typing import Any

try:  # pragma: no cover - optional dependency
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None

from trainer.closed_loop.replay_manifest import build_seeds_payload, now_iso, now_stamp, read_json, read_jsonl, write_json
from trainer.world_model.dataset import build_world_model_dataset
from trainer.world_model.eval import run_world_model_eval
from trainer.world_model.losses import compute_world_model_losses
from trainer.world_model.model import WorldModelConfig, build_world_model, save_world_model_checkpoint
from trainer.world_model.planning_hook import run_world_model_assist_compare


def _require_torch():
    try:
        import torch
        from torch.utils.data import DataLoader, Dataset
    except Exception as exc:  # pragma: no cover - runtime guarded
        raise RuntimeError("PyTorch is required for trainer.world_model.train") from exc
    return torch, DataLoader, Dataset


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


def _seed_everything(seed: int) -> None:
    torch, _, _ = _require_torch()
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _build_rows(rows: list[dict[str, Any]], *, split: str, limit: int) -> list[dict[str, Any]]:
    selected = [row for row in rows if str(row.get("split") or "train").strip().lower() == split]
    if limit > 0:
        selected = selected[:limit]
    return selected


def _mean(values: list[float]) -> float:
    return float(sum(values) / max(1, len(values)))


def _make_dataset_class():
    torch, _DataLoader, Dataset = _require_torch()

    class _RowsDataset(Dataset):
        def __init__(self, rows: list[dict[str, Any]]) -> None:
            self.rows = list(rows)

        def __len__(self) -> int:
            return len(self.rows)

        def __getitem__(self, idx: int) -> dict[str, Any]:
            return dict(self.rows[idx])

    return _RowsDataset


def _collate(items: list[dict[str, Any]], *, device: Any):
    torch, _, _ = _require_torch()
    obs_t = torch.tensor([list(item.get("obs_t") or []) for item in items], dtype=torch.float32, device=device)
    obs_t1 = torch.tensor([list(item.get("obs_t1") or []) for item in items], dtype=torch.float32, device=device)
    action_id = torch.tensor([int(item.get("action_id") or 0) for item in items], dtype=torch.long, device=device)
    reward_t = torch.tensor([_safe_float(item.get("reward_t"), 0.0) for item in items], dtype=torch.float32, device=device)
    score_delta_t = torch.tensor([_safe_float(item.get("score_delta_t"), 0.0) for item in items], dtype=torch.float32, device=device)
    resource_delta_t = torch.tensor([list(item.get("resource_delta_t") or [0.0] * 5) for item in items], dtype=torch.float32, device=device)
    latent_rows = [list(item.get("latent_t1") or []) for item in items]
    latent_t1 = None
    if latent_rows and all(latent_rows) and len(set(len(row) for row in latent_rows)) == 1:
        latent_t1 = torch.tensor(latent_rows, dtype=torch.float32, device=device)
    return {
        "obs_t": obs_t,
        "obs_t1": obs_t1,
        "action_id": action_id,
        "reward_t": reward_t,
        "score_delta_t": score_delta_t,
        "resource_delta_t": resource_delta_t,
        "latent_t1": latent_t1,
    }


def _run_epoch(
    *,
    model: Any,
    loader: Any,
    optimizer: Any | None,
    device: Any,
    loss_weights: dict[str, Any],
) -> dict[str, float]:
    train_mode = optimizer is not None
    metric_store: dict[str, list[float]] = {}
    if train_mode:
        model.train()
    else:
        model.eval()

    torch, _, _ = _require_torch()
    context = torch.enable_grad() if train_mode else torch.no_grad()
    with context:
        for batch in loader:
            outputs = model(batch["obs_t"], batch["action_id"], batch["obs_t1"])
            losses = compute_world_model_losses(outputs=outputs, batch=batch, loss_weights=loss_weights)
            if train_mode and optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                losses["total_loss"].backward()
                optimizer.step()
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
                metric_store.setdefault(key, []).append(_safe_float(losses[key].detach().cpu().item(), 0.0))
    return {key: _mean(values) for key, values in metric_store.items()}


def run_world_model_train(
    *,
    config_path: str | Path,
    out_dir: str | Path | None = None,
    run_id: str = "",
    quick: bool = False,
    seed_override: int | None = None,
) -> dict[str, Any]:
    repo_root = Path(__file__).resolve().parents[2]
    cfg_path = Path(config_path)
    if not cfg_path.is_absolute():
        cfg_path = (repo_root / cfg_path).resolve()
    cfg = _read_yaml_or_json(cfg_path)
    train_cfg = cfg.get("train") if isinstance(cfg.get("train"), dict) else {}
    eval_cfg = cfg.get("eval") if isinstance(cfg.get("eval"), dict) else {}
    arena_cfg = cfg.get("arena_compare") if isinstance(cfg.get("arena_compare"), dict) else {}
    output_cfg = cfg.get("output") if isinstance(cfg.get("output"), dict) else {}

    seed = int(seed_override if seed_override is not None else int(train_cfg.get("seed") or 4501))
    _seed_everything(seed)
    dataset_summary = build_world_model_dataset(config_path=cfg_path, quick=quick)
    dataset_manifest_path = Path(str(dataset_summary.get("dataset_manifest_json") or ""))
    dataset_manifest = read_json(dataset_manifest_path)
    if not isinstance(dataset_manifest, dict):
        raise ValueError(f"dataset manifest unreadable: {dataset_manifest_path}")
    dataset_jsonl = Path(str(dataset_manifest.get("dataset_jsonl") or ""))
    if not dataset_jsonl.exists():
        raise ValueError(f"dataset jsonl missing: {dataset_jsonl}")

    rows = [row for row in read_jsonl(dataset_jsonl) if isinstance(row, dict)]
    max_train_samples = max(0, _safe_int(train_cfg.get("max_train_samples"), 0))
    max_val_samples = max(0, _safe_int(train_cfg.get("max_val_samples"), 0))
    if quick:
        max_train_samples = min(max_train_samples or 256, 256)
        max_val_samples = min(max_val_samples or 96, 96)
    train_rows = _build_rows(rows, split="train", limit=max_train_samples)
    val_rows = _build_rows(rows, split="val", limit=max_val_samples)
    if not train_rows:
        raise ValueError("world model train split is empty")
    if not val_rows:
        val_rows = train_rows[: max(1, min(32, len(train_rows)))]

    chosen_run_id = str(run_id or output_cfg.get("run_id") or now_stamp())
    out_root = (
        (repo_root / str(train_cfg.get("output_artifacts_root") or "docs/artifacts/p45/wm_train")).resolve()
        if out_dir is None
        else (Path(out_dir) if Path(out_dir).is_absolute() else (repo_root / out_dir).resolve())
    )
    run_dir = out_root / chosen_run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    progress_path = run_dir / "progress.jsonl"

    model_cfg = WorldModelConfig.from_mapping(
        {
            **(cfg.get("model") if isinstance(cfg.get("model"), dict) else {}),
            "input_dim": int(dataset_manifest.get("feature_dim") or ((cfg.get("model") or {}).get("input_dim") if isinstance(cfg.get("model"), dict) else 48) or 48),
            "action_vocab_size": int(dataset_manifest.get("action_vocab_size") or ((cfg.get("model") or {}).get("action_vocab_size") if isinstance(cfg.get("model"), dict) else 4096) or 4096),
        }
    )
    torch, DataLoader, _Dataset = _require_torch()
    device = torch.device("cpu")
    model = build_world_model(model_cfg).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=max(1e-5, _safe_float(train_cfg.get("lr"), 1e-3)),
        weight_decay=max(0.0, _safe_float(train_cfg.get("weight_decay"), 1e-5)),
    )

    DatasetCls = _make_dataset_class()
    batch_size = max(8, _safe_int(train_cfg.get("batch_size"), 64))
    if quick:
        batch_size = min(batch_size, 64)
    train_loader = DataLoader(DatasetCls(train_rows), batch_size=batch_size, shuffle=True, collate_fn=lambda items: _collate(items, device=device))
    val_loader = DataLoader(DatasetCls(val_rows), batch_size=batch_size, shuffle=False, collate_fn=lambda items: _collate(items, device=device))

    loss_weights = train_cfg.get("loss_weights") if isinstance(train_cfg.get("loss_weights"), dict) else {}
    epochs = max(1, _safe_int(train_cfg.get("epochs"), 3))
    if quick:
        epochs = min(epochs, 3)

    best_metric_name = str(train_cfg.get("checkpoint_metric") or "val_total_loss")
    best_metric = float("inf")
    best_checkpoint_path = ""
    best_epoch = 0
    best_val_metrics: dict[str, Any] = {}
    grad_clip_norm = max(0.0, _safe_float(train_cfg.get("grad_clip_norm"), 1.0))

    for epoch in range(1, epochs + 1):
        model.train()
        metric_store: dict[str, list[float]] = {}
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            outputs = model(batch["obs_t"], batch["action_id"], batch["obs_t1"])
            losses = compute_world_model_losses(outputs=outputs, batch=batch, loss_weights=loss_weights)
            losses["total_loss"].backward()
            if grad_clip_norm > 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            optimizer.step()
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
                metric_store.setdefault(key, []).append(_safe_float(losses[key].detach().cpu().item(), 0.0))
        train_metrics = {f"train_{key}": _mean(values) for key, values in metric_store.items()}
        val_metrics = {f"val_{key}": value for key, value in _run_epoch(model=model, loader=val_loader, optimizer=None, device=device, loss_weights=loss_weights).items()}

        epoch_payload = {
            "schema": "p45_world_model_train_progress_v1",
            "generated_at": now_iso(),
            "run_id": chosen_run_id,
            "epoch": epoch,
            "metrics": {**train_metrics, **val_metrics},
        }
        _append_jsonl(progress_path, epoch_payload)

        current_metric = _safe_float(
            epoch_payload["metrics"].get(best_metric_name),
            _safe_float(epoch_payload["metrics"].get("val_total_loss"), 0.0),
        )
        last_checkpoint_path = save_world_model_checkpoint(
            path=run_dir / "last.pt",
            model=model,
            optimizer=optimizer,
            config=model_cfg,
            extra={
                "epoch": epoch,
                "config_path": str(cfg_path),
                "dataset_manifest": str(dataset_manifest_path),
                "dataset_summary": dataset_summary,
                "train_metrics": train_metrics,
                "val_metrics": val_metrics,
            },
        )
        if current_metric <= best_metric:
            best_metric = current_metric
            best_epoch = epoch
            best_val_metrics = dict(val_metrics)
            best_checkpoint_path = save_world_model_checkpoint(
                path=run_dir / "best.pt",
                model=model,
                optimizer=optimizer,
                config=model_cfg,
                extra={
                    "epoch": epoch,
                    "config_path": str(cfg_path),
                    "dataset_manifest": str(dataset_manifest_path),
                    "dataset_summary": dataset_summary,
                    "train_metrics": train_metrics,
                    "val_metrics": val_metrics,
                },
            )

    if not best_checkpoint_path:
        best_checkpoint_path = str((run_dir / "last.pt").resolve())

    (run_dir / "best_checkpoint.txt").write_text(str(best_checkpoint_path) + "\n", encoding="utf-8")
    loss_weights_path = run_dir / "reward_config_or_loss_weights.json"
    write_json(loss_weights_path, {"loss_weights": loss_weights})

    dataset_seeds_payload = read_json(Path(str(dataset_summary.get("seeds_used_json") or "")))
    dataset_seeds = dataset_seeds_payload.get("seeds") if isinstance(dataset_seeds_payload, dict) and isinstance(dataset_seeds_payload.get("seeds"), list) else []
    seeds_payload = build_seeds_payload([str(seed_token) for seed_token in dataset_seeds], seed_policy_version="p45.world_model.train")
    seeds_payload["metadata"] = {
        "training_seed": int(seed),
        "dataset_seed_count": len(dataset_seeds),
    }
    write_json(run_dir / "seeds_used.json", seeds_payload)

    eval_summary = {}
    if best_checkpoint_path:
        eval_summary = run_world_model_eval(
            config_path=cfg_path,
            checkpoint_path=best_checkpoint_path,
            dataset_manifest_path=dataset_manifest_path,
            run_id=f"{chosen_run_id}-eval",
            quick=quick,
        )

    assist_summary = {}
    if bool((cfg.get("planning") or {}).get("enabled", True)) and bool(arena_cfg.get("enabled", False)) and best_checkpoint_path:
        assist_summary = run_world_model_assist_compare(
            config_path=cfg_path,
            checkpoint_path=best_checkpoint_path,
            run_id=f"{chosen_run_id}-assist",
            quick=quick,
        )

    metrics_payload = {
        "schema": "p45_world_model_train_metrics_v1",
        "generated_at": now_iso(),
        "run_id": chosen_run_id,
        "best_epoch": best_epoch,
        "best_metric_name": best_metric_name,
        "best_metric": best_metric,
        "best_checkpoint": best_checkpoint_path,
        "last_checkpoint": str((run_dir / "last.pt").resolve()),
        "train_samples": len(train_rows),
        "val_samples": len(val_rows),
        "dataset_manifest": str(dataset_manifest_path),
        "eval_summary": eval_summary,
        "assist_summary": assist_summary,
        "best_val_metrics": best_val_metrics,
    }
    manifest_payload = {
        "schema": "p45_world_model_train_manifest_v1",
        "generated_at": now_iso(),
        "run_id": chosen_run_id,
        "status": "ok",
        "config_path": str(cfg_path),
        "run_dir": str(run_dir),
        "dataset_summary": dataset_summary,
        "dataset_manifest": str(dataset_manifest_path),
        "checkpoints": {
            "best": best_checkpoint_path,
            "last": str((run_dir / "last.pt").resolve()),
            "best_checkpoint_txt": str((run_dir / "best_checkpoint.txt").resolve()),
        },
        "loss_weights_json": str(loss_weights_path.resolve()),
        "progress_jsonl": str(progress_path.resolve()),
        "metrics_json": str((run_dir / "metrics.json").resolve()),
        "eval_summary": eval_summary,
        "assist_summary": assist_summary,
    }
    write_json(run_dir / "metrics.json", metrics_payload)
    write_json(run_dir / "train_manifest.json", manifest_payload)

    return {
        "status": "ok",
        "run_id": chosen_run_id,
        "run_dir": str(run_dir),
        "train_manifest_json": str(run_dir / "train_manifest.json"),
        "metrics_json": str(run_dir / "metrics.json"),
        "progress_jsonl": str(progress_path),
        "best_checkpoint": best_checkpoint_path,
        "eval_summary": eval_summary,
        "assist_summary": assist_summary,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a P45 latent world model.")
    parser.add_argument("--config", default="configs/experiments/p45_world_model_smoke.yaml")
    parser.add_argument("--out-dir", default="")
    parser.add_argument("--run-id", default="")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--seed", type=int, default=-1)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = run_world_model_train(
        config_path=args.config,
        out_dir=(args.out_dir if str(args.out_dir).strip() else None),
        run_id=str(args.run_id or ""),
        quick=bool(args.quick),
        seed_override=(args.seed if int(args.seed) >= 0 else None),
    )
    print(json.dumps(summary, ensure_ascii=False))
    return 0 if str(summary.get("status")) == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
