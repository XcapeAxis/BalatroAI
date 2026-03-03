from __future__ import annotations

if __package__ is None or __package__ == "":
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

import json
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:  # pragma: no cover
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None

from trainer.self_supervised.datasets import (
    P33DatasetRow,
    build_dataset_rows,
    collect_trajectories,
    write_dataset_jsonl,
    write_dataset_stats,
)
from trainer.self_supervised.models import P33SelfSupMLP
from trainer.utils import setup_logger, warn_if_unstable_python


def _require_torch():
    try:
        import torch
        import torch.nn.functional as F
        from torch.utils.data import DataLoader, Dataset
    except Exception as exc:
        raise RuntimeError("PyTorch is required for p33 self-supervised training") from exc
    return torch, F, Dataset, DataLoader


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def _read_yaml_or_json(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in {".yaml", ".yml"}:
        if yaml is not None:
            payload = yaml.safe_load(text)
        else:
            try:
                payload = json.loads(text)
            except Exception:
                sidecar = path.with_suffix(".json")
                if sidecar.exists():
                    payload = json.loads(sidecar.read_text(encoding="utf-8"))
                else:
                    raise
    else:
        payload = json.loads(text)
    if not isinstance(payload, dict):
        raise ValueError(f"config must be mapping: {path}")
    return payload


def _split_rows(rows: list[P33DatasetRow], val_ratio: float, seed: int) -> tuple[list[P33DatasetRow], list[P33DatasetRow]]:
    if not rows:
        return [], []
    idx = list(range(len(rows)))
    random.Random(seed).shuffle(idx)
    cut = int(len(idx) * (1.0 - val_ratio))
    cut = max(1, min(cut, len(idx) - 1))
    train = [rows[i] for i in idx[:cut]]
    val = [rows[i] for i in idx[cut:]]
    return train, val


def run_p33_selfsup_training(
    *,
    config_path: str | Path,
    out_dir: str | Path | None = None,
    seed_override: int | None = None,
    max_samples_override: int | None = None,
    quiet: bool = False,
) -> dict[str, Any]:
    torch, F, Dataset, DataLoader = _require_torch()
    logger = setup_logger("trainer.self_supervised.train")
    warn_if_unstable_python(logger)

    repo_root = Path(__file__).resolve().parents[2]
    cfg_path = (repo_root / str(config_path)).resolve() if not Path(config_path).is_absolute() else Path(config_path)
    cfg = _read_yaml_or_json(cfg_path)

    data_cfg = cfg.get("data") if isinstance(cfg.get("data"), dict) else {}
    model_cfg = cfg.get("model") if isinstance(cfg.get("model"), dict) else {}
    train_cfg = cfg.get("training") if isinstance(cfg.get("training"), dict) else {}
    task_cfg = cfg.get("task") if isinstance(cfg.get("task"), dict) else {}
    out_cfg = cfg.get("output") if isinstance(cfg.get("output"), dict) else {}

    seed = int(seed_override if seed_override is not None else int(train_cfg.get("seed") or 33))
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    thresholds = tuple(float(x) for x in (task_cfg.get("bucket_thresholds") or [0.0, 120.0])[:2])
    if len(thresholds) != 2:
        thresholds = (0.0, 120.0)
    horizon_steps = int(task_cfg.get("horizon_steps") or 3)

    trajectories, source_summaries = collect_trajectories(repo_root=repo_root, data_cfg=data_cfg)
    max_samples = int(max_samples_override if max_samples_override is not None else int(data_cfg.get("max_samples") or 0))
    rows = build_dataset_rows(
        trajectories,
        bucket_thresholds=(float(thresholds[0]), float(thresholds[1])),
        horizon_steps=horizon_steps,
        max_samples=max_samples,
    )
    if not rows:
        raise RuntimeError("p33 selfsup dataset is empty")

    dataset_path = (repo_root / str(data_cfg.get("dataset_out") or "trainer_data/p33/selfsup_dataset.jsonl")).resolve()
    stats_path = (repo_root / str(data_cfg.get("stats_out") or "docs/artifacts/p33/selfsup_dataset_stats.json")).resolve()
    write_dataset_jsonl(dataset_path, rows)
    dataset_stats = write_dataset_stats(
        out_path=stats_path,
        rows=rows,
        source_summaries=source_summaries,
        dataset_path=dataset_path,
        horizon_steps=horizon_steps,
        bucket_thresholds=(float(thresholds[0]), float(thresholds[1])),
    )

    train_rows, val_rows = _split_rows(rows, float(train_cfg.get("val_ratio") or 0.2), seed)
    if not train_rows:
        raise RuntimeError("p33 selfsup train split is empty")
    if not val_rows:
        val_rows = train_rows[:1]

    class _Dataset(Dataset):
        def __init__(self, samples: list[P33DatasetRow]) -> None:
            self.samples = samples

        def __len__(self) -> int:
            return len(self.samples)

        def __getitem__(self, idx: int) -> dict[str, Any]:
            item = self.samples[idx]
            return {
                "x": item.features,
                "y": int(item.target_next_score_bucket),
            }

    def _collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "x": torch.tensor([x["x"] for x in batch], dtype=torch.float32),
            "y": torch.tensor([x["y"] for x in batch], dtype=torch.long),
        }

    batch_size = int(train_cfg.get("batch_size") or 64)
    train_loader = DataLoader(_Dataset(train_rows), batch_size=batch_size, shuffle=True, collate_fn=_collate)
    val_loader = DataLoader(_Dataset(val_rows), batch_size=batch_size, shuffle=False, collate_fn=_collate)

    device_name = str(train_cfg.get("device") or "auto").lower()
    if device_name == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_name)

    input_dim = len(train_rows[0].features)
    model = P33SelfSupMLP(
        input_dim=input_dim,
        hidden_dim=int(model_cfg.get("hidden_dim") or 64),
        dropout=float(model_cfg.get("dropout") or 0.1),
        num_classes=3,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg.get("lr") or 1e-3),
        weight_decay=float(train_cfg.get("weight_decay") or 1e-4),
    )

    if out_dir is not None:
        run_dir = Path(out_dir).resolve()
    else:
        run_root = str(out_cfg.get("run_root") or "trainer_runs/p33_selfsup")
        run_dir = (repo_root / run_root / _now_stamp()).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    progress_path = run_dir / "progress.jsonl"

    def _append_progress(payload: dict[str, Any]) -> None:
        with progress_path.open("a", encoding="utf-8", newline="\n") as fp:
            fp.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def _evaluate(loader) -> dict[str, float]:
        model.eval()
        total = 0
        loss_sum = 0.0
        correct = 0
        with torch.no_grad():
            for batch in loader:
                x = batch["x"].to(device)
                y = batch["y"].to(device)
                logits = model(x)
                loss = F.cross_entropy(logits, y)
                loss_sum += float(loss.detach().cpu()) * int(y.numel())
                pred = torch.argmax(logits, dim=-1)
                correct += int((pred == y).sum().detach().cpu())
                total += int(y.numel())
        return {
            "loss": float(loss_sum / max(1, total)),
            "acc": float(correct / max(1, total)),
            "count": int(total),
        }

    epochs = int(train_cfg.get("epochs") or 2)
    log_every = int(train_cfg.get("log_every") or 10)
    step = 0
    for epoch in range(1, epochs + 1):
        model.train()
        for batch in train_loader:
            step += 1
            x = batch["x"].to(device)
            y = batch["y"].to(device)
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            if step == 1 or (step % max(1, log_every) == 0):
                _append_progress(
                    {
                        "ts": _now_iso(),
                        "epoch": epoch,
                        "step": step,
                        "train_loss": float(loss.detach().cpu()),
                    }
                )
        train_eval = _evaluate(train_loader)
        val_eval = _evaluate(val_loader)
        _append_progress(
            {
                "ts": _now_iso(),
                "epoch": epoch,
                "step": step,
                "train_loss_epoch": train_eval["loss"],
                "train_acc_epoch": train_eval["acc"],
                "val_loss_epoch": val_eval["loss"],
                "val_acc_epoch": val_eval["acc"],
            }
        )
        ckpt = run_dir / f"selfsup_p33_epoch{epoch}.pt"
        torch.save({"model": model.state_dict(), "epoch": epoch}, ckpt)
        if not quiet:
            logger.info(
                "[p33-selfsup] epoch=%d train_loss=%.4f train_acc=%.4f val_loss=%.4f val_acc=%.4f",
                epoch,
                train_eval["loss"],
                train_eval["acc"],
                val_eval["loss"],
                val_eval["acc"],
            )

    final_val = _evaluate(val_loader)
    final_train = _evaluate(train_loader)
    summary = {
        "schema": "p33_selfsup_training_summary_v1",
        "status": "ok",
        "generated_at": _now_iso(),
        "config_path": str(cfg_path),
        "run_dir": str(run_dir),
        "dataset_path": str(dataset_path),
        "dataset_stats_path": str(stats_path),
        "seed": seed,
        "epochs": epochs,
        "steps": step,
        "train_metrics": final_train,
        "final_metrics": {
            "val_loss": final_val["loss"],
            "val_acc": final_val["acc"],
            "val_count": final_val["count"],
        },
        "dataset_stats": dataset_stats,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


__all__ = ["run_p33_selfsup_training"]
