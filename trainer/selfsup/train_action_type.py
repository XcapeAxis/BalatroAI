from __future__ import annotations

if __package__ is None or __package__ == "":
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

import argparse
import csv
import json
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:  # pragma: no cover
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None

from trainer.selfsup.data import (
    build_samples_from_trajectories,
    load_trajectories_from_sources,
    parse_source_tokens,
)
from trainer.selfsup.tasks import SelfSupActionTypeTask, build_action_type_batch, load_dataset_rows


def _require_torch():
    try:
        import torch
        from torch.utils.data import DataLoader, Dataset
    except Exception as exc:
        raise RuntimeError("PyTorch is required for train_action_type.py") from exc
    return torch, Dataset, DataLoader


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


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as fp:
        for row in rows:
            fp.write(json.dumps(row, ensure_ascii=False) + "\n")


def _parse_sources_from_cfg(data_cfg: dict[str, Any]) -> list[str]:
    out: list[str] = []
    raw = data_cfg.get("sources")
    if not isinstance(raw, list):
        return out
    for item in raw:
        if isinstance(item, dict):
            kind = str(item.get("type") or item.get("kind") or "auto").strip().lower()
            path = str(item.get("path") or "").strip()
            if path:
                out.append(f"{kind}:{path}")
        else:
            token = str(item).strip()
            if token:
                out.append(token)
    return out


def _materialize_rows_from_config(
    *,
    repo_root: Path,
    data_cfg: dict[str, Any],
    max_samples_override: int | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    dataset_path_raw = str(data_cfg.get("dataset_path") or data_cfg.get("dataset_out") or "").strip()
    max_samples = int(max_samples_override if max_samples_override is not None else int(data_cfg.get("max_samples") or 0))
    if dataset_path_raw:
        dataset_path = (repo_root / dataset_path_raw).resolve() if not Path(dataset_path_raw).is_absolute() else Path(dataset_path_raw)
        if dataset_path.exists():
            rows = load_dataset_rows(dataset_path, max_samples=max_samples)
            return rows, {"dataset_path": str(dataset_path), "source": "existing_dataset"}

    source_tokens = _parse_sources_from_cfg(data_cfg)
    if not source_tokens:
        raise RuntimeError("action_type training requires data.dataset_path or data.sources")

    specs = parse_source_tokens(source_tokens)
    trajectories, source_stats = load_trajectories_from_sources(
        repo_root=repo_root,
        sources=specs,
        max_trajectories_per_source=max(1, int(data_cfg.get("max_trajectories_per_source") or 20)),
        require_steps=True,
    )
    samples = build_samples_from_trajectories(
        trajectories,
        lookahead_k=max(1, int(data_cfg.get("lookahead_k") or 3)),
        max_samples=max_samples,
    )
    rows = [s.to_dict() for s in samples]
    if dataset_path_raw:
        dataset_path = (repo_root / dataset_path_raw).resolve() if not Path(dataset_path_raw).is_absolute() else Path(dataset_path_raw)
        _write_jsonl(dataset_path, rows)
    return rows, {"dataset_path": dataset_path_raw, "source": "materialized_from_sources", "source_stats": source_stats}


def run_train_action_type(
    *,
    config_path: str | Path,
    out_dir: str | Path | None = None,
    seed_override: int | None = None,
    max_samples_override: int | None = None,
    quiet: bool = False,
) -> dict[str, Any]:
    torch, Dataset, DataLoader = _require_torch()
    repo_root = Path(__file__).resolve().parents[2]
    cfg_path = (repo_root / str(config_path)).resolve() if not Path(config_path).is_absolute() else Path(config_path)
    cfg = _read_yaml_or_json(cfg_path)

    data_cfg = cfg.get("data") if isinstance(cfg.get("data"), dict) else {}
    model_cfg = cfg.get("model") if isinstance(cfg.get("model"), dict) else {}
    train_cfg = cfg.get("training") if isinstance(cfg.get("training"), dict) else {}
    output_cfg = cfg.get("output") if isinstance(cfg.get("output"), dict) else {}

    seed = int(seed_override if seed_override is not None else int(train_cfg.get("seed") or 3602))
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    rows, dataset_meta = _materialize_rows_from_config(
        repo_root=repo_root,
        data_cfg=data_cfg,
        max_samples_override=max_samples_override,
    )
    batch = build_action_type_batch(rows)
    if not batch.features_t:
        raise RuntimeError("action_type dataset is empty")

    indices = list(range(len(batch.features_t)))
    random.Random(seed).shuffle(indices)
    val_ratio = float(train_cfg.get("val_ratio") or 0.2)
    cut = int(len(indices) * (1.0 - val_ratio))
    cut = max(1, min(cut, len(indices) - 1))
    train_idx = indices[:cut]
    val_idx = indices[cut:]

    class _DS(Dataset):
        def __init__(self, idxs: list[int]) -> None:
            self.idxs = idxs

        def __len__(self) -> int:
            return len(self.idxs)

        def __getitem__(self, i: int) -> dict[str, Any]:
            j = self.idxs[i]
            return {"x_t": batch.features_t[j], "x_tp1": batch.features_tp1[j], "y": batch.labels[j]}

    def _collate(items: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "x_t": torch.tensor([x["x_t"] for x in items], dtype=torch.float32),
            "x_tp1": torch.tensor([x["x_tp1"] for x in items], dtype=torch.float32),
            "y": torch.tensor([x["y"] for x in items], dtype=torch.long),
        }

    batch_size = int(train_cfg.get("batch_size") or 32)
    train_loader = DataLoader(_DS(train_idx), batch_size=batch_size, shuffle=True, collate_fn=_collate)
    val_loader = DataLoader(_DS(val_idx), batch_size=batch_size, shuffle=False, collate_fn=_collate)

    device_name = str(train_cfg.get("device") or "auto").lower()
    device = torch.device("cuda" if (device_name == "auto" and torch.cuda.is_available()) else ("cpu" if device_name == "auto" else device_name))

    task = SelfSupActionTypeTask(
        input_dim=len(batch.features_t[0]),
        num_classes=max(2, len(batch.label_vocab)),
        latent_dim=int(model_cfg.get("latent_dim") or 64),
        hidden_dim=int(model_cfg.get("hidden_dim") or 128),
        dropout=float(model_cfg.get("dropout") or 0.1),
    ).to(device)
    optimizer = torch.optim.AdamW(task.parameters(), lr=float(train_cfg.get("lr") or 1e-3), weight_decay=float(train_cfg.get("weight_decay") or 1e-4))

    if out_dir is not None:
        run_dir = Path(out_dir).resolve()
    else:
        artifacts_root = str(output_cfg.get("artifacts_root") or "docs/artifacts/p36/action_type")
        run_dir = (repo_root / artifacts_root / _now_stamp()).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    progress_jsonl = run_dir / "progress.jsonl"
    loss_curve_csv = run_dir / "loss_curve.csv"
    metrics_json = run_dir / "metrics.json"

    epochs = int(train_cfg.get("epochs") or 1)
    log_every = int(train_cfg.get("log_every") or 10)
    global_step = 0
    loss_rows: list[dict[str, Any]] = []

    def _append_progress(payload: dict[str, Any]) -> None:
        with progress_jsonl.open("a", encoding="utf-8", newline="\n") as fp:
            fp.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def _evaluate(loader) -> tuple[float, float]:
        task.model.eval()
        total_loss = 0.0
        total_count = 0
        total_correct = 0
        with torch.no_grad():
            for batch_t in loader:
                x_t = batch_t["x_t"].to(device)
                x_tp1 = batch_t["x_tp1"].to(device)
                y = batch_t["y"].to(device)
                logits = task.forward(x_t, x_tp1)
                loss = task.loss(logits, y)
                total_loss += float(loss.item()) * int(y.numel())
                total_count += int(y.numel())
                total_correct += int((logits.argmax(dim=-1) == y).sum().item())
        return (total_loss / max(1, total_count), float(total_correct / max(1, total_count)))

    for epoch in range(1, epochs + 1):
        task.model.train()
        epoch_loss = 0.0
        epoch_count = 0
        for batch_t in train_loader:
            global_step += 1
            x_t = batch_t["x_t"].to(device)
            x_tp1 = batch_t["x_tp1"].to(device)
            y = batch_t["y"].to(device)
            logits = task.forward(x_t, x_tp1)
            loss = task.loss(logits, y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item()) * int(y.numel())
            epoch_count += int(y.numel())
            if global_step == 1 or global_step % max(1, log_every) == 0:
                _append_progress(
                    {
                        "schema": "p36_action_type_progress_v1",
                        "ts": _now_iso(),
                        "epoch": epoch,
                        "step": global_step,
                        "train_loss": float(epoch_loss / max(1, epoch_count)),
                    }
                )

        train_loss = float(epoch_loss / max(1, epoch_count))
        val_loss, val_acc = _evaluate(val_loader)
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "step": global_step,
        }
        loss_rows.append(row)
        _append_progress({"schema": "p36_action_type_progress_v1", "ts": _now_iso(), **row, "stage": "epoch_done"})
        ckpt = run_dir / f"action_type_epoch{epoch}.pt"
        torch.save({"model": task.model.state_dict(), "epoch": epoch, "seed": seed, "label_vocab": batch.label_vocab}, ckpt)
        if not quiet:
            print(
                "[p36-action] epoch={epoch} train_loss={train:.6f} val_loss={val:.6f} val_acc={acc:.4f}".format(
                    epoch=epoch,
                    train=train_loss,
                    val=val_loss,
                    acc=val_acc,
                )
            )

    with loss_curve_csv.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=["epoch", "train_loss", "val_loss", "val_acc", "step"])
        writer.writeheader()
        writer.writerows(loss_rows)

    final = loss_rows[-1] if loss_rows else {"val_loss": 0.0, "val_acc": 0.0, "train_loss": 0.0}
    summary = {
        "schema": "p36_action_type_summary_v1",
        "status": "ok",
        "generated_at": _now_iso(),
        "config_path": str(cfg_path),
        "run_dir": str(run_dir),
        "dataset_meta": dataset_meta,
        "label_vocab": batch.label_vocab,
        "seed": seed,
        "sample_count": len(batch.features_t),
        "train_count": len(train_idx),
        "val_count": len(val_idx),
        "epochs": epochs,
        "steps": global_step,
        "final_metrics": {
            "train_loss": float(final.get("train_loss") or 0.0),
            "val_loss": float(final.get("val_loss") or 0.0),
            "val_acc": float(final.get("val_acc") or 0.0),
            "val_count": len(val_idx),
        },
    }
    _write_json(metrics_json, summary)
    return summary


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="P36 self-supervised task: inverse dynamics action type prediction.")
    p.add_argument("--config", default="configs/experiments/p36_selfsup_action_type.yaml")
    p.add_argument("--out-dir", default="")
    p.add_argument("--seed", type=int, default=-1)
    p.add_argument("--max-samples", type=int, default=0)
    p.add_argument("--quiet", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    summary = run_train_action_type(
        config_path=args.config,
        out_dir=(args.out_dir if args.out_dir else None),
        seed_override=(args.seed if args.seed >= 0 else None),
        max_samples_override=(args.max_samples if args.max_samples > 0 else None),
        quiet=bool(args.quiet),
    )
    print(json.dumps({"status": summary.get("status"), "run_dir": summary.get("run_dir"), "schema": summary.get("schema")}))
    return 0 if str(summary.get("status")) == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())

