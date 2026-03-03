from __future__ import annotations

if __package__ is None or __package__ == "":
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

import argparse
import csv
import json
import random
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:  # pragma: no cover - optional
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None

from trainer.experiments.selfsup_model import ReplaySelfSupModel
from trainer.experiments.selfsup_objectives import (
    build_action_vocab,
    build_training_rows,
    compute_objective_loss,
    load_replay_steps,
)
from trainer.replay.storage import build_replay_dataset


def _require_torch():
    try:
        import torch
        from torch.utils.data import DataLoader, Dataset
    except Exception as exc:
        raise RuntimeError("PyTorch is required for trainer.experiments.selfsup_train") from exc
    return torch, Dataset, DataLoader


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def _read_yaml_or_json(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8-sig")
    if path.suffix.lower() in {".yaml", ".yml"}:
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
        raise ValueError(f"config must be a mapping: {path}")
    return payload


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as fp:
        fp.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _is_replay_v1_config(cfg: dict[str, Any]) -> bool:
    schema = str(cfg.get("schema") or "").strip().lower()
    if schema.startswith("p36_selfsup_replay"):
        return True
    replay_cfg = cfg.get("replay")
    return isinstance(replay_cfg, dict) and bool(replay_cfg)


def _materialize_replay_dataset(
    *,
    repo_root: Path,
    cfg: dict[str, Any],
    run_dir: Path,
) -> tuple[Path, dict[str, Any]]:
    data_cfg = cfg.get("data") if isinstance(cfg.get("data"), dict) else {}
    replay_cfg = cfg.get("replay") if isinstance(cfg.get("replay"), dict) else {}

    dataset_path_raw = str(data_cfg.get("replay_steps_path") or "").strip()
    if dataset_path_raw:
        dataset_path = (repo_root / dataset_path_raw).resolve() if not Path(dataset_path_raw).is_absolute() else Path(dataset_path_raw)
        if dataset_path.exists():
            return dataset_path, {
                "source": "existing_replay_steps",
                "dataset_path": str(dataset_path),
            }

    real_roots_raw = replay_cfg.get("real_roots") if isinstance(replay_cfg.get("real_roots"), list) else []
    sim_roots_raw = replay_cfg.get("sim_roots") if isinstance(replay_cfg.get("sim_roots"), list) else []
    if not real_roots_raw and not sim_roots_raw:
        real_roots_raw = ["docs/artifacts/p32", "docs/artifacts/p13"]

    real_roots = [str((repo_root / str(p)).resolve()) if not Path(str(p)).is_absolute() else str(Path(str(p))) for p in real_roots_raw]
    sim_roots = [str((repo_root / str(p)).resolve()) if not Path(str(p)).is_absolute() else str(Path(str(p))) for p in sim_roots_raw]

    replay_out_raw = str(replay_cfg.get("out_dir") or "").strip()
    replay_out = (
        (run_dir / "replay_dataset").resolve()
        if not replay_out_raw
        else ((repo_root / replay_out_raw).resolve() if not Path(replay_out_raw).is_absolute() else Path(replay_out_raw))
    )

    replay_result = build_replay_dataset(
        real_roots=real_roots,
        sim_roots=sim_roots,
        out_dir=replay_out,
        max_episodes_per_source=max(0, int(replay_cfg.get("max_episodes_per_source") or 0)),
    )

    latest_dir = (repo_root / "docs/artifacts/p36/replay/latest").resolve()
    latest_dir.mkdir(parents=True, exist_ok=True)
    for fname in ("replay_steps.jsonl", "replay_summary.json", "replay_summary.md"):
        src = replay_out / fname
        if src.exists():
            shutil.copy2(src, latest_dir / fname)

    dataset_path = replay_out / "replay_steps.jsonl"
    if not dataset_path.exists():
        raise RuntimeError(f"replay dataset missing replay_steps.jsonl at {dataset_path}")
    return dataset_path, replay_result


def _train_replay_selfsup(
    *,
    config_path: Path,
    cfg: dict[str, Any],
    out_dir: str | Path | None,
    seed_override: int | None,
    max_steps_override: int | None,
    quiet: bool,
) -> dict[str, Any]:
    torch, Dataset, DataLoader = _require_torch()
    repo_root = Path(__file__).resolve().parents[2]

    output_cfg = cfg.get("output") if isinstance(cfg.get("output"), dict) else {}
    if out_dir is not None:
        run_dir = Path(out_dir).resolve()
    else:
        artifacts_root = str(output_cfg.get("artifacts_root") or "docs/artifacts/p36/selfsup_replay")
        run_dir = (repo_root / artifacts_root / _now_stamp()).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    training_cfg = cfg.get("training") if isinstance(cfg.get("training"), dict) else {}
    model_cfg = cfg.get("model") if isinstance(cfg.get("model"), dict) else {}
    data_cfg = cfg.get("data") if isinstance(cfg.get("data"), dict) else {}

    seed = int(seed_override if seed_override is not None else int(training_cfg.get("seed") or 3606))
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    replay_steps_path, replay_meta = _materialize_replay_dataset(repo_root=repo_root, cfg=cfg, run_dir=run_dir)

    max_samples = int(data_cfg.get("max_samples") or 0)
    valid_only = bool(data_cfg.get("valid_only", True))
    rows = load_replay_steps(replay_steps_path, max_samples=max_samples, valid_only=valid_only)
    if not rows:
        raise RuntimeError(f"no replay rows loaded from {replay_steps_path}")

    action_vocab = build_action_vocab(rows)
    objective_type = str(training_cfg.get("objective_type") or "hybrid").strip().lower()
    mask_ratio = float(training_cfg.get("mask_ratio") or 0.15)
    rng = random.Random(seed)
    training_rows = build_training_rows(rows, action_vocab=action_vocab, mask_ratio=mask_ratio, rng=rng)

    indices = list(range(len(training_rows)))
    random.Random(seed).shuffle(indices)
    val_ratio = float(training_cfg.get("val_ratio") or 0.2)
    cut = int(len(indices) * (1.0 - val_ratio))
    cut = max(1, min(cut, len(indices) - 1)) if len(indices) > 1 else 1
    train_idx = indices[:cut]
    val_idx = indices[cut:] if len(indices) > 1 else indices

    class _ReplayDS(Dataset):
        def __init__(self, idxs: list[int]):
            self.idxs = idxs

        def __len__(self) -> int:
            return len(self.idxs)

        def __getitem__(self, i: int) -> dict[str, Any]:
            return training_rows[self.idxs[i]]

    def _collate(items: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "input_action_ids": torch.tensor([x["input_action_id"] for x in items], dtype=torch.long),
            "target_action_ids": torch.tensor([x["target_action_id"] for x in items], dtype=torch.long),
            "mask_target": torch.tensor([x["mask_target"] for x in items], dtype=torch.float32),
            "numeric": torch.tensor([x["numeric"] for x in items], dtype=torch.float32),
            "next_delta_target": torch.tensor([x["next_delta_target"] for x in items], dtype=torch.float32),
        }

    batch_size = int(training_cfg.get("batch_size") or 32)
    train_loader = DataLoader(_ReplayDS(train_idx), batch_size=batch_size, shuffle=True, collate_fn=_collate)
    val_loader = DataLoader(_ReplayDS(val_idx), batch_size=batch_size, shuffle=False, collate_fn=_collate)

    numeric_dim = len(training_rows[0]["numeric"])
    model = ReplaySelfSupModel(
        action_vocab_size=len(action_vocab),
        numeric_dim=numeric_dim,
        action_embed_dim=int(model_cfg.get("action_embed_dim") or 16),
        hidden_dim=int(model_cfg.get("hidden_dim") or 96),
        latent_dim=int(model_cfg.get("latent_dim") or 64),
        dropout=float(model_cfg.get("dropout") or 0.1),
    )

    device_raw = str(training_cfg.get("device") or "auto").lower()
    if device_raw == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_raw)
    model = model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training_cfg.get("lr") or 1e-3),
        weight_decay=float(training_cfg.get("weight_decay") or 1e-4),
    )

    mask_weight = float(training_cfg.get("mask_weight") or 1.0)
    next_delta_weight = float(training_cfg.get("next_delta_weight") or 1.0)
    epochs = int(training_cfg.get("epochs") or 1)
    log_every = int(training_cfg.get("log_every") or 10)
    max_steps = int(max_steps_override if max_steps_override is not None else int(training_cfg.get("max_steps") or 0))

    progress_jsonl = run_dir / "progress.jsonl"
    loss_curve_csv = run_dir / "loss_curve.csv"
    metrics_json = run_dir / "summary.json"

    global_step = 0
    epoch_rows: list[dict[str, Any]] = []

    def _evaluate(loader) -> dict[str, float]:
        model.eval()
        total_loss = 0.0
        total_mask_loss = 0.0
        total_next_loss = 0.0
        total_count = 0
        mask_correct = 0
        mask_total = 0
        next_abs = 0.0
        with torch.no_grad():
            for batch in loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                losses = compute_objective_loss(
                    objective_type=objective_type,
                    model=model,
                    batch=batch,
                    torch_mod=torch,
                    mask_weight=mask_weight,
                    next_delta_weight=next_delta_weight,
                )
                n = int(batch["input_action_ids"].numel())
                total_count += n
                total_loss += float(losses["total"].item()) * n
                total_mask_loss += float(losses["mask_loss"].item()) * n
                total_next_loss += float(losses["next_delta_loss"].item()) * n
                if losses["logits"] is not None:
                    preds = losses["logits"].argmax(dim=-1)
                    mask_flags = batch["mask_target"] > 0.0
                    if bool(mask_flags.any()):
                        mask_correct += int((preds[mask_flags] == batch["target_action_ids"][mask_flags]).sum().item())
                        mask_total += int(mask_flags.sum().item())
                if losses["preds"] is not None:
                    next_abs += float((losses["preds"] - batch["next_delta_target"]).abs().sum().item())
        return {
            "loss": float(total_loss / max(1, total_count)),
            "mask_loss": float(total_mask_loss / max(1, total_count)),
            "next_delta_loss": float(total_next_loss / max(1, total_count)),
            "mask_acc": float(mask_correct / max(1, mask_total)),
            "next_delta_mae": float(next_abs / max(1, total_count)),
            "sample_count": int(total_count),
        }

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss_sum = 0.0
        epoch_mask_sum = 0.0
        epoch_next_sum = 0.0
        epoch_count = 0

        for batch in train_loader:
            global_step += 1
            batch = {k: v.to(device) for k, v in batch.items()}
            losses = compute_objective_loss(
                objective_type=objective_type,
                model=model,
                batch=batch,
                torch_mod=torch,
                mask_weight=mask_weight,
                next_delta_weight=next_delta_weight,
            )
            loss = losses["total"]
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            n = int(batch["input_action_ids"].numel())
            epoch_count += n
            epoch_loss_sum += float(loss.item()) * n
            epoch_mask_sum += float(losses["mask_loss"].item()) * n
            epoch_next_sum += float(losses["next_delta_loss"].item()) * n

            if global_step == 1 or global_step % max(1, log_every) == 0:
                _append_jsonl(
                    progress_jsonl,
                    {
                        "schema": "p36_selfsup_train_progress_v1",
                        "ts": _now_iso(),
                        "stage": "train",
                        "objective_type": objective_type,
                        "epoch": epoch,
                        "step": global_step,
                        "train_loss": float(epoch_loss_sum / max(1, epoch_count)),
                        "train_mask_loss": float(epoch_mask_sum / max(1, epoch_count)),
                        "train_next_delta_loss": float(epoch_next_sum / max(1, epoch_count)),
                    },
                )

            if max_steps > 0 and global_step >= max_steps:
                break

        train_metrics = {
            "loss": float(epoch_loss_sum / max(1, epoch_count)),
            "mask_loss": float(epoch_mask_sum / max(1, epoch_count)),
            "next_delta_loss": float(epoch_next_sum / max(1, epoch_count)),
        }
        val_metrics = _evaluate(val_loader)
        row = {
            "epoch": epoch,
            "step": global_step,
            "train_loss": train_metrics["loss"],
            "train_mask_loss": train_metrics["mask_loss"],
            "train_next_delta_loss": train_metrics["next_delta_loss"],
            "val_loss": val_metrics["loss"],
            "val_mask_loss": val_metrics["mask_loss"],
            "val_next_delta_loss": val_metrics["next_delta_loss"],
            "val_mask_acc": val_metrics["mask_acc"],
            "val_next_delta_mae": val_metrics["next_delta_mae"],
        }
        epoch_rows.append(row)

        _append_jsonl(
            progress_jsonl,
            {
                "schema": "p36_selfsup_train_progress_v1",
                "ts": _now_iso(),
                "stage": "epoch_done",
                "objective_type": objective_type,
                **row,
            },
        )

        ckpt_path = run_dir / f"selfsup_replay_epoch{epoch}.pt"
        torch.save(
            {
                "epoch": epoch,
                "step": global_step,
                "seed": seed,
                "action_vocab": action_vocab,
                "model_state": model.state_dict(),
                "objective_type": objective_type,
            },
            ckpt_path,
        )

        if not quiet:
            print(
                "[p36-selfsup-replay] epoch={epoch} train_loss={train:.6f} val_loss={val:.6f} "
                "val_mask_acc={mask_acc:.4f} val_next_delta_mae={mae:.4f}".format(
                    epoch=epoch,
                    train=row["train_loss"],
                    val=row["val_loss"],
                    mask_acc=row["val_mask_acc"],
                    mae=row["val_next_delta_mae"],
                )
            )

        if max_steps > 0 and global_step >= max_steps:
            break

    with loss_curve_csv.open("w", encoding="utf-8", newline="") as fp:
        fieldnames = [
            "epoch",
            "step",
            "train_loss",
            "train_mask_loss",
            "train_next_delta_loss",
            "val_loss",
            "val_mask_loss",
            "val_next_delta_loss",
            "val_mask_acc",
            "val_next_delta_mae",
        ]
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(epoch_rows)

    final_row = epoch_rows[-1] if epoch_rows else {}
    total_rows = load_replay_steps(replay_steps_path, max_samples=0, valid_only=False)
    valid_rows = load_replay_steps(replay_steps_path, max_samples=0, valid_only=True)
    invalid_fraction = 1.0 - (float(len(valid_rows)) / float(max(1, len(total_rows))))

    summary = {
        "schema": "p36_selfsup_train_summary_v1",
        "status": "ok",
        "generated_at": _now_iso(),
        "config_path": str(config_path),
        "run_dir": str(run_dir),
        "objective_type": objective_type,
        "seed": seed,
        "action_vocab_size": len(action_vocab),
        "sample_count": len(training_rows),
        "train_count": len(train_idx),
        "val_count": len(val_idx),
        "epochs": len(epoch_rows),
        "steps": global_step,
        "replay_steps_path": str(replay_steps_path),
        "replay_meta": replay_meta,
        "invalid_fraction": float(invalid_fraction),
        "final_metrics": {
            "train_loss": float(final_row.get("train_loss") or 0.0),
            "val_loss": float(final_row.get("val_loss") or 0.0),
            "val_mask_acc": float(final_row.get("val_mask_acc") or 0.0),
            "val_next_delta_mae": float(final_row.get("val_next_delta_mae") or 0.0),
            "val_mask_loss": float(final_row.get("val_mask_loss") or 0.0),
            "val_next_delta_loss": float(final_row.get("val_next_delta_loss") or 0.0),
        },
    }
    _write_json(metrics_json, summary)
    return summary


def run_selfsup_training(
    *,
    config_path: str | Path,
    out_dir: str | Path | None = None,
    seed_override: int | None = None,
    max_steps_override: int | None = None,
    run_name: str = "",
    quiet: bool = False,
) -> dict[str, Any]:
    repo_root = Path(__file__).resolve().parents[2]
    cfg_path = (repo_root / str(config_path)).resolve() if not Path(config_path).is_absolute() else Path(config_path)
    cfg = _read_yaml_or_json(cfg_path)

    # Backward-compatible path for existing P31 selfsup config and tooling.
    if not _is_replay_v1_config(cfg):
        from trainer.selfsup_train import run_selfsup_training as legacy_run_selfsup_training

        return legacy_run_selfsup_training(
            config_path=cfg_path,
            out_dir=out_dir,
            seed_override=seed_override,
            max_steps_override=max_steps_override,
            run_name=run_name,
            quiet=quiet,
        )

    return _train_replay_selfsup(
        config_path=cfg_path,
        cfg=cfg,
        out_dir=out_dir,
        seed_override=seed_override,
        max_steps_override=max_steps_override,
        quiet=quiet,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run self-supervised training (legacy P31 or P36 replay-v1).")
    parser.add_argument("--config", required=True, help="Path to YAML/JSON config.")
    parser.add_argument("--out-dir", default="", help="Optional output run directory.")
    parser.add_argument("--seed", type=int, default=0, help="Optional seed override.")
    parser.add_argument("--max-steps", type=int, default=0, help="Optional max-step override for smoke runs.")
    parser.add_argument("--quiet", action="store_true", help="Reduce console output.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = run_selfsup_training(
        config_path=args.config,
        out_dir=(args.out_dir if str(args.out_dir).strip() else None),
        seed_override=(int(args.seed) if int(args.seed) > 0 else None),
        max_steps_override=(int(args.max_steps) if int(args.max_steps) > 0 else None),
        quiet=bool(args.quiet),
    )
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
