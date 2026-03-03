from __future__ import annotations

if __package__ is None or __package__ == "":
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

import argparse
import csv
import json
import random
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:  # pragma: no cover - optional in some local environments
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None

from trainer.experiments.ssl_dataset import build_ssl_dataloaders
from trainer.models.ssl_state_encoder import SSLProjectionHead, StateEncoder, StateEncoderConfig


def _require_torch():
    try:
        import torch
        import torch.nn.functional as F
    except Exception as exc:  # pragma: no cover - runtime dependency
        raise RuntimeError("PyTorch is required for trainer.experiments.ssl_trainer") from exc
    return torch, F


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def _seed_to_int(seed: int | str) -> int:
    token = str(seed)
    try:
        return int(token)
    except Exception:
        return int.from_bytes(token.encode("utf-8"), "little", signed=False) % (2**31 - 1)


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


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as fp:
        fp.write(json.dumps(payload, ensure_ascii=False) + "\n")


def run_ssl_pretrain(
    *,
    config_path: str | Path,
    out_dir: str | Path | None = None,
    seed_override: int | None = None,
    max_samples_override: int | None = None,
    quiet: bool = False,
) -> dict[str, Any]:
    torch, F = _require_torch()
    repo_root = Path(__file__).resolve().parents[2]
    cfg_path = (repo_root / str(config_path)).resolve() if not Path(config_path).is_absolute() else Path(config_path)
    cfg = _read_yaml_or_json(cfg_path)

    data_cfg = cfg.get("data") if isinstance(cfg.get("data"), dict) else {}
    model_cfg = cfg.get("model") if isinstance(cfg.get("model"), dict) else {}
    train_cfg = cfg.get("training") if isinstance(cfg.get("training"), dict) else {}
    output_cfg = cfg.get("output") if isinstance(cfg.get("output"), dict) else {}

    seed = int(seed_override if seed_override is not None else int(train_cfg.get("seed") or 3701))
    seeded = _seed_to_int(seed)
    random.seed(seeded)
    torch.manual_seed(seeded)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seeded)

    built = build_ssl_dataloaders(
        config=cfg,
        seed=seeded,
        repo_root=repo_root,
        max_samples_override=max_samples_override,
    )
    train_loader = built["train_loader"]
    val_loader = built["val_loader"]
    input_dim = int(built["input_dim"])
    if input_dim <= 0:
        raise RuntimeError("ssl pretrain input_dim must be > 0")

    device_name = str(train_cfg.get("device") or "auto").lower()
    if device_name == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_name)

    encoder = StateEncoder(
        StateEncoderConfig(
            input_dim=input_dim,
            latent_dim=int(model_cfg.get("latent_dim") or 64),
            hidden_dim=int(model_cfg.get("hidden_dim") or 128),
            dropout=float(model_cfg.get("dropout") or 0.1),
        )
    ).to(device)
    projection = SSLProjectionHead(
        input_dim=int(model_cfg.get("latent_dim") or 64),
        projection_dim=int(model_cfg.get("projection_dim") or 64),
        dropout=float(model_cfg.get("dropout") or 0.1),
    ).to(device)
    optimizer = torch.optim.AdamW(
        list(encoder.parameters()) + list(projection.parameters()),
        lr=float(train_cfg.get("lr") or 1e-3),
        weight_decay=float(train_cfg.get("weight_decay") or 1e-4),
    )

    epochs = int(train_cfg.get("epochs") or 1)
    log_every = int(train_cfg.get("log_every") or 20)
    temperature = float(train_cfg.get("temperature") or 0.15)

    if out_dir is not None:
        run_dir = Path(out_dir).resolve()
    else:
        artifacts_root = str(output_cfg.get("artifacts_root") or "docs/artifacts/p37/ssl_pretrain")
        run_dir = (repo_root / artifacts_root / _now_stamp()).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    progress_path = run_dir / "progress.jsonl"
    curve_path = run_dir / "loss_curve.csv"
    metrics_path = run_dir / "metrics.json"

    def _ssl_loss(obs_a, obs_b) -> tuple[Any, float]:
        z_a = F.normalize(projection(encoder(obs_a)), dim=-1)
        z_b = F.normalize(projection(encoder(obs_b)), dim=-1)
        logits = torch.matmul(z_a, z_b.transpose(0, 1)) / max(1e-6, temperature)
        labels = torch.arange(logits.shape[0], device=logits.device)
        loss = 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.transpose(0, 1), labels))
        # embedding collapse diagnostic (larger is healthier for tiny smoke runs).
        emb_std = float(z_a.std(dim=0).mean().item())
        return loss, emb_std

    def _evaluate(loader) -> tuple[float, float, float]:
        encoder.eval()
        projection.eval()
        total_loss = 0.0
        total_pos_cos = 0.0
        total_emb_std = 0.0
        total_count = 0
        with torch.no_grad():
            for batch in loader:
                obs = batch["obs"].to(device)
                next_obs = batch["next_obs"].to(device)
                z_a = F.normalize(projection(encoder(obs)), dim=-1)
                z_b = F.normalize(projection(encoder(next_obs)), dim=-1)
                logits = torch.matmul(z_a, z_b.transpose(0, 1)) / max(1e-6, temperature)
                labels = torch.arange(logits.shape[0], device=logits.device)
                loss = 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.transpose(0, 1), labels))
                pos_cos = float((z_a * z_b).sum(dim=-1).mean().item())
                emb_std = float(z_a.std(dim=0).mean().item())
                n = int(obs.shape[0])
                total_count += n
                total_loss += float(loss.item()) * n
                total_pos_cos += pos_cos * n
                total_emb_std += emb_std * n
        denom = max(1, total_count)
        return (
            float(total_loss / denom),
            float(total_pos_cos / denom),
            float(total_emb_std / denom),
        )

    started = time.time()
    global_step = 0
    history: list[dict[str, Any]] = []
    best_val_loss = float("inf")
    best_ckpt = run_dir / "ssl_encoder_best.pt"

    for epoch in range(1, epochs + 1):
        encoder.train()
        projection.train()
        epoch_loss = 0.0
        epoch_emb_std = 0.0
        epoch_items = 0
        for batch in train_loader:
            global_step += 1
            obs = batch["obs"].to(device)
            next_obs = batch["next_obs"].to(device)
            loss, emb_std = _ssl_loss(obs, next_obs)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            n = int(obs.shape[0])
            epoch_items += n
            epoch_loss += float(loss.item()) * n
            epoch_emb_std += float(emb_std) * n

            if global_step == 1 or global_step % max(1, log_every) == 0:
                _append_jsonl(
                    progress_path,
                    {
                        "schema": "p37_ssl_progress_v1",
                        "ts": _now_iso(),
                        "stage": "train",
                        "epoch": epoch,
                        "step": global_step,
                        "train_loss": float(epoch_loss / max(1, epoch_items)),
                        "embedding_std": float(epoch_emb_std / max(1, epoch_items)),
                        "lr": float(optimizer.param_groups[0]["lr"]),
                    },
                )

        train_loss = float(epoch_loss / max(1, epoch_items))
        train_emb_std = float(epoch_emb_std / max(1, epoch_items))
        val_loss, val_pos_cos, val_emb_std = _evaluate(val_loader)
        epoch_row = {
            "epoch": epoch,
            "step": global_step,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_pos_cos": val_pos_cos,
            "train_embedding_std": train_emb_std,
            "val_embedding_std": val_emb_std,
            "lr": float(optimizer.param_groups[0]["lr"]),
            "elapsed_sec": float(time.time() - started),
        }
        history.append(epoch_row)
        _append_jsonl(
            progress_path,
            {
                "schema": "p37_ssl_progress_v1",
                "ts": _now_iso(),
                "stage": "epoch_done",
                **epoch_row,
            },
        )
        ckpt = run_dir / f"ssl_encoder_epoch{epoch}.pt"
        torch.save(
            {
                "epoch": epoch,
                "step": global_step,
                "seed": seed,
                "encoder_state_dict": encoder.state_dict(),
                "projection_state_dict": projection.state_dict(),
                "input_dim": input_dim,
                "model_cfg": model_cfg,
            },
            ckpt,
        )
        if val_loss <= best_val_loss:
            best_val_loss = float(val_loss)
            torch.save(
                {
                    "epoch": epoch,
                    "step": global_step,
                    "seed": seed,
                    "encoder_state_dict": encoder.state_dict(),
                    "projection_state_dict": projection.state_dict(),
                    "input_dim": input_dim,
                    "model_cfg": model_cfg,
                    "best_val_loss": best_val_loss,
                },
                best_ckpt,
            )

        if not quiet:
            print(
                "[p37-ssl] epoch={epoch} train_loss={train:.6f} val_loss={val:.6f} val_pos_cos={cos:.4f} emb_std={std:.4f}".format(
                    epoch=epoch,
                    train=train_loss,
                    val=val_loss,
                    cos=val_pos_cos,
                    std=val_emb_std,
                )
            )

    with curve_path.open("w", encoding="utf-8", newline="") as fp:
        fieldnames = [
            "epoch",
            "step",
            "train_loss",
            "val_loss",
            "val_pos_cos",
            "train_embedding_std",
            "val_embedding_std",
            "lr",
            "elapsed_sec",
        ]
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history)

    final = history[-1] if history else {}
    summary = {
        "schema": "p37_ssl_pretrain_summary_v1",
        "status": "ok",
        "generated_at": _now_iso(),
        "config_path": str(cfg_path),
        "run_dir": str(run_dir),
        "seed": seed,
        "sample_count": int(built.get("sample_count") or 0),
        "train_count": int(built.get("train_count") or 0),
        "val_count": int(built.get("val_count") or 0),
        "input_dim": input_dim,
        "epochs": len(history),
        "steps": global_step,
        "dataset_meta": built.get("dataset_meta") or {},
        "materialize_meta": built.get("materialize_meta") or {},
        "best_checkpoint": str(best_ckpt),
        "final_metrics": {
            "train_loss": float(final.get("train_loss") or 0.0),
            "val_loss": float(final.get("val_loss") or 0.0),
            "val_pos_cos": float(final.get("val_pos_cos") or 0.0),
            "train_embedding_std": float(final.get("train_embedding_std") or 0.0),
            "val_embedding_std": float(final.get("val_embedding_std") or 0.0),
            "lr": float(final.get("lr") or 0.0),
        },
    }
    _write_json(metrics_path, summary)
    return summary


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="P37 SSL state encoder pretraining (next-step contrastive).")
    p.add_argument("--config", default="configs/experiments/p37_ssl_pretrain.yaml")
    p.add_argument("--out-dir", default="")
    p.add_argument("--seed", type=int, default=-1)
    p.add_argument("--max-samples", type=int, default=0)
    p.add_argument("--quiet", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    summary = run_ssl_pretrain(
        config_path=args.config,
        out_dir=(args.out_dir if str(args.out_dir).strip() else None),
        seed_override=(int(args.seed) if int(args.seed) >= 0 else None),
        max_samples_override=(int(args.max_samples) if int(args.max_samples) > 0 else None),
        quiet=bool(args.quiet),
    )
    print(json.dumps({"status": summary.get("status"), "run_dir": summary.get("run_dir")}, ensure_ascii=False))
    return 0 if str(summary.get("status")) == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
