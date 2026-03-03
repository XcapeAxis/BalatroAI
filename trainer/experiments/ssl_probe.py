from __future__ import annotations

if __package__ is None or __package__ == "":
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

import argparse
import json
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:  # pragma: no cover
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None

from trainer.experiments.ssl_dataset import build_ssl_dataloaders
from trainer.experiments.ssl_trainer import run_ssl_pretrain
from trainer.models.ssl_state_encoder import StateEncoder, StateEncoderConfig


def _require_torch():
    try:
        import torch
        import torch.nn.functional as F
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("PyTorch is required for trainer.experiments.ssl_probe") from exc
    return torch, F


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


def _write_md(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _seed_to_int(seed: int | str) -> int:
    token = str(seed)
    try:
        return int(token)
    except Exception:
        return int.from_bytes(token.encode("utf-8"), "little", signed=False) % (2**31 - 1)


def _train_linear_probe(
    *,
    encoder: Any,
    train_loader: Any,
    val_loader: Any,
    probe_cfg: dict[str, Any],
    device: Any,
    seeded: int,
):
    torch, F = _require_torch()
    random.seed(seeded)
    torch.manual_seed(seeded)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seeded)

    for p in encoder.parameters():
        p.requires_grad = False
    encoder.eval()

    probe_batch = next(iter(train_loader))
    probe_obs = probe_batch["obs"][:1].to(device)
    with torch.no_grad():
        probe_in_dim = int(encoder(probe_obs).shape[-1])
    classifier = torch.nn.Linear(probe_in_dim, 3).to(device)
    optimizer = torch.optim.AdamW(
        classifier.parameters(),
        lr=float(probe_cfg.get("lr") or 1e-2),
        weight_decay=float(probe_cfg.get("weight_decay") or 1e-5),
    )
    epochs = int(probe_cfg.get("epochs") or 8)

    def _eval(loader) -> tuple[float, float]:
        classifier.eval()
        total_loss = 0.0
        total = 0
        correct = 0
        with torch.no_grad():
            for batch in loader:
                obs = batch["obs"].to(device)
                y = batch["reward_bucket"].to(device)
                features = encoder(obs)
                logits = classifier(features)
                loss = F.cross_entropy(logits, y)
                n = int(y.shape[0])
                total += n
                total_loss += float(loss.item()) * n
                correct += int((logits.argmax(dim=-1) == y).sum().item())
        return float(total_loss / max(1, total)), float(correct / max(1, total))

    for _epoch in range(1, epochs + 1):
        classifier.train()
        for batch in train_loader:
            obs = batch["obs"].to(device)
            y = batch["reward_bucket"].to(device)
            with torch.no_grad():
                features = encoder(obs)
            logits = classifier(features)
            loss = F.cross_entropy(logits, y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

    val_loss, val_acc = _eval(val_loader)
    return {"val_loss": val_loss, "val_acc": val_acc}


def run_ssl_probe(
    *,
    config_path: str | Path,
    out_dir: str | Path | None = None,
    seed_override: int | None = None,
    max_samples_override: int | None = None,
    quiet: bool = False,
) -> dict[str, Any]:
    torch, _F = _require_torch()
    repo_root = Path(__file__).resolve().parents[2]
    cfg_path = (repo_root / str(config_path)).resolve() if not Path(config_path).is_absolute() else Path(config_path)
    cfg = _read_yaml_or_json(cfg_path)

    model_cfg = cfg.get("model") if isinstance(cfg.get("model"), dict) else {}
    train_cfg = cfg.get("training") if isinstance(cfg.get("training"), dict) else {}
    probe_cfg = cfg.get("probe") if isinstance(cfg.get("probe"), dict) else {}
    pretrain_cfg = cfg.get("ssl_pretrain") if isinstance(cfg.get("ssl_pretrain"), dict) else {}
    output_cfg = cfg.get("output") if isinstance(cfg.get("output"), dict) else {}

    seed = int(seed_override if seed_override is not None else int(train_cfg.get("seed") or 3702))
    seeded = _seed_to_int(seed)
    random.seed(seeded)
    torch.manual_seed(seeded)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seeded)

    if out_dir is not None:
        run_dir = Path(out_dir).resolve()
    else:
        artifacts_root = str(output_cfg.get("artifacts_root") or "docs/artifacts/p37/ssl_probe")
        run_dir = (repo_root / artifacts_root / _now_stamp()).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    pretrain_config = str(pretrain_cfg.get("config") or "configs/experiments/p37_ssl_pretrain.yaml")
    pretrain_run = run_ssl_pretrain(
        config_path=pretrain_config,
        out_dir=run_dir / "ssl_pretrain",
        seed_override=seed,
        max_samples_override=max_samples_override,
        quiet=True,
    )
    best_ckpt_path = Path(str(pretrain_run.get("best_checkpoint") or "")).resolve()
    if not best_ckpt_path.exists():
        raise RuntimeError(f"ssl probe requires best checkpoint from pretrain, got: {best_ckpt_path}")

    built = build_ssl_dataloaders(
        config=cfg,
        seed=seeded,
        repo_root=repo_root,
        max_samples_override=max_samples_override,
    )
    train_loader = built["train_loader"]
    val_loader = built["val_loader"]
    input_dim = int(built.get("input_dim") or 0)
    if input_dim <= 0:
        raise RuntimeError("ssl probe input_dim must be > 0")

    device_name = str(train_cfg.get("device") or "auto").lower()
    if device_name == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_name)

    baseline_encoder = StateEncoder(
        StateEncoderConfig(
            input_dim=input_dim,
            latent_dim=int(model_cfg.get("latent_dim") or 64),
            hidden_dim=int(model_cfg.get("hidden_dim") or 128),
            dropout=float(model_cfg.get("dropout") or 0.1),
        )
    ).to(device)

    ssl_encoder = StateEncoder(
        StateEncoderConfig(
            input_dim=input_dim,
            latent_dim=int(model_cfg.get("latent_dim") or 64),
            hidden_dim=int(model_cfg.get("hidden_dim") or 128),
            dropout=float(model_cfg.get("dropout") or 0.1),
        )
    ).to(device)
    ckpt = torch.load(best_ckpt_path, map_location=device)
    ssl_encoder.load_state_dict(ckpt.get("encoder_state_dict") or {}, strict=False)

    baseline = _train_linear_probe(
        encoder=baseline_encoder,
        train_loader=train_loader,
        val_loader=val_loader,
        probe_cfg=probe_cfg,
        device=device,
        seeded=seeded + 17,
    )
    warm = _train_linear_probe(
        encoder=ssl_encoder,
        train_loader=train_loader,
        val_loader=val_loader,
        probe_cfg=probe_cfg,
        device=device,
        seeded=seeded + 31,
    )
    delta_acc = float(warm["val_acc"] - baseline["val_acc"])

    summary = {
        "schema": "p37_ssl_probe_summary_v1",
        "status": "ok",
        "generated_at": _now_iso(),
        "config_path": str(cfg_path),
        "run_dir": str(run_dir),
        "seed": seed,
        "sample_count": int(built.get("sample_count") or 0),
        "train_count": int(built.get("train_count") or 0),
        "val_count": int(built.get("val_count") or 0),
        "ssl_pretrain": {
            "run_dir": pretrain_run.get("run_dir"),
            "best_checkpoint": str(best_ckpt_path),
            "final_metrics": pretrain_run.get("final_metrics") or {},
        },
        "probe": {
            "baseline_val_loss": float(baseline["val_loss"]),
            "baseline_val_acc": float(baseline["val_acc"]),
            "ssl_val_loss": float(warm["val_loss"]),
            "ssl_val_acc": float(warm["val_acc"]),
            "delta_val_acc": delta_acc,
        },
        "final_metrics": {
            "baseline_val_loss": float(baseline["val_loss"]),
            "baseline_val_acc": float(baseline["val_acc"]),
            "ssl_val_loss": float(warm["val_loss"]),
            "ssl_val_acc": float(warm["val_acc"]),
            "delta_val_acc": delta_acc,
        },
    }
    _write_json(run_dir / "probe_metrics.json", summary)
    _write_md(
        run_dir / "probe_report.md",
        "\n".join(
            [
                "# P37 SSL Probe Report",
                "",
                f"- run_dir: {run_dir}",
                f"- baseline_val_acc: {baseline['val_acc']:.4f}",
                f"- ssl_val_acc: {warm['val_acc']:.4f}",
                f"- delta_val_acc: {delta_acc:.4f}",
                "",
                "This is a preliminary frozen-encoder linear probe on reward-bucket prediction.",
            ]
        )
        + "\n",
    )
    if not quiet:
        print(
            "[p37-ssl-probe] baseline_acc={base:.4f} ssl_acc={ssl:.4f} delta={delta:.4f}".format(
                base=baseline["val_acc"],
                ssl=warm["val_acc"],
                delta=delta_acc,
            )
        )
    return summary


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="P37 SSL downstream linear probe.")
    p.add_argument("--config", default="configs/experiments/p37_ssl_probe.yaml")
    p.add_argument("--out-dir", default="")
    p.add_argument("--seed", type=int, default=-1)
    p.add_argument("--max-samples", type=int, default=0)
    p.add_argument("--quiet", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    summary = run_ssl_probe(
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
