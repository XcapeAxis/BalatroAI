from __future__ import annotations

if __package__ is None or __package__ == "":
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).resolve().parent.parent))

import argparse
import json
import random
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:  # pragma: no cover - optional in some environments
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None

from trainer.data.trajectory import (
    DecisionStep,
    Trajectory,
    load_trajectories_from_oracle_traces,
    load_trajectories_from_p13_drift_fixture,
)
from trainer.models.selfsup_encoder import SelfSupModelConfig, SelfSupMultiHeadModel
from trainer.utils import setup_logger, warn_if_unstable_python


def _require_torch():
    try:
        import torch
        import torch.nn.functional as F
        from torch.utils.data import DataLoader, Dataset
    except Exception as exc:
        raise RuntimeError("PyTorch is required for self-supervised training.") from exc
    return torch, F, Dataset, DataLoader


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def _read_config(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    suffix = path.suffix.lower()
    if suffix in {".yaml", ".yml"}:
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
                    raise RuntimeError(
                        f"PyYAML not available and no JSON sidecar found for config: {path}"
                    )
    else:
        payload = json.loads(text)
    if not isinstance(payload, dict):
        raise ValueError(f"selfsup config must be a mapping: {path}")
    return payload


def _state_hash_signal(state_hashes: dict[str, str]) -> float:
    if not state_hashes:
        return 0.0
    values: list[float] = []
    for value in state_hashes.values():
        token = str(value).strip().lower()
        if len(token) < 8:
            continue
        try:
            values.append((int(token[:8], 16) % 1000000) / 1000000.0)
        except Exception:
            continue
    if not values:
        return 0.0
    return float(sum(values) / len(values))


@dataclass
class SelfSupSample:
    phase: str
    action_type: str
    numeric: list[float]
    score_delta_target: float
    score_delta_mask: float
    hand_type: str
    hand_type_mask: float


def _step_to_sample(step: DecisionStep) -> SelfSupSample:
    score_target = float(step.score_delta if step.score_delta is not None else 0.0)
    reward = float(step.reward if step.reward is not None else 0.0)
    numeric = [
        _state_hash_signal(step.state_hashes),
        min(1.0, len(step.state_hashes) / 24.0),
        min(1.0, len(step.action_args) / 8.0),
        max(-2.0, min(2.0, reward / 200.0)),
        1.0 if step.done else 0.0,
        1.0 if step.action_type == "PLAY" else 0.0,
    ]
    return SelfSupSample(
        phase=str(step.phase or "UNKNOWN").upper(),
        action_type=str(step.action_type or "UNKNOWN").upper(),
        numeric=numeric,
        score_delta_target=score_target,
        score_delta_mask=1.0 if step.score_delta is not None else 0.0,
        hand_type=str(step.hand_type or "UNKNOWN").upper(),
        hand_type_mask=1.0 if step.hand_type is not None else 0.0,
    )


def _collect_trajectories(
    *,
    repo_root: Path,
    config: dict[str, Any],
    logger,
) -> list[Trajectory]:
    data_cfg = config.get("data") if isinstance(config.get("data"), dict) else {}
    source_cfgs = data_cfg.get("sources")
    if not isinstance(source_cfgs, list) or not source_cfgs:
        raise ValueError("config.data.sources must be a non-empty list")
    max_per_source = int(data_cfg.get("max_trajectories_per_source") or 16)
    require_steps = bool(data_cfg.get("require_steps", True))

    trajectories: list[Trajectory] = []
    for source in source_cfgs:
        if not isinstance(source, dict):
            continue
        source_type = str(source.get("type") or "").strip().lower()
        rel_path = str(source.get("path") or "").strip()
        if not rel_path:
            continue
        path = (repo_root / rel_path).resolve()
        if source_type == "oracle_traces":
            rows = load_trajectories_from_oracle_traces(
                path,
                max_trajectories=max_per_source,
                require_steps=require_steps,
            )
        elif source_type in {"p13_drift_fixture", "p13_fixture"}:
            rows = load_trajectories_from_p13_drift_fixture(
                path,
                max_trajectories=max_per_source,
                require_steps=False,
            )
            if require_steps:
                rows = [x for x in rows if x.steps]
        else:
            logger.warning("skip unsupported source type: %s", source_type)
            continue
        logger.info("loaded source=%s path=%s trajectories=%d", source_type, path, len(rows))
        trajectories.extend(rows)
    return trajectories


def _build_samples(trajectories: list[Trajectory], *, max_steps: int | None) -> list[SelfSupSample]:
    samples: list[SelfSupSample] = []
    for traj in trajectories:
        for step in traj.steps:
            samples.append(_step_to_sample(step))
            if max_steps is not None and len(samples) >= int(max_steps):
                return samples
    return samples


def _split_train_val(samples: list[SelfSupSample], val_ratio: float, seed: int) -> tuple[list[SelfSupSample], list[SelfSupSample]]:
    if not samples:
        return [], []
    idx = list(range(len(samples)))
    random.Random(seed).shuffle(idx)
    cut = int(len(idx) * (1.0 - val_ratio))
    cut = max(1, min(cut, len(idx) - 1))
    train = [samples[i] for i in idx[:cut]]
    val = [samples[i] for i in idx[cut:]]
    return train, val


def run_selfsup_training(
    *,
    config_path: str | Path,
    out_dir: str | Path | None = None,
    seed_override: int | None = None,
    max_steps_override: int | None = None,
    run_name: str = "",
    quiet: bool = False,
) -> dict[str, Any]:
    torch, F, Dataset, DataLoader = _require_torch()
    logger = setup_logger("trainer.selfsup_train")
    warn_if_unstable_python(logger)

    repo_root = Path(__file__).resolve().parents[1]
    cfg_path = (repo_root / str(config_path)).resolve() if not Path(config_path).is_absolute() else Path(config_path)
    cfg = _read_config(cfg_path)
    train_cfg = cfg.get("training") if isinstance(cfg.get("training"), dict) else {}
    model_cfg = cfg.get("model") if isinstance(cfg.get("model"), dict) else {}
    losses_cfg = cfg.get("losses") if isinstance(cfg.get("losses"), dict) else {}
    output_cfg = cfg.get("output") if isinstance(cfg.get("output"), dict) else {}

    seed = int(seed_override if seed_override is not None else int(train_cfg.get("seed") or 7))
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    trajectories = _collect_trajectories(repo_root=repo_root, config=cfg, logger=logger)
    max_data_steps = int(cfg.get("data", {}).get("max_steps") or 0) if isinstance(cfg.get("data"), dict) else 0
    if max_steps_override is not None:
        max_data_steps = int(max_steps_override)
    samples = _build_samples(trajectories, max_steps=(max_data_steps if max_data_steps > 0 else None))
    if not samples:
        raise RuntimeError("selfsup dataset is empty after loading configured sources")

    train_samples, val_samples = _split_train_val(samples, float(train_cfg.get("val_ratio") or 0.1), seed)
    if not train_samples:
        raise RuntimeError("selfsup train split is empty")
    if not val_samples:
        val_samples = train_samples[:1]

    phase_tokens = sorted({s.phase for s in samples})
    action_tokens = sorted({s.action_type for s in samples})
    hand_tokens = sorted({s.hand_type for s in samples})
    if "UNKNOWN" not in hand_tokens:
        hand_tokens = ["UNKNOWN", *hand_tokens]
    phase_to_id = {token: i for i, token in enumerate(["UNKNOWN", *phase_tokens])}
    action_to_id = {token: i for i, token in enumerate(["UNKNOWN", *action_tokens])}
    hand_to_id = {token: i for i, token in enumerate(hand_tokens)}

    class _SelfSupDataset(Dataset):
        def __init__(self, rows: list[SelfSupSample]):
            self.rows = rows

        def __len__(self) -> int:
            return len(self.rows)

        def __getitem__(self, idx: int) -> dict[str, Any]:
            row = self.rows[idx]
            return {
                "phase_id": phase_to_id.get(row.phase, 0),
                "action_id": action_to_id.get(row.action_type, 0),
                "numeric": row.numeric,
                "score_target": row.score_delta_target,
                "score_mask": row.score_delta_mask,
                "hand_type_id": hand_to_id.get(row.hand_type, hand_to_id.get("UNKNOWN", 0)),
                "hand_mask": row.hand_type_mask,
            }

    def _collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "phase_ids": torch.tensor([x["phase_id"] for x in batch], dtype=torch.long),
            "action_ids": torch.tensor([x["action_id"] for x in batch], dtype=torch.long),
            "numeric_features": torch.tensor([x["numeric"] for x in batch], dtype=torch.float32),
            "score_targets": torch.tensor([x["score_target"] for x in batch], dtype=torch.float32),
            "score_mask": torch.tensor([x["score_mask"] for x in batch], dtype=torch.float32),
            "hand_type_ids": torch.tensor([x["hand_type_id"] for x in batch], dtype=torch.long),
            "hand_mask": torch.tensor([x["hand_mask"] for x in batch], dtype=torch.float32),
        }

    batch_size = int(train_cfg.get("batch_size") or 64)
    train_loader = DataLoader(_SelfSupDataset(train_samples), batch_size=batch_size, shuffle=True, collate_fn=_collate)
    val_loader = DataLoader(_SelfSupDataset(val_samples), batch_size=batch_size, shuffle=False, collate_fn=_collate)

    device_raw = str(train_cfg.get("device") or "auto").lower()
    if device_raw == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_raw)

    cfg_obj = SelfSupModelConfig(
        phase_vocab_size=max(2, len(phase_to_id)),
        action_vocab_size=max(2, len(action_to_id)),
        hand_type_vocab_size=max(2, len(hand_to_id)),
        numeric_dim=len(train_samples[0].numeric),
        embed_dim=int(model_cfg.get("embed_dim") or 24),
        hidden_dim=int(model_cfg.get("hidden_dim") or 128),
        dropout=float(model_cfg.get("dropout") or 0.1),
    )
    model = SelfSupMultiHeadModel(cfg_obj).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg.get("lr") or 1e-3),
        weight_decay=float(train_cfg.get("weight_decay") or 1e-4),
    )

    loss_w_score = float(losses_cfg.get("score_delta") or 1.0)
    loss_w_hand = float(losses_cfg.get("hand_type") or 0.5)
    epochs = int(train_cfg.get("epochs") or 1)
    log_every = int(train_cfg.get("log_every") or 10)
    max_steps = int(max_steps_override if max_steps_override is not None else int(train_cfg.get("max_steps") or 0))

    if out_dir is not None:
        run_dir = Path(out_dir)
    else:
        base = str(output_cfg.get("run_root") or "trainer_runs/p31_selfsup")
        run_dir = (repo_root / base / (run_name or _now_stamp())).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    progress_path = run_dir / "progress.jsonl"

    def _append_progress(payload: dict[str, Any]) -> None:
        with progress_path.open("a", encoding="utf-8", newline="\n") as fp:
            fp.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def _eval(loader) -> dict[str, float]:
        model.eval()
        total_items = 0
        total_loss = 0.0
        score_abs = 0.0
        score_count = 0.0
        hand_correct = 0.0
        hand_count = 0.0
        with torch.no_grad():
            for batch in loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                out = model(
                    phase_ids=batch["phase_ids"],
                    action_ids=batch["action_ids"],
                    numeric_features=batch["numeric_features"],
                )

                score_mask = batch["score_mask"] > 0
                if bool(score_mask.any().item()):
                    score_pred = out["score_delta"][score_mask]
                    score_target = batch["score_targets"][score_mask]
                    score_loss = F.smooth_l1_loss(score_pred, score_target)
                    score_abs += float((score_pred - score_target).abs().sum().item())
                    score_count += float(score_mask.sum().item())
                else:
                    score_loss = torch.tensor(0.0, device=device)

                hand_mask = batch["hand_mask"] > 0
                if bool(hand_mask.any().item()):
                    hand_logits = out["hand_type_logits"][hand_mask]
                    hand_target = batch["hand_type_ids"][hand_mask]
                    hand_loss = F.cross_entropy(hand_logits, hand_target)
                    hand_pred = hand_logits.argmax(dim=-1)
                    hand_correct += float((hand_pred == hand_target).sum().item())
                    hand_count += float(hand_mask.sum().item())
                else:
                    hand_loss = torch.tensor(0.0, device=device)

                batch_loss = (loss_w_score * score_loss) + (loss_w_hand * hand_loss)
                batch_n = int(batch["phase_ids"].shape[0])
                total_items += batch_n
                total_loss += float(batch_loss.item()) * batch_n

        return {
            "loss": (total_loss / max(1, total_items)),
            "score_delta_mae": (score_abs / max(1.0, score_count)),
            "hand_type_acc": (hand_correct / max(1.0, hand_count)),
        }

    global_step = 0
    epoch_summaries: list[dict[str, Any]] = []
    started_ts = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_items = 0
        epoch_total_loss = 0.0
        epoch_score_loss = 0.0
        epoch_hand_loss = 0.0
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(
                phase_ids=batch["phase_ids"],
                action_ids=batch["action_ids"],
                numeric_features=batch["numeric_features"],
            )
            score_mask = batch["score_mask"] > 0
            if bool(score_mask.any().item()):
                score_loss = F.smooth_l1_loss(out["score_delta"][score_mask], batch["score_targets"][score_mask])
            else:
                score_loss = torch.tensor(0.0, device=device)

            hand_mask = batch["hand_mask"] > 0
            if bool(hand_mask.any().item()):
                hand_loss = F.cross_entropy(out["hand_type_logits"][hand_mask], batch["hand_type_ids"][hand_mask])
            else:
                hand_loss = torch.tensor(0.0, device=device)

            total_loss = (loss_w_score * score_loss) + (loss_w_hand * hand_loss)
            optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            optimizer.step()

            batch_n = int(batch["phase_ids"].shape[0])
            epoch_items += batch_n
            epoch_total_loss += float(total_loss.item()) * batch_n
            epoch_score_loss += float(score_loss.item()) * batch_n
            epoch_hand_loss += float(hand_loss.item()) * batch_n
            global_step += 1

            if (global_step % max(1, log_every) == 0) and (not quiet):
                msg = {
                    "ts": _now_iso(),
                    "epoch": epoch,
                    "global_step": global_step,
                    "train_loss": epoch_total_loss / max(1, epoch_items),
                    "train_score_loss": epoch_score_loss / max(1, epoch_items),
                    "train_hand_loss": epoch_hand_loss / max(1, epoch_items),
                }
                print(
                    "[selfsup] epoch={epoch} step={step} loss={loss:.6f} score={score:.6f} hand={hand:.6f}".format(
                        epoch=epoch,
                        step=global_step,
                        loss=msg["train_loss"],
                        score=msg["train_score_loss"],
                        hand=msg["train_hand_loss"],
                    )
                )
                _append_progress(msg)

            if max_steps > 0 and global_step >= max_steps:
                break

        val_metrics = _eval(val_loader)
        epoch_summary = {
            "epoch": epoch,
            "train_loss": (epoch_total_loss / max(1, epoch_items)),
            "train_score_loss": (epoch_score_loss / max(1, epoch_items)),
            "train_hand_loss": (epoch_hand_loss / max(1, epoch_items)),
            "val_loss": val_metrics["loss"],
            "val_score_delta_mae": val_metrics["score_delta_mae"],
            "val_hand_type_acc": val_metrics["hand_type_acc"],
            "global_step": global_step,
            "elapsed_sec": time.time() - started_ts,
        }
        epoch_summaries.append(epoch_summary)
        _append_progress({"ts": _now_iso(), **epoch_summary, "stage": "epoch_done"})

        ckpt_path = run_dir / f"selfsup_encoder_epoch{epoch}.pt"
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch,
                "global_step": global_step,
                "config": cfg,
                "model_config": cfg_obj.__dict__,
                "phase_to_id": phase_to_id,
                "action_to_id": action_to_id,
                "hand_type_to_id": hand_to_id,
            },
            ckpt_path,
        )

        if max_steps > 0 and global_step >= max_steps:
            break

    final_epoch = epoch_summaries[-1] if epoch_summaries else {
        "train_loss": 0.0,
        "val_loss": 0.0,
        "val_score_delta_mae": 0.0,
        "val_hand_type_acc": 0.0,
    }
    recommendation = "promote" if float(final_epoch.get("val_hand_type_acc") or 0.0) >= 0.55 else "hold"
    summary = {
        "schema": "p31_selfsup_train_summary_v1",
        "generated_at": _now_iso(),
        "config_path": str(cfg_path),
        "run_dir": str(run_dir),
        "status": "ok",
        "seed": seed,
        "source_trajectory_count": len(trajectories),
        "sample_count": len(samples),
        "train_sample_count": len(train_samples),
        "val_sample_count": len(val_samples),
        "global_step": global_step,
        "epoch_count": len(epoch_summaries),
        "final_metrics": {
            "train_loss": float(final_epoch.get("train_loss") or 0.0),
            "val_loss": float(final_epoch.get("val_loss") or 0.0),
            "val_score_delta_mae": float(final_epoch.get("val_score_delta_mae") or 0.0),
            "val_hand_type_acc": float(final_epoch.get("val_hand_type_acc") or 0.0),
        },
        "recommendation": recommendation,
        "checkpoints": sorted([str(p) for p in run_dir.glob("selfsup_encoder_epoch*.pt")]),
        "vocab_sizes": {
            "phase": len(phase_to_id),
            "action": len(action_to_id),
            "hand_type": len(hand_to_id),
        },
        "loss_weights": {
            "score_delta": loss_w_score,
            "hand_type": loss_w_hand,
        },
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (run_dir / "summary.md").write_text(
        "\n".join(
            [
                "# P31 Self-Supervised Train Summary",
                "",
                f"- run_dir: {run_dir}",
                f"- sample_count: {len(samples)}",
                f"- epochs: {len(epoch_summaries)}",
                f"- global_step: {global_step}",
                f"- val_loss: {summary['final_metrics']['val_loss']:.6f}",
                f"- val_score_delta_mae: {summary['final_metrics']['val_score_delta_mae']:.6f}",
                f"- val_hand_type_acc: {summary['final_metrics']['val_hand_type_acc']:.6f}",
                f"- recommendation: {recommendation}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return summary


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="P31 self-supervised encoder training entrypoint.")
    p.add_argument("--config", default="configs/experiments/p31_selfsup.yaml")
    p.add_argument("--out-dir", default="")
    p.add_argument("--seed", type=int, default=-1)
    p.add_argument("--max-steps", type=int, default=0)
    p.add_argument("--run-name", default="")
    p.add_argument("--quiet", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    summary = run_selfsup_training(
        config_path=args.config,
        out_dir=(args.out_dir or None),
        seed_override=(args.seed if int(args.seed) >= 0 else None),
        max_steps_override=(args.max_steps if int(args.max_steps) > 0 else None),
        run_name=str(args.run_name or _now_stamp()),
        quiet=bool(args.quiet),
    )
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
