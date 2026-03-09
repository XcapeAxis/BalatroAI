from __future__ import annotations

if __package__ is None or __package__ == "":
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).resolve().parent.parent))

import argparse
import csv
import json
import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from demo.model_inference import MAX_ACTIONS, MVPHandPolicy


def _require_torch():
    try:
        import torch
        import torch.nn.functional as F
        from torch.utils.data import DataLoader, Dataset
    except Exception as exc:
        raise RuntimeError("PyTorch is required to train the MVP model") from exc
    return torch, F, Dataset, DataLoader


def now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def default_run_dir() -> Path:
    run_id = now_stamp()
    return Path(__file__).resolve().parent.parent / "docs" / "artifacts" / "mvp" / "model_train" / run_id


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the first MVP demo hand-policy model.")
    parser.add_argument("--dataset", default="", help="Path to dataset.jsonl created by build_mvp_dataset.py")
    parser.add_argument("--run-dir", default="", help="Output directory under docs/artifacts/mvp/model_train.")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=96)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    return parser.parse_args()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fp:
        for line in fp:
            text = line.strip()
            if not text:
                continue
            rows.append(json.loads(text))
    return rows


def _resolve_dataset_path(args: argparse.Namespace, run_dir: Path) -> Path:
    if args.dataset:
        return Path(args.dataset).resolve()
    candidate = run_dir / "dataset.jsonl"
    if candidate.exists():
        return candidate
    latest_runs = sorted(
        [path for path in run_dir.parent.iterdir() if path.is_dir() and (path / "dataset.jsonl").exists()],
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if latest_runs:
        return latest_runs[0] / "dataset.jsonl"
    raise FileNotFoundError("dataset.jsonl not found; run demo/build_mvp_dataset.py first")


@dataclass
class Sample:
    rank: list[int]
    suit: list[int]
    chip: list[int]
    enh: list[int]
    edt: list[int]
    seal: list[int]
    pad: list[int]
    context: list[int]
    target: int
    legal_action_ids: list[int]


def _sample_from_record(record: dict[str, Any]) -> Sample:
    features = record["features"]
    return Sample(
        rank=list(features["card_rank_ids"]),
        suit=list(features["card_suit_ids"]),
        chip=list(features.get("card_chip_hint") or [0] * 10),
        enh=list(features["card_has_enhancement"]),
        edt=list(features["card_has_edition"]),
        seal=list(features["card_has_seal"]),
        pad=list(features["hand_pad_mask"]),
        context=list(features["context"]),
        target=int(record["expert_action_id"]),
        legal_action_ids=[int(aid) for aid in record["legal_action_ids"]],
    )


def _split_samples(samples: list[Sample], val_ratio: float, seed: int) -> tuple[list[Sample], list[Sample]]:
    rng = random.Random(seed)
    shuffled = list(samples)
    rng.shuffle(shuffled)
    cut = int(len(shuffled) * (1.0 - val_ratio))
    cut = max(1, min(cut, len(shuffled) - 1))
    return shuffled[:cut], shuffled[cut:]


def main() -> int:
    args = parse_args()
    run_dir = Path(args.run_dir).resolve() if args.run_dir else default_run_dir()
    run_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = _resolve_dataset_path(args, run_dir)
    records = _read_jsonl(dataset_path)
    if not records:
        raise RuntimeError(f"dataset is empty: {dataset_path}")
    dataset_stats_source = dataset_path.with_name("dataset_stats.json")

    samples = [_sample_from_record(record) for record in records]
    train_samples, val_samples = _split_samples(samples, val_ratio=float(args.val_ratio), seed=int(args.seed))

    torch, F, Dataset, DataLoader = _require_torch()
    random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))

    class _SamplesDataset(Dataset):
        def __init__(self, rows: list[Sample]):
            self.rows = rows

        def __len__(self) -> int:
            return len(self.rows)

        def __getitem__(self, index: int) -> Sample:
            return self.rows[index]

    def _collate(batch: list[Sample]) -> dict[str, Any]:
        max_actions = MAX_ACTIONS
        legal_mask = torch.zeros((len(batch), max_actions), dtype=torch.float32)
        for row_idx, sample in enumerate(batch):
            for action_id in sample.legal_action_ids:
                if 0 <= int(action_id) < max_actions:
                    legal_mask[row_idx, int(action_id)] = 1.0
        return {
            "rank": torch.tensor([sample.rank for sample in batch], dtype=torch.long),
            "suit": torch.tensor([sample.suit for sample in batch], dtype=torch.long),
            "chip": torch.tensor([sample.chip for sample in batch], dtype=torch.float32),
            "enh": torch.tensor([sample.enh for sample in batch], dtype=torch.float32),
            "edt": torch.tensor([sample.edt for sample in batch], dtype=torch.float32),
            "seal": torch.tensor([sample.seal for sample in batch], dtype=torch.float32),
            "pad": torch.tensor([sample.pad for sample in batch], dtype=torch.float32),
            "context": torch.tensor([sample.context for sample in batch], dtype=torch.float32),
            "targets": torch.tensor([sample.target for sample in batch], dtype=torch.long),
            "legal_mask": legal_mask,
        }

    train_loader = DataLoader(_SamplesDataset(train_samples), batch_size=max(8, int(args.batch_size)), shuffle=True, collate_fn=_collate)
    val_loader = DataLoader(_SamplesDataset(val_samples), batch_size=max(8, int(args.batch_size)), shuffle=False, collate_fn=_collate)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    import torch.nn as nn

    model = MVPHandPolicy(nn, max_actions=MAX_ACTIONS).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))

    def _masked_loss(logits, targets, legal_mask):
        masked_logits = torch.where(legal_mask > 0, logits, torch.full_like(logits, -1e9))
        return F.cross_entropy(masked_logits, targets), masked_logits

    def _evaluate(loader) -> dict[str, float]:
        model.eval()
        total = 0
        total_loss = 0.0
        correct1 = 0
        correct3 = 0
        with torch.no_grad():
            for batch in loader:
                batch = {key: value.to(device) for key, value in batch.items()}
                logits = model(batch)
                loss, masked_logits = _masked_loss(logits, batch["targets"], batch["legal_mask"])
                total += int(batch["targets"].shape[0])
                total_loss += float(loss.item()) * int(batch["targets"].shape[0])
                top1 = masked_logits.argmax(dim=1)
                correct1 += int((top1 == batch["targets"]).sum().item())
                topk = torch.topk(masked_logits, k=min(3, masked_logits.shape[1]), dim=1).indices
                for idx in range(int(topk.shape[0])):
                    if int(batch["targets"][idx].item()) in topk[idx].tolist():
                        correct3 += 1
        return {
            "loss": total_loss / max(1, total),
            "acc1": correct1 / max(1, total),
            "acc3": correct3 / max(1, total),
        }

    history: list[dict[str, Any]] = []
    best_val = None
    best_path = run_dir / "mvp_policy.pt"

    for epoch in range(1, max(1, int(args.epochs)) + 1):
        model.train()
        train_loss_sum = 0.0
        train_items = 0
        online_correct1 = 0
        online_correct3 = 0
        for batch in train_loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            logits = model(batch)
            loss, masked_logits = _masked_loss(logits, batch["targets"], batch["legal_mask"])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_items += int(batch["targets"].shape[0])
            train_loss_sum += float(loss.item()) * int(batch["targets"].shape[0])
            top1 = masked_logits.argmax(dim=1)
            online_correct1 += int((top1 == batch["targets"]).sum().item())
            topk = torch.topk(masked_logits, k=min(3, masked_logits.shape[1]), dim=1).indices
            for idx in range(int(topk.shape[0])):
                if int(batch["targets"][idx].item()) in topk[idx].tolist():
                    online_correct3 += 1

        val_metrics = _evaluate(val_loader)
        row = {
            "epoch": epoch,
            "train_loss": train_loss_sum / max(1, train_items),
            "train_acc1": online_correct1 / max(1, train_items),
            "train_acc3": online_correct3 / max(1, train_items),
            "val_loss": val_metrics["loss"],
            "val_acc1": val_metrics["acc1"],
            "val_acc3": val_metrics["acc3"],
        }
        history.append(row)
        if best_val is None or float(val_metrics["loss"]) < float(best_val):
            best_val = float(val_metrics["loss"])
            torch.save(model.state_dict(), best_path)

    metrics = {
        "schema": "mvp_demo_train_metrics_v1",
        "dataset_path": str(dataset_path),
        "train_samples": len(train_samples),
        "val_samples": len(val_samples),
        "best_val_loss": best_val,
        "history": history,
        "final": history[-1],
    }
    (run_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    with (run_dir / "loss_curve.csv").open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=["epoch", "train_loss", "val_loss", "val_acc1", "val_acc3"])
        writer.writeheader()
        for row in history:
            writer.writerow(
                {
                    "epoch": row["epoch"],
                    "train_loss": row["train_loss"],
                    "val_loss": row["val_loss"],
                    "val_acc1": row["val_acc1"],
                    "val_acc3": row["val_acc3"],
                }
            )

    model_config = {
        "schema": "mvp_demo_model_config_v1",
        "model_name": "mvp_hand_policy_v1",
        "checkpoint_name": "mvp_policy.pt",
        "max_actions": MAX_ACTIONS,
        "dataset_path": str(dataset_path),
        "train_samples": len(train_samples),
        "val_samples": len(val_samples),
        "device_used": str(device),
    }
    (run_dir / "model_config.json").write_text(json.dumps(model_config, ensure_ascii=False, indent=2), encoding="utf-8")
    if dataset_stats_source.exists():
        (run_dir / "dataset_stats.json").write_text(dataset_stats_source.read_text(encoding="utf-8"), encoding="utf-8")
    (run_dir / "checkpoint_path.txt").write_text(str(best_path), encoding="utf-8")

    summary = "\n".join(
        [
            f"# MVP Model Training {run_dir.name}",
            "",
            f"- Dataset: `{dataset_path}`",
            f"- Train samples: `{len(train_samples)}`",
            f"- Validation samples: `{len(val_samples)}`",
            f"- Best validation loss: `{best_val:.4f}`" if best_val is not None else "- Best validation loss: `n/a`",
            f"- Final top-1 accuracy: `{history[-1]['val_acc1']:.4f}`",
            f"- Final top-3 accuracy: `{history[-1]['val_acc3']:.4f}`",
            f"- Checkpoint: `{best_path}`",
        ]
    )
    (run_dir / "training_summary.md").write_text(summary + "\n", encoding="utf-8")
    latest_hint = run_dir.parent / "latest_run.txt"
    latest_hint.write_text(run_dir.name, encoding="utf-8")
    print(json.dumps({"run_dir": str(run_dir), "checkpoint": str(best_path), "best_val_loss": best_val}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
