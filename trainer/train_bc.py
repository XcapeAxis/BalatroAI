if __package__ is None or __package__ == "":
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).resolve().parent.parent))

import argparse
import json
import random
from pathlib import Path

from trainer import action_space
from trainer.dataset import iter_train_samples
from trainer.utils import setup_logger, warn_if_unstable_python


def _require_torch():
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        from torch.utils.data import DataLoader, Dataset
    except Exception as exc:
        raise RuntimeError(
            "PyTorch is required for train_bc.py. Install dependencies in trainer/requirements.txt."
        ) from exc
    return torch, nn, F, Dataset, DataLoader


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train behavior cloning policy from rollout jsonl.")
    parser.add_argument("--train-jsonl", required=True, help="Training dataset jsonl path.")
    parser.add_argument("--val-jsonl", default=None, help="Validation dataset path. If omitted, auto-split train.")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-actions", type=int, default=action_space.max_actions())
    parser.add_argument("--out-dir", default="trainer_runs/bc")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    return parser.parse_args()


def _as_sample(record: dict) -> dict:
    f = record["features"]
    hand_size = int(record["hand_size"])
    legal_ids = list(record["legal_action_ids"])
    target = int(record["expert_action_id"])
    chip_hint = list(f.get("card_chip_hint") or [0] * action_space.MAX_HAND)
    return {
        "rank": list(f["card_rank_ids"]),
        "suit": list(f["card_suit_ids"]),
        "chip": chip_hint,
        "enh": list(f["card_has_enhancement"]),
        "edt": list(f["card_has_edition"]),
        "seal": list(f["card_has_seal"]),
        "pad": list(f["hand_pad_mask"]),
        "context": list(f["context"]),
        "hand_size": hand_size,
        "legal_ids": legal_ids,
        "target": target,
    }


def load_samples(path: str) -> list[dict]:
    return [_as_sample(r) for r in iter_train_samples(path)]


def split_samples(samples: list[dict], val_ratio: float, seed: int) -> tuple[list[dict], list[dict]]:
    rng = random.Random(seed)
    idx = list(range(len(samples)))
    rng.shuffle(idx)
    cut = int(len(samples) * (1 - val_ratio))
    if cut <= 0:
        cut = max(1, len(samples) - 1)
    if cut >= len(samples):
        cut = max(1, len(samples) - 1)
    train_idx = idx[:cut]
    val_idx = idx[cut:]
    train = [samples[i] for i in train_idx]
    val = [samples[i] for i in val_idx] if val_idx else [samples[i] for i in train_idx[:1]]
    return train, val


def build_model(nn, max_actions: int):
    class BCModel(nn.Module):
        def __init__(self, max_actions: int):
            super().__init__()
            self.rank_emb = nn.Embedding(16, 16)
            self.suit_emb = nn.Embedding(8, 8)
            self.card_proj = nn.Sequential(
                nn.Linear(16 + 8 + 4, 64),
                nn.ReLU(),
                nn.Linear(64, 64),
                nn.ReLU(),
            )
            self.ctx_proj = nn.Sequential(
                nn.Linear(12, 64),
                nn.ReLU(),
                nn.Linear(64, 64),
                nn.ReLU(),
            )
            self.head = nn.Sequential(
                nn.Linear(128, 128),
                nn.ReLU(),
                nn.Linear(128, max_actions),
            )

        def forward(self, batch):
            rank = batch["rank"]
            suit = batch["suit"]
            chip = batch["chip"].unsqueeze(-1)
            enh = batch["enh"].unsqueeze(-1)
            edt = batch["edt"].unsqueeze(-1)
            seal = batch["seal"].unsqueeze(-1)
            pad = batch["pad"]

            r = self.rank_emb(rank)
            s = self.suit_emb(suit)
            card_x = __import__("torch").cat([r, s, chip, enh, edt, seal], dim=-1)
            card_h = self.card_proj(card_x)

            pad_sum = pad.sum(dim=1, keepdim=True).clamp(min=1.0)
            pooled = (card_h * pad.unsqueeze(-1)).sum(dim=1) / pad_sum

            ctx_h = self.ctx_proj(batch["context"])
            fused = __import__("torch").cat([pooled, ctx_h], dim=-1)
            logits = self.head(fused)
            return logits

    return BCModel(max_actions)


def main() -> int:
    args = parse_args()
    logger = setup_logger("trainer.train_bc")
    warn_if_unstable_python(logger)

    try:
        torch, nn, F, Dataset, DataLoader = _require_torch()
    except RuntimeError as exc:
        logger.error("%s", exc)
        return 2

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    samples_train = load_samples(args.train_jsonl)
    if not samples_train:
        logger.error("No training samples found in %s", args.train_jsonl)
        return 2

    if args.val_jsonl:
        samples_val = load_samples(args.val_jsonl)
        if not samples_val:
            logger.warning("Validation dataset empty, fallback to split from train")
            samples_train, samples_val = split_samples(samples_train, val_ratio=0.1, seed=args.seed)
    else:
        samples_train, samples_val = split_samples(samples_train, val_ratio=0.1, seed=args.seed)

    max_actions = args.max_actions

    class BCDataset(Dataset):
        def __init__(self, samples):
            self.samples = samples

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, idx):
            return self.samples[idx]

    def collate(batch):
        bsz = len(batch)
        rank = torch.tensor([x["rank"] for x in batch], dtype=torch.long)
        suit = torch.tensor([x["suit"] for x in batch], dtype=torch.long)
        chip = torch.tensor([x["chip"] for x in batch], dtype=torch.float32)
        enh = torch.tensor([x["enh"] for x in batch], dtype=torch.float32)
        edt = torch.tensor([x["edt"] for x in batch], dtype=torch.float32)
        seal = torch.tensor([x["seal"] for x in batch], dtype=torch.float32)
        pad = torch.tensor([x["pad"] for x in batch], dtype=torch.float32)
        context = torch.tensor([x["context"] for x in batch], dtype=torch.float32)
        targets = torch.tensor([x["target"] for x in batch], dtype=torch.long)

        legal_mask = torch.zeros((bsz, max_actions), dtype=torch.float32)
        for i, x in enumerate(batch):
            for aid in x["legal_ids"]:
                if 0 <= aid < max_actions:
                    legal_mask[i, aid] = 1.0
        return {
            "rank": rank,
            "suit": suit,
            "chip": chip,
            "enh": enh,
            "edt": edt,
            "seal": seal,
            "pad": pad,
            "context": context,
            "targets": targets,
            "legal_mask": legal_mask,
        }

    train_loader = DataLoader(BCDataset(samples_train), batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(BCDataset(samples_val), batch_size=args.batch_size, shuffle=False, collate_fn=collate)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    model = build_model(nn, max_actions=max_actions).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    def masked_ce(logits, targets, legal_mask):
        neg = torch.full_like(logits, -1e9)
        masked_logits = torch.where(legal_mask > 0, logits, neg)
        return F.cross_entropy(masked_logits, targets)

    def evaluate(loader):
        model.eval()
        total_loss = 0.0
        total = 0
        correct1 = 0
        correct3 = 0
        with torch.no_grad():
            for batch in loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                logits = model(batch)
                loss = masked_ce(logits, batch["targets"], batch["legal_mask"])
                total_loss += loss.item() * batch["targets"].shape[0]
                total += batch["targets"].shape[0]

                masked_logits = torch.where(batch["legal_mask"] > 0, logits, torch.full_like(logits, -1e9))
                top1 = masked_logits.argmax(dim=1)
                correct1 += (top1 == batch["targets"]).sum().item()

                topk = torch.topk(masked_logits, k=min(3, masked_logits.shape[1]), dim=1).indices
                for i in range(topk.shape[0]):
                    if int(batch["targets"][i].item()) in topk[i].tolist():
                        correct3 += 1

        if total == 0:
            return {"loss": 0.0, "acc1": 0.0, "acc3": 0.0}
        return {
            "loss": total_loss / total,
            "acc1": correct1 / total,
            "acc3": correct3 / total,
        }

    history = []
    best_val = None
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(batch)
            loss = masked_ce(logits, batch["targets"], batch["legal_mask"])
            optim.zero_grad()
            loss.backward()
            optim.step()

        train_metrics = evaluate(train_loader)
        val_metrics = evaluate(val_loader)
        row = {
            "epoch": epoch,
            "train": train_metrics,
            "val": val_metrics,
        }
        history.append(row)
        logger.info(
            "epoch=%d train_loss=%.4f train_acc1=%.4f val_loss=%.4f val_acc1=%.4f val_acc3=%.4f",
            epoch,
            train_metrics["loss"],
            train_metrics["acc1"],
            val_metrics["loss"],
            val_metrics["acc1"],
            val_metrics["acc3"],
        )

        torch.save(model.state_dict(), out_dir / "last.pt")
        if best_val is None or val_metrics["loss"] < best_val:
            best_val = val_metrics["loss"]
            torch.save(model.state_dict(), out_dir / "best.pt")

    metrics = {
        "history": history,
        "train_size": len(samples_train),
        "val_size": len(samples_val),
        "max_actions": max_actions,
    }
    (out_dir / "train_metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "config.json").write_text(
        json.dumps(
            {
                "max_actions": max_actions,
                "context_dim": 12,
                "rank_vocab": 16,
                "suit_vocab": 8,
                "uses_chip_hint": True,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    logger.info("Training complete. Outputs in %s", out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
