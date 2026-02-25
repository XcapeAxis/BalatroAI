from __future__ import annotations

if __package__ is None or __package__ == "":
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).resolve().parent.parent))

import argparse
import json
import random
from pathlib import Path

from trainer import action_space
from trainer import action_space_shop
from trainer.dataset import iter_shop_samples, iter_train_samples
from trainer.features_shop import SHOP_CONTEXT_DIM
from trainer.models.policy_value import PolicyValueModel
from trainer.utils import setup_logger, warn_if_unstable_python


def _require_torch():
    try:
        import torch
        import torch.nn.functional as F
        from torch.utils.data import DataLoader, Dataset
    except Exception as exc:
        raise RuntimeError("PyTorch is required for train_pv.py") from exc
    return torch, F, Dataset, DataLoader


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train policy-value model from search-labeled dataset.")
    p.add_argument("--train-jsonl", required=True)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--value-weight", type=float, default=0.2)
    p.add_argument("--shop-weight", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--max-actions", type=int, default=action_space.max_actions())
    p.add_argument("--max-shop-actions", type=int, default=action_space_shop.max_actions())
    p.add_argument("--out-dir", default="trainer_runs/p15_pv")
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    return p.parse_args()


def _hand_sample(r: dict) -> dict:
    f = r["features"]
    chip_hint = list(f.get("card_chip_hint") or [0] * action_space.MAX_HAND)
    value_target = float(r.get("value_target", r.get("reward", 0.0)))
    return {
        "rank": list(f["card_rank_ids"]),
        "suit": list(f["card_suit_ids"]),
        "chip": chip_hint,
        "enh": list(f["card_has_enhancement"]),
        "edt": list(f["card_has_edition"]),
        "seal": list(f["card_has_seal"]),
        "pad": list(f["hand_pad_mask"]),
        "context": list(f["context"]),
        "legal_ids": list(r.get("legal_action_ids") or []),
        "target": int(r["expert_action_id"]),
        "value_target": value_target,
    }


def _shop_sample(r: dict) -> dict:
    sf = r.get("shop_features") if isinstance(r.get("shop_features"), dict) else {}
    ctx = list(sf.get("shop_context") or [0.0] * SHOP_CONTEXT_DIM)
    if len(ctx) != SHOP_CONTEXT_DIM:
        ctx = (ctx + [0.0] * SHOP_CONTEXT_DIM)[:SHOP_CONTEXT_DIM]
    value_target = float(r.get("value_target", r.get("reward", 0.0)))
    return {
        "shop_context": ctx,
        "legal_ids": list(r.get("shop_legal_action_ids") or []),
        "target": int(r.get("shop_expert_action_id")),
        "value_target": value_target,
    }


def _split(samples: list[dict], val_ratio: float, seed: int) -> tuple[list[dict], list[dict]]:
    if not samples:
        return [], []
    rng = random.Random(seed)
    idx = list(range(len(samples)))
    rng.shuffle(idx)
    cut = int(len(samples) * (1.0 - val_ratio))
    cut = max(1, min(cut, len(samples) - 1))
    tr = [samples[i] for i in idx[:cut]]
    va = [samples[i] for i in idx[cut:]]
    return tr, va


def main() -> int:
    args = parse_args()
    logger = setup_logger("trainer.train_pv")
    warn_if_unstable_python(logger)

    torch, F, Dataset, DataLoader = _require_torch()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    hand_all = [_hand_sample(r) for r in iter_train_samples(args.train_jsonl)]
    shop_all = [_shop_sample(r) for r in iter_shop_samples(args.train_jsonl)]
    if not hand_all:
        logger.error("No hand samples found in %s", args.train_jsonl)
        return 2

    hand_train, hand_val = _split(hand_all, 0.1, args.seed)
    shop_train, shop_val = _split(shop_all, 0.1, args.seed)

    class HandDS(Dataset):
        def __init__(self, rows):
            self.rows = rows

        def __len__(self):
            return len(self.rows)

        def __getitem__(self, i):
            return self.rows[i]

    class ShopDS(Dataset):
        def __init__(self, rows):
            self.rows = rows

        def __len__(self):
            return len(self.rows)

        def __getitem__(self, i):
            return self.rows[i]

    def collate_hand(batch):
        bsz = len(batch)
        legal = torch.zeros((bsz, args.max_actions), dtype=torch.float32)
        for i, x in enumerate(batch):
            for aid in x["legal_ids"]:
                if 0 <= int(aid) < args.max_actions:
                    legal[i, int(aid)] = 1.0
        return {
            "rank": torch.tensor([x["rank"] for x in batch], dtype=torch.long),
            "suit": torch.tensor([x["suit"] for x in batch], dtype=torch.long),
            "chip": torch.tensor([x["chip"] for x in batch], dtype=torch.float32),
            "enh": torch.tensor([x["enh"] for x in batch], dtype=torch.float32),
            "edt": torch.tensor([x["edt"] for x in batch], dtype=torch.float32),
            "seal": torch.tensor([x["seal"] for x in batch], dtype=torch.float32),
            "pad": torch.tensor([x["pad"] for x in batch], dtype=torch.float32),
            "context": torch.tensor([x["context"] for x in batch], dtype=torch.float32),
            "targets": torch.tensor([x["target"] for x in batch], dtype=torch.long),
            "value_targets": torch.tensor([x["value_target"] for x in batch], dtype=torch.float32),
            "legal_mask": legal,
        }

    def collate_shop(batch):
        bsz = len(batch)
        legal = torch.zeros((bsz, args.max_shop_actions), dtype=torch.float32)
        for i, x in enumerate(batch):
            for aid in x["legal_ids"]:
                if 0 <= int(aid) < args.max_shop_actions:
                    legal[i, int(aid)] = 1.0
        return {
            "shop_context": torch.tensor([x["shop_context"] for x in batch], dtype=torch.float32),
            "targets": torch.tensor([x["target"] for x in batch], dtype=torch.long),
            "value_targets": torch.tensor([x["value_target"] for x in batch], dtype=torch.float32),
            "legal_mask": legal,
        }

    hand_tr_loader = DataLoader(HandDS(hand_train), batch_size=args.batch_size, shuffle=True, collate_fn=collate_hand)
    hand_va_loader = DataLoader(HandDS(hand_val), batch_size=args.batch_size, shuffle=False, collate_fn=collate_hand)
    shop_tr_loader = DataLoader(ShopDS(shop_train), batch_size=args.batch_size, shuffle=True, collate_fn=collate_shop) if shop_train else None
    shop_va_loader = DataLoader(ShopDS(shop_val), batch_size=args.batch_size, shuffle=False, collate_fn=collate_shop) if shop_val else None

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    model = PolicyValueModel(args.max_actions, args.max_shop_actions).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    def masked_ce(logits, targets, legal_mask):
        neg = torch.full_like(logits, -1e9)
        masked = torch.where(legal_mask > 0, logits, neg)
        return F.cross_entropy(masked, targets)

    def evaluate_hand(loader):
        model.eval()
        total = 0
        pol_loss = 0.0
        val_loss = 0.0
        acc1 = 0
        acc3 = 0
        with torch.no_grad():
            for b in loader:
                b = {k: v.to(device) for k, v in b.items()}
                logits, values = model.forward_hand(b)
                lpol = masked_ce(logits, b["targets"], b["legal_mask"])
                lval = F.smooth_l1_loss(values, b["value_targets"])
                n = b["targets"].shape[0]
                total += n
                pol_loss += lpol.item() * n
                val_loss += lval.item() * n

                masked = torch.where(b["legal_mask"] > 0, logits, torch.full_like(logits, -1e9))
                top1 = masked.argmax(dim=1)
                acc1 += (top1 == b["targets"]).sum().item()
                topk = torch.topk(masked, k=min(3, masked.shape[1]), dim=1).indices
                for i in range(topk.shape[0]):
                    if int(b["targets"][i].item()) in topk[i].tolist():
                        acc3 += 1
        if total <= 0:
            return {"policy_loss": 0.0, "value_loss": 0.0, "acc1": 0.0, "acc3": 0.0}
        return {"policy_loss": pol_loss / total, "value_loss": val_loss / total, "acc1": acc1 / total, "acc3": acc3 / total}

    def evaluate_shop(loader):
        if loader is None:
            return {"policy_loss": 0.0, "value_loss": 0.0, "acc1": 0.0, "illegal_rate": 0.0, "total": 0}
        model.eval()
        total = 0
        pol_loss = 0.0
        val_loss = 0.0
        acc1 = 0
        illegal = 0
        with torch.no_grad():
            for b in loader:
                b = {k: v.to(device) for k, v in b.items()}
                logits, values = model.forward_shop(b)
                lpol = masked_ce(logits, b["targets"], b["legal_mask"])
                lval = F.smooth_l1_loss(values, b["value_targets"])
                n = b["targets"].shape[0]
                total += n
                pol_loss += lpol.item() * n
                val_loss += lval.item() * n
                masked = torch.where(b["legal_mask"] > 0, logits, torch.full_like(logits, -1e9))
                top1 = masked.argmax(dim=1)
                acc1 += (top1 == b["targets"]).sum().item()
                for i in range(top1.shape[0]):
                    if b["legal_mask"][i, top1[i]].item() <= 0:
                        illegal += 1
        if total <= 0:
            return {"policy_loss": 0.0, "value_loss": 0.0, "acc1": 0.0, "illegal_rate": 0.0, "total": 0}
        return {
            "policy_loss": pol_loss / total,
            "value_loss": val_loss / total,
            "acc1": acc1 / total,
            "illegal_rate": illegal / total,
            "total": total,
        }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    best_key = None
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        for b in hand_tr_loader:
            b = {k: v.to(device) for k, v in b.items()}
            logits, values = model.forward_hand(b)
            lpol = masked_ce(logits, b["targets"], b["legal_mask"])
            lval = F.smooth_l1_loss(values, b["value_targets"])
            loss = lpol + float(args.value_weight) * lval
            opt.zero_grad()
            loss.backward()
            opt.step()

        if shop_tr_loader is not None:
            for b in shop_tr_loader:
                b = {k: v.to(device) for k, v in b.items()}
                logits, values = model.forward_shop(b)
                lpol = masked_ce(logits, b["targets"], b["legal_mask"])
                lval = F.smooth_l1_loss(values, b["value_targets"])
                loss = float(args.shop_weight) * (lpol + float(args.value_weight) * lval)
                opt.zero_grad()
                loss.backward()
                opt.step()

        train_hand = evaluate_hand(hand_tr_loader)
        val_hand = evaluate_hand(hand_va_loader)
        train_shop = evaluate_shop(shop_tr_loader)
        val_shop = evaluate_shop(shop_va_loader)
        row = {
            "epoch": epoch,
            "train_hand": train_hand,
            "val_hand": val_hand,
            "train_shop": train_shop,
            "val_shop": val_shop,
        }
        history.append(row)
        logger.info(
            "epoch=%d hand_acc1=%.4f/%.4f hand_val_ploss=%.4f shop_acc1=%.4f/%.4f shop_illegal=%.4f",
            epoch,
            train_hand["acc1"],
            val_hand["acc1"],
            val_hand["policy_loss"],
            train_shop["acc1"],
            val_shop["acc1"],
            val_shop.get("illegal_rate", 0.0),
        )

        torch.save(model.state_dict(), out_dir / "last.pt")
        key = float(val_hand["policy_loss"]) + float(val_hand["value_loss"]) + float(val_shop["policy_loss"]) + float(val_shop["value_loss"])
        if best_key is None or key < best_key:
            best_key = key
            torch.save(model.state_dict(), out_dir / "best.pt")

    metrics = {
        "history": history,
        "train_hand_size": len(hand_train),
        "val_hand_size": len(hand_val),
        "train_shop_size": len(shop_train),
        "val_shop_size": len(shop_val),
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    logger.info("saved model and metrics to %s", out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
