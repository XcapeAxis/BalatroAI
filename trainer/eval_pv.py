from __future__ import annotations

if __package__ is None or __package__ == "":
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).resolve().parent.parent))

import argparse
import json
from pathlib import Path

from trainer import action_space
from trainer import action_space_shop
from trainer.dataset import iter_shop_samples, iter_train_samples
from trainer.features_shop import SHOP_CONTEXT_DIM
from trainer.models.policy_value import PolicyValueModel
from trainer.utils import setup_logger, timestamp, warn_if_unstable_python


def _require_torch():
    try:
        import torch
    except Exception as exc:
        raise RuntimeError("PyTorch is required for eval_pv.py") from exc
    return torch


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Offline eval for policy-value model.")
    p.add_argument("--model", required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--max-actions", type=int, default=action_space.max_actions())
    p.add_argument("--max-shop-actions", type=int, default=action_space_shop.max_actions())
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    return p.parse_args()


def _hand_rows(path: str):
    out = []
    for r in iter_train_samples(path):
        f = r["features"]
        out.append(
            {
                "rank": list(f["card_rank_ids"]),
                "suit": list(f["card_suit_ids"]),
                "chip": list(f.get("card_chip_hint") or [0] * action_space.MAX_HAND),
                "enh": list(f["card_has_enhancement"]),
                "edt": list(f["card_has_edition"]),
                "seal": list(f["card_has_seal"]),
                "pad": list(f["hand_pad_mask"]),
                "context": list(f["context"]),
                "target": int(r["expert_action_id"]),
                "legal": list(r.get("legal_action_ids") or []),
                "value_target": float(r.get("value_target", r.get("reward", 0.0))),
            }
        )
    return out


def _shop_rows(path: str):
    out = []
    for r in iter_shop_samples(path):
        sf = r.get("shop_features") if isinstance(r.get("shop_features"), dict) else {}
        ctx = list(sf.get("shop_context") or [0.0] * SHOP_CONTEXT_DIM)
        if len(ctx) != SHOP_CONTEXT_DIM:
            ctx = (ctx + [0.0] * SHOP_CONTEXT_DIM)[:SHOP_CONTEXT_DIM]
        out.append(
            {
                "ctx": ctx,
                "target": int(r.get("shop_expert_action_id")),
                "legal": list(r.get("shop_legal_action_ids") or []),
                "value_target": float(r.get("value_target", r.get("reward", 0.0))),
            }
        )
    return out


def main() -> int:
    args = parse_args()
    logger = setup_logger("trainer.eval_pv")
    warn_if_unstable_python(logger)

    torch = _require_torch()
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    model = PolicyValueModel(args.max_actions, args.max_shop_actions).to(device)
    state = torch.load(args.model, map_location=device)
    model.load_state_dict(state, strict=True)
    model.eval()

    hand_rows = _hand_rows(args.dataset)
    shop_rows = _shop_rows(args.dataset)
    if not hand_rows:
        logger.error("no hand rows in dataset")
        return 2

    hand_total = 0
    hand_acc1 = 0
    hand_acc3 = 0
    hand_illegal = 0
    hand_val_abs = 0.0

    for row in hand_rows:
        batch = {
            "rank": torch.tensor([row["rank"]], dtype=torch.long, device=device),
            "suit": torch.tensor([row["suit"]], dtype=torch.long, device=device),
            "chip": torch.tensor([row["chip"]], dtype=torch.float32, device=device),
            "enh": torch.tensor([row["enh"]], dtype=torch.float32, device=device),
            "edt": torch.tensor([row["edt"]], dtype=torch.float32, device=device),
            "seal": torch.tensor([row["seal"]], dtype=torch.float32, device=device),
            "pad": torch.tensor([row["pad"]], dtype=torch.float32, device=device),
            "context": torch.tensor([row["context"]], dtype=torch.float32, device=device),
        }
        with torch.no_grad():
            logits, value = model.forward_hand(batch)

        legal = [int(x) for x in row["legal"] if 0 <= int(x) < args.max_actions]
        if not legal:
            continue
        mask = torch.zeros((1, args.max_actions), dtype=torch.float32, device=device)
        for aid in legal:
            mask[0, aid] = 1.0
        masked = torch.where(mask > 0, logits, torch.full_like(logits, -1e9))
        top1 = int(masked.argmax(dim=1).item())
        top3 = torch.topk(masked, k=min(3, masked.shape[1]), dim=1).indices[0].tolist()

        hand_total += 1
        if top1 == row["target"]:
            hand_acc1 += 1
        if row["target"] in top3:
            hand_acc3 += 1
        if top1 not in legal:
            hand_illegal += 1
        hand_val_abs += abs(float(value.item()) - float(row["value_target"]))

    shop_total = 0
    shop_acc1 = 0
    shop_illegal = 0
    shop_val_abs = 0.0

    for row in shop_rows:
        with torch.no_grad():
            logits, value = model.forward_shop({"shop_context": torch.tensor([row["ctx"]], dtype=torch.float32, device=device)})
        legal = [int(x) for x in row["legal"] if 0 <= int(x) < args.max_shop_actions]
        if not legal:
            continue
        mask = torch.zeros((1, args.max_shop_actions), dtype=torch.float32, device=device)
        for aid in legal:
            mask[0, aid] = 1.0
        masked = torch.where(mask > 0, logits, torch.full_like(logits, -1e9))
        top1 = int(masked.argmax(dim=1).item())

        shop_total += 1
        if top1 == row["target"]:
            shop_acc1 += 1
        if top1 not in legal:
            shop_illegal += 1
        shop_val_abs += abs(float(value.item()) - float(row["value_target"]))

    out = {
        "generated_at": timestamp(),
        "model": args.model,
        "dataset": args.dataset,
        "hand": {
            "total": hand_total,
            "top1": (hand_acc1 / hand_total) if hand_total else 0.0,
            "top3": (hand_acc3 / hand_total) if hand_total else 0.0,
            "illegal_rate": (hand_illegal / hand_total) if hand_total else 0.0,
            "value_mae": (hand_val_abs / hand_total) if hand_total else 0.0,
        },
        "shop": {
            "total": shop_total,
            "top1": (shop_acc1 / shop_total) if shop_total else 0.0,
            "illegal_rate": (shop_illegal / shop_total) if shop_total else 0.0,
            "value_mae": (shop_val_abs / shop_total) if shop_total else 0.0,
        },
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    logger.info("wrote %s", out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
