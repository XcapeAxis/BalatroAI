"""Train a deploy student model from ensemble distillation data (multi-head hand+shop)."""
from __future__ import annotations

if __package__ is None or __package__ == "":
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).resolve().parent.parent))

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from trainer import action_space, action_space_shop
from trainer.features_shop import SHOP_CONTEXT_DIM


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _require_torch():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    return torch, nn, F


def _build_student(nn, max_actions: int, max_shop_actions: int):
    """Lightweight student: smaller hidden dims than the full policy model."""

    class DeployStudent(nn.Module):
        def __init__(self):
            super().__init__()
            self.rank_emb = nn.Embedding(16, 8)
            self.suit_emb = nn.Embedding(8, 4)
            self.card_proj = nn.Sequential(nn.Linear(8 + 4 + 4, 32), nn.ReLU(), nn.Linear(32, 32), nn.ReLU())
            self.ctx_proj = nn.Sequential(nn.Linear(12, 32), nn.ReLU(), nn.Linear(32, 32), nn.ReLU())
            self.hand_head = nn.Sequential(nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, max_actions))
            self.shop_proj = nn.Sequential(nn.Linear(SHOP_CONTEXT_DIM, 32), nn.ReLU(), nn.Linear(32, 32), nn.ReLU())
            self.shop_head = nn.Sequential(nn.Linear(32, 32), nn.ReLU(), nn.Linear(32, max_shop_actions))

        def forward_hand(self, batch):
            import torch
            r = self.rank_emb(batch["rank"])
            s = self.suit_emb(batch["suit"])
            card_x = torch.cat([r, s, batch["chip"].unsqueeze(-1), batch["enh"].unsqueeze(-1),
                                batch["edt"].unsqueeze(-1), batch["seal"].unsqueeze(-1)], dim=-1)
            card_h = self.card_proj(card_x)
            pad = batch["pad"]
            pad_sum = pad.sum(dim=1, keepdim=True).clamp(min=1.0)
            pooled = (card_h * pad.unsqueeze(-1)).sum(dim=1) / pad_sum
            ctx_h = self.ctx_proj(batch["context"])
            fused = torch.cat([pooled, ctx_h], dim=-1)
            return self.hand_head(fused)

        def forward_shop(self, batch):
            h = self.shop_proj(batch["shop_context"])
            return self.shop_head(h)

        def forward(self, batch):
            return self.forward_hand(batch)

    return DeployStudent()


def _parse_distill_record(record: dict, max_hand: int) -> dict[str, Any]:
    features = record.get("state_features") or {}
    phase = record.get("phase", "HAND")
    topk = record.get("teacher_topk") or []
    return {"phase": phase, "features": features, "teacher_topk": topk}


def _prepare_hand_batch(records: list[dict], torch, device, max_actions: int):
    rank_ids, suit_ids, chips, enhs, edts, seals, pads, ctxs = [], [], [], [], [], [], [], []
    targets = []
    for r in records:
        f = r["features"]
        rank_ids.append(f.get("card_rank_ids", [0] * action_space.MAX_HAND))
        suit_ids.append(f.get("card_suit_ids", [0] * action_space.MAX_HAND))
        chips.append(list(f.get("card_chip_hint") or [0.0] * action_space.MAX_HAND))
        enhs.append(f.get("card_has_enhancement", [0.0] * action_space.MAX_HAND))
        edts.append(f.get("card_has_edition", [0.0] * action_space.MAX_HAND))
        seals.append(f.get("card_has_seal", [0.0] * action_space.MAX_HAND))
        pads.append(f.get("hand_pad_mask", [1.0] * action_space.MAX_HAND))
        ctxs.append(f.get("context", [0.0] * 12))
        topk = r.get("teacher_topk") or [0]
        targets.append(topk[0] if topk else 0)

    batch = {
        "rank": torch.tensor(rank_ids, dtype=torch.long, device=device),
        "suit": torch.tensor(suit_ids, dtype=torch.long, device=device),
        "chip": torch.tensor(chips, dtype=torch.float32, device=device),
        "enh": torch.tensor(enhs, dtype=torch.float32, device=device),
        "edt": torch.tensor(edts, dtype=torch.float32, device=device),
        "seal": torch.tensor(seals, dtype=torch.float32, device=device),
        "pad": torch.tensor(pads, dtype=torch.float32, device=device),
        "context": torch.tensor(ctxs, dtype=torch.float32, device=device),
    }
    labels = torch.tensor(targets, dtype=torch.long, device=device)
    return batch, labels


def _prepare_shop_batch(records: list[dict], torch, device, max_shop_actions: int):
    ctxs = []
    targets = []
    for r in records:
        f = r["features"]
        ctx = list(f.get("shop_context") or [0.0] * SHOP_CONTEXT_DIM)
        if len(ctx) != SHOP_CONTEXT_DIM:
            ctx = (ctx + [0.0] * SHOP_CONTEXT_DIM)[:SHOP_CONTEXT_DIM]
        ctxs.append(ctx)
        topk = r.get("teacher_topk") or [0]
        targets.append(topk[0] if topk else 0)

    batch = {"shop_context": torch.tensor(ctxs, dtype=torch.float32, device=device)}
    labels = torch.tensor(targets, dtype=torch.long, device=device)
    return batch, labels


def main() -> int:
    p = argparse.ArgumentParser(description="Train deploy student from distillation data.")
    p.add_argument("--train-jsonl", required=True)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--artifacts-dir", default="")
    p.add_argument("--device", default="auto")
    args = p.parse_args()

    torch, nn, F = _require_torch()
    device_str = "cuda" if args.device == "auto" and torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)

    max_a = action_space.max_actions()
    max_sa = action_space_shop.max_actions()
    model = _build_student(nn, max_a, max_sa).to(device)

    # Load data
    hand_records: list[dict] = []
    shop_records: list[dict] = []
    data_path = Path(args.train_jsonl)
    for line in data_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        parsed = _parse_distill_record(rec, action_space.MAX_HAND)
        if parsed["phase"] == "HAND":
            hand_records.append(parsed)
        elif parsed["phase"] == "SHOP":
            shop_records.append(parsed)

    print(f"loaded hand={len(hand_records)} shop={len(shop_records)} records")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    art_dir = Path(args.artifacts_dir) if args.artifacts_dir else out_dir
    art_dir.mkdir(parents=True, exist_ok=True)
    log_lines: list[str] = []

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    best_loss = float("inf")
    train_start = time.time()

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        # Hand batches
        import random
        rng = random.Random(epoch)
        indices = list(range(len(hand_records)))
        rng.shuffle(indices)
        for i in range(0, len(indices), args.batch_size):
            batch_idx = indices[i:i + args.batch_size]
            batch_recs = [hand_records[j] for j in batch_idx]
            batch, labels = _prepare_hand_batch(batch_recs, torch, device, max_a)
            logits = model.forward_hand(batch)
            loss = F.cross_entropy(logits, labels.clamp(0, max_a - 1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        # Shop batches
        shop_indices = list(range(len(shop_records)))
        rng.shuffle(shop_indices)
        for i in range(0, len(shop_indices), args.batch_size):
            batch_idx = shop_indices[i:i + args.batch_size]
            batch_recs = [shop_records[j] for j in batch_idx]
            batch, labels = _prepare_shop_batch(batch_recs, torch, device, max_sa)
            logits = model.forward_shop(batch)
            loss = F.cross_entropy(logits, labels.clamp(0, max_sa - 1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / max(n_batches, 1)
        msg = f"epoch={epoch} avg_loss={avg_loss:.6f} batches={n_batches}"
        print(msg)
        log_lines.append(msg)

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), out_dir / "best.pt")

    train_time = time.time() - train_start

    # Measure inference latency
    model.eval()
    latencies: list[float] = []
    if hand_records:
        for rec in hand_records[:100]:
            batch, _ = _prepare_hand_batch([rec], torch, device, max_a)
            t0 = time.perf_counter()
            with torch.no_grad():
                model.forward_hand(batch)
            latencies.append((time.perf_counter() - t0) * 1000)

    avg_latency_ms = sum(latencies) / len(latencies) if latencies else 0.0
    p95_latency_ms = sorted(latencies)[int(len(latencies) * 0.95)] if latencies else 0.0

    # Save final model
    torch.save(model.state_dict(), out_dir / "final.pt")
    if not (out_dir / "best.pt").exists():
        torch.save(model.state_dict(), out_dir / "best.pt")

    summary: dict[str, Any] = {
        "schema": "distill_train_summary_v1",
        "generated_at": _now_iso(),
        "train_jsonl": str(data_path),
        "hand_records": len(hand_records),
        "shop_records": len(shop_records),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "best_loss": round(best_loss, 6),
        "train_time_sec": round(train_time, 2),
        "avg_inference_latency_ms": round(avg_latency_ms, 3),
        "p95_inference_latency_ms": round(p95_latency_ms, 3),
        "model_path": str(out_dir / "best.pt"),
        "device": device_str,
    }
    (art_dir / "distill_train_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (art_dir / "distill_train_log.txt").write_text("\n".join(log_lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
