"""Evaluate deploy student model: offline metrics (top1/top3/illegal) + long-horizon."""
from __future__ import annotations

if __package__ is None or __package__ == "":
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).resolve().parent.parent))

import argparse
import json
import subprocess
import sys
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
    return torch, nn


def _load_student(model_path: str, torch, nn, max_a: int, max_sa: int, device):
    from trainer.train_distill import _build_student
    model = _build_student(nn, max_a, max_sa)
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)
    model.to(device)
    model.eval()
    return model


def _offline_eval(model, data_path: str, torch, device, max_a: int, max_sa: int) -> dict[str, Any]:
    from trainer.features import extract_features

    hand_top1 = 0
    hand_top3 = 0
    hand_illegal = 0
    hand_total = 0
    shop_top1 = 0
    shop_top3 = 0
    shop_total = 0

    for line in Path(data_path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        phase = rec.get("phase", "HAND")
        topk = rec.get("teacher_topk") or []
        if not topk:
            continue
        target = topk[0]
        features = rec.get("state_features") or {}

        if phase == "HAND":
            rank_ids = features.get("card_rank_ids", [0] * action_space.MAX_HAND)
            suit_ids = features.get("card_suit_ids", [0] * action_space.MAX_HAND)
            chip_hint = list(features.get("card_chip_hint") or [0.0] * action_space.MAX_HAND)
            batch = {
                "rank": torch.tensor([rank_ids], dtype=torch.long, device=device),
                "suit": torch.tensor([suit_ids], dtype=torch.long, device=device),
                "chip": torch.tensor([chip_hint], dtype=torch.float32, device=device),
                "enh": torch.tensor([features.get("card_has_enhancement", [0.0] * action_space.MAX_HAND)], dtype=torch.float32, device=device),
                "edt": torch.tensor([features.get("card_has_edition", [0.0] * action_space.MAX_HAND)], dtype=torch.float32, device=device),
                "seal": torch.tensor([features.get("card_has_seal", [0.0] * action_space.MAX_HAND)], dtype=torch.float32, device=device),
                "pad": torch.tensor([features.get("hand_pad_mask", [1.0] * action_space.MAX_HAND)], dtype=torch.float32, device=device),
                "context": torch.tensor([features.get("context", [0.0] * 12)], dtype=torch.float32, device=device),
            }
            with torch.no_grad():
                logits = model.forward_hand(batch)
            pred_top3 = torch.topk(logits, k=min(3, logits.shape[1]), dim=1).indices[0].tolist()
            pred_top1 = pred_top3[0] if pred_top3 else -1

            hand_total += 1
            if pred_top1 == target:
                hand_top1 += 1
            if target in pred_top3:
                hand_top3 += 1
            if pred_top1 < 0 or pred_top1 >= max_a:
                hand_illegal += 1
        elif phase == "SHOP":
            ctx = list(features.get("shop_context") or [0.0] * SHOP_CONTEXT_DIM)
            if len(ctx) != SHOP_CONTEXT_DIM:
                ctx = (ctx + [0.0] * SHOP_CONTEXT_DIM)[:SHOP_CONTEXT_DIM]
            batch = {"shop_context": torch.tensor([ctx], dtype=torch.float32, device=device)}
            with torch.no_grad():
                logits = model.forward_shop(batch)
            pred_top3 = torch.topk(logits, k=min(3, logits.shape[1]), dim=1).indices[0].tolist()
            pred_top1 = pred_top3[0] if pred_top3 else -1
            shop_total += 1
            if pred_top1 == target:
                shop_top1 += 1
            if target in pred_top3:
                shop_top3 += 1

    return {
        "hand_top1_acc": round(hand_top1 / max(hand_total, 1), 4),
        "hand_top3_acc": round(hand_top3 / max(hand_total, 1), 4),
        "hand_illegal_rate": round(hand_illegal / max(hand_total, 1), 4),
        "hand_total": hand_total,
        "shop_top1_acc": round(shop_top1 / max(shop_total, 1), 4),
        "shop_top3_acc": round(shop_top3 / max(shop_total, 1), 4),
        "shop_total": shop_total,
    }


def _long_horizon_eval(model_path: str, seeds_file: str, episodes: int) -> dict[str, Any]:
    """Run eval_long_horizon with deploy_student policy."""
    out_json = Path(model_path).parent / "eval_lh_distill.json"
    cmd = [
        sys.executable, "-B", "trainer/eval_long_horizon.py",
        "--backend", "sim",
        "--stake", "gold",
        "--episodes", str(episodes),
        "--seeds-file", seeds_file,
        "--policy", "deploy_student",
        "--model", model_path,
        "--max-steps-per-episode", "120",
        "--out", str(out_json),
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if proc.returncode == 0 and out_json.exists():
            return json.loads(out_json.read_text(encoding="utf-8"))
        return {"error": f"returncode={proc.returncode}", "stderr": (proc.stderr or "")[-500:]}
    except Exception as e:
        return {"error": str(e)}


def main() -> int:
    p = argparse.ArgumentParser(description="Evaluate deploy student model.")
    p.add_argument("--model", required=True)
    p.add_argument("--dataset", default="", help="Distill jsonl for offline eval.")
    p.add_argument("--seeds-file", default="balatro_mechanics/derived/eval_seeds_100.txt")
    p.add_argument("--lh-episodes", type=int, default=100, help="Long-horizon episodes.")
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="auto")
    args = p.parse_args()

    torch, nn = _require_torch()
    device_str = "cuda" if args.device == "auto" and torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)

    max_a = action_space.max_actions()
    max_sa = action_space_shop.max_actions()
    model = _load_student(args.model, torch, nn, max_a, max_sa, device)

    result: dict[str, Any] = {
        "schema": "eval_distill_summary_v1",
        "generated_at": _now_iso(),
        "model_path": args.model,
    }

    if args.dataset and Path(args.dataset).exists():
        offline = _offline_eval(model, args.dataset, torch, device, max_a, max_sa)
        result["offline"] = offline
    else:
        result["offline"] = {"skipped": True, "reason": "no dataset provided"}

    # Inference latency benchmark
    latencies: list[float] = []
    dummy_batch = {
        "rank": torch.zeros((1, action_space.MAX_HAND), dtype=torch.long, device=device),
        "suit": torch.zeros((1, action_space.MAX_HAND), dtype=torch.long, device=device),
        "chip": torch.zeros((1, action_space.MAX_HAND), dtype=torch.float32, device=device),
        "enh": torch.zeros((1, action_space.MAX_HAND), dtype=torch.float32, device=device),
        "edt": torch.zeros((1, action_space.MAX_HAND), dtype=torch.float32, device=device),
        "seal": torch.zeros((1, action_space.MAX_HAND), dtype=torch.float32, device=device),
        "pad": torch.ones((1, action_space.MAX_HAND), dtype=torch.float32, device=device),
        "context": torch.zeros((1, 12), dtype=torch.float32, device=device),
    }
    for _ in range(200):
        t0 = time.perf_counter()
        with torch.no_grad():
            model.forward_hand(dummy_batch)
        latencies.append((time.perf_counter() - t0) * 1000)
    result["inference_benchmark"] = {
        "avg_ms": round(sum(latencies) / len(latencies), 3),
        "p95_ms": round(sorted(latencies)[int(len(latencies) * 0.95)], 3),
        "samples": len(latencies),
    }

    # Long-horizon eval
    if Path(args.seeds_file).exists():
        lh = _long_horizon_eval(args.model, args.seeds_file, args.lh_episodes)
        result["long_horizon"] = lh
    else:
        result["long_horizon"] = {"skipped": True, "reason": f"seeds file not found: {args.seeds_file}"}

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
