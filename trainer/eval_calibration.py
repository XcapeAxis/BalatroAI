from __future__ import annotations

if __package__ is None or __package__ == "":
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).resolve().parent.parent))

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from trainer import action_space
from trainer import action_space_shop
from trainer.calibration import compute_ece
from trainer.env_client import create_backend
from trainer.expert_policy import choose_action
from trainer.expert_policy_shop import choose_shop_action
from trainer.features import extract_features
from trainer.features_shop import extract_shop_features
from trainer.models.policy_value import PolicyValueModel
from trainer.utils import timestamp


def _require_torch():
    import torch

    return torch


def _state_to_hand_batch(state, torch):
    f = extract_features(state)
    chip_hint = list(f.get("card_chip_hint") or [0] * action_space.MAX_HAND)
    return {
        "rank": torch.tensor([f["card_rank_ids"]], dtype=torch.long),
        "suit": torch.tensor([f["card_suit_ids"]], dtype=torch.long),
        "chip": torch.tensor([chip_hint], dtype=torch.float32),
        "enh": torch.tensor([f["card_has_enhancement"]], dtype=torch.float32),
        "edt": torch.tensor([f["card_has_edition"]], dtype=torch.float32),
        "seal": torch.tensor([f["card_has_seal"]], dtype=torch.float32),
        "pad": torch.tensor([f["hand_pad_mask"]], dtype=torch.float32),
        "context": torch.tensor([f["context"]], dtype=torch.float32),
    }


def _state_to_shop_batch(state, torch):
    sf = extract_shop_features(state)
    return {"shop_context": torch.tensor([sf["shop_context"]], dtype=torch.float32)}


def _masked_top1(logits, legal_ids: list[int], max_actions: int, torch):
    legal = [int(x) for x in legal_ids if 0 <= int(x) < max_actions]
    if not legal:
        return None, 0.0
    mask = torch.zeros((1, max_actions), dtype=torch.float32, device=logits.device)
    for aid in legal:
        mask[0, aid] = 1.0
    masked = torch.where(mask > 0, logits, torch.full_like(logits, -1e9))
    probs = torch.softmax(masked, dim=1)
    top_vals, top_ids = torch.topk(probs, k=1, dim=1)
    return int(top_ids[0, 0].item()), float(top_vals[0, 0].item())


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="P19 calibration smoke evaluator.")
    p.add_argument("--backend", choices=["sim"], default="sim")
    p.add_argument("--stake", default="gold")
    p.add_argument("--episodes", type=int, default=50)
    p.add_argument("--seeds-file", required=True)
    p.add_argument("--pv-model", required=True)
    p.add_argument("--rl-model", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--max-steps-per-episode", type=int, default=240)
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    return p.parse_args()


def _load_model(path: str, max_actions: int, max_shop_actions: int, device, torch):
    state = torch.load(path, map_location=device)
    model = PolicyValueModel(max_actions, max_shop_actions).to(device)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def main() -> int:
    args = _parse_args()
    torch = _require_torch()
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    max_actions = action_space.max_actions()
    max_shop_actions = action_space_shop.max_actions()
    pv_model = _load_model(args.pv_model, max_actions, max_shop_actions, device, torch)
    rl_model = _load_model(args.rl_model, max_actions, max_shop_actions, device, torch)

    seeds = [line.strip() for line in Path(args.seeds_file).read_text(encoding="utf-8").splitlines() if line.strip()]
    if not seeds:
        raise RuntimeError("empty seeds file")

    backend = create_backend("sim", seed="AAAAAAA", timeout_sec=8.0, logger=None)
    records: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    try:
        for ep in range(int(args.episodes)):
            seed = seeds[ep % len(seeds)]
            state = backend.reset(seed=seed)
            if str(args.stake).lower() not in {"", "white"}:
                try:
                    state, _, _, _ = backend.step({"action_type": "START", "seed": seed, "stake": str(args.stake).upper()})
                except Exception:
                    pass
            for _ in range(int(args.max_steps_per_episode)):
                phase = str(state.get("state") or "")
                if phase == "GAME_OVER":
                    break
                if phase == "SELECTING_HAND":
                    hand = (state.get("hand") or {}).get("cards") or []
                    hand_size = min(len(hand), action_space.MAX_HAND)
                    if hand_size <= 0:
                        state, _, done, _ = backend.step({"action_type": "WAIT"})
                        if done:
                            break
                        continue
                    legal = action_space.legal_action_ids(hand_size)
                    batch = _state_to_hand_batch(state, torch)
                    batch = {k: v.to(device) for k, v in batch.items()}
                    with torch.no_grad():
                        pv_logits, _ = pv_model.forward_hand(batch)
                        rl_logits, _ = rl_model.forward_hand(batch)
                    pv_id, pv_conf = _masked_top1(pv_logits, legal, max_actions, torch)
                    rl_id, rl_conf = _masked_top1(rl_logits, legal, max_actions, torch)
                    teacher = choose_action(state, start_seed=seed)
                    teacher_id = None
                    if teacher.action_type and teacher.mask_int is not None:
                        try:
                            teacher_id = action_space.encode(hand_size, teacher.action_type, int(teacher.mask_int))
                        except Exception:
                            teacher_id = None
                    if pv_id is not None and teacher_id is not None:
                        records["pv"]["hand"].append({"conf": pv_conf, "ok": int(pv_id) == int(teacher_id)})
                    if rl_id is not None and teacher_id is not None:
                        records["rl"]["hand"].append({"conf": rl_conf, "ok": int(rl_id) == int(teacher_id)})
                    progress = {"action_type": "WAIT"}
                    if teacher.action_type and teacher.mask_int is not None:
                        progress = {
                            "action_type": str(teacher.action_type).upper(),
                            "indices": action_space.mask_to_indices(int(teacher.mask_int), hand_size),
                        }
                    try:
                        state, _, done, _ = backend.step(progress)
                    except Exception:
                        try:
                            state, _, done, _ = backend.step({"action_type": "AUTO"})
                        except Exception:
                            break
                    if done:
                        break
                    continue

                if phase in action_space_shop.SHOP_PHASES:
                    legal = action_space_shop.legal_action_ids(state)
                    batch = _state_to_shop_batch(state, torch)
                    batch = {k: v.to(device) for k, v in batch.items()}
                    with torch.no_grad():
                        pv_logits, _ = pv_model.forward_shop(batch)
                        rl_logits, _ = rl_model.forward_shop(batch)
                    pv_id, pv_conf = _masked_top1(pv_logits, legal, max_shop_actions, torch)
                    rl_id, rl_conf = _masked_top1(rl_logits, legal, max_shop_actions, torch)
                    teacher_action = choose_shop_action(state).action
                    teacher_id = None
                    try:
                        teacher_id = action_space_shop.encode(str(teacher_action.get("action_type") or "WAIT"), dict(teacher_action.get("params") or {}))
                    except Exception:
                        teacher_id = None
                    if pv_id is not None and teacher_id is not None:
                        records["pv"]["shop"].append({"conf": pv_conf, "ok": int(pv_id) == int(teacher_id)})
                    if rl_id is not None and teacher_id is not None:
                        records["rl"]["shop"].append({"conf": rl_conf, "ok": int(rl_id) == int(teacher_id)})
                    progress_shop = dict(teacher_action) if isinstance(teacher_action, dict) else {"action_type": "NEXT_ROUND"}
                    try:
                        state, _, done, _ = backend.step(progress_shop)
                    except Exception:
                        try:
                            state, _, done, _ = backend.step({"action_type": "NEXT_ROUND"})
                        except Exception:
                            break
                    if done:
                        break
                    continue

                try:
                    state, _, done, _ = backend.step({"action_type": "AUTO"})
                except Exception:
                    break
                if done:
                    break
    finally:
        backend.close()

    summary: dict[str, Any] = {
        "schema": "p19_calibration_v1",
        "generated_at": timestamp(),
        "episodes": int(args.episodes),
        "stake": str(args.stake),
        "models": {"pv": str(args.pv_model), "rl": str(args.rl_model)},
        "metrics": {},
    }

    for model_name in ("pv", "rl"):
        summary["metrics"][model_name] = {}
        for phase in ("hand", "shop"):
            rows = records.get(model_name, {}).get(phase, [])
            metric = compute_ece(rows, bins=10)
            metric["count"] = len(rows)
            summary["metrics"][model_name][phase] = metric

    (out_dir / "calibration_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    md_lines = [
        "# P19 Calibration Summary",
        "",
        f"- generated_at: {summary['generated_at']}",
        f"- episodes: {summary['episodes']}",
        f"- stake: {summary['stake']}",
        "",
        "## Metrics",
    ]
    for model_name in ("pv", "rl"):
        for phase in ("hand", "shop"):
            m = summary["metrics"][model_name][phase]
            md_lines.append(f"- {model_name}/{phase}: ECE={m['ece']:.4f}, count={m['count']}")
    (out_dir / "calibration_summary.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": "ok", "out_dir": str(out_dir)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
