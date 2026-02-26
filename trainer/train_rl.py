from __future__ import annotations

if __package__ is None or __package__ == "":
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).resolve().parent.parent))

import argparse
import json
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from trainer import action_space
from trainer import action_space_shop
from trainer.algos.awr import train_awr_epoch
from trainer.dataset import iter_shop_samples, iter_train_samples
from trainer.env_client import create_backend
from trainer.features import extract_features
from trainer.features_shop import SHOP_CONTEXT_DIM, extract_shop_features
from trainer.models.policy_value import PolicyValueModel
from trainer.replay_buffer import ReplayBuffer, ReplayItem
from trainer.utils import setup_logger, warn_if_unstable_python


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _require_torch():
    try:
        import torch
        import torch.nn.functional as F
    except Exception as exc:
        raise RuntimeError("PyTorch is required for train_rl.py") from exc
    return torch, F


def _load_cfg(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        try:
            import yaml  # type: ignore

            payload = yaml.safe_load(text)
            return payload if isinstance(payload, dict) else {}
        except Exception as exc:
            raise RuntimeError(f"failed to parse config: {path}") from exc


def _to_hand_payload(record: dict[str, Any]) -> dict[str, Any]:
    f = record["features"]
    return {
        "rank": list(f["card_rank_ids"]),
        "suit": list(f["card_suit_ids"]),
        "chip": list(f.get("card_chip_hint") or [0.0] * action_space.MAX_HAND),
        "enh": list(f["card_has_enhancement"]),
        "edt": list(f["card_has_edition"]),
        "seal": list(f["card_has_seal"]),
        "pad": list(f["hand_pad_mask"]),
        "context": list(f["context"]),
    }


def _to_shop_payload(record: dict[str, Any]) -> dict[str, Any]:
    sf = record.get("shop_features") if isinstance(record.get("shop_features"), dict) else {}
    ctx = list(sf.get("shop_context") or [0.0] * SHOP_CONTEXT_DIM)
    if len(ctx) != SHOP_CONTEXT_DIM:
        ctx = (ctx + [0.0] * SHOP_CONTEXT_DIM)[:SHOP_CONTEXT_DIM]
    return {"shop_context": ctx}


def _read_offline_into_buffer(dataset_path: Path, buffer: ReplayBuffer) -> tuple[int, int]:
    hand = 0
    shop = 0
    for r in iter_train_samples(dataset_path):
        legal = [int(x) for x in (r.get("legal_action_ids") or [])]
        aid = int(r.get("expert_action_id"))
        value_target = float(r.get("value_target", r.get("reward", 0.0)))
        buffer.add(
            ReplayItem(
                phase="SELECTING_HAND",
                source="offline",
                stake=str(r.get("stake") or "unknown"),
                action_id=aid,
                legal_ids=legal,
                value_target=value_target,
                reward=float(r.get("reward", 0.0)),
                payload=_to_hand_payload(r),
                tags=["offline"],
            )
        )
        hand += 1
    for r in iter_shop_samples(dataset_path):
        legal = [int(x) for x in (r.get("shop_legal_action_ids") or [])]
        aid = int(r.get("shop_expert_action_id"))
        value_target = float(r.get("value_target", r.get("reward", 0.0)))
        buffer.add(
            ReplayItem(
                phase="SHOP",
                source="offline",
                stake=str(r.get("stake") or "unknown"),
                action_id=aid,
                legal_ids=legal,
                value_target=value_target,
                reward=float(r.get("reward", 0.0)),
                payload=_to_shop_payload(r),
                tags=["offline"],
            )
        )
        shop += 1
    return hand, shop


def _sample_stake(plan: dict[str, Any], rng: random.Random) -> str:
    probs = plan.get("sampling_probabilities")
    if not isinstance(probs, dict) or not probs:
        return "gold"
    items = [(str(k), float(v)) for k, v in probs.items()]
    total = sum(max(0.0, v) for _, v in items)
    if total <= 0:
        return items[0][0]
    p = rng.random() * total
    acc = 0.0
    for k, v in items:
        acc += max(0.0, v)
        if p <= acc:
            return k
    return items[-1][0]


def _mask_argmax(logits, legal_ids: list[int], width: int, torch, device):
    legal = [int(a) for a in legal_ids if 0 <= int(a) < width]
    if not legal:
        return None
    mask = torch.zeros((1, width), dtype=torch.float32, device=device)
    for aid in legal:
        mask[0, aid] = 1.0
    masked = torch.where(mask > 0, logits, torch.full_like(logits, -1e9))
    return int(masked.argmax(dim=1).item())


def _collect_online(
    *,
    model,
    torch,
    device,
    cfg: dict[str, Any],
    mode: str,
    seed_prefix: str,
    curriculum_plan: dict[str, Any],
    logger,
) -> tuple[list[ReplayItem], dict[str, float]]:
    rng = random.Random(7)
    backend = create_backend("sim", seed=seed_prefix, timeout_sec=8.0, logger=logger)
    rows: list[ReplayItem] = []
    rewards = {
        "score_delta_term": 0.0,
        "survival_term": 0.0,
        "resource_term": 0.0,
        "econ_term": 0.0,
        "ante_progress_term": 0.0,
        "illegal_action_penalty": 0.0,
        "total": 0.0,
    }
    reward_w = cfg.get("reward_terms") if isinstance(cfg.get("reward_terms"), dict) else {}
    epsilon = float(cfg.get("epsilon_explore") or 0.05)
    episodes = int(cfg.get("episodes_for_online") or 16)
    max_steps = int(cfg.get("max_steps_per_episode") or 180)
    online_steps_target = int(cfg.get("online_steps") or 400)
    if mode == "smoke":
        episodes = min(episodes, 8)
        max_steps = min(max_steps, 120)
        online_steps_target = min(online_steps_target, 220)

    model.eval()
    collected = 0
    for ep in range(episodes):
        stake = _sample_stake(curriculum_plan, rng)
        ep_seed = f"{seed_prefix}-p18rl-{ep}"
        state = backend.reset(seed=ep_seed)
        if stake and stake.lower() not in {"", "white"}:
            try:
                state, _, _, _ = backend.step({"action_type": "START", "seed": ep_seed, "stake": stake.upper()})
            except Exception:
                pass
        prev = state
        for _ in range(max_steps):
            phase = str(state.get("state") or "")
            if phase == "GAME_OVER":
                break

            action = {"action_type": "WAIT"}
            action_id = -1
            legal_ids: list[int] = []
            payload: dict[str, Any] = {}
            row_phase = "OTHER"

            if phase == "SELECTING_HAND":
                row_phase = "SELECTING_HAND"
                f = extract_features(state)
                hand_size = min(len((state.get("hand") or {}).get("cards") or []), action_space.MAX_HAND)
                legal_ids = action_space.legal_action_ids(hand_size) if hand_size > 0 else []
                payload = {
                    "rank": list(f["card_rank_ids"]),
                    "suit": list(f["card_suit_ids"]),
                    "chip": list(f.get("card_chip_hint") or [0.0] * action_space.MAX_HAND),
                    "enh": list(f["card_has_enhancement"]),
                    "edt": list(f["card_has_edition"]),
                    "seal": list(f["card_has_seal"]),
                    "pad": list(f["hand_pad_mask"]),
                    "context": list(f["context"]),
                }
                if legal_ids:
                    if rng.random() < epsilon:
                        action_id = int(rng.choice(legal_ids))
                    else:
                        batch = {k: torch.tensor([v], dtype=(torch.long if k in {"rank", "suit"} else torch.float32), device=device) for k, v in payload.items()}
                        with torch.no_grad():
                            logits, _ = model.forward_hand(batch)
                        pred = _mask_argmax(logits, legal_ids, action_space.max_actions(), torch, device)
                        action_id = int(pred if pred is not None else legal_ids[0])
                    atype, mask_int = action_space.decode(hand_size, action_id)
                    action = {"action_type": atype, "indices": action_space.mask_to_indices(mask_int, hand_size)}

            elif phase in action_space_shop.SHOP_PHASES:
                row_phase = "SHOP"
                sf = extract_shop_features(state)
                ctx = list(sf.get("shop_context") or [0.0] * SHOP_CONTEXT_DIM)
                if len(ctx) != SHOP_CONTEXT_DIM:
                    ctx = (ctx + [0.0] * SHOP_CONTEXT_DIM)[:SHOP_CONTEXT_DIM]
                payload = {"shop_context": ctx}
                legal_ids = action_space_shop.legal_action_ids(state)
                if legal_ids:
                    if rng.random() < epsilon:
                        action_id = int(rng.choice(legal_ids))
                    else:
                        batch = {"shop_context": torch.tensor([ctx], dtype=torch.float32, device=device)}
                        with torch.no_grad():
                            logits, _ = model.forward_shop(batch)
                        pred = _mask_argmax(logits, legal_ids, action_space_shop.max_actions(), torch, device)
                        action_id = int(pred if pred is not None else legal_ids[0])
                    action = action_space_shop.action_from_id(state, action_id)
            else:
                # keep loop moving
                action = {"action_type": "WAIT", "sleep": 0.01}

            try:
                next_state, _, done_flag, _ = backend.step(action)
            except Exception:
                # no silent crash, skip this step
                break

            round_prev = prev.get("round") if isinstance(prev.get("round"), dict) else {}
            round_cur = next_state.get("round") if isinstance(next_state.get("round"), dict) else {}
            score_delta = float(round_cur.get("chips") or 0.0) - float(round_prev.get("chips") or 0.0)
            h_prev = float(round_prev.get("hands_left") or 0.0)
            d_prev = float(round_prev.get("discards_left") or 0.0)
            h_cur = float(round_cur.get("hands_left") or 0.0)
            d_cur = float(round_cur.get("discards_left") or 0.0)
            res_delta = (h_cur + d_cur) - (h_prev + d_prev)
            money_prev = float(prev.get("money") or 0.0)
            money_cur = float(next_state.get("money") or 0.0)
            money_delta = money_cur - money_prev
            ante_prev = float(round_prev.get("ante") or prev.get("ante_num") or 0.0)
            ante_cur = float(round_cur.get("ante") or next_state.get("ante_num") or 0.0)
            ante_delta = ante_cur - ante_prev
            survival = -1.0 if str(next_state.get("state") or "") == "GAME_OVER" else 0.1
            illegal_penalty = 0.0
            if action_id >= 0 and legal_ids and action_id not in legal_ids:
                illegal_penalty = -1.0

            reward = (
                float(reward_w.get("score_delta_term", 1.0)) * score_delta
                + float(reward_w.get("resource_term", 0.3)) * res_delta
                + float(reward_w.get("econ_term", 0.15)) * money_delta
                + float(reward_w.get("ante_progress_term", 3.0)) * ante_delta
                + float(reward_w.get("survival_term", 2.5)) * survival
                + float(reward_w.get("illegal_action_penalty", -4.0)) * (1.0 if illegal_penalty < 0 else 0.0)
            )
            rewards["score_delta_term"] += float(reward_w.get("score_delta_term", 1.0)) * score_delta
            rewards["resource_term"] += float(reward_w.get("resource_term", 0.3)) * res_delta
            rewards["econ_term"] += float(reward_w.get("econ_term", 0.15)) * money_delta
            rewards["ante_progress_term"] += float(reward_w.get("ante_progress_term", 3.0)) * ante_delta
            rewards["survival_term"] += float(reward_w.get("survival_term", 2.5)) * survival
            rewards["illegal_action_penalty"] += float(reward_w.get("illegal_action_penalty", -4.0)) * (1.0 if illegal_penalty < 0 else 0.0)
            rewards["total"] += reward

            if row_phase in {"SELECTING_HAND", "SHOP"} and action_id >= 0 and legal_ids:
                rows.append(
                    ReplayItem(
                        phase=row_phase,
                        source="online",
                        stake=str(stake),
                        action_id=int(action_id),
                        legal_ids=[int(x) for x in legal_ids],
                        value_target=float(reward),
                        reward=float(reward),
                        payload=payload,
                        tags=["online", "p18_rl"],
                    )
                )
                collected += 1
                if collected >= online_steps_target:
                    backend.close()
                    return rows, rewards

            prev = next_state
            state = next_state
            if done_flag:
                break

    backend.close()
    return rows, rewards


def _as_rows(buffer: ReplayBuffer, *, phase: str, limit: int) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in buffer.sample(limit, phase=phase, rng=random.Random(11)):
        if phase == "SELECTING_HAND":
            p = row.payload
            out.append(
                {
                    "rank": p["rank"],
                    "suit": p["suit"],
                    "chip": p["chip"],
                    "enh": p["enh"],
                    "edt": p["edt"],
                    "seal": p["seal"],
                    "pad": p["pad"],
                    "context": p["context"],
                    "action_id": row.action_id,
                    "legal_ids": row.legal_ids,
                    "value_target": row.value_target,
                }
            )
        elif phase == "SHOP":
            out.append(
                {
                    "shop_context": row.payload["shop_context"],
                    "action_id": row.action_id,
                    "legal_ids": row.legal_ids,
                    "value_target": row.value_target,
                }
            )
    return out


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="P18 RL pilot trainer (AWR-lite).")
    p.add_argument("--config", default="trainer/config/p18_rl.yaml")
    p.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    p.add_argument("--curriculum-plan", default="")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--artifacts-dir", required=True)
    p.add_argument("--warm-start-model", default="")
    p.add_argument("--seed", type=int, default=7)
    return p


def main() -> int:
    args = _build_parser().parse_args()
    logger = setup_logger("trainer.train_rl")
    warn_if_unstable_python(logger)

    cfg = _load_cfg(Path(args.config))
    algo = str(cfg.get("algo") or "awr").lower()
    if algo not in {"awr", "awac", "awac-lite", "awr-lite"}:
        logger.error("unsupported algo: %s", algo)
        return 2

    torch, F = _require_torch()
    random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))

    out_dir = Path(args.out_dir)
    art_dir = Path(args.artifacts_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    art_dir.mkdir(parents=True, exist_ok=True)

    warm = Path(args.warm_start_model) if args.warm_start_model else Path(str(cfg.get("warm_start_model") or ""))
    if not warm.exists():
        logger.error("warm_start_model missing: %s", warm)
        return 2

    model = PolicyValueModel(action_space.max_actions(), action_space_shop.max_actions())
    state_dict = torch.load(str(warm), map_location="cpu")
    model.load_state_dict(state_dict, strict=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg.get("learning_rate") or 5e-4),
        weight_decay=float(cfg.get("weight_decay") or 1e-4),
    )

    plan: dict[str, Any] = {}
    if args.curriculum_plan and Path(args.curriculum_plan).exists():
        try:
            plan = json.loads(Path(args.curriculum_plan).read_text(encoding="utf-8"))
        except Exception:
            plan = {}

    buffer = ReplayBuffer()
    offline_path = Path(str(cfg.get("offline_dataset") or "trainer_data/p17_smoke_search.jsonl"))
    if not offline_path.exists():
        logger.error("offline dataset missing: %s", offline_path)
        return 2
    offline_hand, offline_shop = _read_offline_into_buffer(offline_path, buffer)

    offline_steps = int(cfg.get("offline_steps") or 600)
    online_steps = int(cfg.get("online_steps") or 400)
    if args.mode == "smoke":
        offline_steps = min(offline_steps, 300)
        online_steps = min(online_steps, 220)

    hand_rows = _as_rows(buffer, phase="SELECTING_HAND", limit=offline_steps)
    shop_rows = _as_rows(buffer, phase="SHOP", limit=offline_steps)
    off_metrics = train_awr_epoch(
        model=model,
        optimizer=opt,
        torch=torch,
        F=F,
        hand_rows=hand_rows,
        shop_rows=shop_rows,
        batch_size=int(cfg.get("batch_size") or 64),
        beta=float(cfg.get("beta") or 1.0),
        max_weight=float(cfg.get("max_weight") or 20.0),
        value_weight=0.2,
        max_actions=action_space.max_actions(),
        max_shop_actions=action_space_shop.max_actions(),
        device=device,
        seed=int(args.seed),
    )
    torch.save(model.state_dict(), out_dir / "offline.pt")

    online_rows, reward_terms = _collect_online(
        model=model,
        torch=torch,
        device=device,
        cfg={**cfg, "online_steps": online_steps},
        mode=args.mode,
        seed_prefix="AAAAAAA",
        curriculum_plan=plan,
        logger=logger,
    )
    buffer.extend(online_rows)

    online_hand_rows = [
        {
            "rank": r.payload["rank"],
            "suit": r.payload["suit"],
            "chip": r.payload["chip"],
            "enh": r.payload["enh"],
            "edt": r.payload["edt"],
            "seal": r.payload["seal"],
            "pad": r.payload["pad"],
            "context": r.payload["context"],
            "action_id": r.action_id,
            "legal_ids": r.legal_ids,
            "value_target": r.value_target,
        }
        for r in online_rows
        if r.phase == "SELECTING_HAND"
    ]
    online_shop_rows = [
        {
            "shop_context": r.payload["shop_context"],
            "action_id": r.action_id,
            "legal_ids": r.legal_ids,
            "value_target": r.value_target,
        }
        for r in online_rows
        if r.phase == "SHOP"
    ]
    on_metrics = train_awr_epoch(
        model=model,
        optimizer=opt,
        torch=torch,
        F=F,
        hand_rows=online_hand_rows,
        shop_rows=online_shop_rows,
        batch_size=int(cfg.get("batch_size") or 64),
        beta=float(cfg.get("beta") or 1.0),
        max_weight=float(cfg.get("max_weight") or 20.0),
        value_weight=0.2,
        max_actions=action_space.max_actions(),
        max_shop_actions=action_space_shop.max_actions(),
        device=device,
        seed=int(args.seed) + 1,
    )

    torch.save(model.state_dict(), out_dir / "last.pt")
    torch.save(model.state_dict(), out_dir / "best.pt")

    stats = buffer.composition_stats()
    summary = {
        "schema": "p18_rl_train_summary_v1",
        "generated_at": _now_iso(),
        "algo": "awr_lite",
        "mode": args.mode,
        "config": str(Path(args.config)),
        "warm_start_model": str(warm),
        "output_model": str(out_dir / "best.pt"),
        "offline_counts": {"hand": offline_hand, "shop": offline_shop},
        "online_counts": {
            "total": len(online_rows),
            "hand": sum(1 for r in online_rows if r.phase == "SELECTING_HAND"),
            "shop": sum(1 for r in online_rows if r.phase == "SHOP"),
        },
        "buffer_stats": stats,
        "offline_metrics": off_metrics,
        "online_metrics": on_metrics,
        "reward_breakdown": reward_terms,
    }
    (art_dir / "rl_train_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (art_dir / "rl_checkpoints_manifest.json").write_text(
        json.dumps(
            {
                "schema": "p18_rl_ckpt_manifest_v1",
                "generated_at": _now_iso(),
                "offline": str(out_dir / "offline.pt"),
                "best": str(out_dir / "best.pt"),
                "last": str(out_dir / "last.pt"),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (art_dir / "rl_train_log.txt").write_text(
        "\n".join(
            [
                "P18 RL Train",
                f"algo=awr_lite mode={args.mode}",
                f"warm_start_model={warm}",
                f"offline_metrics={off_metrics}",
                f"online_metrics={on_metrics}",
                f"reward_breakdown={reward_terms}",
                f"buffer_stats={stats}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": "ok", "out_dir": str(out_dir), "artifacts_dir": str(art_dir), "online_total": len(online_rows)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
