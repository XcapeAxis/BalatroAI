from __future__ import annotations

if __package__ is None or __package__ == "":
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).resolve().parent.parent))

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from trainer import action_space, action_space_shop
from trainer.dataset import JsonlWriter
from trainer.env_client import create_backend
from trainer.features import extract_features
from trainer.features_shop import extract_shop_features
from trainer.infer_assistant_real import _heuristic_hand_rankings, _heuristic_shop_rankings
from trainer.search_expert_full import choose_full_action
from trainer.utils import setup_logger, timestamp, warn_if_unstable_python


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                rows.append(json.loads(text))
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid json at line {line_no}: {exc}") from exc
    return rows


def _state_from_row(row: dict[str, Any]) -> dict[str, Any] | None:
    if isinstance(row.get("gamestate_raw_before"), dict):
        return row["gamestate_raw_before"]
    if isinstance(row.get("gamestate_raw"), dict):
        return row["gamestate_raw"]
    if isinstance(row.get("gamestate_raw_after"), dict):
        return row["gamestate_raw_after"]
    obs = row.get("gamestate_min")
    if not isinstance(obs, dict):
        return None

    phase = str(obs.get("phase") or row.get("phase") or "UNKNOWN")
    resources = obs.get("resources") if isinstance(obs.get("resources"), dict) else {}
    hand_block = obs.get("hand") if isinstance(obs.get("hand"), dict) else {}
    shop_root = obs.get("shop") if isinstance(obs.get("shop"), dict) else {}

    hand_cards: list[dict[str, Any]] = []
    raw_cards = hand_block.get("cards") if isinstance(hand_block.get("cards"), list) else []
    for card in raw_cards:
        if not isinstance(card, dict):
            continue
        rank = str(card.get("rank") or "")
        suit = str(card.get("suit") or "")
        hand_cards.append(
            {
                "key": str(card.get("key") or ""),
                "value": {
                    "rank": rank,
                    "suit": suit,
                    "effect": str(card.get("effect") or ""),
                },
                "modifier": list(card.get("modifier") or []),
                "state": list(card.get("state") or []),
            }
        )

    def _market(block: Any) -> dict[str, Any]:
        if not isinstance(block, dict):
            return {"count": 0, "cards": []}
        cards: list[dict[str, Any]] = []
        raw = block.get("cards") if isinstance(block.get("cards"), list) else []
        for i, item in enumerate(raw):
            if not isinstance(item, dict):
                continue
            cards.append(
                {
                    "index": int(item.get("index") or i),
                    "key": str(item.get("key") or ""),
                    "label": str(item.get("label") or ""),
                    "cost": float(item.get("cost") or 0.0),
                    "set": str(item.get("set") or ""),
                }
            )
        return {"count": int(block.get("count") or len(cards)), "cards": cards}

    return {
        "state": phase,
        "round": {
            "hands_left": int(resources.get("hands_left") or 0),
            "discards_left": int(resources.get("discards_left") or 0),
            "ante": int(resources.get("ante") or 0),
            "round_num": int(resources.get("round_num") or 0),
            "blind": str(resources.get("blind") or ""),
            "chips": float(resources.get("round_chips") or resources.get("score_chips") or 0.0),
        },
        "score": {
            "chips": float(resources.get("score_chips") or resources.get("round_chips") or 0.0),
            "mult": float(resources.get("score_mult") or 0.0),
            "target_chips": float(resources.get("target_chips") or 0.0),
        },
        "money": float(resources.get("money") or 0.0),
        "economy": {"money": float(resources.get("money") or 0.0)},
        "hand": {"cards": hand_cards},
        "shop": _market(shop_root.get("shop")),
        "vouchers": _market(shop_root.get("vouchers")),
        "packs": _market(shop_root.get("packs")),
        "consumables": _market(shop_root.get("consumables")),
    }


def _hand_record_from_state(
    state: dict[str, Any],
    *,
    episode_id: int,
    step_id: int,
    teacher_action: dict[str, Any],
    student_topk: list[dict[str, Any]],
    valid_reconstruct: bool,
    failure_reason: str | None,
) -> dict[str, Any] | None:
    hand_cards = ((state.get("hand") or {}).get("cards") or []) if isinstance(state.get("hand"), dict) else []
    hand_size = min(len(hand_cards), action_space.MAX_HAND)
    if hand_size <= 0:
        return None

    legal_ids = action_space.legal_action_ids(hand_size)
    at = str(teacher_action.get("action_type") or "PLAY").upper()
    idxs = [int(x) for x in (teacher_action.get("indices") or [])]
    try:
        mask = action_space.indices_to_mask(idxs)
        expert_action_id = action_space.encode(hand_size, at, mask)
    except Exception:
        expert_action_id = legal_ids[0] if legal_ids else None
    if expert_action_id is None:
        return None

    rec = {
        "schema_version": "record_v1",
        "timestamp": timestamp(),
        "episode_id": int(episode_id),
        "step_id": int(step_id),
        "instance_id": 0,
        "base_url": "dagger://real_session",
        "phase": "SELECTING_HAND",
        "done": False,
        "hand_size": hand_size,
        "legal_action_ids": legal_ids,
        "expert_action_id": int(expert_action_id),
        "macro_action": None,
        "reward": 0.0,
        "reward_info": "dagger_teacher",
        "features": extract_features(state),
        "shop_legal_action_ids": [],
        "shop_expert_action_id": None,
        "shop_features": None,
        "student_action_topk": student_topk,
        "teacher_action": teacher_action,
        "valid_reconstruct": bool(valid_reconstruct),
        "failure_reason": failure_reason,
    }
    return rec


def _shop_record_from_state(
    state: dict[str, Any],
    *,
    episode_id: int,
    step_id: int,
    teacher_action: dict[str, Any],
    teacher_action_id: int,
    student_topk: list[dict[str, Any]],
    valid_reconstruct: bool,
    failure_reason: str | None,
) -> dict[str, Any] | None:
    shop_legal = action_space_shop.legal_action_ids(state)
    if not shop_legal:
        return None
    if teacher_action_id not in shop_legal:
        teacher_action_id = shop_legal[0]
        teacher_action = action_space_shop.action_from_id(state, teacher_action_id)

    rec = {
        "schema_version": "record_v1",
        "timestamp": timestamp(),
        "episode_id": int(episode_id),
        "step_id": int(step_id),
        "instance_id": 0,
        "base_url": "dagger://real_session",
        "phase": str(state.get("state") or "SHOP"),
        "done": False,
        "hand_size": min(len(((state.get("hand") or {}).get("cards") or [])), action_space.MAX_HAND),
        "legal_action_ids": [],
        "expert_action_id": None,
        "macro_action": str(teacher_action.get("action_type") or "WAIT"),
        "reward": 0.0,
        "reward_info": "dagger_teacher",
        "features": extract_features(state),
        "shop_legal_action_ids": shop_legal,
        "shop_expert_action_id": int(teacher_action_id),
        "shop_features": extract_shop_features(state),
        "student_action_topk": student_topk,
        "teacher_action": teacher_action,
        "valid_reconstruct": bool(valid_reconstruct),
        "failure_reason": failure_reason,
    }
    return rec


def _macro_progress_action(state: dict[str, Any], seed: str) -> dict[str, Any]:
    phase = str(state.get("state") or "UNKNOWN")
    if phase == "BLIND_SELECT":
        return {"action_type": "SELECT", "index": 0}
    if phase == "ROUND_EVAL":
        return {"action_type": "CASH_OUT"}
    if phase == "SHOP":
        return {"action_type": "NEXT_ROUND"}
    if phase in {"GAME_OVER", "MENU"}:
        return {"action_type": "START", "seed": seed}
    return {"action_type": "WAIT"}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect DAgger dataset from real shadow session + sim teacher.")
    parser.add_argument("--session", required=True, help="session jsonl produced by record_real_session.py")
    parser.add_argument("--backend", choices=["sim"], default="sim")
    parser.add_argument("--out", required=True)
    parser.add_argument("--hand-samples", type=int, default=500)
    parser.add_argument("--shop-samples", type=int, default=200)
    parser.add_argument("--time-budget-ms", type=float, default=20.0)
    parser.add_argument("--hand-max-branch", type=int, default=80)
    parser.add_argument("--hand-max-depth", type=int, default=2)
    parser.add_argument("--shop-max-actions", type=int, default=24)
    parser.add_argument(
        "--allow-sim-augment",
        dest="allow_sim_augment",
        action="store_true",
        help="Fill missing quotas with sim-generated states (default on).",
    )
    parser.add_argument(
        "--no-sim-augment",
        dest="allow_sim_augment",
        action="store_false",
        help="Disable sim augmentation when real samples are insufficient.",
    )
    parser.set_defaults(allow_sim_augment=True)
    parser.add_argument("--summary-out", default="")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    logger = setup_logger("trainer.dagger_collect")
    warn_if_unstable_python(logger)

    rows = _read_jsonl(Path(args.session))
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    stats = Counter()
    failure_reasons = Counter()
    records: list[dict[str, Any]] = []

    def add_record(rec: dict[str, Any], kind: str) -> None:
        records.append(rec)
        stats[f"{kind}_records"] += 1

    for idx, row in enumerate(rows):
        state = _state_from_row(row)
        if not isinstance(state, dict):
            stats["invalid_rows"] += 1
            failure_reasons["missing_gamestate_raw"] += 1
            continue

        phase = str(state.get("state") or "UNKNOWN")
        try:
            decision = choose_full_action(
                state,
                hand_max_branch=int(args.hand_max_branch),
                hand_max_depth=int(args.hand_max_depth),
                shop_max_actions=int(args.shop_max_actions),
                time_budget_ms=float(args.time_budget_ms),
                seed=f"DAGGER-{idx}",
            )
        except Exception as exc:
            stats["invalid_rows"] += 1
            failure_reasons[f"teacher_failed:{type(exc).__name__}"] += 1
            continue

        if phase == "SELECTING_HAND" and stats["hand_records"] < int(args.hand_samples):
            student_topk = _heuristic_hand_rankings(state, topk=3)
            rec = _hand_record_from_state(
                state,
                episode_id=0,
                step_id=idx,
                teacher_action=decision.action,
                student_topk=student_topk,
                valid_reconstruct=True,
                failure_reason=None,
            )
            if rec is not None:
                add_record(rec, "hand")
            else:
                stats["invalid_rows"] += 1
                failure_reasons["hand_record_build_failed"] += 1

        elif phase in action_space_shop.SHOP_PHASES and stats["shop_records"] < int(args.shop_samples):
            student_topk = _heuristic_shop_rankings(state, topk=3)
            aid = int(decision.action_id) if isinstance(decision.action_id, int) else (
                action_space_shop.legal_action_ids(state)[0] if action_space_shop.legal_action_ids(state) else 0
            )
            rec = _shop_record_from_state(
                state,
                episode_id=0,
                step_id=idx,
                teacher_action=decision.action,
                teacher_action_id=aid,
                student_topk=student_topk,
                valid_reconstruct=True,
                failure_reason=None,
            )
            if rec is not None:
                add_record(rec, "shop")
            else:
                stats["invalid_rows"] += 1
                failure_reasons["shop_record_build_failed"] += 1

        if stats["hand_records"] >= int(args.hand_samples) and stats["shop_records"] >= int(args.shop_samples):
            break

    if args.allow_sim_augment and (stats["hand_records"] < int(args.hand_samples) or stats["shop_records"] < int(args.shop_samples)):
        logger.info("insufficient real samples, augmenting from sim...")
        sim_backend = create_backend("sim", seed="DAGGER-AUG")
        try:
            episode = 1
            step = 0
            while stats["hand_records"] < int(args.hand_samples) or stats["shop_records"] < int(args.shop_samples):
                state = sim_backend.reset(seed=f"DAGGER-AUG-{episode}")
                for _ in range(260):
                    phase = str(state.get("state") or "UNKNOWN")
                    phase = str(state.get("state") or "UNKNOWN")
                    if phase == "SELECTING_HAND" or phase in action_space_shop.SHOP_PHASES:
                        decision = choose_full_action(
                            state,
                            hand_max_branch=int(args.hand_max_branch),
                            hand_max_depth=int(args.hand_max_depth),
                            shop_max_actions=int(args.shop_max_actions),
                            time_budget_ms=float(args.time_budget_ms),
                            seed=f"DAGGER-AUG-{episode}-{step}",
                        )
                        chosen_action = dict(decision.action)
                        chosen_action_id = decision.action_id
                    else:
                        chosen_action = _macro_progress_action(state, seed=f"DAGGER-AUG-{episode}-{step}")
                        chosen_action_id = None

                    if phase == "SELECTING_HAND":
                        hand_cards = ((state.get("hand") or {}).get("cards") or []) if isinstance(state.get("hand"), dict) else []
                        round_info = state.get("round") if isinstance(state.get("round"), dict) else {}
                        hands_left = int(round_info.get("hands_left") or 0)
                        discards_left = int(round_info.get("discards_left") or 0)
                        at = str(chosen_action.get("action_type") or "").upper()
                        idxs = [int(x) for x in (chosen_action.get("indices") or [])]
                        if hands_left <= 0 and discards_left > 0 and len(hand_cards) > 0:
                            chosen_action = {"action_type": "DISCARD", "indices": [0]}
                        elif at == "DISCARD" and len(idxs) == 0 and len(hand_cards) > 0 and hands_left > 0:
                            chosen_action = {
                                "action_type": "PLAY",
                                "indices": list(range(min(5, len(hand_cards)))),
                            }
                    if phase == "SELECTING_HAND" and stats["hand_records"] < int(args.hand_samples):
                        student_topk = _heuristic_hand_rankings(state, topk=3)
                        rec = _hand_record_from_state(
                            state,
                            episode_id=episode,
                            step_id=step,
                            teacher_action=chosen_action,
                            student_topk=student_topk,
                            valid_reconstruct=False,
                            failure_reason="sim_augment",
                        )
                        if rec is not None:
                            add_record(rec, "hand")
                    elif phase in action_space_shop.SHOP_PHASES and stats["shop_records"] < int(args.shop_samples):
                        student_topk = _heuristic_shop_rankings(state, topk=3)
                        aid = int(chosen_action_id) if isinstance(chosen_action_id, int) else (
                            action_space_shop.legal_action_ids(state)[0] if action_space_shop.legal_action_ids(state) else 0
                        )
                        rec = _shop_record_from_state(
                            state,
                            episode_id=episode,
                            step_id=step,
                            teacher_action=chosen_action,
                            teacher_action_id=aid,
                            student_topk=student_topk,
                            valid_reconstruct=False,
                            failure_reason="sim_augment",
                        )
                        if rec is not None:
                            add_record(rec, "shop")

                    if stats["hand_records"] >= int(args.hand_samples) and stats["shop_records"] >= int(args.shop_samples):
                        break
                    try:
                        state, _, done, _ = sim_backend.step(chosen_action)
                    except Exception:
                        break
                    step += 1
                    if done:
                        break
                episode += 1
                if episode > 1200:
                    break

            if stats["shop_records"] < int(args.shop_samples):
                logger.info("shop sample quota still unmet, synthesizing SHOP states from sim snapshots...")
                synth_idx = 0
                while stats["shop_records"] < int(args.shop_samples) and synth_idx < 5000:
                    state = sim_backend.reset(seed=f"DAGGER-SHOP-{synth_idx}")
                    state["state"] = "SHOP"
                    decision = choose_full_action(
                        state,
                        hand_max_branch=int(args.hand_max_branch),
                        hand_max_depth=int(args.hand_max_depth),
                        shop_max_actions=int(args.shop_max_actions),
                        time_budget_ms=float(args.time_budget_ms),
                        seed=f"DAGGER-SHOP-{synth_idx}",
                    )
                    student_topk = _heuristic_shop_rankings(state, topk=3)
                    aid = int(decision.action_id) if isinstance(decision.action_id, int) else (
                        action_space_shop.legal_action_ids(state)[0] if action_space_shop.legal_action_ids(state) else 0
                    )
                    rec = _shop_record_from_state(
                        state,
                        episode_id=episode + synth_idx,
                        step_id=step + synth_idx,
                        teacher_action=decision.action,
                        teacher_action_id=aid,
                        student_topk=student_topk,
                        valid_reconstruct=False,
                        failure_reason="sim_augment_shop_synth",
                    )
                    if rec is not None:
                        add_record(rec, "shop")
                    synth_idx += 1
        finally:
            sim_backend.close()

    with JsonlWriter(out_path) as writer:
        for rec in records:
            writer.write_record(rec)

    summary = {
        "session_rows": len(rows),
        "records_written": len(records),
        "hand_records": int(stats["hand_records"]),
        "shop_records": int(stats["shop_records"]),
        "invalid_rows": int(stats["invalid_rows"]),
        "reconstruct_failure_rate": float(stats["invalid_rows"] / max(1, len(rows))),
        "top_failure_reasons": failure_reasons.most_common(10),
        "out": str(out_path),
    }

    summary_out = Path(args.summary_out) if args.summary_out else out_path.with_suffix(".summary.json")
    summary_out.parent.mkdir(parents=True, exist_ok=True)
    summary_out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("dagger collected: hand=%d shop=%d invalid=%d out=%s", summary["hand_records"], summary["shop_records"], summary["invalid_rows"], out_path)
    logger.info("summary: %s", summary_out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
