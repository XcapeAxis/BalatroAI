from __future__ import annotations

if __package__ is None or __package__ == "":
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).resolve().parent.parent))

import copy
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from demo.model_arch import MAX_ACTIONS, MVPHandPolicy, MVPHandPolicyConfig
from demo.state_adapter import action_label, compute_resource_delta, phase_label
from demo.training_status import infer_profile, read_status
from sim.core.score_basic import evaluate_selected_breakdown
from sim.score.expected_basic import compute_expected_for_action
from trainer import action_space
from trainer.expert_policy import choose_action
from trainer.features import extract_features


def _require_torch():
    try:
        import torch
    except Exception as exc:
        raise RuntimeError("MVP Demo 推理需要 PyTorch。") from exc
    return torch


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


@dataclass
class ModelBundle:
    run_dir: Path
    checkpoint_path: Path
    config: dict[str, Any]
    metrics: dict[str, Any]
    history: list[dict[str, Any]]
    train_summary: str
    model_name: str
    device_name: str
    loaded: bool
    error: str = ""
    model: Any = None
    torch: Any = None


def artifact_root() -> Path:
    return Path(__file__).resolve().parent.parent / "docs" / "artifacts" / "mvp" / "model_train"


def latest_run_dir(root: Path | None = None) -> Path | None:
    base = root or artifact_root()
    latest_hint = base / "latest_run.txt"
    if latest_hint.exists():
        name = latest_hint.read_text(encoding="utf-8").strip()
        hinted = base / name
        if hinted.exists():
            return hinted
    candidates = [path for path in base.iterdir() if path.is_dir()] if base.exists() else []
    if not candidates:
        return None
    return sorted(candidates, key=lambda path: path.stat().st_mtime, reverse=True)[0]


def _read_json(path: Path, default: Any | None = None) -> Any:
    if not path.exists():
        return default if default is not None else {}
    return json.loads(path.read_text(encoding="utf-8"))


def load_latest_bundle(run_dir: Path | None = None) -> ModelBundle:
    actual_run_dir = run_dir or latest_run_dir()
    if actual_run_dir is None:
        return ModelBundle(
            run_dir=Path(""),
            checkpoint_path=Path(""),
            config={},
            metrics={},
            history=[],
            train_summary="",
            model_name="unavailable",
            device_name="cpu",
            loaded=False,
            error="no_mvp_model_run_found",
        )

    config = _read_json(actual_run_dir / "model_config.json", default={})
    metrics = _read_json(actual_run_dir / "metrics.json", default={})
    checkpoint_name = str(config.get("checkpoint_name") or "mvp_policy.pt")
    checkpoint_path = actual_run_dir / checkpoint_name
    train_summary = (actual_run_dir / "training_summary.md").read_text(encoding="utf-8") if (actual_run_dir / "training_summary.md").exists() else ""

    try:
        torch = _require_torch()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        arch_payload = config.get("architecture") if isinstance(config.get("architecture"), dict) else config
        model_cfg = MVPHandPolicyConfig.from_dict(arch_payload)
        model = MVPHandPolicy(torch.nn, config=model_cfg)
        state_dict = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state_dict, strict=True)
        model.to(device)
        model.eval()
        return ModelBundle(
            run_dir=actual_run_dir,
            checkpoint_path=checkpoint_path,
            config=config,
            metrics=metrics,
            history=list(metrics.get("history") or []),
            train_summary=train_summary,
            model_name=str(config.get("model_name") or actual_run_dir.name),
            device_name=str(device),
            loaded=True,
            model=model,
            torch=torch,
        )
    except Exception as exc:
        return ModelBundle(
            run_dir=actual_run_dir,
            checkpoint_path=checkpoint_path,
            config=config,
            metrics=metrics,
            history=list(metrics.get("history") or []),
            train_summary=train_summary,
            model_name=str(config.get("model_name") or actual_run_dir.name),
            device_name="cpu",
            loaded=False,
            error=str(exc),
        )


def build_policy_model(nn, raw_config: dict[str, Any] | None = None):
    config = MVPHandPolicyConfig.from_dict(raw_config)
    return MVPHandPolicy(nn, config=config)


def _state_to_batch(state: dict[str, Any], torch):
    features = extract_features(state)
    chip_hint = list(features.get("card_chip_hint") or [0] * action_space.MAX_HAND)
    return {
        "rank": torch.tensor([features["card_rank_ids"]], dtype=torch.long),
        "suit": torch.tensor([features["card_suit_ids"]], dtype=torch.long),
        "chip": torch.tensor([chip_hint], dtype=torch.float32),
        "enh": torch.tensor([features["card_has_enhancement"]], dtype=torch.float32),
        "edt": torch.tensor([features["card_has_edition"]], dtype=torch.float32),
        "seal": torch.tensor([features["card_has_seal"]], dtype=torch.float32),
        "pad": torch.tensor([features["hand_pad_mask"]], dtype=torch.float32),
        "context": torch.tensor([features["context"]], dtype=torch.float32),
    }


def legal_action_ids_from_state(state: dict[str, Any]) -> list[int]:
    cards = (state.get("hand") or {}).get("cards") if isinstance(state.get("hand"), dict) else []
    cards = cards if isinstance(cards, list) else []
    hand_size = min(len(cards), action_space.MAX_HAND)
    if hand_size <= 0:
        return []

    round_info = state.get("round") if isinstance(state.get("round"), dict) else {}
    hands_left = int(round_info.get("hands_left") or 0)
    discards_left = int(round_info.get("discards_left") or 0)
    legal = action_space.legal_action_ids(hand_size)
    out: list[int] = []
    for aid in legal:
        atype, mask_int = action_space.decode(hand_size, aid)
        indices = action_space.mask_to_indices(mask_int, hand_size)
        if atype == action_space.PLAY and hands_left <= 0:
            continue
        if atype == action_space.DISCARD and discards_left <= 0:
            continue
        if atype == action_space.DISCARD and not indices:
            continue
        out.append(aid)
    return out


def _action_from_id(hand_size: int, action_id: int) -> dict[str, Any]:
    atype, mask_int = action_space.decode(hand_size, action_id)
    return {"action_type": atype, "indices": action_space.mask_to_indices(mask_int, hand_size)}


def _score_discard(cards: list[dict[str, Any]], indices: list[int]) -> float:
    if not indices:
        return 0.0
    all_suits = [str((card.get("value") or {}).get("suit") or card.get("suit") or "").upper()[:1] for card in cards]
    suit_count: dict[str, int] = {}
    for suit in all_suits:
        suit_count[suit] = suit_count.get(suit, 0) + 1

    rank_values = {
        "2": 2,
        "3": 3,
        "4": 4,
        "5": 5,
        "6": 6,
        "7": 7,
        "8": 8,
        "9": 9,
        "10": 10,
        "T": 10,
        "J": 11,
        "Q": 12,
        "K": 13,
        "A": 14,
    }
    badness = 0.0
    for idx in indices:
        card = cards[idx]
        rank = str((card.get("value") or {}).get("rank") or card.get("rank") or "").upper()
        rank = "10" if rank == "T" else rank
        suit = str((card.get("value") or {}).get("suit") or card.get("suit") or "").upper()[:1]
        value = rank_values.get(rank, 0)
        badness += max(0, 12 - value)
        badness += max(0, 3 - suit_count.get(suit, 0))
    badness += 0.8 * len(indices)
    return badness


def _heuristic_hand_candidates(state: dict[str, Any], topk: int) -> list[dict[str, Any]]:
    cards = (state.get("hand") or {}).get("cards") if isinstance(state.get("hand"), dict) else []
    cards = cards if isinstance(cards, list) else []
    hand_size = min(len(cards), action_space.MAX_HAND)
    if hand_size <= 0:
        return []

    ranked: list[dict[str, Any]] = []
    for action_id in legal_action_ids_from_state(state):
        atype, mask_int = action_space.decode(hand_size, action_id)
        indices = action_space.mask_to_indices(mask_int, hand_size)
        if atype == action_space.PLAY:
            selected = [cards[idx] for idx in indices]
            breakdown = evaluate_selected_breakdown(selected)
            proxy_score = _safe_float(breakdown.get("total_delta"))
            reason = f"启发式判断这手更接近 {breakdown.get('hand_type') or '高收益出牌'}，预计基础收益 {_safe_float(breakdown.get('total_delta')):.0f}。"
        else:
            proxy_score = _score_discard(cards, indices)
            reason = f"启发式建议先弃掉 {len(indices)} 张弱牌，争取更好的重抽。"
        ranked.append(
            {
                "action_id": int(action_id),
                "action": {"action_type": atype, "indices": indices},
                "score": float(proxy_score),
                "confidence": 0.0,
                "source": "heuristic",
                "reason": reason,
            }
        )
    ranked.sort(key=lambda item: float(item["score"]), reverse=True)
    return ranked[: max(1, int(topk))]


def _phase_fallback_action(state: dict[str, Any], seed: str = "MVP") -> tuple[dict[str, Any], str]:
    decision = choose_action(state, start_seed=seed)
    if decision.action_type and decision.mask_int is not None:
        cards = (state.get("hand") or {}).get("cards") if isinstance(state.get("hand"), dict) else []
        hand_size = min(len(cards or []), action_space.MAX_HAND)
        indices = action_space.mask_to_indices(int(decision.mask_int), hand_size)
        return {"action_type": str(decision.action_type), "indices": indices}, decision.reason
    macro = str(decision.macro_action or "wait").lower()
    params = dict(decision.macro_params or {})
    if macro == "select":
        return {"action_type": "SELECT", "index": int(params.get("index", 0))}, decision.reason
    if macro == "cash_out":
        return {"action_type": "CASH_OUT"}, decision.reason
    if macro == "next_round":
        return {"action_type": "NEXT_ROUND"}, decision.reason
    if macro == "start":
        return {"action_type": "START", "seed": str(params.get("seed") or seed)}, decision.reason
    return {"action_type": "WAIT"}, decision.reason or "default_wait"


def _preview_action(state: dict[str, Any], env: Any | None, action: dict[str, Any]) -> dict[str, Any]:
    if env is None:
        phase_after = str(state.get("state") or "UNKNOWN")
        return {
            "reward": 0.0,
            "done": False,
            "delta": {},
            "phase_after": phase_after,
            "phase_after_text": phase_label(phase_after),
        }
    probe = copy.deepcopy(env)
    before = probe.get_state()
    after, reward, done, _info = probe.step(action)
    phase_after = str(after.get("state") or "UNKNOWN")
    preview = {
        "reward": _safe_float(reward),
        "done": bool(done),
        "phase_after": phase_after,
        "phase_after_text": phase_label(phase_after),
        "delta": compute_resource_delta(before, after),
        "score_after": _safe_float((after.get("score") or {}).get("chips")),
        "money_after": _safe_float(after.get("money")),
    }
    if str(action.get("action_type") or "").upper() == "PLAY":
        expected = compute_expected_for_action(before, action)
        if bool(expected.get("available")):
            preview["expected_score"] = _safe_float(expected.get("score"))
            preview["expected_hand_type"] = str(expected.get("hand_type") or "")
            preview["joker_breakdown"] = list(expected.get("joker_breakdown") or [])
        else:
            selected = [card for idx, card in enumerate((before.get("hand") or {}).get("cards") or []) if idx in set(action.get("indices") or [])]
            breakdown = evaluate_selected_breakdown(selected)
            preview["expected_score"] = _safe_float(breakdown.get("total_delta"))
            preview["expected_hand_type"] = str(breakdown.get("hand_type") or "")
    return preview


def _source_label(source: str) -> str:
    return {
        "model": "模型",
        "heuristic": "启发式",
        "teacher": "教师策略",
    }.get(str(source or "").lower(), str(source or "未知"))


def _recommendation_risk(preview: dict[str, Any], *, teacher_agrees: bool) -> tuple[str, str]:
    delta = preview.get("delta") or {}
    score_gain = _safe_float(preview.get("expected_score") or preview.get("reward"))
    hands_delta = _safe_int(delta.get("hands_left"))
    discards_delta = _safe_int(delta.get("discards_left"))
    if not teacher_agrees and score_gain <= 0:
        return "高", "与教师策略不一致，且当前动作不能立即兑现收益。"
    if hands_delta < 0 or discards_delta < 0:
        return "中", "会消耗当前回合资源，适合强调收益与试错空间的权衡。"
    return "低", "当前动作更偏稳健，适合作为面试演示中的基线建议。"


def _decorate_recommendations(
    state: dict[str, Any],
    env: Any | None,
    candidates: list[dict[str, Any]],
    *,
    source: str,
) -> list[dict[str, Any]]:
    teacher_action, teacher_reason = _phase_fallback_action(state)
    teacher_signature = json.dumps(teacher_action, sort_keys=True)
    out: list[dict[str, Any]] = []
    for rank, item in enumerate(candidates, start=1):
        action = dict(item["action"])
        preview = _preview_action(state, env, action)
        selected_signature = json.dumps(action, sort_keys=True)
        expert_agrees = selected_signature == teacher_signature
        reason = str(item.get("reason") or "").strip()
        if expert_agrees and teacher_reason:
            reason = f"{reason} 教师策略也支持这个动作（{teacher_reason}）。"
        elif teacher_reason:
            reason = f"{reason} 教师策略会优先 {action_label(teacher_action, state)}（{teacher_reason}）。"
        preview_score = _safe_float((preview.get("expected_score") or preview.get("reward")))
        risk_level, risk_reason = _recommendation_risk(preview, teacher_agrees=expert_agrees)
        why_not_next = teacher_reason or "次优动作更多是在保留资源，而不是立刻兑现当前收益。"
        out.append(
            {
                "rank": rank,
                "action": action,
                "label": action_label(action, state),
                "score": round(_safe_float(item.get("score")), 4),
                "confidence": round(_safe_float(item.get("confidence")), 4),
                "confidence_pct": round(_safe_float(item.get("confidence")) * 100.0, 1),
                "source": source,
                "source_label": _source_label(source),
                "source_text": _source_label(source),
                "reason": reason,
                "teacher_reason": teacher_reason,
                "teacher_agrees": expert_agrees,
                "teacher_agrees_label": "与教师一致" if expert_agrees else "与教师不同",
                "risk_level": risk_level,
                "risk_hint": risk_reason,
                "preview": preview,
                "tags": [source == "model" and "训练模型" or "启发式", expert_agrees and "教师一致" or "偏离教师"],
                "why_not_next": why_not_next,
                "summary": f"预计单步收益 {preview_score:.0f}，后续阶段 {preview.get('phase_after') or '未知'}。",
            }
        )
    return out


def recommend_actions(
    state: dict[str, Any],
    *,
    env: Any | None = None,
    policy: str = "model",
    topk: int = 3,
    bundle: ModelBundle | None = None,
) -> dict[str, Any]:
    phase = str(state.get("state") or "UNKNOWN")
    effective_policy = str(policy or "model").lower()
    loaded_bundle = bundle or load_latest_bundle()

    if phase != "SELECTING_HAND":
        action, reason = _phase_fallback_action(state)
        return {
            "phase": phase,
            "policy": "heuristic",
            "policy_label": "启发式",
            "model_loaded": bool(loaded_bundle.loaded),
            "recommendations": _decorate_recommendations(
                state,
                env,
                [{"action": action, "score": 1.0, "confidence": 1.0, "reason": f"当前阶段使用教师/启发式 fallback：{reason}。"}],
                source="heuristic",
            ),
            "model_name": loaded_bundle.model_name,
        }

    cards = (state.get("hand") or {}).get("cards") if isinstance(state.get("hand"), dict) else []
    cards = cards if isinstance(cards, list) else []
    hand_size = min(len(cards), action_space.MAX_HAND)
    legal_ids = legal_action_ids_from_state(state)

    if effective_policy == "model" and loaded_bundle.loaded:
        torch = loaded_bundle.torch
        device = torch.device(loaded_bundle.device_name)
        batch = _state_to_batch(state, torch)
        batch = {key: value.to(device) for key, value in batch.items()}
        legal_mask = torch.zeros((1, MAX_ACTIONS), dtype=torch.float32, device=device)
        for aid in legal_ids:
            legal_mask[0, int(aid)] = 1.0
        with torch.no_grad():
            logits = loaded_bundle.model(batch)
            masked_logits = torch.where(legal_mask > 0, logits, torch.full_like(logits, -1e9))
            probs = torch.softmax(masked_logits, dim=1)
            k = min(max(1, int(topk)), len(legal_ids))
            top_probs, top_ids = torch.topk(probs, k=k, dim=1)
        raw: list[dict[str, Any]] = []
        for idx in range(k):
            action_id = int(top_ids[0, idx].item())
            action = _action_from_id(hand_size, action_id)
            reason = "模型认为这是当前合法动作里的最优选择。"
            if str(action.get("action_type") or "").upper() == "PLAY":
                selected = [cards[i] for i in action.get("indices") or []]
                breakdown = evaluate_selected_breakdown(selected)
                reason = f"模型更偏向 {breakdown.get('hand_type') or '高收益出牌'}，预计基础收益 {_safe_float(breakdown.get('total_delta')):.0f}。"
            elif str(action.get("action_type") or "").upper() == "DISCARD":
                reason = f"模型倾向先弃掉 {len(action.get('indices') or [])} 张牌，换取更高质量重抽。"
            raw.append(
                {
                    "action": action,
                    "score": _safe_float(masked_logits[0, action_id].item()),
                    "confidence": _safe_float(top_probs[0, idx].item()),
                    "reason": reason,
                }
            )
        return {
            "phase": phase,
            "policy": "model",
            "policy_label": "模型",
            "model_loaded": True,
            "recommendations": _decorate_recommendations(state, env, raw, source="model"),
            "model_name": loaded_bundle.model_name,
        }

    heuristics = _heuristic_hand_candidates(state, topk=topk)
    return {
        "phase": phase,
        "policy": "heuristic",
        "policy_label": "启发式",
        "model_loaded": bool(loaded_bundle.loaded),
        "recommendations": _decorate_recommendations(state, env, heuristics, source="heuristic"),
        "model_name": loaded_bundle.model_name,
    }


def model_info(bundle: ModelBundle | None = None) -> dict[str, Any]:
    loaded_bundle = bundle or load_latest_bundle()
    metrics = dict(loaded_bundle.metrics)
    dataset_stats = _read_json(loaded_bundle.run_dir / "dataset_stats.json", default={}) if loaded_bundle.run_dir else {}
    curve_path = loaded_bundle.run_dir / "loss_curve.csv" if loaded_bundle.run_dir else Path("")
    curve_rows: list[dict[str, Any]] = []
    if curve_path.exists():
        with curve_path.open("r", encoding="utf-8", newline="") as fp:
            for row in csv.DictReader(fp):
                curve_rows.append(
                    {
                        "epoch": _safe_int(row.get("epoch")),
                        "train_loss": _safe_float(row.get("train_loss")),
                        "val_loss": _safe_float(row.get("val_loss")),
                        "val_acc1": _safe_float(row.get("val_acc1")),
                        "val_acc3": _safe_float(row.get("val_acc3")),
                        "lr": _safe_float(row.get("lr")),
                    }
                )
    scenario_eval = {}
    if loaded_bundle.run_dir:
        scenario_eval = _read_json(loaded_bundle.run_dir / "demo_scenario_eval.json", default={})
        if not scenario_eval:
            scenario_eval = _read_json(loaded_bundle.run_dir / "scenario_eval.json", default={})
    verdict = _read_json(loaded_bundle.run_dir / "training_verdict.json", default={}) if loaded_bundle.run_dir else {}
    raw_training_status = dict(read_status())
    training_status = {
        "job_id": raw_training_status.get("job_id"),
        "status": raw_training_status.get("status"),
        "status_label": raw_training_status.get("status_label"),
        "message": raw_training_status.get("message"),
        "updated_at": raw_training_status.get("updated_at"),
        "finished_at": raw_training_status.get("finished_at"),
        "budget_minutes": raw_training_status.get("budget_minutes"),
        "profile": infer_profile(raw_training_status),
        "final_run_dir": raw_training_status.get("final_run_dir"),
        "dataset": {
            "episodes_target": ((raw_training_status.get("dataset") or {}).get("episodes_target")),
            "progress": ((raw_training_status.get("dataset") or {}).get("progress")),
            "records": ((raw_training_status.get("dataset") or {}).get("records")),
        },
        "training": {
            "epoch": ((raw_training_status.get("training") or {}).get("epoch")),
            "epochs_total": ((raw_training_status.get("training") or {}).get("epochs_total")),
            "progress": ((raw_training_status.get("training") or {}).get("progress")),
            "best_val_loss": ((raw_training_status.get("training") or {}).get("best_val_loss")),
            "best_epoch": ((raw_training_status.get("training") or {}).get("best_epoch")),
            "eta_sec": ((raw_training_status.get("training") or {}).get("eta_sec")),
        },
    }
    return {
        "loaded": bool(loaded_bundle.loaded),
        "model_name": loaded_bundle.model_name,
        "run_dir": str(loaded_bundle.run_dir),
        "checkpoint_path": str(loaded_bundle.checkpoint_path),
        "error": loaded_bundle.error,
        "config": loaded_bundle.config,
        "metrics": metrics,
        "dataset_stats": dataset_stats,
        "history": curve_rows or list(metrics.get("history") or []),
        "train_summary": loaded_bundle.train_summary,
        "scenario_eval": scenario_eval,
        "verdict": verdict,
        "training_status": training_status,
    }
