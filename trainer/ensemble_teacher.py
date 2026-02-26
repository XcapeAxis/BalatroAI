"""Ensemble teacher: fuse multiple strategy policies via rank aggregation or score fusion."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any


@dataclass
class MemberVote:
    strategy: str
    top1_action: int
    scores: list[float] = field(default_factory=list)
    available: bool = True


@dataclass
class EnsembleDecision:
    chosen_action: int
    member_votes: list[MemberVote]
    method: str
    disagreement: float
    entropy_proxy: float
    confidence: float


def borda_rank_aggregate(
    member_rankings: dict[str, list[int]],
    legal_ids: list[int],
    k: int = 5,
) -> EnsembleDecision:
    """Borda count over members' top-k ranked actions.

    Each member gives k points to rank-1, k-1 to rank-2, etc.
    """
    legal_set = set(int(a) for a in legal_ids)
    scores: dict[int, float] = {a: 0.0 for a in legal_set}
    votes: list[MemberVote] = []

    for strat, ranking in member_rankings.items():
        filtered = [a for a in ranking if a in legal_set][:k]
        for rank_idx, action in enumerate(filtered):
            scores[action] = scores.get(action, 0.0) + (k - rank_idx)
        top1 = filtered[0] if filtered else -1
        votes.append(MemberVote(strategy=strat, top1_action=top1, scores=filtered[:k]))

    if not scores:
        return EnsembleDecision(
            chosen_action=-1, member_votes=votes, method="borda",
            disagreement=1.0, entropy_proxy=0.0, confidence=0.0,
        )

    best_action = max(scores, key=lambda a: scores[a])
    total = sum(scores.values())
    n_members = max(1, len(member_rankings))

    top1_actions = [v.top1_action for v in votes if v.top1_action >= 0]
    unique_top1 = len(set(top1_actions))
    disagreement = 1.0 - (1.0 / max(1, unique_top1)) if unique_top1 > 1 else 0.0

    probs = [(scores.get(a, 0.0) / max(total, 1e-9)) for a in legal_set if scores.get(a, 0.0) > 0]
    entropy = -sum(p * math.log(p + 1e-12) for p in probs) if probs else 0.0
    max_entropy = math.log(max(len(probs), 1)) if probs else 1.0
    entropy_proxy = entropy / max(max_entropy, 1e-9)

    confidence = scores[best_action] / (n_members * k) if (n_members * k) > 0 else 0.0

    return EnsembleDecision(
        chosen_action=best_action,
        member_votes=votes,
        method="borda",
        disagreement=disagreement,
        entropy_proxy=entropy_proxy,
        confidence=confidence,
    )


def weighted_score_fusion(
    member_scores: dict[str, dict[int, float]],
    weights: dict[str, float] | None = None,
    legal_ids: list[int] | None = None,
) -> EnsembleDecision:
    """Weighted fusion of raw logits/scores across members."""
    if weights is None:
        weights = {s: 1.0 for s in member_scores}
    total_w = sum(weights.values()) or 1.0

    legal_set = set(int(a) for a in legal_ids) if legal_ids else None
    fused: dict[int, float] = {}
    votes: list[MemberVote] = []

    for strat, action_scores in member_scores.items():
        w = weights.get(strat, 1.0) / total_w
        top1 = max(action_scores, key=lambda a: action_scores[a]) if action_scores else -1
        votes.append(MemberVote(strategy=strat, top1_action=top1))
        for a, s in action_scores.items():
            if legal_set is not None and a not in legal_set:
                continue
            fused[a] = fused.get(a, 0.0) + w * s

    best_action = max(fused, key=lambda a: fused[a]) if fused else -1
    top1s = [v.top1_action for v in votes if v.top1_action >= 0]
    unique = len(set(top1s))
    disagreement = 1.0 - (1.0 / max(1, unique)) if unique > 1 else 0.0

    vals = list(fused.values())
    total_s = sum(abs(v) for v in vals) or 1.0
    probs = [abs(v) / total_s for v in vals if abs(v) > 0]
    entropy = -sum(p * math.log(p + 1e-12) for p in probs) if probs else 0.0
    max_ent = math.log(max(len(probs), 1)) if probs else 1.0

    confidence = fused.get(best_action, 0.0) / total_s if total_s > 0 else 0.0

    return EnsembleDecision(
        chosen_action=best_action,
        member_votes=votes,
        method="weighted_score_fusion",
        disagreement=disagreement,
        entropy_proxy=entropy / max(max_ent, 1e-9),
        confidence=confidence,
    )


def decision_to_dict(d: EnsembleDecision) -> dict[str, Any]:
    return {
        "chosen_action": d.chosen_action,
        "method": d.method,
        "disagreement": round(d.disagreement, 4),
        "entropy_proxy": round(d.entropy_proxy, 4),
        "confidence": round(d.confidence, 4),
        "members": [
            {"strategy": v.strategy, "top1_action": v.top1_action, "available": v.available}
            for v in d.member_votes
        ],
    }
