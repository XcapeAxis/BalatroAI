from __future__ import annotations

import random
from typing import Any


def normalize_legal_action_ids(raw: Any, *, action_dim: int) -> list[int]:
    out: list[int] = []
    seen: set[int] = set()
    if not isinstance(raw, list):
        return out
    for item in raw:
        try:
            aid = int(item)
        except Exception:
            continue
        if aid < 0 or aid >= int(action_dim):
            continue
        if aid in seen:
            continue
        seen.add(aid)
        out.append(aid)
    return out


def build_action_mask(*, action_dim: int, legal_action_ids: list[int]) -> list[int]:
    dim = max(1, int(action_dim))
    mask = [0] * dim
    for aid in legal_action_ids:
        if 0 <= int(aid) < dim:
            mask[int(aid)] = 1
    return mask


def action_mask_density(mask: list[int]) -> float:
    if not mask:
        return 0.0
    return float(sum(1 for x in mask if int(x) > 0)) / float(len(mask))


def apply_action_mask_policy(
    *,
    requested_action: int,
    action_mask: list[int],
    strategy: str,
    rng: random.Random | None = None,
) -> tuple[int, bool, str]:
    legal_ids = [idx for idx, flag in enumerate(action_mask) if int(flag) > 0]
    if not legal_ids:
        return int(requested_action), True, "no_legal_actions"

    req = int(requested_action)
    if req in legal_ids:
        return req, False, "requested_legal"

    mode = str(strategy or "fallback_first_legal").strip().lower()
    if mode == "fallback_random_legal":
        chooser = rng if rng is not None else random
        return int(chooser.choice(legal_ids)), True, "fallback_random_legal"
    if mode in {"strict", "penalize_only", "pass_through"}:
        return req, True, "pass_through_invalid"
    return int(legal_ids[0]), True, "fallback_first_legal"

