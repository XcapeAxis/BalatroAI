from __future__ import annotations

import math
from typing import Any


def train_awr_epoch(
    *,
    model,
    optimizer,
    torch,
    F,
    hand_rows: list[dict[str, Any]],
    shop_rows: list[dict[str, Any]],
    batch_size: int,
    beta: float,
    max_weight: float,
    value_weight: float,
    max_actions: int,
    max_shop_actions: int,
    device,
    seed: int = 7,
) -> dict[str, float]:
    rng = torch.Generator(device="cpu")
    rng.manual_seed(int(seed))

    def _shuffle(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not rows:
            return []
        idx = torch.randperm(len(rows), generator=rng).tolist()
        return [rows[i] for i in idx]

    def _mask(batch_legal: list[list[int]], width: int) -> Any:
        m = torch.zeros((len(batch_legal), width), dtype=torch.float32, device=device)
        for i, aids in enumerate(batch_legal):
            for a in aids:
                ia = int(a)
                if 0 <= ia < width:
                    m[i, ia] = 1.0
        return m

    def _policy_term(logits, actions, legal_mask, advantages):
        neg = torch.full_like(logits, -1e9)
        masked = torch.where(legal_mask > 0, logits, neg)
        logp = F.log_softmax(masked, dim=1)
        act_lp = logp.gather(1, actions.unsqueeze(1)).squeeze(1)
        w = torch.exp(advantages / max(beta, 1e-6)).clamp(max=max_weight)
        return -(w * act_lp).mean(), float(w.mean().item())

    model.train()
    agg: dict[str, float] = {
        "hand_policy_loss": 0.0,
        "hand_value_loss": 0.0,
        "hand_adv_mean": 0.0,
        "hand_weight_mean": 0.0,
        "shop_policy_loss": 0.0,
        "shop_value_loss": 0.0,
        "shop_adv_mean": 0.0,
        "shop_weight_mean": 0.0,
        "steps_hand": 0.0,
        "steps_shop": 0.0,
    }

    hand_rows = _shuffle(hand_rows)
    shop_rows = _shuffle(shop_rows)

    for start in range(0, len(hand_rows), max(1, batch_size)):
        b = hand_rows[start : start + batch_size]
        if not b:
            continue
        batch = {
            "rank": torch.tensor([x["rank"] for x in b], dtype=torch.long, device=device),
            "suit": torch.tensor([x["suit"] for x in b], dtype=torch.long, device=device),
            "chip": torch.tensor([x["chip"] for x in b], dtype=torch.float32, device=device),
            "enh": torch.tensor([x["enh"] for x in b], dtype=torch.float32, device=device),
            "edt": torch.tensor([x["edt"] for x in b], dtype=torch.float32, device=device),
            "seal": torch.tensor([x["seal"] for x in b], dtype=torch.float32, device=device),
            "pad": torch.tensor([x["pad"] for x in b], dtype=torch.float32, device=device),
            "context": torch.tensor([x["context"] for x in b], dtype=torch.float32, device=device),
        }
        actions = torch.tensor([int(x["action_id"]) for x in b], dtype=torch.long, device=device)
        target_values = torch.tensor([float(x["value_target"]) for x in b], dtype=torch.float32, device=device)
        legal_mask = _mask([x["legal_ids"] for x in b], max_actions)

        logits, values = model.forward_hand(batch)
        advantages = target_values - values.detach()
        ploss, wmean = _policy_term(logits, actions, legal_mask, advantages)
        vloss = F.smooth_l1_loss(values, target_values)
        loss = ploss + float(value_weight) * vloss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        agg["hand_policy_loss"] += float(ploss.item())
        agg["hand_value_loss"] += float(vloss.item())
        agg["hand_adv_mean"] += float(advantages.mean().item())
        agg["hand_weight_mean"] += float(wmean)
        agg["steps_hand"] += 1.0

    for start in range(0, len(shop_rows), max(1, batch_size)):
        b = shop_rows[start : start + batch_size]
        if not b:
            continue
        batch = {"shop_context": torch.tensor([x["shop_context"] for x in b], dtype=torch.float32, device=device)}
        actions = torch.tensor([int(x["action_id"]) for x in b], dtype=torch.long, device=device)
        target_values = torch.tensor([float(x["value_target"]) for x in b], dtype=torch.float32, device=device)
        legal_mask = _mask([x["legal_ids"] for x in b], max_shop_actions)

        logits, values = model.forward_shop(batch)
        advantages = target_values - values.detach()
        ploss, wmean = _policy_term(logits, actions, legal_mask, advantages)
        vloss = F.smooth_l1_loss(values, target_values)
        loss = ploss + float(value_weight) * vloss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        agg["shop_policy_loss"] += float(ploss.item())
        agg["shop_value_loss"] += float(vloss.item())
        agg["shop_adv_mean"] += float(advantages.mean().item())
        agg["shop_weight_mean"] += float(wmean)
        agg["steps_shop"] += 1.0

    for k in (
        "hand_policy_loss",
        "hand_value_loss",
        "hand_adv_mean",
        "hand_weight_mean",
    ):
        d = max(1.0, agg["steps_hand"])
        agg[k] /= d
    for k in (
        "shop_policy_loss",
        "shop_value_loss",
        "shop_adv_mean",
        "shop_weight_mean",
    ):
        d = max(1.0, agg["steps_shop"])
        agg[k] /= d
    agg["steps_total"] = agg["steps_hand"] + agg["steps_shop"]
    if math.isnan(agg["hand_policy_loss"]) or math.isnan(agg["shop_policy_loss"]):
        raise RuntimeError("AWR produced NaN loss")
    return agg
