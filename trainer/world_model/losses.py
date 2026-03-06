from __future__ import annotations

from typing import Any


def _require_torch():
    try:
        import torch
        import torch.nn.functional as F
    except Exception as exc:  # pragma: no cover - runtime guarded
        raise RuntimeError("PyTorch is required for trainer.world_model.losses") from exc
    return torch, F


def compute_world_model_losses(
    *,
    outputs: dict[str, Any],
    batch: dict[str, Any],
    loss_weights: dict[str, Any] | None = None,
) -> dict[str, Any]:
    torch, F = _require_torch()
    weights = loss_weights if isinstance(loss_weights, dict) else {}

    latent_weight = float(weights.get("latent") or 1.0)
    reward_weight = float(weights.get("reward") or 1.0)
    score_weight = float(weights.get("score_delta") or 0.5)
    resource_weight = float(weights.get("resource_delta") or 0.25)
    uncertainty_weight = float(weights.get("uncertainty") or 0.25)

    z_next_target = batch.get("latent_t1")
    if z_next_target is None:
        z_next_target = outputs.get("z_next_target")
    if z_next_target is None:
        raise ValueError("missing latent target for world model loss")

    reward_target = batch["reward_t"].float()
    score_target = batch["score_delta_t"].float()
    resource_target = batch["resource_delta_t"].float()

    latent_error_vec = ((outputs["z_next_pred"] - z_next_target) ** 2).mean(dim=-1)
    reward_error_vec = (outputs["reward_pred"] - reward_target) ** 2
    score_error_vec = (outputs["score_pred"] - score_target) ** 2
    resource_error_vec = ((outputs["resource_pred"] - resource_target) ** 2).mean(dim=-1)

    latent_loss = latent_error_vec.mean()
    reward_loss = reward_error_vec.mean()
    score_loss = score_error_vec.mean()
    resource_loss = resource_error_vec.mean()

    combined_error = (
        latent_weight * latent_error_vec.detach()
        + reward_weight * reward_error_vec.detach()
        + score_weight * score_error_vec.detach()
        + resource_weight * resource_error_vec.detach()
    )
    uncertainty_target = torch.log1p(torch.clamp(combined_error, min=0.0))
    uncertainty_loss = F.smooth_l1_loss(outputs["uncertainty_pred"], uncertainty_target)

    total_loss = (
        latent_weight * latent_loss
        + reward_weight * reward_loss
        + score_weight * score_loss
        + resource_weight * resource_loss
        + uncertainty_weight * uncertainty_loss
    )
    reward_mae = torch.abs(outputs["reward_pred"] - reward_target).mean()
    score_mae = torch.abs(outputs["score_pred"] - score_target).mean()
    resource_mae = torch.abs(outputs["resource_pred"] - resource_target).mean()

    return {
        "total_loss": total_loss,
        "latent_loss": latent_loss,
        "reward_loss": reward_loss,
        "score_loss": score_loss,
        "resource_loss": resource_loss,
        "uncertainty_loss": uncertainty_loss,
        "combined_error_mean": combined_error.mean(),
        "reward_mae": reward_mae,
        "score_mae": score_mae,
        "resource_mae": resource_mae,
        "uncertainty_mean": outputs["uncertainty_pred"].mean(),
        "per_item_combined_error": combined_error,
        "per_item_uncertainty": outputs["uncertainty_pred"],
        "per_item_reward_error_abs": torch.abs(outputs["reward_pred"] - reward_target),
        "per_item_score_error_abs": torch.abs(outputs["score_pred"] - score_target),
        "per_item_latent_error": latent_error_vec,
    }
