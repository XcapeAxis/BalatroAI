from __future__ import annotations

import math
import statistics
from typing import Any


def aggregate_seed_metrics(seed_results: list[dict[str, Any]], primary_metric: str) -> dict[str, Any]:
    values: list[float] = []
    reward_values: list[float] = []
    best_episode_values: list[float] = []
    avg_ante_values: list[float] = []
    median_ante_values: list[float] = []
    win_rate_values: list[float] = []
    hand_top1_values: list[float] = []
    hand_top3_values: list[float] = []
    shop_top1_values: list[float] = []
    illegal_rate_values: list[float] = []
    catastrophic_failures: list[dict[str, Any]] = []
    per_seed_rows: list[dict[str, Any]] = []

    for row in seed_results:
        seed = str(row.get("seed") or "")
        status = str(row.get("status") or "unknown").lower()
        metrics = row.get("metrics") or {}
        value = metrics.get(primary_metric)
        numeric = value if isinstance(value, (int, float)) else None
        if numeric is not None:
            values.append(float(numeric))
        avg_reward_v = metrics.get("avg_reward")
        if isinstance(avg_reward_v, (int, float)):
            reward_values.append(float(avg_reward_v))
        elif numeric is not None:
            reward_values.append(float(numeric))
        best_reward_v = metrics.get("best_episode_reward")
        if isinstance(best_reward_v, (int, float)):
            best_episode_values.append(float(best_reward_v))
        avg_ante_v = metrics.get("avg_ante_reached")
        if isinstance(avg_ante_v, (int, float)):
            avg_ante_values.append(float(avg_ante_v))
        median_ante_v = metrics.get("median_ante")
        if isinstance(median_ante_v, (int, float)):
            median_ante_values.append(float(median_ante_v))
        win_rate_v = metrics.get("win_rate")
        if isinstance(win_rate_v, (int, float)):
            win_rate_values.append(float(win_rate_v))
        hand_top1_v = metrics.get("hand_top1")
        if isinstance(hand_top1_v, (int, float)):
            hand_top1_values.append(float(hand_top1_v))
        hand_top3_v = metrics.get("hand_top3")
        if isinstance(hand_top3_v, (int, float)):
            hand_top3_values.append(float(hand_top3_v))
        shop_top1_v = metrics.get("shop_top1")
        if isinstance(shop_top1_v, (int, float)):
            shop_top1_values.append(float(shop_top1_v))
        illegal_rate_v = metrics.get("illegal_action_rate")
        if isinstance(illegal_rate_v, (int, float)):
            illegal_rate_values.append(float(illegal_rate_v))

        out = {
            "seed": seed,
            "status": status,
            "primary_metric": numeric,
            "avg_ante_reached": avg_ante_v if isinstance(avg_ante_v, (int, float)) else None,
            "median_ante": median_ante_v if isinstance(median_ante_v, (int, float)) else None,
            "win_rate": win_rate_v if isinstance(win_rate_v, (int, float)) else None,
            "hand_top1": hand_top1_v if isinstance(hand_top1_v, (int, float)) else None,
            "hand_top3": hand_top3_v if isinstance(hand_top3_v, (int, float)) else None,
            "shop_top1": shop_top1_v if isinstance(shop_top1_v, (int, float)) else None,
            "illegal_action_rate": illegal_rate_v if isinstance(illegal_rate_v, (int, float)) else None,
            "stage": row.get("stage"),
            "error": row.get("error"),
            "elapsed_sec": row.get("elapsed_sec"),
        }
        per_seed_rows.append(out)

        if status != "ok":
            catastrophic_failures.append(
                {
                    "seed": seed,
                    "stage": row.get("stage"),
                    "error": row.get("error") or "unknown",
                }
            )

    mean_v = statistics.mean(values) if values else math.nan
    std_v = statistics.pstdev(values) if len(values) >= 2 else 0.0
    reward_mean = statistics.mean(reward_values) if reward_values else mean_v
    reward_std = statistics.pstdev(reward_values) if len(reward_values) >= 2 else (std_v if reward_values else 0.0)
    best_episode_reward = max(best_episode_values) if best_episode_values else math.nan
    avg_ante_mean = statistics.mean(avg_ante_values) if avg_ante_values else math.nan
    median_ante_mean = statistics.mean(median_ante_values) if median_ante_values else math.nan
    win_rate_mean = statistics.mean(win_rate_values) if win_rate_values else math.nan
    hand_top1_mean = statistics.mean(hand_top1_values) if hand_top1_values else math.nan
    hand_top3_mean = statistics.mean(hand_top3_values) if hand_top3_values else math.nan
    shop_top1_mean = statistics.mean(shop_top1_values) if shop_top1_values else math.nan
    illegal_rate_mean = statistics.mean(illegal_rate_values) if illegal_rate_values else math.nan

    return {
        "primary_metric": primary_metric,
        "count": len(seed_results),
        "count_valid_metric": len(values),
        "mean": mean_v,
        "std": std_v,
        "avg_reward": reward_mean,
        "reward_std": reward_std,
        "best_episode_reward": best_episode_reward,
        "avg_ante_reached": avg_ante_mean,
        "median_ante": median_ante_mean,
        "win_rate": win_rate_mean,
        "hand_top1": hand_top1_mean,
        "hand_top3": hand_top3_mean,
        "shop_top1": shop_top1_mean,
        "illegal_action_rate": illegal_rate_mean,
        "catastrophic_failure_count": len(catastrophic_failures),
        "catastrophic_failures": catastrophic_failures,
        "per_seed": per_seed_rows,
    }


def is_success(metric_summary: dict[str, Any]) -> bool:
    return int(metric_summary.get("catastrophic_failure_count") or 0) == 0
