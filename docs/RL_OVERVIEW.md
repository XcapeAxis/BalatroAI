# RL Overview (P37)

## Design Principles

- Keep P37 as a research skeleton first: stable interfaces, reproducible artifacts, minimal algorithm assumptions.
- Reuse existing simulator/backend and feature contracts instead of introducing a parallel runtime.
- Keep P22 integration native so RL rows follow the same seed governance, run manifests, and summary paths.

## Environment Interface

`trainer/rl/env.py` provides a Gym-like wrapper:

- `reset(seed=None) -> obs`
- `step(action) -> obs, reward, done, info`
- `render() -> str`

The wrapper uses `trainer.env_client.create_backend(...)` and currently defaults to simulator-backed rollout.

## Reward Design (Current)

Current reward modes are intentionally simple:

- `score_delta` (default): per-step delta of `round.chips`
- `episode_total_score`: direct current total score proxy

This keeps training signals easy to inspect during early integration. Reward shaping beyond this should be treated as future work and reported with seed-robust comparisons.

## Self-Play and Replay Relationship

- Self-play (`trainer/rl/selfplay.py`) generates online trajectories and per-episode progress logs.
- Replay/self-supervised pipelines (P36) remain a separate path focused on representation pretraining from trace artifacts.
- The intended long-term flow is: replay/self-supervised encoder initialization -> RL fine-tune/self-play iteration under P22 multi-seed gating.

## P22 Integration

P37 adds `experiment_type: rl_selfplay` rows to `configs/experiments/p22.yaml`:

- `rl_ppo_smoke`
- `rl_ppo_medium`

Orchestrator behavior:

- resolves seeds into `seeds_used.json`
- executes PPO skeleton per seed
- appends per-episode progress events (reward/length/wall_time)
- writes experiment metrics to `docs/artifacts/p22/runs/<run_id>/<exp_id>/metrics.json`

## Current Scope and Future Directions

Current scope:

- functional env interface
- rollout buffer and return computation
- self-play loop with artifact persistence
- one-step PPO-like policy gradient update

Future directions:

- full PPO loop (mini-batch epochs, clipping, GAE, value bootstrap)
- A2C / vanilla policy gradient baselines
- MCTS + prior policy hybrids
- AlphaZero-like self-play with stronger planning targets
- tighter shared-encoder workflows with P36 checkpoints

