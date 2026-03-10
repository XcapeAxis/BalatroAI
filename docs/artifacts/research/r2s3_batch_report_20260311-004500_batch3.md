# R2-S3 Batch 3 报告 — Controlled Self-Imitation

## 假设

`flat self-imitation` 之前已经被证伪；如果把它限制到后期、低比例、指定 slice / phase / action allowlist，可能会成为可控辅助项。

## 改动

- 扩展 `trainer/rl/ppo_config.py`
- 扩展 `trainer/rl/ppo_lite.py`
- 让 self-imitation 支持：
  - `slice_allowlist`
  - `phase_allowlist`
  - `action_type_allowlist`

## 验证

- Tier 0：`py_compile`、`pytest` 通过
- Tier 1：
  - `r2s3-b3-controlled-selfimit`

关键产物：

- `docs/artifacts/research/r2s3_closed_loop/r2s3-b3-controlled-selfimit/summary_table.json`
- `docs/artifacts/research/r2s3_closed_loop/r2s3-b3-controlled-selfimit/candidate_train/rl_train/best_trajectory_stats.json`

## 结果

- smoke arena：candidate `94.0` vs heuristic `400.0`
- `configured_enabled=true`
- `selected_episode_total=0`
- `mean_replay_ratio=0.0`
- allowlist：
  - `slice_stage:early`
  - `SELECTING_HAND`
  - `PLAY, DISCARD`

## 结论

Batch 3 的提升不是因为 self-imitation 真正提供了训练信号。  
这次 gain 更可能来自 curriculum / reward gating，而不是 imitation 本身。

## 决策

- 继续下一批次
- 方向：去掉对 self-imitation 的过高预期，优先细化 curriculum / reward 联动，争取拿到更强 smoke recipe
