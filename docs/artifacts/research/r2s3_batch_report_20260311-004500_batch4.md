# R2-S3 Batch 4 报告 — Curriculum / Reward Refine

## 假设

在当前 failure source 仍然偏窄的前提下，stage-aware curriculum + reward schedule 仍可能比单纯 bucket replay 更稳定。

## 改动

- 新增 / 调整：
  - `configs/experiments/r2s3_p44_curriculum_bucketed_refined.yaml`
  - `configs/experiments/r2s3_p42_closed_loop_bucketed_curriculum_smoke.yaml`
- 延续 bucket-aware replay，但把 strongest smoke recipe 收敛到 curriculum / reward refine

## 验证

- Tier 0：`py_compile`、sidecar sync 通过
- Tier 1：
  - `r2s3-b4-curriculum-refine`

关键产物：

- `docs/artifacts/research/r2s3_closed_loop/r2s3-b4-curriculum-refine/summary_table.json`
- `docs/artifacts/research/r2s3_closed_loop/r2s3-b4-curriculum-refine/candidate_train/metrics.json`
- `docs/artifacts/research/r2s3_closed_loop/r2s3-b4-curriculum-refine/candidate_train/rl_train/hardcase_stats.json`

## 结果

- smoke arena：candidate `103.5` vs heuristic `400.0`
- `eval_mean_score=75.0`
- `invalid_action_rate=0.0`
- 这是本轮最强 smoke recipe
- 但 `selected_failure_count=1`，failure source 反而进一步收窄

## 结论

Batch 4 说明当前 strongest smoke 提升，更多来自 curriculum / reward refine，而不是 replay coverage 的实质扩大。  
这使它适合进入认证，但也明确暴露出下一阶段真正需要补的是 failure source 扩容。

## 决策

- 进入认证级 compare
- 用 certification 回答：这个 strongest smoke recipe 是否在更严格协议下仍然优于既有 certified pure-RL 基线
