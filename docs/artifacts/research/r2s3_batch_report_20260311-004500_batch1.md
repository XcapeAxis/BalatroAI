# R2-S3 Batch 1 报告 — Failure Taxonomy

## 假设

当前纯 RL 的 replay 压力不足，不只是样本量问题，更是 failure 分类过粗导致的“错误聚类”。  
如果先把 `hard-case` 正式拆成可审计 buckets，后续 replay / curriculum 才有意义。

## 改动

- 更新 `trainer/closed_loop/failure_buckets.py`
- 更新 `trainer/closed_loop/failure_mining.py`
- 输出 `failure_bucket_candidates` / `failure_bucket_signals`

## 验证

- Tier 0：`py_compile` 通过
- Tier 1：`python trainer/closed_loop/failure_mining.py --config configs/experiments/r2s3_p40_failure_mining_bucketed.yaml --run-id r2s3-b1-failure-taxonomy`

关键产物：

- `docs/artifacts/research/r2s3_failure_buckets/r2s3-b1-failure-taxonomy/failure_bucket_stats.json`

## 结果

- 当前样本里主导 bucket 不再是粗粒度 `early_collapse`
- 实际主导 bucket 为 `discard_mismanagement=6`
- 其余高价值 buckets 仍然 scarce

## 结论

Batch 1 成功修正了 failure truthfulness。  
后续训练应围绕“扩大 bucket coverage”，而不是继续把 `early_collapse` 当单一主导标签。

## 决策

- 继续下一批次
- 方向：把 bucket-aware replay 真接进 PPO，并观察它在当前狭窄 source 上是否已经生效
