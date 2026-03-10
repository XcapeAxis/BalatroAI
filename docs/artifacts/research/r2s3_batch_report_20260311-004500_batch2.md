# R2-S3 Batch 2 报告 — Bucket-aware Replay

## 假设

如果 replay 真按 bucket / slice / risk 权重进入 PPO，即使 source 仍较窄，也应当在训练与 arena 结果中看到可验证的结构化信号。

## 改动

- 扩展 `trainer/rl/ppo_config.py`
- 扩展 `trainer/rl/ppo_lite.py`
- 扩展 `trainer/rl/distributed_rollout.py`
- 扩展 `trainer/closed_loop/candidate_train.py`
- 新增 `trainer/tests/test_rl_r2s3_scaling.py`

## 验证

- Tier 0：
  - `py_compile` 通过
  - `pytest trainer/tests/test_rl_r2s3_scaling.py -q` 通过
- Tier 1：
  - `r2s3-b2-bucket-replay`

关键产物：

- `docs/artifacts/research/r2s3_closed_loop/r2s3-b2-bucket-replay/summary_table.json`
- `docs/artifacts/research/r2s3_closed_loop/r2s3-b2-bucket-replay/candidate_train/rl_train/bucket_replay_stats.json`

## 结果

- smoke arena：candidate `61.0` vs heuristic `400.0`
- `selected_failure_count=6`
- `failure_type_coverage=2`
- `failure_bucket_coverage=1`
- 当前被选中的 failure bucket 仍然只有：`discard_mismanagement`

## 结论

Batch 2 证明了 bucket-aware replay 已经真正进入训练，不再只是 manifest。  
但它也明确说明：当前 replay source 结构仍然过窄，训练并没有因为“有 bucket 配置”就自动得到更广 failure coverage。

## 决策

- 继续下一批次
- 方向：测试受控 self-imitation / curriculum gating 是否能在当前窄 source 上带来额外收益
