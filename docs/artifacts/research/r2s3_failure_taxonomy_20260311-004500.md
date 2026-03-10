# R2-S3 Failure Taxonomy（2026-03-11）

## 结论

本轮先把 `hard-case` 进一步拆成可审计的 `failure buckets`，再决定 replay 和 curriculum 是否值得继续加压。  
当前已经确认：

- `legality drift` 不再是主问题。
- 旧的粗粒度 `early_collapse` 在这批样本里，真实主导 bucket 是 `discard_mismanagement`。
- 当前 failure source 仍然过窄，尚未把 `resource_pressure_misplay`、`shop_or_economy_misallocation`、`position_sensitive_misplay` 等高价值 bucket 真正打进训练主分布。

## 当前 bucket 规则

本轮正式落地并保留以下 bucket：

1. `early_collapse`
2. `resource_pressure_misplay`
3. `discard_mismanagement`
4. `position_sensitive_misplay`
5. `stateful_joker_misplay`
6. `shop_or_economy_misallocation`
7. `risk_overcommit`
8. `risk_undercommit`
9. `low_score_survival`
10. `invalid_or_wasted_decision`

判定逻辑已进入：

- `trainer/closed_loop/failure_buckets.py`
- `trainer/closed_loop/failure_mining.py`

并且每条 failure 现在都会保留：

- `failure_bucket`
- `bucket_reason`
- `failure_bucket_candidates`
- `failure_bucket_signals`

## 本轮证据

来源：`docs/artifacts/research/r2s3_failure_buckets/r2s3-b1-failure-taxonomy/failure_bucket_stats.json`

- 当前样本总量：`6`
- 主导 bucket：`discard_mismanagement=6`
- 稀缺 bucket：
  - `early_collapse`
  - `resource_pressure_misplay`
  - `position_sensitive_misplay`
  - `stateful_joker_misplay`
  - `shop_or_economy_misallocation`
  - `risk_overcommit`
  - `risk_undercommit`
  - `low_score_survival`
  - `invalid_or_wasted_decision`

## 研发含义

- 这说明 R2-S2 里大量被粗分类成 `early_collapse` 的失败，至少在当前 failure pack 上，本质上更像 `discard_mismanagement`。
- replay 逻辑本身已经能消费 bucket，但当前数据分布仍然主要来自单一 bucket。
- 因此 R2-S3 后续的主要瓶颈不是 “bucket-aware replay 是否工作”，而是 “failure bucket source 是否足够宽”。

## 下一步

优先级保持为：

1. 扩大真实 failure source，尤其是 `resource_pressure_misplay`、`shop_or_economy_misallocation`、`position_sensitive_misplay`
2. 继续保留 bucket-aware replay，但不再把“更多 reward 微调”当主要杠杆
3. 受控 self-imitation 只作为后期低比例辅助，不单独作为下一阶段主线
