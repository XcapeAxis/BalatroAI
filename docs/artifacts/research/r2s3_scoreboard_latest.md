# R2-S3 Scoreboard（Latest）

## 当前 strongest certified pure-RL recipe

- recipe: `r2s3-c4-certification-candidate-rl`
- checkpoint_id: `rl_policy:p42_rl_candidate:r2s3-c4-certification-candidate-rl:aaaaaaa,bbbbbbb,ccccccc,ddddddd:e3acd6cbae`
- certified score: `98.25`
- unified compare score: `87.0`
- heuristic compare score: `303.0`
- certified gap vs heuristic: `204.75`
- unified compare gap vs heuristic: `216.0`
- status: `observe`

## 批次总表

| Batch | 假设 | 关键改动 | 结果 | 认证状态 | strongest checkpoint | 是否继续 | 备注 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| B1 | 先把 hard-case 拆成真实 buckets，再谈 replay | `failure_buckets.py` / `failure_mining.py` | `discard_mismanagement=6`，纠正了粗粒度 `early_collapse` | 未认证 | N/A | 是 | truthfulness 修复成功，但 source 仍窄 |
| B2 | bucket-aware replay 真进入 PPO 后，至少应看到结构化回放信号 | replay weight / quota / slice / risk 接入 PPO | candidate `61.0` vs heuristic `400.0` | 未认证 | `072474091e` | 否，保留为控制组 | replay 生效，但 `failure_bucket_coverage=1` |
| B3 | 受控 self-imitation 可能成为可用辅助 | late-stage + allowlist self-imitation | candidate `94.0` vs heuristic `400.0` | 未认证 | `25268b1c1b` | 部分继续 | 这次提升并非 imitation 真正生效，`selected_episode_total=0` |
| B4 | curriculum / reward refine 仍可能是当前最强 smoke 配方 | refined curriculum + reward schedule | candidate `103.5` vs heuristic `400.0` | fast_pass | `32e2ac27f4` | 是 | strongest smoke，但 `selected_failure_count=1` |
| C4 | 用认证级协议确认 strongest recipe 是否成立 | `r2s3-c4-certification` + 横向 compare | certified `98.25` vs heuristic `303.0`；统一 compare `87.0` | 已认证 | `e3acd6cbae` | 是 | strongest certified pure-RL recipe 已更新 |

## 本轮确认有效的组件

- failure bucket taxonomy
- bucket-aware replay wiring
- stage-aware curriculum / reward refine
- 低比例、后期、受控 self-imitation 作为可保留辅助约束

## 本轮被证伪或不值得优先继续的组件

- `flat self-imitation`
- 仅靠当前狭窄 failure source 的 bucket replay
- 继续泛泛调 reward，而不先扩展 failure coverage

## 当前最关键瓶颈

1. `failure_bucket_coverage` 仍然太窄
2. dominant source 仍然过度集中在 `discard_mismanagement`
3. `resource_pressure_misplay` / `shop_or_economy_misallocation` / `position_sensitive_misplay` / `stateful_joker_misplay` 样本不足

## 下一阶段建议

继续纯 RL 主线。  
下一阶段默认优先顺序：

1. failure source 扩容
2. slice-targeted replay（early / position-sensitive / resource pressure）
3. richer bucket mix 下重跑 curriculum certification
4. 仅在 richer source 条件下再重新评估 controlled self-imitation
