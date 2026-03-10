# R2-S3 认证报告 — C4 Certification / Compare

## 目标

验证 R2-S3 strongest recipe 是否在认证级协议下，继续优于既有 certified pure-RL 基线，并明确当前 strongest certified recipe。

## 认证运行

- run_id: `r2s3-c4-certification`
- summary:
  - `docs/artifacts/research/r2s3_certification/r2s3-c4-certification/summary_table.json`
- promotion:
  - `docs/artifacts/research/r2s3_certification/r2s3-c4-certification/promotion_decision.json`
- triage:
  - `docs/artifacts/research/r2s3_certification/r2s3-c4-certification/triage_report.json`

## 认证结果

- certified candidate score：`98.25`
- heuristic champion：`303.0`
- certified gap：`204.75`
- recommendation：`observe`
- 相比 R2-S2 认证基线，`score_delta_change_vs_baseline = +24.25`

新认证 checkpoint：

- `rl_policy:p42_rl_candidate:r2s3-c4-certification-candidate-rl:aaaaaaa,bbbbbbb,ccccccc,ddddddd:e3acd6cbae`

## 统一横向 compare

来源：

- `docs/artifacts/research/r2s3_compare/r2s3-c4-compare-20260311-0040/summary_table.json`

统一协议下的排名：

1. `heuristic_baseline = 303.0`
2. `r2s3_bucket_curriculum_cert = 87.0`
3. `reward_survival_baseline = 84.25`
4. `r2s3_controlled_selfimit = 73.75`
5. `r2s1_curriculum_cert = 59.75`
6. `r2s3_bucket_replay = 47.0`

## 结论

- R2-S3 已经把 strongest certified pure-RL recipe 向前推进了一步。
- 当前 strongest certified recipe 是：
  - bucket-aware replay
  - stage-aware curriculum / reward refine
  - self-imitation 保留为受控辅助，但这轮并未成为主要增益来源
- strongest certified recipe 已明确更新为 `r2s3-c4-certification-candidate-rl`
- 但它仍显著弱于 heuristic champion，尚不具备 promotion 条件

## 下一阶段建议

继续纯 RL 主线，不切回 teacher / warm-start。  
下一阶段最值得做的是：

1. 扩大 failure source，尤其是：
   - `resource_pressure_misplay`
   - `shop_or_economy_misallocation`
   - `position_sensitive_misplay`
   - `stateful_joker_misplay`
2. 让 replay 从“bucket-aware 但 source 狭窄”升级到“bucket-aware 且 source 丰富”
3. 继续把 self-imitation 限制在后期、低比例、allowlist 下，只作为次级杠杆
