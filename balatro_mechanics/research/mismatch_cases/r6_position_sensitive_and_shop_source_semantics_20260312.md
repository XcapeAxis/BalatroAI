# R6 机制研究：position-sensitive / shop source 语义边界（2026-03-12）

## 问题描述
R6 的 gap-swap richer-source family 扩容后，`position_sensitive_misplay` 与 `shop_or_economy_misallocation` 已经真实进入训练，但认证结果依然不升反降，需要判断这是否来自规则错配，还是来自 failure source 语义/样本结构问题。

## 触发案例
- `D:\MYFILES\BalatroAI_r1\docs\artifacts\research\r6_certification\r6-c5-gap-swap-balanced-nonoverlap-cert-20260312\triage_report.json`
- `D:\MYFILES\BalatroAI_r1\docs\artifacts\research\r6_certification\r6-c6-gap-swap-scarcity-nonoverlap-cert-20260312\triage_report.json`
- `D:\MYFILES\BalatroAI_r1\docs\artifacts\research\r6_failure_buckets\r6-b8-gap-swap-resource-shop-failure-20260312\failure_pack_stats.json`
- `D:\MYFILES\BalatroAI_r1\docs\artifacts\research\r6_failure_buckets\r6-b10-gap-swap-balanced-nonoverlap-failure-20260312\failure_pack_stats.json`

## 当前实现
当前 failure mining 会把 `position_sensitive_misplay`、`resource_pressure_misplay`、`shop_or_economy_misallocation`、`risk_undercommit` 通过 `candidate_slice_failure_seed / compound_slice_failure_seed / slice_coverage_gap_seed / failure_mining` 等路径送入 replay。B8/B10 已证明这些 source 会被 `selected_for_training`。

## 资料来源
- 本地项目资料与 artifacts；本次未触发外部网络搜索。
- 未触发原因：当前证据更像 failure source 语义和覆盖度问题，而不是 simulator 与真实规则的直接冲突。

## 可信度判断
- 中等。结论足以指导训练路线，但不足以宣告规则完全无误。

## 结论
1. 目前没有直接证据表明 `position_sensitive` 或 `shop/economy` 的 simulator 规则与真实游戏规则发生冲突。
2. 当前更可能的问题是：这些 source 进入训练了，但 source 之间的语义边界仍不够干净，导致 replay 预算虽然变宽，却没有形成足够强的可泛化学习信号。
3. 因此 R6 的下一优先级不应是立即改 simulator 规则，而应是继续做 source overlap / dedup 和 richer-source v2，必要时再回头做外部规则核验。

## 对训练/评测/代码的影响
- 训练层：继续优先优化 non-overlap source expansion 与 quota，而不是先动 simulator。
- 评测层：重点关注 `slice_position_sensitive=false`、`slice_stage=early` 的退化是否随 richer-source v2 而改善。
- 代码层：failure mining 和 replay 逻辑需要继续输出更细的 source provenance，以便下一轮判断是否要升级为真正的规则研究。

## 下一步建议
如果下一轮 richer-source v2 仍然无法改善 `position_sensitive` 相关 certified slice，则应把该主题升级为外部规则核验，并补充网络来源。
