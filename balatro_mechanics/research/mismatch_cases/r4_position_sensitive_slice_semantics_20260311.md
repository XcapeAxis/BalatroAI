# R4 机制研究：`slice_position_sensitive` 长期为 `unknown` 的原因排查

## 问题描述
在 R4 的多轮 certification / diagnostics 中，`slice_position_sensitive` 长期几乎只出现 `unknown`，这会干扰我们判断 pure RL 是否真的覆盖到了 position-sensitive failure 家族。

## 触发样例
- `docs/artifacts/research/r4_certification/20260311-152822/triage_report.json`
- `docs/artifacts/research/r4_certification/20260311-221824/triage_report.json`
- `docs/artifacts/research/r4_candidate_train/20260311-153941/rl_train/slice_metrics.json`

## 当前实现
- `trainer/common/slices.py` 负责 `infer_slice_position_sensitive(...)`
- 旧逻辑会把“已知 jokers 为空列表”的状态也归入 `unknown`
- 在当前 long-episode 训练/评测样本中，很多状态没有可重排或显式 position-sensitive joker 证据，因此统计上持续偏向 `unknown`

## 本地证据
1. 子模型排查表明，当前训练与 arena 流里几乎没有真实 reorder/move 动作样本。
2. 很多 shop / play state 的 `jokers` 字段是已知空列表，而不是缺失。
3. 因此这更像是诊断语义问题与样本覆盖问题，而不是 simulator 规则与真实机制冲突。

## 外部搜索情况
- 本次没有触发外部搜索。
- 原因：当前证据不足以表明存在 sim ↔ 真机规则冲突，更像是本地诊断标签语义与样本稀缺问题；先修本地语义更合理。

## 可信度判断
- 可信度：中等偏高。
- 原因：现象可以被本地实现与样本分布解释，目前没有直接反证表明是规则实现错误。

## 结论
- 这不是当前需要升级为“机制错配”的问题。
- 已完成的有效修复是：把“已知 jokers 为空列表”从 `unknown` 调整为 `false`。
- 下一步真正需要做的不是继续争论规则，而是构造更多真实 `position_sensitive` / `stateful_joker` source，让训练和认证里能出现 `selected > 0` 的样本。

## 对训练 / 评测 / 代码的影响
- 评测结论更可信：`position_sensitive:false` 与 `unknown` 被区分开了。
- replay 设计上，后续应优先补强真正的 position-sensitive source，而不是把 `unknown` 当成 failure family。

## 下一步建议
1. 在 failure mining 中专门扩 `position_sensitive_misplay` 与 `stateful_joker_misplay` source。
2. 在 slice-aware replay 中把 `position_sensitive:false` 与 `position_sensitive:unknown` 分开统计。
3. 只有在出现明确“真实规则疑点”时，再升级到外部资料搜索与规则沉淀。
