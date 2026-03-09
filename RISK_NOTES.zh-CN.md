# 风险说明：P0 v6 计分对齐改动

> 语言切换： [简体中文](RISK_NOTES.zh-CN.md) | [English](RISK_NOTES.en.md)

## 这轮改动新增或放大的主要风险

1. `sim/core/score_basic.py` 中对 scoring-card 选择的假设
- 当前 High Card 只计算最高牌
- Pair / Two Pair / Three / Four-kind 的 scoring-card 子集都是显式选出来的
- 风险：未来 Joker、enhancement、retrigger 等规则层可能会改变哪些牌应参与计分

2. `<5` 张牌的牌型边界
- Straight / Flush 现在只在 5 张牌评估路径里检测
- 风险：如果后续某个模式或机制允许非 5 张牌也形成有效牌型，这里会偏离

3. Four-of-a-kind 的 kicker 处理
- 当前 Four-kind 不把 kicker 的 rank chips 加入 scoring-card 总和
- 风险：如果后续 modifier / retrigger 机制要求混合处理，这里需要重新定义

4. 模拟器中的 `MENU` 转移语义
- 为了和 oracle fixture 对齐，增加了显式 `MENU` 处理
- 风险：如果真实生产流程依赖更深层的菜单栈语义，这里可能只对当前 fixture 过拟合

5. snapshot hand-cap 规范化
- snapshot replay 当前会按可观测 hand-cap 行为做规范化
- 风险：如果调试 fixture 中存在超上限手牌，抽牌逻辑可能会在语义上截断它们

6. scope 设计与身份噪声
- `p0_hand_score_observed_core` 有意避开高身份噪声的手牌比较
- 风险：这样可能会隐藏一些对后续机制重要的顺序 / 身份回归

## 与未来机制可能发生的冲突

1. Enhancements / Editions / Seals
- 当前 P0 计分对齐门禁没有纳入完整 modifier 代数
- 未来冲突：当加法 / 乘法 modifier 与 retrigger 叠加时，计分对齐可能再次失效

2. Joker 图与触发时序
- P0 核心还没有建模 Joker 的遍历时机和触发顺序
- 未来冲突：即使基础分对了，Joker 在手牌前后阶段的作用仍可能让最终观测总分偏离

3. Planet hand level
- 当前 P0 基础逻辑能对齐现有 fixture，但没有完整硬编码 hand level 的状态机
- 未来冲突：如果不按手牌谱系跟踪状态转移，level 相关的 base chips / mult 可能再次偏离

## 建议的隔离方式

1. 按机制族保留 feature flags
- `score_core_base_only`
- `score_core_planet`
- `score_core_modifiers`
- `score_core_jokers`

2. 保持分层 scope
- `p0_hand_score_observed_core`：稳定基础层
- `p1_hand_score_observed_core`：基础层 + 选定机制
- 按机制单独做诊断投影，例如 planet-only / modifier-only，便于精确 diff

3. 保留 analyzer 驱动的门禁方式
- 在大改公式前继续使用 `analyze_score_delta_mismatch.py`
- 修改核心公式前必须先拿到分解表证据

4. 保持确定性的 fixture 构造
- 每个 fixture 只隔离一个机制变量，避免归因混乱

## 实操护栏

- 任何 P1+ 改动后，都先跑一遍 P0 v6 batch 作为非回归门禁
- 把 P1 smoke 放在独立分支并按机制族拆提交，便于快速回滚
- 给已知通过的对齐点打标签（本轮已预留 `sim-p0-v6-pass`）
