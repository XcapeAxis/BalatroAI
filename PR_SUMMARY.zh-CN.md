# PR 摘要：Sim Score Core Parity（P0 v6）

> 语言切换： [简体中文](PR_SUMMARY.zh-CN.md) | [English](PR_SUMMARY.en.md)

## 这次修复达成了什么

这个 PR 让模拟器在手牌级得分上与 Oracle 观测到的 `round.chips` 增量对齐，并移除了 P0 回归里噪声很高的 diff 来源。

演进过程：

- v4 / v5：频繁在不可观测的 `last_base_*` 字段或高身份噪声投影上出现 `diff_fail`
- v5（中间状态）：主对齐信号改成可观测的 `score_observed.delta`，但计分核心与状态转移语义仍未完全对齐
- v6：**P0 回归达到 8/8 通过**，动作后 trace 合约稳定，fixture 流程可重复

最终 v6 统计（`sim/tests/fixtures_runtime/oracle_p0_v6/report_p0.json`）：

- pass：8
- diff_fail：0
- oracle_fail：0
- gen_fail：0
- skipped：0

## 按模块看关键改动

### 1）计分核心

- `sim/core/rank_chips.py`
  - 引入显式的基础 rank -> chip 映射（`A=11`、`K/Q/J/T=10`、`9..2=n`）
- `sim/core/score_basic.py`
  - 重构为基于 breakdown 的计分流程：
    - 手牌类型识别
    - scoring-card 集合选择
    - base chips / mult
    - `total_delta = (base + sum(scoring card chips)) * mult`
  - 修正边界行为：
    - `<5` 张牌不再误判成 Straight / Flush
    - Four-of-a-kind 只统计 4 张匹配牌
    - High Card 只统计最高牌
- `sim/core/engine.py`
  - 出牌计分现在统一消费 `evaluate_selected_breakdown()` 的 `total_delta`

### 2）引擎 step / 状态转移对齐

- `sim/core/engine.py`
  - 改善非法资源处理和回退行为一致性
  - 增加显式 `MENU` 动作语义，用于和 oracle trace 对齐
  - 在 replay 加载 snapshot 时，把 hand-cap 行为规范到可观测范围

### 3）Hash scope 与 diff 信号质量

- `sim/core/hashing.py`
  - 新增 / 迭代 `p0_hand_score_observed_core`，使其更聚焦可观测字段
  - 从该 scope 中移除了高身份噪声比较，例如完整手牌身份顺序，避免误报

### 4）Fixture 生成与 Oracle 回放

- `sim/oracle/generate_p0_trace.py`
  - 确保合成起始状态会被持久化，保证 replay 一致性
  - 改善 round-eval 目标下的确定性生成流程

### 5）调试与诊断工具

- `sim/oracle/analyze_score_delta_mismatch.py`
  - 为每个目标增加分解表（CSV / MD），包括：
    - 选中的牌
    - 识别到的牌型
    - base chips / mult
    - rank chips
    - 预测核心值
    - oracle 与 sim 的 delta 对比
  - 这样可以准确定位失败是来自选牌、牌型识别、rank chips，还是最终公式

## 设计决策

- 使用 `round.chips` 增量（`score_observed.delta`）作为主对齐信号：
  - 它来自真实运行时，可观测且稳定
  - 它避免依赖不可移植或缺失的 `last_base_*` 字段
- 保持 hash scope 分层：
  - P0 scope 只比较当前里程碑真正需要的稳定语义
  - 高身份噪声的区域比较保留在计分门禁之外，以降低噪声

## 回归覆盖与当前限制

P0 v6 已覆盖：

- PLAY / DISCARD 的手牌级语义
- 资源计数，如 `hands_left`、`discards_left`
- 核心牌型识别与基础计分对齐
- 带 oracle 差分检查的定向 fixture

尚未覆盖：

- Joker trigger graph
- 完整宏观经济 / 商店决策对齐
- 全量 modifier / edition / seal 组合
- RNG 内部状态的完整等价

## 可复现命令（已执行）

```powershell
python -B sim\oracle\batch_build_p0_oracle_fixtures.py --base-url http://127.0.0.1:12346 --out-dir sim\tests\fixtures_runtime\oracle_p0_v6 --max-steps 160 --scope p0_hand_score_observed_core --seed AAAAAAA --resume --dump-on-diff sim\tests\fixtures_runtime\oracle_p0_v6\dumps

python -B sim\oracle\analyze_score_delta_mismatch.py --fixtures-dir sim\tests\fixtures_runtime\oracle_p0_v6
```

产物：

- report：`sim/tests/fixtures_runtime/oracle_p0_v6/report_p0.json`
- analyzer markdown：`sim/tests/fixtures_runtime/oracle_p0_v6/score_mismatch_table.md`
- analyzer csv：`sim/tests/fixtures_runtime/oracle_p0_v6/score_mismatch_table.csv`

## 提交轨迹（高层）

- debug analyzer：`e88e0ea`
- rank chip mapping：`4c78d88`
- hand detection & scoring-card fixes：`4f41a6f`
- core formula parity：`9a0b48f`
- observed scope stabilization：`77b2860`
- menu / fixture deterministic flow：`649c314`
