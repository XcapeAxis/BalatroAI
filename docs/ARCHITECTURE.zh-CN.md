# BalatroAI 架构与数据流

> 语言切换： [简体中文](ARCHITECTURE.zh-CN.md) | [English](ARCHITECTURE.en.md)

## 高层流程

```mermaid
flowchart LR
  A["Balatro 游戏运行时"] -->|RPC| B["balatrobot 服务"]
  B --> C["Oracle 采集与快照"]
  C --> D["规范化 trace 与 scope hash"]
  D --> E["模拟器对齐检查与 fixtures"]
  E --> F["Trainer 流水线：Search / BC / DAgger / SelfSup / RL"]
  F --> G["P22 / P23 实验编排"]
  G --> H["docs/artifacts 下的产物与报告"]
```

## 模块职责（每块 1-2 句）

### Oracle / Sim 对齐层

对齐层把真实运行时产生的 oracle trace 作为事实来源，并持续检查模拟器行为是否与这些 trace 一致。它支撑 `P0-P10` 范围内的保真回归，确保模拟器改动不会在进入训练与实验结论前就失真。

### Hashing / Scope（`hand_score_observed_core` 等）

Scope 用来定义某个里程碑下哪些状态或动作结果属于契约关键内容，hashing 用来为这些观测生成确定性的指纹。这样即便非 scope 字段继续演进，回归结果也能保持稳定、可比较，并且可以按机制簇做有针对性的门禁。

### Trainer 层（fixtures + real sessions，P13 + P36）

Trainer 流水线既能消费模拟器 fixture，也能消费 `P13` / `P32` 风格引入的真实会话产物，用来降低 sim-to-real 漂移。当前这层既支持策略搜索，也支持表征学习预训练，主路径大致是 `Search -> BC -> DAgger -> SelfSup -> RL`，同时把指标写成可追踪产物。

### 实验编排层

编排器负责执行带 seed、预算控制、可恢复状态和逐 run 报告的实验矩阵。它通过统一的 `summary_table`、排名和 artifact 化决策来驱动长时评估、策略比较以及 champion / candidate 的状态更新。

## 以产物为中心的数据流

1. Oracle 采集把运行时 trace 和 fixture 写到模拟器相关产物目录。
2. 对齐门禁生成带 scope 的报告，如 `report_p*.json` 和 diff 诊断。
3. 训练与评测输出逐 run 和逐 seed 的指标结果。
4. 编排器把实验行聚合成：
   - `run_plan.json`
   - `telemetry.jsonl`
   - `live_summary_snapshot.json`
   - `summary_table.{csv,json,md}`
   - 每个实验对应的 `run_manifest.json`、`progress.jsonl`、`seeds_used.json`
5. 状态发布与 dashboard 步骤再消费这些产物，为 README、dashboard 和报告视图提供输入。
6. `P36` 自监督路径会把 trace 转成 `SelfSupSample`，训练共享编码器任务，如 `future_value`、`action_type`，再把摘要回写到 P22 矩阵报告。
7. `P36 replay v1` 通过 `trainer/replay/*` 把 real / sim trace 统一成 replay-step 合约，并用 `valid_for_training` 标记每一步是否适合用于训练。

## 相关文档

- [SIM_ALIGNMENT_STATUS.md](SIM_ALIGNMENT_STATUS.md)
- [EXPERIMENTS_P22.md](EXPERIMENTS_P22.md)
- [P36_SELF_SUP_LEARNING.md](P36_SELF_SUP_LEARNING.md)
- [ARCHITECTURE_P25.md](ARCHITECTURE_P25.md)
