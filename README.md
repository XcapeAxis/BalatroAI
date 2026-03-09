# BalatroAI

> English README: [README.en.md](README.en.md)

<!-- BADGES:START -->
[![Latest Gate](https://img.shields.io/badge/Latest_Gate-RunP29_PASS-2EA44F)](scripts/run_regressions.ps1)
[![Workflow](https://img.shields.io/badge/Workflow-mainline--only-2EA44F)](scripts/git_sync.ps1)
[![Seed Governance](https://img.shields.io/badge/Seed_Governance-P23%2B_enabled-0E8A16)](configs/experiments/seeds_p23.yaml)
[![Experiment Orchestrator](https://img.shields.io/badge/Experiment_Orchestrator-P22%2B_enabled-1F6FEB)](scripts/run_p22.ps1)
[![Trend Warehouse](https://img.shields.io/badge/Trend_Warehouse-P26%2B_enabled-0E8A16)](docs/TREND_WAREHOUSE_P26.md)
[![Docs Coverage](https://img.shields.io/badge/Docs_Coverage-P15--P38-6E7781)](docs/)
[![Platform](https://img.shields.io/badge/Platform-Windows-0078D6)](USAGE_GUIDE.md)
[![Python](https://img.shields.io/badge/Python-3.12%2B-3776AB)](trainer/requirements.txt)
[![License](https://img.shields.io/badge/License-Not_Specified-6E7781)](#license-and-contributing)
[![CI Smoke](https://github.com/XcapeAxis/BalatroAI/actions/workflows/ci-smoke.yml/badge.svg)](https://github.com/XcapeAxis/BalatroAI/actions/workflows/ci-smoke.yml)
[![GitHub Stars](https://img.shields.io/github/stars/XcapeAxis/BalatroAI?style=social)](https://github.com/XcapeAxis/BalatroAI/stargazers)
[![GitHub Issues](https://img.shields.io/github/issues/XcapeAxis/BalatroAI)](https://github.com/XcapeAxis/BalatroAI/issues)
<!-- BADGES:END -->

BalatroAI 是一个围绕 Balatro 风格决策研究搭建的本地工程仓库。它把模拟器、训练、评测、实验编排、运维视图和本地 Web Demo 放在同一条可复现工作流里，方便在一台 Windows 机器上继续开发、验证和演示。

## Quick Start / 快速开始

如果你现在最关心的是演示，直接运行：

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_mvp_demo.ps1 -OpenBrowser
```

默认地址：

```text
http://127.0.0.1:8050/
```

如果本地还没有可用模型，先执行：

```powershell
powershell -ExecutionPolicy Bypass -File scripts\bootstrap_mvp_demo.ps1 -OpenBrowser
```

如果你要接手整个仓库，推荐顺序是：

1. 克隆仓库并进入目录。

```powershell
git clone https://github.com/XcapeAxis/BalatroAI.git
cd BalatroAI
```

2. 初始化本地 Windows 环境。

```powershell
powershell -ExecutionPolicy Bypass -File scripts\setup_windows.ps1 -Mode auto -SkipSmoke
```

3. 运行环境体检。

```powershell
powershell -ExecutionPolicy Bypass -File scripts\doctor.ps1
```

4. 跑一轮默认 quick 流程。

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_p22.ps1 -Quick
```

## What This Project Is / 项目定位

这个仓库当前有两条清晰主线：

- 面向演示的本地 AI 决策沙盘：浏览器里可以直接看场景、局面、推荐、执行后变化和训练状态。
- 面向研发的实验与运维主线：覆盖 `P22` 编排、回放验证、种子治理、夜间运行、Ops UI 和交接脚本。

当前 Demo 已经能稳定展示这些能力：

- 默认打开是薄顶栏 + 左中右三栏 + 底部摘要带：左边选场景，中间看局面，右边看模型 vs 基线，下方看结果、时间线和训练成果。
- 界面会根据分辨率和窗口比例自适应：宽屏优先保持一屏总览，较窄窗口自动收成更稳的单列布局。
- `3` 个内置高质量场景，可直接用于面试演示。
- 模型与启发式并排推荐，能直接看出差异。
- 点击推荐后，局面、结果预览和时间线会联动更新。
- 页面内能看到真实训练出的模型和训练过程状态。

当前最适合现场演示的顺序是：

1. 选一个场景故事。
2. 看当前局面为什么关键。
3. 看模型和启发式是否一致、差异在哪里。
4. 执行当前动作。
5. 看一步后的因果变化和训练成果。

## Scope and Boundaries / 适用边界

适合：

- Balatro 风格 AI 决策可视化与交互演示
- 模拟器对齐、回放验证和可复现实验
- 带 seed 治理的训练、评估与对比
- 本地 dashboard、Ops UI、attention queue 和 morning summary

不适合：

- 直接控制商业游戏客户端
- 绕过门禁直接做高风险 promotion
- 没有来源记录的环境改动
- 把当前模型包装成“已经完成的最优代理”

## Architecture Overview / 架构总览

主流程可以概括为：

1. 运行时或 `balatrobot` 产出 trace 与回放数据。
2. 模拟器负责做局部对齐和一致性验证。
3. 训练与评测消费这些数据，生成模型、指标和摘要。
4. `P22` 统一编排实验、seed、产物和比较结果。
5. Demo、dashboard、Ops UI 和 attention queue 读取最新产物做展示。

继续阅读：

- Demo 说明：[DEMO_README.zh-CN.md](DEMO_README.zh-CN.md)
- 使用指南：[USAGE_GUIDE.zh-CN.md](USAGE_GUIDE.zh-CN.md)
- 架构说明：[docs/ARCHITECTURE.zh-CN.md](docs/ARCHITECTURE.zh-CN.md)
- 演示脚本：[docs/MVP_DEMO_SCRIPT.md](docs/MVP_DEMO_SCRIPT.md)

## Reproducibility / 可复现性

仓库默认强调“能复现、能追踪、能回看”：

- 主参考文档：[docs/REPRODUCIBILITY_P25.md](docs/REPRODUCIBILITY_P25.md)
- Seed 与运行追踪：[docs/SEEDS_AND_REPRODUCIBILITY.md](docs/SEEDS_AND_REPRODUCIBILITY.md)
- 推荐体检脚本：`scripts\doctor.ps1`
- 推荐主线门禁：`powershell -ExecutionPolicy Bypass -File scripts\run_regressions.ps1 -RunP22`

## Example Outputs / 示例产物

你可以直接从这些文件快速判断仓库当前状态：

- Demo API smoke：`docs/artifacts/mvp/api_smoke_20260309_154858.json`
- Demo 训练状态：`docs/artifacts/mvp/training_status/latest.json`
- P22 汇总示例：`docs/artifacts/p22/runs/<run_id>/summary_table.json`

## Roadmap / 路线图

接下来的重点不是无约束扩张，而是在保持可验证性的前提下继续增强：

- Demo 的可视化表达、解释力和训练可信度
- 模拟器对齐、回放验证和数据质量
- 训练、评估、路由与自治入口之间的衔接

路线图文档：

- 中文版：[docs/ROADMAP.zh-CN.md](docs/ROADMAP.zh-CN.md)
- English version: [docs/ROADMAP.en.md](docs/ROADMAP.en.md)

## Known Limitations / 当前限制

- 更深层的 milestone 文档仍保留仓库历史写法，还没有全部重写成统一中文。
- 当前 Demo 仍然是 scenario-driven，模型也还是最小可用版本，不是最终形态。
- 一部分研发文档和实验脚本仍然默认面向维护者，而不是第一次接触项目的读者。

## Repository Status / 仓库状态

<!-- STATUS:START -->
<!-- README_STATUS:BEGIN -->
### Repository Status (Auto-generated)

- branch: main
- latest_gate: RunP29 (PASS)
- recent_trend_signal: regression
- trend_warehouse_last_updated: 2026-03-02T17:16:27.519440+00:00
- trend_rows_count: 20115
- champion: p40_closed_loop_smoke (champion)
- candidate:  (decision: hold)
- docs_coverage: P15-P38
<!-- README_STATUS:END -->
<!-- STATUS:END -->

<a id="license-and-contributing"></a>

## License and Contributing / 许可与贡献

- 许可：仓库顶层目前没有单独的 `LICENSE` 文件。
- 贡献方式：默认直接在 `main` 上推进，保持变更可审计，并在修改运行默认值前执行对应门禁。
