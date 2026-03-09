# BalatroAI

> English README: [README.en.md](README.en.md)

BalatroAI 是一个面向 Balatro 的本地工程仓库。它把模拟器对齐、训练、评测、夜间运行和本地展示整合成一条可复现、可追踪、可交接的工作流。

## 项目定位

这个仓库现在有两条清晰主线：

- 面向演示的本地 Web Demo：可直接在浏览器里看场景、局面、推荐、执行后变化和训练状态
- 面向研发的主线系统：覆盖 `P22` 编排、`P53` 本地运维、`P57` 夜跑自治、`P58` Windows 交接、`P60` AGENTS 规范、`P61` 验证金字塔

## 本地 MVP Demo

当前仓库已经包含一个适合面试直接展示的本地 Web Demo。它不是研究后台，而是一个可以打开浏览器、切场景、比较模型与基线、执行一步并展示训练过程的本地 AI 决策沙盘。

你可以在 2 分钟内展示这些内容：

- 本地浏览器 UI，直接由 simulator 驱动，不是静态 mock 数据
- `3` 个内置高质量场景：高收益起手、高压弃牌转折、Joker 协同爆发
- 模型与启发式基线的并排推荐对比
- 点击推荐后，局面高亮、结果预览、时间线同步联动
- 一个真实训练出的最小可用模型，以及页面内可见的训练过程

一键启动：

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_mvp_demo.ps1 -OpenBrowser
```

如需先补一个可用模型再启动：

```powershell
powershell -ExecutionPolicy Bypass -File scripts\bootstrap_mvp_demo.ps1 -OpenBrowser
```

默认地址：

```text
http://127.0.0.1:8050/
```

## 快速开始

如果你要接手这个仓库，优先走这条路径：

1. 克隆仓库并进入目录。

```powershell
git clone https://github.com/XcapeAxis/BalatroAI.git
cd BalatroAI
```

2. 初始化本地 Windows 环境。

```powershell
powershell -ExecutionPolicy Bypass -File scripts\setup_windows.ps1 -Mode auto -SkipSmoke
```

3. 做环境体检。

```powershell
powershell -ExecutionPolicy Bypass -File scripts\doctor.ps1
```

4. 运行默认 quick 流程。

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_p22.ps1 -Quick
```

如果你只想演示 Demo，则直接运行 `scripts\run_mvp_demo.ps1` 即可。

## 项目边界

适合：

- Balatro 风格 AI 决策可视化
- simulator 对齐与回放验证
- 带 seed 治理的训练、评测与对比
- 本地 dashboard、Ops UI、attention queue、morning summary
- Windows 机器上的可复现实验交接

不适合：

- 直接控制商业游戏客户端
- 不经过门禁的高风险 promotion
- 没有来源记录的环境变更
- 把当前模型包装成“通用最优代理”

## 当前架构

仓库主线可以概括为：

1. 真实运行时或 `balatrobot` 产出 Oracle trace。
2. 模拟器与 hash scope 做对齐验证。
3. 训练与评测流水线消费这些 trace 和回放数据。
4. `P22` 聚合实验矩阵、seed、产物和比较结果。
5. dashboard、Ops UI、attention queue、morning summary 在本地展示这些产物。

更详细说明：

- 架构总览：[docs/ARCHITECTURE.zh-CN.md](docs/ARCHITECTURE.zh-CN.md)
- 路线图：[docs/ROADMAP.zh-CN.md](docs/ROADMAP.zh-CN.md)
- Demo 说明：[DEMO_README.zh-CN.md](DEMO_README.zh-CN.md)
- 演示脚本：[docs/MVP_DEMO_SCRIPT.zh-CN.md](docs/MVP_DEMO_SCRIPT.zh-CN.md)

## 常用入口

| 目标 | 命令 | 主要输出 |
|---|---|---|
| 启动 Demo | `powershell -ExecutionPolicy Bypass -File scripts\run_mvp_demo.ps1 -OpenBrowser` | `http://127.0.0.1:8050/` |
| 体检环境 | `powershell -ExecutionPolicy Bypass -File scripts\doctor.ps1` | `docs/artifacts/p58/latest_doctor.json` |
| 运行 quick | `powershell -ExecutionPolicy Bypass -File scripts\run_p22.ps1 -Quick` | `summary_table.json`、dashboard |
| 运行 RunP22 门禁 | `powershell -ExecutionPolicy Bypass -File scripts\run_regressions.ps1 -RunP22` | regression artifacts |
| 启动 Ops UI | `powershell -ExecutionPolicy Bypass -File scripts\run_ops_ui.ps1` | `http://127.0.0.1:8765/` |
| 跑自治入口 | `powershell -ExecutionPolicy Bypass -File scripts\run_autonomy.ps1 -Quick` | `docs/artifacts/p60/latest_autonomy_entry.json` |

## 进一步阅读

- Demo 说明：[DEMO_README.zh-CN.md](DEMO_README.zh-CN.md)
- 使用指南：[USAGE_GUIDE.zh-CN.md](USAGE_GUIDE.zh-CN.md)
- 架构说明：[docs/ARCHITECTURE.zh-CN.md](docs/ARCHITECTURE.zh-CN.md)
- 路线图：[docs/ROADMAP.zh-CN.md](docs/ROADMAP.zh-CN.md)
- P22 编排说明：[docs/EXPERIMENTS_P22.md](docs/EXPERIMENTS_P22.md)
- Windows 交接：[docs/P58_WINDOWS_BOOTSTRAP.md](docs/P58_WINDOWS_BOOTSTRAP.md)

## 许可与贡献

- 许可：仓库顶层目前没有单独的 `LICENSE` 文件。
- 贡献：默认在 `main` 分支工作，保持变更可审计，并在修改运行默认值前执行相应门禁。
