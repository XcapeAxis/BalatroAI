# BalatroAI 本地 MVP Demo

> 语言切换： [简体中文](DEMO_README.zh-CN.md) | [English](DEMO_README.en.md)

## 这是什么

这是一个面向面试演示的本地 Web Demo。它把 Balatro 风格局面的可视化、AI 推荐、一步结果预览、时间线和训练过程放在同一个页面里，不依赖原版游戏窗口，也不要求联网。

当前 Demo 的主故事是：

- 我已经做出一个本地可交互的 AI 决策沙盘
- 用户可以直接看到局面、推荐动作、解释、执行结果
- 页面里既有启发式基线，也有真实训练出来的模型
- 训练过程本身也能在 UI 中实时展示

## 你能直接演示什么

- `3` 个内置高质量场景：高收益起手、高压弃牌转折、Joker 协同爆发
- 模型 vs 启发式 Top-K 推荐对比
- 当前资源、盲注、筹码进度、Joker、手牌可视化
- 选中推荐后的单步预览
- 手动执行与自动演示
- UI 内发起训练，并在训练面板看到状态、进度、loss 曲线和关键指标

## 快速启动

如果本地已经有可用 checkpoint：

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_mvp_demo.ps1 -OpenBrowser
```

默认地址：

```text
http://127.0.0.1:8050/
```

如果你希望先确保有一个可用模型，再启动 Demo：

```powershell
powershell -ExecutionPolicy Bypass -File scripts\bootstrap_mvp_demo.ps1 -OpenBrowser
```

说明：

- `scripts\run_mvp_demo.ps1` 会自动解析合适的 Python 环境并启动本地服务
- `scripts\bootstrap_mvp_demo.ps1` 默认会先跑一轮 `smoke` 训练；如需 2 小时正式训练，可传 `-TrainProfile standard`
- 页面内也可以直接点击“快速烟雾训练”或“开始 2 小时训练”

## 推荐演示顺序

1. `高收益起手`
   - 适合开场，局面直观，动作收益一眼能看懂
   - 推荐重点：模型/启发式都倾向同一手高收益出牌
2. `高压弃牌转折`
   - 适合说明 AI 不是只看眼前收益，还会考虑资源压力和后续机会
   - 推荐重点：为什么先弃牌而不是硬打
3. `Joker 协同爆发`
   - 适合强调“可解释 AI 推荐”
   - 推荐重点：Joker 如何改变预期收益

## 训练与模型说明

当前 Demo 使用的是轻量监督学习路线：

- 数据构建：`demo/build_mvp_dataset.py`
- 模型训练：`demo/train_mvp_model.py`
- 流水线编排：`demo/train_mvp_pipeline.py`
- 推理接入：`demo/model_inference.py`

训练产物统一落在：

```text
docs/artifacts/mvp/model_train/<run_id>/
```

重点文件：

- `dataset_stats.json`
- `metrics.json`
- `loss_curve.csv`
- `mvp_policy.pt`
- `training_summary.md`

最新默认模型以这里为准：

```text
docs/artifacts/mvp/model_train/latest_run.txt
```

训练状态和 UI 轮询源：

```text
docs/artifacts/mvp/training_status/latest.json
```

## 当前已知能力边界

适合：

- 本地 AI 决策可视化
- Balatro 风格局面讲解
- 场景驱动的产品 Demo
- 最小监督模型训练与接入

不适合：

- 直接控制商业游戏客户端
- 依赖联网的演示流程
- 声称已经得到完整最优代理
- 把这轮 MVP 当成长期主线基础设施成果展示

## 建议在面试前准备的素材

- 打开本地 Demo 页
- 准备好 `docs/MVP_DEMO_SCRIPT.zh-CN.md`
- 备好 fallback 截图：
  - `docs/artifacts/mvp/fallback/basic_play_demo.png`
  - `docs/artifacts/mvp/fallback/high_risk_discard_demo.png`
  - `docs/artifacts/mvp/fallback/joker_synergy_demo.png`
- 备好 smoke / API 证据：
  - `docs/artifacts/mvp/api_smoke_20260309_154858.json`
  - `docs/artifacts/mvp/training_status/latest.json`

## 一句话定位

这不是一个研究控制台，而是一个已经可以“打开浏览器直接讲故事”的本地 AI 决策产品原型。
