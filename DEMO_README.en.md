# BalatroAI Local MVP Demo

> Language: [简体中文](DEMO_README.zh-CN.md) | [English](DEMO_README.en.md)

## What This Is

This is a local web demo designed for interview walkthroughs. It puts Balatro-style state visualization, AI recommendations, one-step previews, timeline updates, and training progress on a single page. It does not depend on the original game window and does not require network access.

The current demo story is straightforward:

- a local interactive AI decision sandbox already exists
- the user can see the current state, recommended action, explanation, and execution result
- the page compares a heuristic baseline with a genuinely trained model
- the training process itself is visible in the UI

## What You Can Demo Immediately

- `3` built-in scenarios: strong opening hand, high-risk discard pivot, and Joker synergy burst
- Top-K comparison between model and heuristic
- Visualized resources, blind target, chip progress, Jokers, and hand cards
- One-step preview after selecting a recommendation
- Manual execution and autoplay
- UI-triggered training with visible status, progress, loss curve, and key metrics

## Quick Start

If a usable checkpoint already exists locally:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_mvp_demo.ps1 -OpenBrowser
```

Default URL:

```text
http://127.0.0.1:8050/
```

If you want to make sure a usable model is available before launching the demo:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\bootstrap_mvp_demo.ps1 -OpenBrowser
```

Notes:

- `scripts\run_mvp_demo.ps1` resolves the appropriate Python environment and starts the local service
- `scripts\bootstrap_mvp_demo.ps1` runs a `smoke` training pass by default; use `-TrainProfile standard` for the 2-hour profile
- the page also allows direct clicks on “Quick Smoke Training” and “Start 2-Hour Training”

## Recommended Demo Order

1. `高收益起手`
   - best for opening because the board is easy to understand
   - key point: both model and heuristic often converge on the same strong scoring line
2. `高压弃牌转折`
   - best for explaining resource pressure and delayed payoff
   - key point: why discarding first is better than forcing a weak play
3. `Joker 协同爆发`
   - best for explaining interpretable recommendations
   - key point: how Joker synergy changes expected value

## Training And Model Notes

The demo currently follows a lightweight supervised-learning path:

- dataset build: `demo/build_mvp_dataset.py`
- model training: `demo/train_mvp_model.py`
- pipeline orchestration: `demo/train_mvp_pipeline.py`
- inference bridge: `demo/model_inference.py`

Training artifacts live under:

```text
docs/artifacts/mvp/model_train/<run_id>/
```

Important files:

- `dataset_stats.json`
- `metrics.json`
- `loss_curve.csv`
- `mvp_policy.pt`
- `training_summary.md`

The default model pointer is:

```text
docs/artifacts/mvp/model_train/latest_run.txt
```

The UI polls training status from:

```text
docs/artifacts/mvp/training_status/latest.json
```

## Current Capability Boundaries

Good fit for:

- local AI decision visualization
- Balatro-style scenario walkthroughs
- scenario-driven product demos
- minimum viable supervised model training and deployment

Not intended for:

- direct control of a commercial game client
- network-dependent demo flows
- claiming the project already has a full optimal agent
- presenting this MVP sprint as the full long-term infrastructure story

## Suggested Materials Before an Interview

- keep the demo page open
- keep `docs/MVP_DEMO_SCRIPT.en.md` ready
- prepare fallback screenshots:
  - `docs/artifacts/mvp/fallback/basic_play_demo.png`
  - `docs/artifacts/mvp/fallback/high_risk_discard_demo.png`
  - `docs/artifacts/mvp/fallback/joker_synergy_demo.png`
- keep smoke / API evidence handy:
  - `docs/artifacts/mvp/api_smoke_20260309_154858.json`
  - `docs/artifacts/mvp/training_status/latest.json`

## One-Sentence Positioning

This is not a research console. It is a local AI decision product prototype that is already presentation-ready in a browser.
