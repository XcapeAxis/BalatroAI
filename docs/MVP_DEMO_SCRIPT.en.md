# MVP Demo Script

> Language: [简体中文](MVP_DEMO_SCRIPT.zh-CN.md) | [English](MVP_DEMO_SCRIPT.en.md)

## First 15 Seconds

Suggested opening:

“This is a local AI decision sandbox for Balatro-style scenarios. I can pick a scenario on the left, inspect the current board in the center, compare model vs baseline recommendations on the right, and then show what changes after the action is executed. The model recommendation on this page comes from a real trained checkpoint, and the training process is visible in the same UI.”

## Recommended Demo Order

1. Strong opening hand
2. High-risk discard pivot
3. Joker synergy burst

Why this order works:

- start with the easiest scene to read
- then show why the assistant is not just chasing immediate chips
- finish with a more explainable Joker-driven recommendation

## Scenario 1: Strong Opening Hand

What to emphasize:

- this is the best opening scene
- the board is easy to parse
- recommendation and payoff are visually obvious
- it proves the page is using the local simulator rather than static cards

Suggested flow:

1. Load `高收益起手`.
2. Point to the board in the center, then to the `model vs baseline` panel on the right.
3. Note that both sides often agree on the top choice.
4. Click “执行当前动作”.
5. Show the board highlight, preview change, and timeline update together.

Suggested line:

“This scene works well as an opening because anyone can understand it quickly. The page is not just showing a recommendation card. It connects the current state, the recommendation, and the result of the next step.”

## Scenario 2: High-Risk Discard Pivot

What to emphasize:

- the assistant is not only optimizing for immediate chips
- resources are already tight, so mistakes are expensive
- the key tradeoff is discard now vs better follow-up later

Suggested flow:

1. Switch to `高压弃牌转折`.
2. Point out `手数 1 / 弃牌 1` and the high-risk label.
3. Show that the top card recommends a setup move rather than instant scoring.
4. Compare the second-best options of the model and the heuristic.
5. Either click autoplay or execute the action manually.

Suggested line:

“Here the assistant is doing resource scheduling. It is not just asking whether this hand scores now. It is asking whether spending the last discard creates a much better next hand.”

## Scenario 3: Joker Synergy Burst

What to emphasize:

- this is the strongest explainability scene
- the recommendation explains not only what to do, but why the payoff expands
- it is the best place to discuss Joker-driven value shifts

Suggested flow:

1. Switch to `Joker 协同爆发`.
2. Look at the `Joker / 机制` panel in the center.
3. Then compare the top recommendation and the preview area.
4. Explain why Joker synergy amplifies this line.
5. Execute once so the timeline completes the explanation chain.

Suggested line:

“This is not just ranking cards. It is explaining how the special mechanics in the state change the value of the action.”

## How To Present The Training Panel

The training panel is one of the strongest credibility points in the whole page.

Recommended order:

1. Start with the result summary and say the current model already powers the page.
2. Point to the `run_id` and sample count.
3. Point to `Val Loss / Top-1 / Top-3`.
4. Show the loss curve and scenario-alignment summary.
5. Mention that training can be triggered directly from the UI.
6. If the setting allows it, start a quick smoke run and show the status transitions.

Suggested line:

“Training is not hidden in a terminal. It has been turned into a product surface, so you can directly see the current stage, the curve, and the best checkpoint.”

## How To Describe The Model Correctly

Recommended framing:

- this is a genuinely trained minimum viable supervised model
- it is not the final strongest agent
- the current goal is to connect training, inference, explanation, and visualization in one demo product
- unsupported phases still fall back to heuristics by design

Avoid saying:

- “it is already close to a complete universal Balatro agent”
- “it can fully replace the original client”
- “the long-term roadmap is already complete”

## If You Want To Trigger Training Live

Recommended order:

1. Click “快速烟雾训练” first.
2. Show the state moving from queueing to dataset build to training or evaluation.
3. Point to the progress, curve, and latest status.
4. If there is time, mention that a full 2-hour profile is also available.

If you mention the 2-hour path:

- say that it launches a more realistic training budget in the background
- do not start from zero if the interview slot is very short

## Fallback Plan

Recommended priority:

1. Relaunch the demo.

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_mvp_demo.ps1 -OpenBrowser
```

2. Bootstrap a small model first if needed.

```powershell
powershell -ExecutionPolicy Bypass -File scripts\bootstrap_mvp_demo.ps1 -OpenBrowser
```

3. If the browser state is unstable, show fallback screenshots:

- `docs/artifacts/mvp/fallback/basic_play_demo.png`
- `docs/artifacts/mvp/fallback/high_risk_discard_demo.png`
- `docs/artifacts/mvp/fallback/joker_synergy_demo.png`

4. If you need non-interactive evidence, show:

- `docs/artifacts/mvp/training_status/latest.json`
- `docs/artifacts/mvp/model_train/latest_run.txt`
- `docs/artifacts/mvp/api_smoke_20260309_154858.json`
- `docs/artifacts/mvp/fallback_assets_20260309_144449.md`

## 30-Second Pre-Interview Checklist

- the demo page is already open
- the theme is set to the one you prefer to present
- the current scene is `高收益起手`
- the training panel is visible
- fallback screenshots are ready
- `latest_run.txt` and `training_status/latest.json` are ready as proof that the model and training outputs are real
