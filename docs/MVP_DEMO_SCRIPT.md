# MVP Demo Script

## 15-Second Opening

"This is a local Balatro-style decision sandbox. I can load a scenario, show the current position, compare heuristic and model recommendations, preview the one-step resource delta, and execute the action in the simulator. The model behind the recommendation is a real trained checkpoint, not a hardcoded rule list."

## Suggested Scenario Order

1. `basic_play`
2. `high_risk_discard`
3. `joker_synergy`

## Scenario Talking Points

### 1. `basic_play`

- Show the hand and the Top-3 model recommendations.
- Point out that the model and teacher agree on the straight flush.
- Click `Execute Selected` and show the chips jump plus the phase move to `ROUND_EVAL`.
- Emphasize: "The UI is not just static analysis. It is stepping a real local simulator."

### 2. `high_risk_discard`

- Explain the resource pressure: only one hand and one discard remain.
- Show that the top recommendation is a full redraw, not a greedy low-value play.
- Use `Autoplay 3 Steps` to show discard first, then the improved follow-up play.
- Emphasize: "The assistant is reasoning about future opportunity cost, not only immediate chips."

### 3. `joker_synergy`

- Point out `Even Steven` in the Joker panel.
- Show the preview card: expected score is much larger than the raw base flush score.
- Explain that the recommendation panel includes the projected one-step result and Joker-aware expected score.
- Emphasize: "This is where the product story becomes explainable AI rather than just card ranking."

## How To Show The Model Training Result

- Open the `Model & Training` panel in the lower-right.
- Mention the latest run id: `20260309_121811_fast`.
- Mention the dataset size: `4254` samples, split into `3615` train and `639` validation.
- Mention the validation metrics briefly:
  - `val_acc1`: about `0.12`
  - `val_acc3`: about `0.16`
- Frame it correctly: "This is the first deployable supervised checkpoint for the demo, not the final strongest agent."

## If Something Goes Wrong Live

Fallback order:

1. Re-run `powershell -ExecutionPolicy Bypass -File scripts\run_mvp_demo.ps1 -OpenBrowser`
2. If the checkpoint is missing, run `powershell -ExecutionPolicy Bypass -File scripts\bootstrap_mvp_demo.ps1 -OpenBrowser`
3. If the browser is unreliable, open the fallback screenshots in `docs/artifacts/mvp/fallback/`
4. If you need a non-interactive proof artifact, show:
   - `docs/artifacts/mvp/api_smoke_20260309_123508.json`
   - `docs/artifacts/mvp/model_train/20260309_121811_fast/training_summary.md`

## Backup Assets To Keep Open

- `docs/artifacts/mvp/fallback/basic_play_demo.png`
- `docs/artifacts/mvp/fallback/high_risk_discard_demo.png`
- `docs/artifacts/mvp/fallback/joker_synergy_demo.png`
- `docs/artifacts/mvp/api_smoke_20260309_123508.json`
