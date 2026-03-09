# BalatroAI MVP Demo

This repository now includes a local web demo focused on interview-friendly decision support.

## What You Can Show

- A browser-based Balatro-style decision sandbox
- Three built-in scenarios with clear AI recommendations
- A real trained first-pass hand-policy model served locally
- Manual step execution and autoplay inside the local simulator

## Quick Start

If a trained checkpoint already exists:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_mvp_demo.ps1 -OpenBrowser
```

If you want the repo to ensure a checkpoint exists first:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\bootstrap_mvp_demo.ps1 -OpenBrowser
```

Default URL:

```text
http://127.0.0.1:8050/
```

## Demo Scenarios

- `basic_play`: obvious high-value play and immediate blind clear
- `high_risk_discard`: resource-tight redraw decision
- `joker_synergy`: Joker-aware explanation and projected score spike

## Model Provenance

- Dataset builder: `demo/build_mvp_dataset.py`
- Trainer: `demo/train_mvp_model.py`
- Latest trained run: `docs/artifacts/mvp/model_train/latest_run.txt`
- Current model artifact family: `docs/artifacts/mvp/model_train/<run_id>/`

The first interview-ready run produced:

- dataset size: `3615` hand-phase samples
- checkpoint: `docs/artifacts/mvp/model_train/20260309_121811_fast/mvp_policy.pt`
- metrics: `docs/artifacts/mvp/model_train/20260309_121811_fast/metrics.json`

## Useful Commands

Train a fresh model into a new run directory:

```powershell
D:\MYFILES\BalatroAI\.venv_trainer_cuda\Scripts\python.exe demo\build_mvp_dataset.py --episodes 220 --max-steps 32 --scenario-copies 64 --run-dir docs\artifacts\mvp\model_train\NEW_RUN
D:\MYFILES\BalatroAI\.venv_trainer\Scripts\python.exe demo\train_mvp_model.py --dataset docs\artifacts\mvp\model_train\NEW_RUN\dataset.jsonl --run-dir docs\artifacts\mvp\model_train\NEW_RUN_fast --epochs 4 --batch-size 256 --device cpu
```

Smoke the API:

```powershell
D:\MYFILES\BalatroAI\.venv_trainer\Scripts\python.exe demo\api_smoke.py --base-url http://127.0.0.1:8050
```

## Scope

Good fit:

- AI decision visualization
- Balatro-style simulator demos
- Small supervised model training and replayable scenarios

Not the goal of this MVP:

- direct control of the commercial game client
- network-dependent demo flows
- full long-horizon optimal play
- new registry/router/world-model infrastructure

