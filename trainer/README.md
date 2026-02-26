# Trainer Pipeline (Balatro + balatrobot HTTP / simulator)

This directory provides a modular trainer scaffold for hand-level behavior cloning:

`rollout -> dataset(jsonl) -> train_bc -> eval/infer`

## 1. Environment Setup

Recommended Python: **3.12/3.13 stable**.
Current 3.15 alpha can run, but may be unstable in long-running sessions.

Install dependencies:

```bash
python -m pip install -r trainer/requirements.txt
```

## E2E Smoke (One-Click)

1. Create stable trainer venv and install dependencies:

```powershell
.\trainer\scripts\setup_trainer_env.ps1
```

2. Run end-to-end smoke (`rollout -> train_bc(1 epoch) -> eval --offline -> infer_assistant --once`):

```powershell
.\trainer\scripts\smoke_e2e.ps1 --base-urls "http://127.0.0.1:12346"
```

## 2. Generate Dataset

### Real backend (balatrobot)

```bash
python trainer/rollout.py \
  --backend real \
  --base-urls http://127.0.0.1:12346 \
  --episodes 20 \
  --out trainer_data/dataset_real.jsonl
```

### Real backend + managed launch

```bash
python trainer/rollout.py \
  --backend real \
  --launch-instances \
  --launcher uvx \
  --uvx-path uvx \
  --base-urls http://127.0.0.1:12346,http://127.0.0.1:12347 \
  --love-path "D:\\SteamLibrary\\steamapps\\common\\Balatro\\Balatro.exe" \
  --lovely-path "D:\\SteamLibrary\\steamapps\\common\\Balatro\\version.dll" \
  --episodes 50 \
  --restart-on-fail \
  --out trainer_data/dataset_real.jsonl
```

### Sim backend

```bash
python trainer/rollout.py \
  --backend sim \
  --episodes 50 \
  --workers 4 \
  --out trainer_data/dataset_sim.jsonl
```

Notes:
- Dataset records all phases.
- `SELECTING_HAND` records include `legal_action_ids` and `expert_action_id`.
- By default `obs_raw` is not written. Add `--include-obs-raw` if needed.

## 3. Train BC Model

```bash
python trainer/train_bc.py \
  --train-jsonl trainer_data/dataset_real.jsonl \
  --epochs 8 \
  --batch-size 64 \
  --out-dir trainer_runs/bc_v1
```

Outputs:
- `best.pt`
- `last.pt`
- `train_metrics.json`
- `config.json`

## 4. Evaluate

### Offline

```bash
python trainer/eval.py \
  --offline \
  --model trainer_runs/bc_v1/best.pt \
  --dataset trainer_data/dataset_real.jsonl
```

### Online (real backend)

```bash
python trainer/eval.py \
  --online \
  --backend real \
  --model trainer_runs/bc_v1/best.pt \
  --base-url http://127.0.0.1:12346 \
  --episodes 10
```

### Online (sim backend)

```bash
python trainer/eval.py \
  --online \
  --backend sim \
  --model trainer_runs/bc_v1/best.pt \
  --episodes 10
```

## 5. Inference Assistant

### Suggest only (real backend)

```bash
python trainer/infer_assistant.py \
  --backend real \
  --base-url http://127.0.0.1:12346 \
  --model trainer_runs/bc_v1/best.pt
```

### Suggest + execute top-1 (sim backend)

```bash
python trainer/infer_assistant.py \
  --backend sim \
  --model trainer_runs/bc_v1/best.pt \
  --execute
```

## Regression gates (P18 / P19)

Layered regression is driven from the repo root:

- **P18** (RL pilot + ablation 100 + champion decision + DAgger v3 + canary):  
  `powershell -ExecutionPolicy Bypass -File scripts\run_regressions.ps1 -RunP18`
- **P19** (risk-aware controller + calibration + champion v3 rollback + ablation 100/1000 + DAgger v4 + canary):  
  `powershell -ExecutionPolicy Bypass -File scripts\run_regressions.ps1 -RunP19`
- **P19 quick gate** (skip 1000-seed milestone):  
  `powershell -ExecutionPolicy Bypass -File scripts\run_regressions.ps1 -RunP19 -SkipMilestone1000`

P19 requires P18 to have run at least once (for baseline and champion model). See `docs/P19_SPEC.md` for gate definitions, artifact layout, and `-RunPerfGateOnly`.

**Regression tests** (strategy routing, gate decision schema, canary synthetic label):  
`python -m unittest trainer.tests.test_ablation_and_gates -v`

**Avoiding hangs:** For long or flaky runs, wrap in **safe_run** so the process is killed after a timeout and stdout/stderr are written to `.safe_run/logs/`. From repo root:  
`powershell -ExecutionPolicy Bypass -File scripts\safe_run.ps1 -TimeoutSec 1200 -- powershell -ExecutionPolicy Bypass -File scripts\run_regressions.ps1 -RunP18`

## Common Errors

- `WinError 10061` / connection refused
  - Instance crashed or port not ready.
  - Use `--restart-on-fail` for rollout.
  - Check balatrobot logs in `logs/`.

- Health check timeout
  - Launch command mismatch or game startup too slow.
  - Verify `uvx balatrobot serve --help` works.
  - Increase timeout and use staggered startup.

- Frequent instance crashes
  - Reduce concurrent instances.
  - Check mod compatibility and lovely runtime logs.
  - Prefer Python 3.12/3.13 stable for training scripts.

- Slow action execution warning (`act_batch fallback`)
  - The client fell back to `select + play/discard` sequence.
  - Add server-side batch RPC in future for speed.

## Data Schema (RecordV1)

Each jsonl line contains:
- `timestamp, episode_id, step_id, instance_id, base_url`
- `phase, done`
- `hand_size, legal_action_ids, expert_action_id`
- `macro_action`
- `reward, reward_info`
- `features`
- `obs_raw` (optional)
