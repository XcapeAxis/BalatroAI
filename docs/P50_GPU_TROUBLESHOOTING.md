# P50 GPU Troubleshooting

Use this checklist when the runtime says "GPU-ready" but the actual learner path is not healthy.

## First Commands

Bootstrap / doctor:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\setup_windows.ps1 -Mode auto -SkipSmoke
powershell -ExecutionPolicy Bypass -File scripts\doctor.ps1
```

Interpreter selection:

```powershell
python -m trainer.runtime.python_resolver
powershell -ExecutionPolicy Bypass -File scripts\resolve_training_python.ps1 -Emit json
```

Environment diagnosis:

```powershell
python -m trainer.runtime.gpu_diagnose --profile single_gpu_mainline
nvidia-smi
```

Minimal end-to-end validation:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_p22.ps1 -RunP50
```

## What "Healthy" Looks Like

- resolver selects `.venv_trainer_cuda`
- `torch.cuda.is_available() == True`
- device name resolves to `NVIDIA GeForce RTX 3080 Ti`
- P42/P45 smoke artifacts show `learner_device = cuda:0`
- `runtime_profile.json` does not contain `cuda_requested_but_unavailable`
- `progress.unified.jsonl` records non-zero `gpu_mem_mb`

## Common Problems

### Resolver falls back to CPU

Symptoms:

- `python_resolver` selects `.venv_trainer`
- `selected_env_type = cpu`

What to check:

- `.venv_trainer_cuda\Scripts\python.exe` exists
- that interpreter imports torch successfully
- torch reports `cuda = True`

Likely cause:

- CUDA wheel missing
- broken environment
- accidental use of the CPU-only venv

Action:

- recreate or repair `.venv_trainer_cuda`
- keep `.venv_trainer` untouched as the fallback

### Torch imports, but `cuda=False`

Symptoms:

- torch version lacks `+cu*`
- `device_count = 0`

Likely cause:

- CPU-only torch wheel
- mismatched install source

Action:

- confirm the CUDA wheel version in `.venv_trainer_cuda`
- compare with `docs/artifacts/p50/cuda_env_probe_20260307-105836.json`

### Dashboard has progress, but learner is still on CPU

Symptoms:

- run is active
- dashboard updates
- artifacts show `learner_device = cpu`

Likely cause:

- runtime profile resolved to CPU
- selected interpreter was CPU-only
- an older bug caused nested runtime-profile resolution to stop at the wrong config layer

Action:

- inspect `runtime_profile.json`
- inspect `gpu_mainline_summary.json`
- inspect `progress.unified.jsonl`
- confirm the selected interpreter in `run_manifest.json`

### P45 crashes with pin-memory / CUDA tensor errors

Symptoms:

- error similar to `cannot pin 'torch.cuda.FloatTensor'`

Likely cause:

- tensors were created on GPU inside DataLoader collation

Current status:

- P50 fixed this by keeping collation on CPU and moving batches afterward

Action:

- if the error reappears, verify local code still matches `trainer/world_model/train.py`

### OOM on larger profiles

Symptoms:

- CUDA out-of-memory
- repeated learner restart or batch shrink

Action:

1. switch to `gpu_debug_small`
2. reduce `batch_size`
3. reduce `micro_batch_size`
4. increase `grad_accum_steps`
5. keep `oom_fallback_policy = reduce_batch` unless debugging

### AMP or BF16 instability

Symptoms:

- NaN loss
- exploding metrics
- unstable reward/value numbers

Action:

- disable BF16 first
- then disable AMP if instability remains
- prefer `gpu_debug_small` for triage
- check `warnings.log`, `metrics.json`, and `progress.unified.jsonl`

### GPU utilization stays lower than expected

Why this can be normal:

- rollout still runs on CPU
- simulator/env stepping can dominate wall time
- smoke configs are intentionally tiny

Action:

- inspect benchmark artifacts before assuming the GPU path is broken
- compare throughput across `single_gpu_mainline` and nightly profiles
- only expect higher sustained usage on larger learners and longer runs

### Readiness guard passes health, but runs still race

Symptoms:

- service starts responding
- immediate follow-up run still fails during cold start

Action:

- inspect `service_readiness_report.json`
- keep warmup grace and consecutive-success checks enabled
- do not bypass the readiness guard in nightly runs

Reference reports:

- `docs/artifacts/p49/readiness/p22-20260307-113848/service_readiness_report.json`
- `docs/artifacts/p49/readiness/p22-20260307-114104/service_readiness_report.json`
- `docs/artifacts/p50/readiness_validation_20260307-115740.md`

## Artifact Checklist

When debugging, collect these first:

- `docs/artifacts/p50/cuda_env_probe_20260307-105836.json`
- `docs/artifacts/p50/python_resolver_20260307-110546.json`
- `docs/artifacts/p50/gpu_diagnose_20260307-111838.json`
- `docs/artifacts/p50/p42_gpu_smoke/20260307-112100/`
- `docs/artifacts/p50/p45_gpu_smoke/20260307-111650/`
- `docs/artifacts/p50/benchmarks/20260307-113049/`
- `docs/artifacts/dashboard/latest/index.html`

## Escalation Guidance

Escalate the issue as an environment problem when:

- resolver consistently selects the CUDA env but torch cannot see the GPU
- `nvidia-smi` works but torch cannot enumerate devices
- the same smoke command works in CPU mode and fails before the first CUDA step

Treat it as a training/runtime tuning problem when:

- CUDA is visible
- the learner reaches `cuda:0`
- failure appears only after batch growth, AMP enablement, or longer nightly budgets

## P58 Portability Note

If a new Windows machine cannot continue the project, start with `doctor.ps1` rather than guessing which environment is broken. P58 writes the latest bootstrap and doctor state into `docs/artifacts/p58/`, and the main entrypoints now surface the resolved training env in P22 summaries and the Ops UI environment page.
