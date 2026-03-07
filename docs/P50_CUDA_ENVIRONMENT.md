# P50 Real CUDA Environment Bring-up

P50 converts the P49 runtime surface into a verified local CUDA training path on this workstation.

Result:

- local CUDA training is working
- CPU fallback remains intact
- all training/nightly scripts now resolve the training interpreter through a single resolver instead of hardcoded `.venv_trainer`

## Decision

P50 chooses the safer split-environment strategy:

- keep `.venv_trainer` as the known-good CPU fallback
- add `.venv_trainer_cuda` for real GPU training
- route all training entrypoints through `trainer.runtime.python_resolver`

Why this was the right choice here:

- `.venv_trainer` was already stable and CPU-only
- upgrading that environment in place would have increased rollback risk
- a dedicated CUDA environment keeps failure isolation simple while the resolver hides most of the multi-env maintenance cost

## Local Audit Summary

Observed on this host:

- GPU: `NVIDIA GeForce RTX 3080 Ti`
- VRAM: `12288 MB`
- driver: `591.59`
- `nvidia-smi` reported CUDA runtime compatibility: `13.1`
- CPU fallback env: `.venv_trainer`
- CUDA env: `.venv_trainer_cuda`
- CUDA torch stack: `torch 2.10.0+cu128`, `torchvision 0.25.0+cu128`, `torchaudio 2.10.0+cu128`

Reference artifacts:

- `docs/artifacts/p50/p50_bootstrap_20260307-104314.md`
- `docs/artifacts/p50/cuda_env_probe_20260307-105836.json`
- `docs/artifacts/p50/python_resolver_20260307-110546.json`
- `docs/artifacts/p50/gpu_diagnose_20260307-111838.json`

## Resolver Flow

The training interpreter priority is now:

1. explicit python or env override
2. `.venv_trainer_cuda` when `torch.cuda.is_available() == true`
3. `.venv_trainer`
4. current/system Python as the last fallback

Entry points using this resolver:

- `scripts/run_p22.ps1`
- `scripts/run_regressions.ps1`
- `scripts/run_dashboard.ps1`
- `scripts/wait_for_service_ready.ps1`
- `trainer.closed_loop.candidate_train`

CLI:

```powershell
python -m trainer.runtime.python_resolver
powershell -ExecutionPolicy Bypass -File scripts\resolve_training_python.ps1 -Emit json
```

## Real GPU Validation

### P42 RL Candidate Smoke

Validated with:

- training python: `.venv_trainer_cuda\Scripts\python.exe`
- profile: `single_gpu_mainline`
- learner device: `cuda:0`
- rollout device: `cpu`

Reference artifacts:

- `docs/artifacts/p50/p42_gpu_smoke/20260307-112100/runtime_profile.json`
- `docs/artifacts/p50/p42_gpu_smoke/20260307-112100/metrics.json`
- `docs/artifacts/p50/p42_gpu_smoke/20260307-112100/progress.unified.jsonl`
- `docs/artifacts/p50/p42_gpu_smoke/20260307-112100/gpu_probe.json`
- `docs/artifacts/p50/p42_gpu_smoke/20260307-112100/summary.md`

Observed smoke-scale result:

- policy training completed end to end
- `invalid_action_rate = 0.0`
- non-zero GPU allocation was recorded

### P45 World Model Smoke

Validated with:

- training python: `.venv_trainer_cuda\Scripts\python.exe`
- profile: `single_gpu_mainline`
- learner device: `cuda:0`

Reference artifacts:

- `docs/artifacts/p50/p45_gpu_smoke/20260307-111650/runtime_profile.json`
- `docs/artifacts/p50/p45_gpu_smoke/20260307-111650/metrics.json`
- `docs/artifacts/p50/p45_gpu_smoke/20260307-111650/progress.unified.jsonl`
- `docs/artifacts/p50/p45_gpu_smoke/20260307-111650/gpu_probe.json`
- `docs/artifacts/p50/p45_gpu_smoke/20260307-111650/summary.md`

Observed smoke-scale result:

- world-model train and eval both completed on CUDA
- `gpu_mem_mb = 26.0` in eval artifacts
- the P50 work fixed the old pin-memory/device mismatch by keeping DataLoader collation on CPU and moving batches after load

### P46 Compatibility

P46 is primarily a generation/inference lane, not a heavy learner. P50 validated that it remains compatible with the GPU world-model path:

- `docs/artifacts/p50/p46_gpu_smoke/20260307-113700/runtime_profile.json`
- `docs/artifacts/p50/p46_gpu_smoke/20260307-113700/imagined_stats.json`
- `docs/artifacts/p50/p46_gpu_smoke/20260307-113700/progress.unified.jsonl`
- `docs/artifacts/p50/p46_gpu_smoke/20260307-113700/gpu_probe.json`

## Recommended Profiles for RTX 3080 Ti 12GB

These recommendations come from the real P50 benchmark sweep in:

- `docs/artifacts/p50/benchmarks/20260307-113049/benchmark_matrix.csv`
- `docs/artifacts/p50/benchmarks/20260307-113049/benchmark_summary.json`
- `docs/artifacts/p50/benchmarks/20260307-113049/recommended_profiles.json`

Recommended defaults:

| Profile | Use case | Batch | Micro-batch | Grad accum | Workers | AMP | BF16 | Max GPU MB |
|---|---|---:|---:|---:|---:|---|---|---:|
| `gpu_debug_small` | smoke / debugging / fast failure isolation | 32 | 16 | 2 | 1 | on | off | 6144 |
| `single_gpu_mainline` | default local training | 128 | 128 | 1 | 2 | on | on | 11264 |
| `single_gpu_nightly_balanced` | safer nightly | 192 | 96 | 2 | 3 | on | on | 11264 |
| `single_gpu_nightly_aggressive` | optional push profile | 256 | 128 | 2 | 4 | on | on | 11776 |

Interpretation note:

- smoke-scale models are tiny, so throughput and memory numbers are not representative of full future nightlies
- `gpu_debug_small` can appear faster simply because the smoke work is too small to saturate the GPU
- use `single_gpu_mainline` as the default until larger runs prove otherwise

## P22 / Nightly Usage

Real CUDA validation row:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_p22.ps1 -RunP50
```

Quick smoke including P50:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_p22.ps1 -Quick
```

P50-specific experiment configs:

- `configs/experiments/p50_gpu_validation_smoke.yaml`
- `configs/experiments/p50_gpu_validation_nightly.yaml`

Useful artifacts from a successful P50 run:

- `docs/artifacts/p22/runs/20260307-113905/p50_gpu_validation_smoke/run_manifest.json`
- `docs/artifacts/p22/runs/20260307-113905/p50_gpu_validation_smoke/gpu_mainline_runs/seed_001_AAAAAAA/gpu_mainline_summary.json`
- `docs/artifacts/p22/runs/20260307-113905/summary_table.json`
- `docs/artifacts/dashboard/latest/index.html`

## Boundaries

- P50 validates one local Windows + CUDA + PyTorch stack; it is not a portability guarantee.
- P44 distributed RL now has the correct resolver/runtime plumbing, but P50's direct real-CUDA smoke focused on P42 and P45 first.
- The CPU environment remains part of the design, not a deprecated leftover.
- Real simulator/oracle gates remain the quality authority; GPU enablement is about throughput and reliability, not automatic policy uplift.
