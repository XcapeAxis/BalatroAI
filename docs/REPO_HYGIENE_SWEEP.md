# Repo Hygiene Sweep

- Timestamp (UTC): `2026-02-19T23:55:04.334012+00:00`
- Branch: `chore/repo-hygiene-and-untracked-sweep`
- Root: `D:/??/???AI`

## Snapshot Before Sweep
- `git status --porcelain=v1`: clean
- `git status --porcelain=v1 --untracked-files=all`: clean (untracked runtime already ignored)
- Total size baseline: **1.5079 GB**

## Actions Performed
- Added branch-local hygiene audit and decision log in this document.
- Hardened cleanup script to preserve `docs/artifacts/**`, optionally keep fixtures runtime, and stop local Balatro/balatrobot processes before deletion to reduce file lock failures.
- Cleaned runtime directories (`logs`, `sim/tests/fixtures_runtime`, `sim/runtime`, `trainer_runs`) and attempted `trainer_data` cleanup (one locked file remains; ignored by git).
- Re-ran full regression gate `-RunP5` and confirmed green before final cleanup pass.

## Decisions (Track / Ignore / Delete)
- **Track**: source and docs changes under `sim/`, `scripts/`, `docs/` required for reproducibility and hygiene.
- **Ignore**: `.venv*`, `trainer_data/`, `trainer_runs/`, `logs/`, `sim/tests/fixtures_runtime/`, caches and temp files via existing `.gitignore` rules.
- **Delete**: runtime outputs and logs via `scripts/cleanup.ps1` (non-destructive to tracked source).
- **Keep evidence**: `docs/artifacts/p3`, `docs/artifacts/p4`, `docs/artifacts/p5` are preserved and not removed by cleanup.

## Snapshot After Sweep
- Total size after cleanup: **0.5161 GB**
- Approx saved: **0.9918 GB** (mostly log/runtime artifacts).
- Note: `trainer_data/sim_p5_rollout.jsonl` remained locked by another process during cleanup; path is ignored and does not affect git cleanliness.

## Top 20 Largest Files After Cleanup
| Rank | Bytes | Category | Path |
| --- | ---: | --- | --- |
| 1 | 264942080 | venv | `.venv_trainer/Lib/site-packages/torch/lib/torch_cpu.dll` |
| 2 | 29597848 | venv | `.venv_trainer/Lib/site-packages/torch/lib/torch_cpu.lib` |
| 3 | 20404736 | venv | `.venv_trainer/Lib/site-packages/numpy.libs/libscipy_openblas64_-74a408729250596b0973e69fdd954eea.dll` |
| 4 | 18800128 | venv | `.venv_trainer/Lib/site-packages/torch/lib/torch_python.dll` |
| 5 | 8812252 | venv | `.venv_trainer/Lib/site-packages/torch/lib/sleef.lib` |
| 6 | 7832064 | venv | `.venv_trainer/Lib/site-packages/PIL/_avif.cp314-win_amd64.pyd` |
| 7 | 7384679 | training-artifact | `trainer_data/sim_p5_rollout.jsonl` |
| 8 | 7130112 | venv | `.venv_trainer/Lib/site-packages/torchvision/python314.dll` |
| 9 | 3721728 | venv | `.venv_trainer/Lib/site-packages/numpy/_core/_multiarray_umath.cp314-win_amd64.pyd` |
| 10 | 2891768 | venv | `.venv_trainer/Lib/site-packages/torch/lib/microkernels-prod.lib` |
| 11 | 2800640 | venv | `.venv_trainer/Lib/site-packages/torch/bin/protoc.exe` |
| 12 | 2703490 | venv | `.venv_trainer/Lib/site-packages/torch/lib/XNNPACK.lib` |
| 13 | 2476032 | venv | `.venv_trainer/Lib/site-packages/PIL/_imaging.cp314-win_amd64.pyd` |
| 14 | 2268492 | venv | `.venv_trainer/Lib/site-packages/torch/include/ATen/RedispatchFunctions.h` |
| 15 | 2060288 | venv | `.venv_trainer/Lib/site-packages/PIL/_imagingft.cp314-win_amd64.pyd` |
| 16 | 1791519 | venv | `.venv_trainer/Lib/site-packages/torch/include/ATen/VmapGeneratedPlumbing.h` |
| 17 | 1614192 | venv | `.venv_trainer/Lib/site-packages/torch/lib/libiomp5md.dll` |
| 18 | 1399507 | venv | `.venv_trainer/Lib/site-packages/torch-2.10.0+cpu.dist-info/RECORD` |
| 19 | 1309238 | venv | `.venv_trainer/Lib/site-packages/torch/lib/fmt.lib` |
| 20 | 1243879 | venv | `.venv_trainer/Lib/site-packages/torch/testing/_internal/common_methods_invocations.py` |

## Artifact Preservation Check
- `docs/artifacts/p3/report_*.json`: present
- `docs/artifacts/p4/report_*.json`: present
- `docs/artifacts/p5/report_*.json`: present


## Final Gate Validation
- Command: `powershell -ExecutionPolicy Bypass -File scripts/run_regressions.ps1 -RunP5`
- Result: **PASS**
- Summary: P0 8/8, P1 6/6, P2 12/12, P2b 18/18, P3 58/58, P4 24/24, P5 42/42.

## Post-Cleanup Verification
- Command: `powershell -ExecutionPolicy Bypass -File scripts/cleanup.ps1`
- `docs/artifacts/p3`, `docs/artifacts/p4`, `docs/artifacts/p5` report files still present.
