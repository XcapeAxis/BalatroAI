# REPO_HYGIENE

Generated: 2026-02-19
Working directory: `D:\我的\小丑牌AI`
Branch: `chore/repo-hygiene-after-p3v2`

## 1. Findings (pre-clean)

### A) Runtime artifacts / caches
- `logs/**` (very large run log)
- `sim/tests/fixtures_runtime/**`
- `sim/runtime/**`
- Python caches: `**/__pycache__/**`
- Sweep raw folders (`sweep_*_raw`)
- Local virtual env `.venv_trainer/`
- Local source dump `Balatro_Mechanics_CSV_UPDATED_20260219/`

### B) Should-be-tracked source/docs
- Modified tracked trainer sources:
  - `trainer/env_client.py`
  - `trainer/rollout.py`
  - `trainer/eval.py`
  - `trainer/infer_assistant.py`
  - `trainer/README.md`
- Untracked sim/source/spec/test docs and fixtures:
  - `sim/core/*.py`, `sim/spec/*`, `sim/pybind/*`, `sim/tests/*`, `sim/oracle/*`
- Project usage guide:
  - `项目和命令使用指南.md`

### C) Local-only uncertain files
- `BALATRO_RULES_AND_MECHANICS.md`
- `RESEARCH_NOTES_WITH_CITATIONS.md`
- `scripts/_test.txt`

These were moved out of the repository surface and treated as local-only scratch.

## 2. Actions taken

1. Added ignore rules for local scratch paths:
- `local_scratch/`
- `local_only/`

2. Committed missing source/doc files that are part of reproducible code paths.

3. Ran cleanup commands:
- `git clean -fd`
- `git clean -fdX` (kept repository code, removed ignored/regenerated artifacts)

4. Ran full gate regression after cleanup:
- `powershell -ExecutionPolicy Bypass -File scripts\run_regressions.ps1 -RunP3`
- Result: P0/P1/P2/P2b/P3 all green.

5. Ran cleanup script again:
- `powershell -ExecutionPolicy Bypass -File scripts\cleanup.ps1`
- `docs/artifacts/**` preserved by design.

## 3. Size comparison

- Before: **0.86 GB**
- After: **0.36 GB**
- Saved: **0.50 GB**

Primary savings source:
- Removed local Python env and large ignored runtime directories.

## 4. Top large files (largest changes)

### Removed/downsized from pre-clean top list
- `.venv_trainer/Lib/site-packages/torch/lib/torch_cpu.dll` (~264.94 MB) removed
- `.venv_trainer/Lib/site-packages/torch/lib/torch_cpu.lib` (~29.60 MB) removed
- `.venv_trainer/Lib/site-packages/numpy.libs/libscipy_openblas...dll` (~20.40 MB) removed
- `.venv_trainer/Lib/site-packages/torch/lib/torch_python.dll` (~18.80 MB) removed
- Multiple other `.venv_trainer/**` binaries and libs removed
- `Balatro_Mechanics_CSV_UPDATED_20260219/research/master_items_with_sources.csv` removed

### Still large post-clean files
- `logs/2026-02-19T19-22-10/12346.log` (~388 MB, runtime-open log)
- `balatro_mechanics/derived/joker_template_map.json` (~75 KB)
- `benchmark_balatrobot.py` (~67 KB)

Note:
- A runtime-open log can survive cleanup while service is active. Stop serve process first, then rerun cleanup to remove it.

## 5. Trainer policy outcome

- Kept and committed trainer code/docs needed for reproducible training/eval/infer backend flow.
- Runtime products remain ignored and are not committed:
  - `trainer_data/**`
  - `trainer_runs/**`

## 6. Keep-out local area

- `local_scratch/` is ignored and reserved for non-reproducible personal notes and temporary files.
- Anything needed for reproducibility must be promoted into tracked code/docs instead of staying in `local_scratch/`.
