# P43 Training Strategy Refocus

## Summary

P43 formally switches the default training route to:

- mainline: self-supervised + RL candidate + closed-loop (P40/P41/P42)
- legacy baseline: BC/DAgger (retained, opt-in, non-default)

This is a policy and engineering-default change, not a hard deletion.

## Why This Decision

### Background

- high-quality expert trajectories are limited and costly to scale continuously
- real-session recording and annotation still have high operational friction
- replay-first self-supervised and RL candidate loops now have stronger artifact, gate, and triage support

### Why "Demote" Instead of "Delete"

- BC/DAgger still provide useful baseline controls and warm-start probes
- legacy code paths are needed for regression isolation and historical comparisons
- full deletion would remove low-cost sanity checks when mainline behavior regresses

## Mainline vs Legacy Mapping

| Lane | Training modes | Primary modules/configs | Script entrypoints | Default |
|---|---|---|---|---|
| Mainline | `rl_ppo_lite`, `selfsup_warm_bc` (stub placeholder), closed-loop replay/curriculum | `trainer/closed_loop/candidate_train.py`, `trainer/closed_loop/closed_loop_runner.py`, `trainer/rl/ppo_lite.py`, `configs/experiments/p40*`, `p41*`, `p42*` | `scripts/run_p22.ps1 -Quick`, `python -m trainer.closed_loop.closed_loop_runner --quick` | enabled |
| Legacy baseline | `bc_finetune`, `dagger_refresh` | `trainer/train_bc.py`, `trainer/dagger_collect.py`, `trainer/dagger_collect_v4.py`, P22 legacy rows | `scripts/run_p22.ps1 -Quick -IncludeLegacy` (or `-LegacyOnly`), `scripts/run_regressions.ps1 -RunLegacySmoke` | disabled by default |

## Default Behavior Changes

## P22

- Added category/default controls in matrix and orchestrator:
  - `category`: `mainline`, `legacy_baseline`, `required_validation`
  - `default_enabled` respected with backward-compatible inference when missing
- Default quick/nightly/gate selection excludes legacy baseline rows.
- Added opt-in legacy rows:
  - `legacy_bc_dagger_probe`
  - `legacy_bc_dagger_smoke`
- `run_plan.json` and `summary_table.*` now record experiment category fields.
- `scripts/run_p22.ps1` adds:
  - `-IncludeLegacy`
  - `-LegacyOnly`
  - category banner output in console.

## P40 / P41 / P42 Closed-loop Candidate Defaults

- Candidate configs now prioritize mainline mode order via `candidate_modes`:
  - `rl_ppo_lite`
  - `selfsup_warm_bc`
- BC fallback exists but is explicit and controlled:
  - `allow_legacy_fallback`
  - `legacy_fallback_modes`
- Default configs keep legacy fallback disabled.

### Manifest Unification (candidate + closed-loop)

Closed-loop and candidate artifacts now include:

- `training_mode`
- `training_mode_category`
- `fallback_used`
- `fallback_reason`
- `legacy_paths_used`

This makes triage and promotion review lane-aware by default.

## Manual Legacy Baseline Usage

Run legacy probe inside quick set:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_p22.ps1 -Quick -IncludeLegacy
```

Run only legacy rows:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_p22.ps1 -Quick -LegacyOnly
```

Run minimal legacy smoke in regression script:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_regressions.ps1 -RunLegacySmoke
```

Direct legacy tools (retained):

```powershell
python -B trainer/train_bc.py --help
python -B trainer/dagger_collect.py --help
python -B trainer/dagger_collect_v4.py --help
```

## Risks and Mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| Mainline candidate instability in specific environments | candidate quality or arena recommendation noise | explicit mode list ordering + per-run `training_mode*` manifest fields + quick triage artifacts |
| Missing RL prerequisites in local runs | candidate stage degrades to stub/fail | fallback metadata and explicit `fallback_reason`; optional legacy fallback when configured |
| Legacy path bit-rot | baseline comparisons become unreliable | keep lightweight legacy smoke (`--help`/probe rows) and non-default P22 legacy entries |
| Confusion over which lane produced candidate | review/debug friction | unified manifest fields in candidate + closed-loop summaries |
