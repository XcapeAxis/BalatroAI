# Reproducibility Guide (P25)

This guide defines the minimum reproducibility contract for BalatroAI experiments and gates.

## 1) Environment Baseline

- Platform: Windows (PowerShell examples in this guide)
- Python: 3.12+ (3.13 recommended in this repo)
- Trainer dependencies:

```powershell
python -m pip install -r trainer/requirements.txt
```

- Optional dedicated environment:

```powershell
powershell -ExecutionPolicy Bypass -File trainer\scripts\setup_trainer_env.ps1
```

## 2) Seed Policy (P23+)

Source of truth:

- `configs/experiments/seeds_p23.yaml`

Seed sets:

- `contract_regression`: fixed set for stable regression checks
- `perf_gate_100`: fixed-size generated gate set
- `milestone_500` / `milestone_1000`: larger fixed generated sets for milestone validation
- `nightly_extra_random`: reproducible random extension for nightly runs

Important policy flag:

- `disallow_single_seed_default: true`

## 3) Config Entry Points

- Experiment matrix: `configs/experiments/p23.yaml`
- Seed governance: `configs/experiments/seeds_p23.yaml`
- Campaign manager configs:
  - `configs/experiments/campaigns/p24_quick.yaml`
  - `configs/experiments/campaigns/p24_nightly.yaml`
- Ranking config: `configs/experiments/ranking_p24.yaml`

## 4) Reproduction Commands

### 4.1 Quick reproducibility check

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_p23.ps1 -DryRun
```

### 4.2 Full gate reproducibility check

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_regressions.ps1 -RunP24
```

### 4.3 Docs productization gate check

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_regressions.ps1 -RunP25
```

## 5) Metric Semantics

Gate decisions (default policy):

- Functional/experiment gates: `avg_ante_reached` and `median_ante`
- Milestone-level checks: fixed-seed `win_rate`
- Reliability checks: seed-policy compliance, coverage, flake/variance checks

## 6) Artifact Paths

Main roots:

- `docs/artifacts/p23/`
- `docs/artifacts/p24/`
- `docs/artifacts/p25/`

Latest run aliases:

- `docs/artifacts/p24/runs/latest/`

Expected key files include:

- `summary_table.{json,md,csv}`
- `telemetry.jsonl`
- `campaign_summary.json` (campaign runs)
- `gate_*.json` and `report_*_gate.json`

## 7) Champion/Candidate Revalidation

Inspect these files after gated runs:

- `docs/artifacts/p24/champion.json`
- `docs/artifacts/p24/candidate.json`
- `docs/artifacts/p24/nightly_decision.json`

The decision is valid only under the exact seed policy, config, and gate version used in that run.

## 8) Common Sources of Non-Reproducibility

- Different seed set / different seed-policy version
- Different budget settings (time, retries, matrix size)
- Service instability or mismatched balatrobot/Balatro runtime
- Script/version drift between runs
