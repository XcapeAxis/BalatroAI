<h1 align="center">BalatroAI</h1>
<p align="center">
  Simulator-first Balatro research platform for parity, reproducibility, and gated policy iteration.
</p>

<!-- BADGES:START -->
[![Latest Gate](https://img.shields.io/badge/Latest_Gate-RunP29_PASS-2EA44F)](scripts/run_regressions.ps1)
[![Workflow](https://img.shields.io/badge/Workflow-mainline--only-2EA44F)](scripts/git_sync.ps1)
[![Seed Governance](https://img.shields.io/badge/Seed_Governance-P23%2B_enabled-0E8A16)](configs/experiments/seeds_p23.yaml)
[![Experiment Orchestrator](https://img.shields.io/badge/Experiment_Orchestrator-P22%2B_enabled-1F6FEB)](scripts/run_p22.ps1)
[![Trend Warehouse](https://img.shields.io/badge/Trend_Warehouse-P26%2B_enabled-0E8A16)](docs/TREND_WAREHOUSE_P26.md)
[![Docs Coverage](https://img.shields.io/badge/Docs_Coverage-P15--P36-6E7781)](docs/)
[![Platform](https://img.shields.io/badge/Platform-Windows-0078D6)](USAGE_GUIDE.md)
[![Python](https://img.shields.io/badge/Python-3.12%2B-3776AB)](trainer/requirements.txt)
[![License](https://img.shields.io/badge/License-Not_Specified-6E7781)](#license-and-contributing)
[![Latest Tag](https://img.shields.io/github/v/tag/XcapeAxis/BalatroAI)](https://github.com/XcapeAxis/BalatroAI/tags)
[![CI Smoke](https://github.com/XcapeAxis/BalatroAI/actions/workflows/ci-smoke.yml/badge.svg)](https://github.com/XcapeAxis/BalatroAI/actions/workflows/ci-smoke.yml)
[![GitHub Stars](https://img.shields.io/github/stars/XcapeAxis/BalatroAI?style=social)](https://github.com/XcapeAxis/BalatroAI/stargazers)
[![GitHub Issues](https://img.shields.io/github/issues/XcapeAxis/BalatroAI)](https://github.com/XcapeAxis/BalatroAI/issues)
<!-- BADGES:END -->

BalatroAI is a high-parity simulator plus strategy experimentation stack for Balatro, backed by oracle traces, seed governance, and gated regressions. Current maturity covers Gold Stake alignment workflows and major mechanics (jokers including stateful behavior, consumables, shop/vouchers/tags, and artifactized experiment operations), and now includes P31/P33/P36 self-supervised entries plus a unified action replay contract. It is designed for mechanism research and Search/BC/DAgger/RL/self-supervised iteration, not as a cheat injector or memory-hook tool.

Badge/status refresh source:

- `docs/artifacts/status/latest_badges.json` and `latest_status.json` from `python -m trainer.experiments.status_publish`
- `powershell -ExecutionPolicy Bypass -File scripts\update_readme_badges.ps1 -DryRun|-Apply`

## What This Project Is

BalatroAI exists to make policy development and validation for Balatro engineering-grade instead of one-off scripting:

- real-game integration via `balatrobot` RPC
- simulator-driven high-throughput experiments
- oracle/simulator parity checks with canonical traces
- gated experiment operations (orchestrator, campaign manager, triage, ranking)
- reproducibility by seed governance and artifactized reports

## Scope and Boundaries

Suitable for:

- simulator parity and canonical trace alignment work
- offline-online policy iteration (Search -> BC -> DAgger -> Self-Supervised (P33/P36) -> RL)
- regression-gated experiment automation (P22+ / P23+ / P24+)
- engineering workflows around champion/candidate decisions

Not suitable for:

- a plug-and-play "always-win" agent
- uncontrolled real-game execution without safety rails
- interpreting metrics outside seed/budget/config/version context
- claiming universal performance without reproducible gate artifacts

## Learning Modes

Current learning modes are complementary rather than mutually exclusive:

- BC (`train_bc.py`): supervised imitation on curated datasets; fast baseline policy shaping.
- DAgger (`dagger_collect.py` + BC refresh): interactive correction with policy-in-the-loop traces.
- Self-Supervised P36 (`trainer/selfsup/*`): representation pretraining from trace artifacts without teacher labels.
- RL (existing pilot paths): downstream policy improvement after stable encoder/policy initialization.

Data dependency by stage:

- early bootstrap: sim traces + oracle fixtures
- alignment-informed iteration: real/P13/P32 replay-compatible traces
- larger-scale policy comparison: P22 orchestrator multi-seed matrix + gate reports

## Quick Start (Windows + PowerShell, < 5 min)

1. Clone and enter the repo.

```powershell
git clone https://github.com/XcapeAxis/BalatroAI.git
cd BalatroAI
```

2. Create and activate a virtual environment, then install dependencies.

```powershell
python -m venv .venv_trainer
.\.venv_trainer\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r trainer/requirements.txt
```

3. Start `balatrobot` (required for oracle/alignment gates and full regression suites).

```powershell
uvx balatrobot serve --headless --fast --port 12346
```

4. Run a baseline alignment regression (P0-P10 path).

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_regressions.ps1 -RunP10
```

5. Run the P22 quick orchestration matrix (includes `quick_selfsup_pretrain`, `quick_selfsup_p33`, and P36 rows `quick_selfsup_future_value` / `quick_selfsup_action_type`).

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_p22.ps1 -Quick
```

Optional verbose progress:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_p22.ps1 -Quick -VerboseLogs
```

Optional multi-seed comparison smoke (2 experiments x 3 seeds):

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_p22.ps1 -Only quick_baseline,quick_candidate -Seeds "AAAAAAA,BBBBBBB,CCCCCCC"
```

6. Inspect generated artifacts.

- `docs/artifacts/p22/runs/<run_id>/summary_table.md`
- `docs/artifacts/p22/runs/<run_id>/run_plan.json`
- `docs/artifacts/p22/runs/<run_id>/<exp_id>/{run_manifest.json,progress.jsonl,seeds_used.json}`
- optional live snapshot view: `powershell -ExecutionPolicy Bypass -File scripts\show_p22_live.ps1`

Expected console excerpt (trimmed):

```text
[RunFast] PASS (P0/P1 baseline completed)
[P22] Experiment 1/6: quick_baseline (seeds 2, mode=gate)
[P22] Experiment 5/6: quick_selfsup_future_value (seeds 2, mode=gate)
[P22] Experiment 6/6: quick_selfsup_action_type (seeds 2, mode=gate)
[P22] Completed 6/6: quick_selfsup_action_type status=passed | avg_ante=2.8545 median_ante=2.855 win_rate=49.09% hand_top1=54.55% hand_top3=69.55% shop_top1=80.26% illegal=3.95%
[P23] run_id=20260303-005315 mode=gate status=PASS
[P23] live_snapshot=.../docs/artifacts/p22/runs/20260303-005315/live_summary_snapshot.json
[P23] summary_json=.../docs/artifacts/p22/runs/20260303-005315/summary_table.json
```

Real-game step note: P0-P10/P13 oracle-alignment workflows require a legal local Balatro install plus `balatrobot` runtime. P22 synthetic modes can run simulator-only when gate stages are disabled.

More details:

- [trainer/README.md](trainer/README.md)
- [USAGE_GUIDE.md](USAGE_GUIDE.md)
- [docs/EXPERIMENTS_P22.md](docs/EXPERIMENTS_P22.md)
- [docs/SELF_SUPERVISED_OVERVIEW.md](docs/SELF_SUPERVISED_OVERVIEW.md)
- [docs/REPRODUCIBILITY_P25.md](docs/REPRODUCIBILITY_P25.md)

## Architecture Overview

```mermaid
flowchart LR
  A["Balatro Game real runtime"] -->|RPC| B["balatrobot server"]
  B --> C["Oracle trace and snapshots (P0-P10)"]
  C --> D["Sim engine and canonical hashing (core, P1-P10)"]
  D --> E["Fixtures and coverage reports (P0-P10, P3-P8)"]
  E --> F["Trainer and policies (Search, BC, DAgger, RL)"]
  F --> G["P22 orchestrator (matrix, multi-seed, resume)"]
  G --> H["Artifacts and status surfaces (docs/artifacts and dashboard)"]

  subgraph OBS["P22 Runtime Observability"]
    O1["telemetry.jsonl"]
    O2["live_summary_snapshot.json"]
    O3["summary_table.csv/json/md"]
    O4["progress.jsonl and seeds_used.json"]
  end

  G --> O1
  G --> O2
  G --> O3
  G --> O4
```

Data-flow details: [docs/ARCHITECTURE_P25.md](docs/ARCHITECTURE_P25.md)
For a detailed architecture and dataflow overview, see [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).
All BC/DAgger/Self-Supervised (P33/P36) experiment paths now normalize actions through a unified replay adapter so single-step behavior is consistent across sim and real-runtime adapters.

## Core Workflows

| Workflow | Entry | Output |
|---|---|---|
| Regression gates | `scripts/run_regressions.ps1` (`-RunP22/-RunP23/-RunP24/-RunP25`) | `docs/artifacts/p22|p23|p24|p25/*` |
| Orchestrator | `scripts/run_p22.ps1`, `scripts/run_p23.ps1` | run plans, telemetry, live snapshot, summary tables |
| Campaign ops | `scripts/run_p24.ps1` | campaign status/summary, triage, ranking |
| Training | `trainer/train_bc.py`, `trainer/train_rl.py`, `trainer/train_pv.py` | checkpoints + eval metrics |
| Self-supervised replay v1 (P36) | `python -B -m trainer.replay.storage`, `python -B -m trainer.experiments.selfsup_train --config configs/experiments/p22_selfsup_smoke.yaml` | replay-validated dataset + pretrain metrics + checkpoints |
| Self-supervised (P33 plumbing) | `scripts/run_p33_selfsup.ps1`, `python -B -m trainer.experiments.selfsupervised_p33` | dataset stats + selfsup summary + checkpoints |
| Self-supervised core (P36) | `python -B -m trainer.selfsup.build_selfsup_dataset`, `python -B -m trainer.selfsup.train_future_value`, `python -B -m trainer.selfsup.train_action_type` | unified dataset + task metrics + checkpoints |
| Inference | `trainer/infer_assistant.py` | suggestions / optional controlled execution |

## Experiments & Telemetry (P34)

P22 now emits a consistent telemetry event schema across run-level and per-experiment streams so you can trace progress without parsing ad-hoc logs.

- run-level stream: `docs/artifacts/p22/runs/<run_id>/telemetry.jsonl` (`schema: p34_telemetry_event_v1`)
- per-experiment stream: `docs/artifacts/p22/runs/<run_id>/<exp_id>/progress.jsonl` (`schema: p34_progress_event_v1`)
- live queue snapshot: `docs/artifacts/p22/runs/<run_id>/live_summary_snapshot.json`
- final roll-up: `docs/artifacts/p22/runs/<run_id>/summary_table.{csv,json,md}`

Each telemetry/progress event includes `run_id`, `exp_id`, `seed`, `phase`, `stage`, `status`, `step_or_epoch`, `metrics`, and elapsed/wall time fields.

## Reproducibility & Seeds

P22 experiments are config-first and artifactized:

- experiment matrix config: `configs/experiments/p22.yaml`
- seed governance config: `configs/experiments/seeds_p23.yaml`
- local default policy block: `configs/experiments/p22.yaml -> seed_policy`
- entrypoint: `powershell -ExecutionPolicy Bypass -File scripts\run_p22.ps1 ...`

Each P22 run writes:

- `docs/artifacts/p22/runs/<run_id>/run_plan.json`
- `docs/artifacts/p22/runs/<run_id>/summary_table.{csv,json,md}`
- `docs/artifacts/p22/runs/<run_id>/<exp_id>/run_manifest.json`
- `docs/artifacts/p22/runs/<run_id>/<exp_id>/progress.jsonl`
- `docs/artifacts/p22/runs/<run_id>/<exp_id>/seeds_used.json`

Seed policy clarification:

- default sets are split by intent (`regression_smoke`, `train_default`, `eval_default`) and nightly can add deterministic extras.
- `scripts/run_p22.ps1 -Quick` keeps runtime bounded via `--seed-limit 2`, but still uses multi-seed execution.
- actual seeds are persisted in both `run_plan.json -> experiments_with_seeds[]` and each experiment `seeds_used.json`.
- optional CLI override is available with `scripts/run_p22.ps1 -Seeds "AAAAAAA,BBBBBBB,CCCCCCC"` and is recorded in artifacts.
- summary tables now expose `seed_set_name`, `seeds_used`, and final metrics (`final_win_rate`, `final_loss` when applicable).

For details and repro patterns: [docs/EXPERIMENTS_P22.md](docs/EXPERIMENTS_P22.md), [docs/SEEDS_AND_REPRODUCIBILITY.md](docs/SEEDS_AND_REPRODUCIBILITY.md), [docs/REPRODUCIBILITY_P25.md](docs/REPRODUCIBILITY_P25.md)

P31 self-supervised backbone reproducibility:

- config: `configs/experiments/p31_selfsup.yaml`
- training entrypoint: `python -B trainer/selfsup_train.py --config configs/experiments/p31_selfsup.yaml --max-steps 100`
- orchestrated run: `powershell -ExecutionPolicy Bypass -File scripts\run_p22.ps1 -Quick` (includes `quick_selfsup_pretrain`)
- reference docs: [docs/EXPERIMENTS_P31.md](docs/EXPERIMENTS_P31.md), [docs/COVERAGE_P31_STATUS.md](docs/COVERAGE_P31_STATUS.md)

P33 self-supervised plumbing reproducibility (experimental entry):

- config: `configs/experiments/p33_selfsup.yaml`
- direct run: `python -B trainer/experiments/selfsupervised_p33.py --config configs/experiments/p33_selfsup.yaml`
- wrapper: `powershell -ExecutionPolicy Bypass -File scripts\run_p33_selfsup.ps1`
- artifacts:
  - `docs/artifacts/p33/selfsup_dataset_stats.json`
  - `docs/artifacts/p33/selfsup_training_summary_<timestamp>.json`
- details: [docs/EXPERIMENTS_P33.md](docs/EXPERIMENTS_P33.md), [docs/COVERAGE_P33_STATUS.md](docs/COVERAGE_P33_STATUS.md)

P32 representation-pretrain stub (P35 skeleton):

- config: `configs/experiments/p32_self_supervised.yaml`
- orchestrator wrapper: `powershell -ExecutionPolicy Bypass -File scripts\run_p32_self_supervised.ps1 -Quick`
- direct stub run: `python -B -m trainer.self_supervised.run_pretrain --config configs/experiments/p32_self_supervised.yaml`
- artifacts:
  - `docs/artifacts/p32_selfsup/runs/<run_id>/run_plan.json`
  - `docs/artifacts/p32_selfsup/runs/<run_id>/summary_table.{csv,json,md}`
  - `docs/artifacts/p32_selfsup/runs/<run_id>/<exp_id>/progress.jsonl`
- details: [docs/EXPERIMENTS_P32_SELF_SUPERVISED.md](docs/EXPERIMENTS_P32_SELF_SUPERVISED.md)

P36 self-supervised core (dataset + two tasks):

- configs:
  - `configs/experiments/p36_selfsup_future_value.yaml`
  - `configs/experiments/p36_selfsup_action_type.yaml`
- dataset build:
  - `python -B -m trainer.selfsup.build_selfsup_dataset --sources "oracle:sim/tests/fixtures_runtime/oracle_p0_v6_regression" "real:docs/artifacts/p32" --out-dir docs/artifacts/p36/selfsup_datasets/<run_id>`
- task runs:
  - `python -B -m trainer.selfsup.train_future_value --config configs/experiments/p36_selfsup_future_value.yaml`
  - `python -B -m trainer.selfsup.train_action_type --config configs/experiments/p36_selfsup_action_type.yaml`
- P22 matrix integration rows:
  - `quick_selfsup_future_value`
  - `quick_selfsup_action_type`
- details: [docs/P36_SELF_SUP_LEARNING.md](docs/P36_SELF_SUP_LEARNING.md)

<!-- STATUS:START -->
<!-- README_STATUS:BEGIN -->
### Repository Status (Auto-generated)

- branch: main
- latest_gate: RunP29 (PASS)
- recent_trend_signal: regression
- trend_warehouse_last_updated: 2026-03-02T17:16:27.519440+00:00
- trend_rows_count: 20115
- champion: quick_risk_aware (champion)
- candidate:  (decision: hold)
- docs_coverage: P15-P33
<!-- README_STATUS:END -->
<!-- STATUS:END -->

## Example Outputs

Assets directory: [docs/assets/readme/](docs/assets/readme/)
These samples are lightweight snapshots derived from recent local gate artifacts.

1. Gate log snippet: [sample_run_log.txt](docs/assets/readme/sample_run_log.txt)
2. Summary table snippet: [sample_summary_table.md](docs/assets/readme/sample_summary_table.md)
3. Data-flow visual source: [architecture_dataflow.mmd](docs/assets/readme/architecture_dataflow.mmd)
4. Dashboard snippet: [sample_dashboard_log.txt](docs/assets/readme/sample_dashboard_log.txt)

Example log excerpt:

```text
[RunP24] gate_status=PASS
artifact_dir=docs/artifacts/p24/20260302-150119
functional.pass=true campaign.pass=true reliability.pass=true ops.pass=true
```

Example summary table excerpt:

| exp_id | status | avg_ante | median_ante | win_rate | seeds |
|---|---:|---:|---:|---:|---:|
| quick_risk_aware | passed | 3.8352 | 3.5750 | 0.4041 | 8 |
| quick_hybrid | passed | 3.7396 | 3.4875 | 0.4876 | 8 |
| quick_baseline | passed | 3.5838 | 3.7125 | 0.4143 | 8 |

Example self-supervised metrics excerpt (P22 `quick_selfsup_pretrain`):

| metric | value |
|---|---:|
| selfsup_val_loss | 11.4079 |
| selfsup_score_delta_mae | 1.0000 |
| selfsup_hand_type_acc | 1.0000 |

## Action Fidelity and RealAction Contract (P32)

P32 introduces a unified single-step `RealAction` contract shared by simulator execution, real runtime translation, and fixture replay. The contract now includes position operations (`REORDER_HAND`, `SWAP_HAND_CARDS`, `REORDER_JOKERS`, `SWAP_JOKERS`) and an order-sensitive replay hash scope (`p32_real_action_position_observed_core`) so hand/joker ordering drift is detectable.

Current coverage notes:

- sim engine executes position actions deterministically.
- real executor/env client accepts the same action schema; on runtimes without reorder RPC methods, actions degrade with explicit `degraded_reason` instead of silent failure.
- `real_trace_to_fixture` can infer reorder/swap actions from raw before/after snapshots when explicit action logs are absent.
- shop/rng micro-alignment is augmented with artifactized reroll sampling reports.

See [docs/P32_REAL_ACTION_CONTRACT_STATUS.md](docs/P32_REAL_ACTION_CONTRACT_STATUS.md), [docs/P32_REAL_ACTION_CONTRACT_SPEC.md](docs/P32_REAL_ACTION_CONTRACT_SPEC.md), and [docs/P32_SHOP_RNG_ALIGNMENT.md](docs/P32_SHOP_RNG_ALIGNMENT.md).

## Project Structure

| Path | Purpose |
|---|---|
| `trainer/` | rollout/train/eval/infer pipelines |
| `sim/` | simulator, oracle, canonical schemas, parity fixtures |
| `scripts/` | gates, smokes, maintenance scripts |
| `configs/experiments/` | experiment matrix, seeds, campaign, ranking configs |
| `docs/` | specs, status docs, architecture/repro guides |
| `docs/artifacts/` | persisted run artifacts and gate outputs |

## Roadmap

Done:

- P0-P13 alignment and oracle/sim parity gates
- P22 orchestrator (modes, telemetry, summaries)
- P23 seed governance + coverage/flake harness
- P24 campaign manager + triage/bisect + ranking + dashboard
- P25 README/docs productization with RunP25 docs gate
- P31 self-supervised trajectory backbone (`DecisionStep/Trajectory`, encoder, quick pretrain)
- P33 unified action replay adapter (`trainer/actions/replay.py`) + minimal self-supervised experimental plumbing
- P36 self-supervised core (`trainer/selfsup/*`) with unified dataset builder and two task heads (`future_value`, `action_type`) integrated into P22

In progress:

- RL pilot stabilization and promotion criteria hardening
- stronger nightly scheduling policy and cost controls
- tighter docs and gate coupling for operator onboarding

Planned:

- deeper real canary safeguards and rollout channels
- broader mechanism coverage and richer failure taxonomy
- self-play/RL integration on top of the P31 self-supervised encoder

## Known Limitations

- Real runtime depends on local Balatro + lovely + balatrobot setup.
- Performance claims are seed/budget/version dependent.
- Some gate logic still uses local/manual artifacts, not centralized CI.
- Simulator/mechanic coverage is still expanding across milestones.
- Generated status/readme snippets are local-run artifacts and should be refreshed before release notes.
- P31 self-supervised backbone is alpha-grade: current heads focus on `score_delta` and `hand_type`; broader tactical targets will be added incrementally.
- P33 self-supervised entry is experimental plumbing: it validates data->train->summary flow, not production-strength policy gains.
- P32 representation pretrain line is currently a stub baseline for data plumbing and observability; it is not yet a full contrastive/world-model stack.
- P36 self-supervised core is representation pretraining only; policy gain still requires downstream BC/DAgger/search/RL integration and seed-robust evaluation.

## Further Reading

- [docs/SIM_ALIGNMENT_STATUS.md](docs/SIM_ALIGNMENT_STATUS.md)
- [docs/EXPERIMENTS_P22.md](docs/EXPERIMENTS_P22.md)
- [docs/SEEDS_AND_REPRODUCIBILITY.md](docs/SEEDS_AND_REPRODUCIBILITY.md)
- [docs/EXPERIMENTS_P31.md](docs/EXPERIMENTS_P31.md)
- [docs/EXPERIMENTS_P33.md](docs/EXPERIMENTS_P33.md)
- [docs/P36_SELF_SUP_LEARNING.md](docs/P36_SELF_SUP_LEARNING.md)
- [docs/EXPERIMENTS_P32_SELF_SUPERVISED.md](docs/EXPERIMENTS_P32_SELF_SUPERVISED.md)
- [docs/ROADMAP.md](docs/ROADMAP.md)
- [docs/P32_REAL_ACTION_CONTRACT_STATUS.md](docs/P32_REAL_ACTION_CONTRACT_STATUS.md)
- [docs/P32_REAL_ACTION_CONTRACT_SPEC.md](docs/P32_REAL_ACTION_CONTRACT_SPEC.md)
- [docs/P32_SHOP_RNG_ALIGNMENT.md](docs/P32_SHOP_RNG_ALIGNMENT.md)
- Coverage snapshots: `docs/COVERAGE_P*_STATUS.md`
- [docs/COVERAGE_P31_STATUS.md](docs/COVERAGE_P31_STATUS.md)
- [docs/COVERAGE_P33_STATUS.md](docs/COVERAGE_P33_STATUS.md)
- [docs/COVERAGE_P32_STATUS.md](docs/COVERAGE_P32_STATUS.md)
- [docs/COVERAGE_P35_STATUS.md](docs/COVERAGE_P35_STATUS.md)
- [docs/COVERAGE_P36_STATUS.md](docs/COVERAGE_P36_STATUS.md)
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)

## Documentation Index

- [trainer/README.md](trainer/README.md)
- [sim/README.md](sim/README.md)
- [USAGE_GUIDE.md](USAGE_GUIDE.md)
- [docs/P24_SPEC.md](docs/P24_SPEC.md)
- [docs/P25_SPEC.md](docs/P25_SPEC.md)
- [docs/P26_SPEC.md](docs/P26_SPEC.md)
- [docs/P27_SPEC.md](docs/P27_SPEC.md)
- [docs/P29_SPEC.md](docs/P29_SPEC.md)
- [docs/P30_SPEC.md](docs/P30_SPEC.md)
- [docs/COVERAGE_P30_STATUS.md](docs/COVERAGE_P30_STATUS.md)
- [docs/STATUS_PUBLISHING_P27.md](docs/STATUS_PUBLISHING_P27.md)
- [docs/RELEASE_TRAIN_P27.md](docs/RELEASE_TRAIN_P27.md)
- [docs/EXPERIMENTS_P22.md](docs/EXPERIMENTS_P22.md)
- [docs/SEEDS_AND_REPRODUCIBILITY.md](docs/SEEDS_AND_REPRODUCIBILITY.md)
- [docs/EXPERIMENTS_P31.md](docs/EXPERIMENTS_P31.md)
- [docs/EXPERIMENTS_P33.md](docs/EXPERIMENTS_P33.md)
- [docs/EXPERIMENTS_P32_SELF_SUPERVISED.md](docs/EXPERIMENTS_P32_SELF_SUPERVISED.md)
- [docs/ROADMAP.md](docs/ROADMAP.md)
- [docs/P32_REAL_ACTION_CONTRACT_STATUS.md](docs/P32_REAL_ACTION_CONTRACT_STATUS.md)
- [docs/P32_REAL_ACTION_CONTRACT_SPEC.md](docs/P32_REAL_ACTION_CONTRACT_SPEC.md)
- [docs/P32_SHOP_RNG_ALIGNMENT.md](docs/P32_SHOP_RNG_ALIGNMENT.md)
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)
- [docs/REPRODUCIBILITY_P25.md](docs/REPRODUCIBILITY_P25.md)
- [docs/ARCHITECTURE_P25.md](docs/ARCHITECTURE_P25.md)
- [docs/COVERAGE_P31_STATUS.md](docs/COVERAGE_P31_STATUS.md)
- [docs/COVERAGE_P33_STATUS.md](docs/COVERAGE_P33_STATUS.md)
- [docs/COVERAGE_P35_STATUS.md](docs/COVERAGE_P35_STATUS.md)

## License and Contributing

- License: currently not specified by a top-level `LICENSE` file.
- Contributions: use mainline-only workflow and run gates before proposing changes.


