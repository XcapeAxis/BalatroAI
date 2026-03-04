# P40 Closed-loop Improvement v1

P40 adds a runnable improvement loop that unifies replay mixing, arena failure mining, candidate training, and arena-gated promotion recommendation.

## Architecture

```mermaid
flowchart LR
  A["ReplayMixer\n(p10/p13/p36/p40-failure-pack)"] --> B["Mixed Replay Manifest\nreplay_mix_manifest.json"]
  C["Failure Mining\n(from P39 arena artifacts)"] --> D["Hard-failure pack\nfailure_pack_manifest.json"]
  D --> A
  B --> E["Candidate Trainer v1\n(bc_finetune + stub fallback)"]
  E --> F["Candidate checkpoint\nbest_checkpoint.txt"]
  F --> G["P39 Arena Runner\n(candidate vs champion)"]
  G --> H["P39 Champion Rules"]
  H --> I["P40 Promotion Decision\npromotion_decision.json/.md"]
  I --> J["P22 summary_table + reports"]
```

## Module I/O Contracts

| Module | Inputs | Outputs |
|---|---|---|
| `trainer.closed_loop.replay_mixer` | `configs/experiments/p40_replay_mix_*.yaml`; source roots | `docs/artifacts/p40/replay_mixer/<run_id>/replay_mix_manifest.json`, `replay_mix_stats.json`, `replay_mix_stats.md`, `seeds_used.json` |
| `trainer.closed_loop.failure_mining` | P39 `episode_records.jsonl`, `summary_table.json`, `bucket_metrics.json`, optional `candidate_decision.json` | `docs/artifacts/p40/failure_mining/<run_id>/failure_pack_manifest.json`, `failure_pack_stats.json`, `failure_pack_stats.md`, `hard_failure_replay.jsonl` |
| `trainer.closed_loop.candidate_train` | replay mix manifest + training config | `docs/artifacts/p40/candidate_train/<run_id>/candidate_train_manifest.json`, `metrics.json`, `progress.jsonl`, `seeds_used.json`, `best_checkpoint.txt` |
| `trainer.closed_loop.closed_loop_runner` | P40 closed-loop config | `docs/artifacts/p40/closed_loop_runs/<run_id>/run_manifest.json`, `promotion_decision.json/.md`, `summary_table.{json,csv,md}` |

### Key Manifest Fields

- replay mix: `sources[]`, `selected_entries[]`, `totals`, `seed_hash`
- failure pack: `candidate_policy`, `champion_policy`, `low_score_threshold`, `failures[]`, `replay_jsonl_path`
- candidate train: `mode`, `seed_results[]`, `candidate_checkpoint`
- promotion decision: `recommendation`, `recommend_promotion`, `arena_status`, `candidate_score`, `champion_score`, `score_delta`, `reasons[]`

## Failure Mining Logic (v1)

P40 marks episodes as hard failures when one or more conditions match:

- episode status not `ok` (`episode_failure_status`)
- invalid action / timeout / execution exception markers
- score in bottom quantile (configurable `bottom_quantile`)
- risk bucket indicates `resource_tight` with short survival or low score
- candidate-vs-champion regression segment when arena comparison indicates no uplift

If P39 artifacts are unavailable, failure mining returns `status=stub` and emits a valid empty manifest instead of crashing.

## Promotion Decision Logic

P40 reuses P39 `trainer.policy_arena.champion_rules` outputs, then applies a conservative overlay:

- insufficient seeds -> force `observe`
- very high candidate variance -> force `observe`
- no clear uplift (score delta <= 0 and win delta <= 0) -> force `observe`

P40 v1 only emits recommendations. It does **not** auto-replace champion metadata.

## Run Modes

Quick smoke:

```powershell
python -m trainer.closed_loop.closed_loop_runner --quick
```

P22 quick (includes `p40_closed_loop_smoke`):

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_p22.ps1 -Quick
```

Nightly-style closed loop:

```powershell
python -m trainer.closed_loop.closed_loop_runner --config configs/experiments/p40_closed_loop_nightly.yaml
```

## Known Gaps / Degrade Paths

- `p10_long_episode` source can be `stub` if local P10 runtime traces are missing.
- `bc_finetune` requires BC-compatible rows and PyTorch; otherwise candidate training degrades to `stub_checkpoint`.
- `model_policy` adapter in P39 currently keeps a stable fallback path; arena deltas should be interpreted as infrastructure smoke unless model inference path is fully wired.
- closed-loop runner continues with `arena_status=skipped` when arena execution is disabled or unavailable.

