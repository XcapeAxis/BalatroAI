# COVERAGE P33 Status

## Milestone

- milestone: `P33`
- title: `Action Replay Contract + Self-Supervised Plumbing`

## Scope Summary

### 1) Unified Action Replay Contract

- new module: `trainer/actions/replay.py`
- core API:
  - `normalize_high_level_action(...)`
  - `ActionReplayer(mode="sim"|"real").replay_single_action(...)`
- integration status:
  - `trainer/record_real_session.py`: integrated for execute path
  - `trainer/real_executor.py`: integrated for execute path
  - `trainer/dagger_collect.py`: integrated for sim augment replay path
  - `trainer/real_trace_to_fixture.py`: integrated action normalization on emitted trace actions

Current coverage:

- stable support for common hand/shop actions and P32 position actions (`REORDER_*`, `SWAP_*`)
- reserved hooks for future complex operations (`CARD_REORDER`, `APPLY_TAROT_SWAP`, etc.) are marked as unimplemented and explicitly reported

### 2) Self-Supervised Plumbing

- new package:
  - `trainer/self_supervised/datasets.py`
  - `trainer/self_supervised/models.py`
  - `trainer/self_supervised/train.py`
- experiment runner:
  - `trainer/experiments/selfsupervised_p33.py`
  - wrapper: `scripts/run_p33_selfsup.ps1`
- config:
  - `configs/experiments/p33_selfsup.yaml`
  - `configs/experiments/p33_selfsup.json` (no-PyYAML fallback)

Current task:

- predict next `score_delta` bucket (`low/mid/high`) from state/action features
- emit dataset stats and training summary artifacts

### 3) P22 Integration

- orchestrator supports `experiment_type: selfsup_p33`
- `configs/experiments/p22.yaml` and `p22.json` include `quick_selfsup_p33`
- quick wrapper now includes `quick_selfsup_p33` by default

## Gate Impact

- required gates remain:
  - `RunP13`
  - `RunP22`
- optional/experimental:
  - standalone `selfsupervised_p33.py` run
  - `quick_selfsup_p33` row in P22 matrix

No mandatory promotion gate depends on P33 selfsup metrics yet.

## Artifacts

- action replay smoke:
  - `docs/artifacts/p33/action_replayer_smoke_summary.json`
- selfsup dataset stats:
  - `docs/artifacts/p33/selfsup_dataset_stats.json`
- selfsup train summary:
  - `docs/artifacts/p33/selfsup_training_summary_<timestamp>.json`

## Known Limits / TODO

- Real runtime support for some complex positional/consumable gestures is still partial.
- P33 selfsup model is intentionally small; no convergence or policy-improvement guarantees.
- Additional heads (e.g., bust risk / tactical long-horizon targets) are planned but not gate-critical yet.

