# P29 Data Flywheel

P29 data flywheel pipeline:

1. Mine top weakness buckets from existing artifacts (`weakness_mining_v3`).
2. Generate weighted targeted dataset (`gen_targeted_dataset`).
3. Train candidates in batch (`train_batch_p29`).
4. Evaluate candidates at larger seed scales (`run_ablation`).
5. Rank and compare candidates (`ranking`, `candidate_compare_p29`).
6. Validate reliability via repeated flake evaluation (`experiments.flake --mode candidate`).

## Design Notes

- Failure-heavy buckets get higher sampling weight.
- Distribution safety is preserved through phase/stake/ante balancing.
- Source diversity includes heuristic/search/risk-aware/failure-replay/champion-prior.
- Reliability check can veto default promote when variance is high.
