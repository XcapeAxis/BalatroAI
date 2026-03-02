# P23 Seed Governance Policy

## Policy File
- `configs/experiments/seeds_p23.yaml`

## Seed Sets
- `contract_regression`: fixed small stable set for contract/regression checks (12 seeds).
- `perf_gate_100`: deterministic fixed set with 100 seeds for gate metrics.
- `milestone_500`: deterministic fixed set with 500 seeds for milestone scoring.
- `milestone_1000`: deterministic fixed set with 1000 seeds for larger milestone checks.
- `nightly_extra_random`: reproducible extra seeds layered on top of `perf_gate_100` for nightly stress/flake coverage.

## Governance Rules
- `disallow_single_seed_default: true`
- Any implicit single-seed default is rejected unless explicit override is provided.
- Every orchestrated experiment run must write:
  - `seed_policy.json`
  - `seeds_used.json`
  - seed hash and policy version in `run_manifest.json`

## Reproducibility
- All generated seed sets are deterministic from:
  - `seed_policy_version`
  - set name/salt
  - index/nonce
- Nightly extra seeds also include:
  - `git_commit`
  - `date_bucket`
  - `run_id`
- The seed list hash is recorded to detect accidental drift.

