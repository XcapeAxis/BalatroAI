#!/usr/bin/env bash
set -euo pipefail

for p in \
  trainer/candidates_hand.py \
  trainer/candidates_shop.py \
  trainer/rollout_search_p15.py \
  trainer/train_pv.py \
  trainer/eval_pv.py \
  trainer/models/policy_value.py \
  scripts/run_p15_smoke.ps1 \
  docs/COVERAGE_P15_STATUS.md \
  docs/P16_SPEC.md \
  docs/COVERAGE_P16_STATUS.md \
  trainer/p16_loop.py
do
  if [[ -e "$p" ]]; then
    echo "$p: OK"
  else
    echo "$p: MISSING"
  fi
done
