---
name: experiment-cycle
description: Run or design an experiment with fixed/varying variables, metrics, and recorded results. Use for ablations, evals, or hyperparameter runs.
---

# Experiment cycle

## When to use
- User wants to run an experiment (ablation, eval, hyperparameter, comparison).
- Task involves "vary X, measure Y, record Z".

## Input/Output
- **Input**: What varies, what is fixed, metrics, where to record (path or format).
- **Output**: Command(s) run, where results were written, facts (numbers), inference (optional), recommendations.

## Steps (SOP)
1. **Define**: Fixed variables, varying variables, metrics, acceptance (e.g. run completes, metric threshold).
2. **Run**: Execute with project scripts if any (e.g. `run_ablation.py`). Use stated config.
3. **Record**: Write results to agreed path (file, table). Do not leave only in chat.
4. **Conclude**: Facts (observed values) | Inference ("likely because...") | Recommendations ("next: ...").

## Acceptance criteria
- Results written to a file or artifact. Facts separated from inference. No "improved" without evidence or baseline.

## Common failures
- **No record**: Always write results somewhere reproducible.
- **Mixing fact and inference**: Label clearly. Do not state "better" without numbers or baseline.
- **Under-spec**: If config is vague, state assumptions (e.g. seed, single run) and note replication needs.

## When not to use
- One-off script with no metrics (use plan-then-implement).
- Bug fix (use debug-regression).
- Feature implementation (use implement-feature-with-gates).
