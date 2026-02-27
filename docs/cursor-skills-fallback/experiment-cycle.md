# SOP: Experiment cycle

## When to use
Designing or running an experiment: define fixed vs varying variables, metrics, and how to record results.

## Inputs
- Experiment goal (e.g. compare A vs B, measure metric M).
- Optional: fixed variables, candidate values, metric names, result path.

## Outputs
- Table or list: fixed variables, varying variables, metrics, acceptance (e.g. "record only" or threshold).
- Path or artifact where results are stored.
- Summary: facts (observed numbers/logs), inference (interpretation), recommendations (next steps).

## Steps (SOP)
1. **Define**: Write down fixed variables, varying variables, metrics, and acceptance (threshold or "record only").
2. **Run**: Execute the experiment (script, command, or manual steps). Save outputs to the agreed path.
3. **Summarize**: Report facts (numbers, logs) separately; then inference; then recommendations. Do not claim "improved" without numbers.

## Acceptance criteria
- At least one of: run completed, results file exists, or metric values reported.
- No "better" or "worse" without reported metrics or evidence.

## Common failures
- Mixing facts and inference: always label (Facts / Inference / Recommendations).
- Vague metrics: define measurable criteria before running.
- Missing result path: agree where results go before running.

## When not to use
- Single bug fix (use debug-regression).
- Feature implementation with acceptance tests (use implement-feature-with-gates).
