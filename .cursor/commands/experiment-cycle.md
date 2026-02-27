# experiment-cycle

## When to use
Run or design an experiment: fixed vs varying variables, metrics, and how to record results.

## Inputs
- Experiment goal (e.g. compare X vs Y, measure Z).
- (Optional) Fixed variables, candidate values for varying variables, and metric names.

## Steps
1. **Define**: Fixed variables, varying variables, metrics, acceptance (e.g. threshold or "record only").
2. **Run**: Execute the experiment (script, command, or steps). Record outputs in the agreed place.
3. **Summarize**: Report facts (numbers, logs) separately from inference and recommendations.

## Outputs
- Fixed / varying variables and metrics (short table or list).
- Where results are stored (path or artifact).
- Summary: facts (observed), inference (interpretation), recommendations (next steps).

## Verification
- At least one of: run completes, results file exists, or metric values are reported. No "improved" without numbers.

## Uncertainty handling
- If metric or criterion is unclear: define it before running. Document assumptions.

## Prohibited actions
- Do not claim "better" or "worse" without reported metrics or evidence.
- Do not mix facts and inference without labeling.
