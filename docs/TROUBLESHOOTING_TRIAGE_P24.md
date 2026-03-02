# P24 Triage / Bisect Troubleshooting

## Triage Categories
- `config_error`
- `runtime_crash`
- `service_instability`
- `gate_fail`
- `metric_regression`
- `flake_failure`
- `seed_specific_failure`
- `timeout_budget_cut`
- `unknown`

## Typical Follow-up
- `config_error`: validate campaign/stage config and rerun sanity stage.
- `runtime_crash`: inspect stack trace in `stage_results.json` and patch runtime path.
- `service_instability`: restart service, check health endpoint and transport logs.
- `gate_fail`: fix baseline regressions first.
- `metric_regression`: compare against champion baseline and inspect ranking report.
- `flake_failure`: run deterministic profile and seed bisect.
- `seed_specific_failure`: use `bisect_lite --mode seed_bisect`.
- `timeout_budget_cut`: increase budget or reduce matrix/seed size.

## Bisect-lite Modes
- `seed_bisect`: localize minimal seed subset likely triggering failure/regression.
- `config_bisect`: compare best/worst config and infer high-impact parameter.

