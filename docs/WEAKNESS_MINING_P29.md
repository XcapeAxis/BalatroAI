# P29 Weakness Mining v3

This document defines the weakness-priority loop for P29.

## Inputs

- `docs/artifacts/p24/**` campaign + triage + ranking summaries
- `docs/artifacts/p26/**` regression alert artifacts
- `docs/artifacts/trends/trend_rows.jsonl` trend warehouse rows

## Outputs

- `weakness_priority_report.json`
- `weakness_priority_report.md`
- `weakness_priority_table.csv`

## Priority Logic

Priority score is computed from:

1. Frequency of weak signal.
2. Penalty proxy (avg/median ante and win-rate loss proxies).
3. Fix-type weight (`model > data > search > heuristic > simulator coverage`).

The top buckets are consumed by targeted data generation (`p29_targeted_data.yaml`).
