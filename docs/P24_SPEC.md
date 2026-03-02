# P24 SPEC: Campaign + Triage/Bisect + Dashboard + Multi-objective Ranking

## Functional Gate (RunP24.functional)
- `RunP23` baseline must pass.
- P24 quick campaign must finish and persist required artifacts.

## Campaign Gate (RunP24.campaign)
- campaign yaml parsing pass.
- quick campaign executes at least 2 stages.
- each experiment emits:
  - `run_manifest.json`
  - `progress.jsonl`
  - `seeds_used.json`
- campaign summary/report generated.

## Reliability Gate (RunP24.reliability)
- seed policy usage required (implicit single-seed disallowed by policy).
- coverage report generated.
- flake report available when configured.
- gate fails if mandatory flake checks fail.

## Ops Gate (RunP24.ops)
- dashboard headless smoke pass.
- triage smoke pass with categorized outputs.
- bisect-lite smoke pass with report.
- ranking smoke pass with candidate recommendations.

## Core Outputs
- `campaign_plan.json`
- `campaign_status.json`
- `campaign_summary.json` / `campaign_summary.md`
- `telemetry.jsonl`
- `live_summary_snapshot.json`
- `triage_report.*`
- `bisect_report.*`
- `ranking_summary.*`
- `dashboard_headless_log.txt`
- gate reports:
  - `gate_functional.json`
  - `gate_campaign.json`
  - `gate_reliability.json`
  - `gate_ops.json`
  - `report_p24_gate.json`

