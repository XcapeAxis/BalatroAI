# P60 AGENTS Standardization + Autonomous Iteration Entry

## Goal

P60 standardizes the repository rule layer for AI coding agents and unattended local execution. The milestone keeps the existing P51/P53/P57/P58 control plane intact and makes it easier to enter the same flow from one place.

Concrete outcomes:

- a short root `AGENTS.md` plus scoped subdirectory `AGENTS.md` files
- a unified `scripts/run_autonomy.ps1` entry that decides whether to continue, resume, or stop for a human
- a consistent P60 artifact surface for autonomy decisions and AGENTS consistency checks

## AGENTS Hierarchy

The hierarchy is intentionally shallow:

- root rules: `AGENTS.md`
- trainer-local rules: `trainer/AGENTS.md`
- simulator-local rules: `sim/AGENTS.md`
- script-local rules: `scripts/AGENTS.md`
- docs-local rules: `docs/AGENTS.md`
- config-local rules: `configs/AGENTS.md`

Role split:

- root `AGENTS.md` defines repo-wide defaults, entrypoints, artifact expectations, and stop boundaries
- subdirectory `AGENTS.md` files define only the rules that are specific to that subtree

The root file is meant to be readable in a few minutes. Detailed policy remains in `docs/DECISION_POLICY.md`.

## Relationship To Decision Policy

The control boundary is layered on purpose:

1. `AGENTS.md` + subdirectory `AGENTS.md`
2. `docs/DECISION_POLICY.md`
3. `configs/runtime/decision_policy.yaml`

Interpretation:

- `AGENTS.md` tells an agent how to work in this repo
- `docs/DECISION_POLICY.md` explains the human decision boundary
- `decision_policy.yaml` is the machine-usable source used by campaign and autonomy code

`trainer/autonomy/agents_consistency_check.py` validates that these layers do not drift too far apart.

## Autonomous Iteration Entry

Primary entrypoint:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_autonomy.ps1 -DryRun
powershell -ExecutionPolicy Bypass -File scripts\run_autonomy.ps1 -Quick
powershell -ExecutionPolicy Bypass -File scripts\run_autonomy.ps1 -Overnight
powershell -ExecutionPolicy Bypass -File scripts\run_autonomy.ps1 -ResumeLatest
```

`run_autonomy.ps1` resolves the training interpreter, then calls `trainer.autonomy.run_autonomy`.

The autonomy entry does the following:

1. reads AGENTS and decision-policy layers
2. builds current ops state from registry, campaigns, dashboard, doctor/bootstrap status, attention queue, and morning summary
3. runs AGENTS consistency checks
4. decides whether to:
   - start a new smoke run
   - start or resume overnight work
   - resume the latest campaign
   - stop because a human gate is active
5. writes a structured autonomy decision artifact

Main output artifacts:

- `docs/artifacts/p60/latest_autonomy_entry.json`
- `docs/artifacts/p60/latest_autonomy_entry.md`
- `docs/artifacts/p60/latest_agents_consistency.json`
- `docs/artifacts/p60/latest_agents_consistency.md`

Ops surfaces read P60 first and fall back to P59 when older artifacts are the latest available copy.

## Continue / Block / Resume Semantics

The unified autonomy entry follows the existing P57 model:

- continue when there is no unresolved human gate
- resume when a campaign is resumable and no blocking attention item exists
- block when:
  - attention queue contains blocking items
  - a campaign is blocked by a human gate
  - AGENTS / decision-policy consistency fails

When blocked, the system still refreshes readable handoff outputs instead of leaving only logs.

## Attention Queue And Morning Summary

The attention queue remains the human handoff surface under `docs/artifacts/attention_required/`.

Attention items support richer context such as:

- `blocking_scope`
- `related_campaign`
- `related_checkpoint_ids`
- `decision_deadline_hint`
- `suggested_commands`
- `summary_for_human`

Morning summary highlights:

- tasks completed / failed / skipped
- latest checkpoints and promotion state
- open attention items
- top-priority next action

The latest summary is written to:

- `docs/artifacts/morning_summary/latest.md`
- `docs/artifacts/morning_summary/latest.json`

`run_autonomy` refreshes morning summary both after normal execution and after blocked decisions.

## Dashboard / Ops UI Integration

P60 keeps the same operator surfaces and exposes the same autonomy state more consistently:

- dashboard shows AGENTS/autonomy status, open attention count, and latest morning-summary excerpt
- Ops UI exposes `Autonomy Overview`, `Attention Queue`, `Morning Summary`, and AGENTS / decision-policy links
- P22 summary rows can carry autonomy references such as:
  - `agents_root_present`
  - `autonomy_entry_ref`
  - `decision_policy_path`
  - `attention_queue_path`
  - `morning_summary_path`

## Quick / Overnight Usage

Recommended commands:

```powershell
# inspect what autonomy would do
powershell -ExecutionPolicy Bypass -File scripts\run_autonomy.ps1 -DryRun

# run the short autonomy lane
powershell -ExecutionPolicy Bypass -File scripts\run_autonomy.ps1 -Quick

# run the unattended overnight lane
powershell -ExecutionPolicy Bypass -File scripts\run_autonomy.ps1 -Overnight

# resume the latest resumable campaign
powershell -ExecutionPolicy Bypass -File scripts\run_autonomy.ps1 -ResumeLatest
```

For long-running local commands, prefer:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\safe_run.ps1 -TimeoutSec 7200 -- powershell -ExecutionPolicy Bypass -File scripts\run_autonomy.ps1 -Quick
```

## Known Limitations

- `run_autonomy` is an entry coordinator, not a planner that invents new research tasks from scratch.
- Blocking attention items intentionally stop progress even when a lower-value fallback might exist.
- AGENTS consistency checks are structural. They do not prove that every document sentence matches runtime behavior.
- Human approval is still required for live promotion, destructive git actions, environment repair, and statistically weak route-changing decisions.
