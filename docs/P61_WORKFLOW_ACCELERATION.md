# P61 Workflow Acceleration

P61 changes the default engineering loop from “wait for the full gate every time” to “fast loop first, certification second”.

## What Changed

- validation is now split into Tier 0 / 1 / 2 / 3
- change scope decides the smallest required validation set
- fast checks produce an explicit `pending_certification` result instead of pretending a quick pass is fully certified
- slow certification is queued and tracked separately

## Core Components

- validation pyramid: `D:\MYFILES\BalatroAI\docs\P61_VALIDATION_PYRAMID.md`
- gate config: `D:\MYFILES\BalatroAI\configs\runtime\gate_plan.yaml`
- change-scope classifier: `D:\MYFILES\BalatroAI\trainer\runtime\change_scope.py`
- validation planner: `D:\MYFILES\BalatroAI\trainer\runtime\validation_planner.py`
- fast loop runner: `D:\MYFILES\BalatroAI\scripts\run_fast_checks.ps1`
- certification queue: `D:\MYFILES\BalatroAI\trainer\autonomy\certification_queue.py`
- certification runner: `D:\MYFILES\BalatroAI\scripts\run_certification.ps1`

## Default Fast Loop

Run this for normal development:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_fast_checks.ps1
```

This flow:

1. detects changed files
2. builds a validation plan
3. runs Tier 0
4. runs the required Tier 1 checks
5. runs Tier 2 only when the planner requires it
6. records whether certification is still pending

Artifacts land in:

- `D:\MYFILES\BalatroAI\docs\artifacts\p61\validation_plan_*.json`
- `D:\MYFILES\BalatroAI\docs\artifacts\p61\fast_checks\<timestamp>\fast_check_report.json`
- `D:\MYFILES\BalatroAI\docs\artifacts\certification_queue\certification_queue.json`

## Deferred Certification

When fast checks pass but full certification is still required, P61 writes a certification item instead of forcing the current edit loop to wait.

Run the latest pending certification with:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_certification.ps1 -LatestPending
```

Typical certification commands are:

- `powershell -ExecutionPolicy Bypass -File scripts\run_regressions.ps1 -RunP22`
- `powershell -ExecutionPolicy Bypass -File scripts\run_p22.ps1 -Overnight`

## Autonomy Integration

`run_autonomy.ps1 -Quick` now prefers the fast loop.

- if there is no blocking human gate, autonomy runs fast checks first
- if fast checks pass and certification is still needed, the queue records that state
- overnight autonomy can then consume the certification queue without inventing a parallel workflow

## P22 / Dashboard / Ops UI Semantics

P61 adds explicit status fields so UI and summaries can distinguish:

- `fast_check_status`
- `validation_tiers_completed`
- `certification_status`
- `pending_certification`
- `certification_queue_ref`
- `recommended_next_gate`

The important distinction is:

- `fast-pass` means the current edit scope is healthy enough to continue engineering work
- `certified-pass` means the deferred Tier 3 gate also completed

## Background-Safe Certification

P61 does not assume that every machine should run long certification tasks in parallel.

- if the environment is suitable, certification can run as an independent queued task
- if not, the queue remains in `pending` and the system reports deferred certification explicitly

This avoids false parallelism and keeps the mainline state auditable.

## Quick Commands

Fast loop:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_fast_checks.ps1
```

Autonomy quick:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_autonomy.ps1 -Quick
```

Certification queue dry-run:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_certification.ps1 -LatestPending -DryRun
```

Certification execution:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_certification.ps1 -LatestPending
```

## Known Limitations

- the first version uses rule-based change-scope mapping, not semantic diff understanding
- simulator changes still escalate aggressively because parity risk is high
- certification queue currently executes the first pending item; it does not yet prioritize across multiple branches
- fast-loop status is explicit, but not every historical P22 artifact will have retrofitted P61 fields
