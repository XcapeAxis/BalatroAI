# P61 Validation Pyramid

P61 defines a four-tier validation model so the default edit loop stays fast, while full certification remains explicit and auditable.

## Tier 0 — Instant

Tier 0 is the fail-fast layer. It should finish in seconds or low minutes and catch obvious breakage before any smoke run starts.

- `py_compile` on changed Python files
- config sidecar consistency check
- AGENTS / decision-policy consistency check
- doctor / resolver precheck
- README lint when the README changes

## Tier 1 — Targeted Smoke

Tier 1 is selected from the actual change scope. It is narrow by design.

- `trainer/hybrid/**` → router smoke
- `trainer/world_model/**` → world-model smoke and imagination smoke
- `trainer/rl/**` → RL smoke
- `trainer/closed_loop/**` → closed-loop smoke
- `scripts/**`, `configs/**`, `trainer/experiments/**`, `trainer/runtime/**` → `run_p22.ps1 -DryRun`

## Tier 2 — Subsystem Gate

Tier 2 is used when a change crosses subsystem boundaries and needs a stronger integration signal.

- `run_p22.ps1 -Quick` for experiment/runtime/integration changes
- `run_regressions.ps1 -RunP22` as a sim-heavy subsystem gate when simulator semantics are touched

## Tier 3 — Certification

Tier 3 is the deferred certification layer. It is intentionally not required for every edit cycle.

- `run_regressions.ps1 -RunP22`
- overnight or nightly P22 lanes
- long-horizon or multi-seed validation already defined by P54/P56/P57

## Fast-Pass vs Certified-Pass

- `fast-pass` means Tier 0/1 and any required Tier 2 checks for the current change scope passed.
- `certified-pass` means the deferred Tier 3 command also passed.
- A run can therefore be healthy for continued engineering work while certification is still pending.

## When Tier 0/1 Is Enough

Use only Tier 0 or Tier 0/1 when the change is narrow and local:

- docs-only edits
- README changes
- metadata/report formatting changes
- local wrapper changes that still pass `run_p22.ps1 -DryRun`

## When To Escalate

Escalate to Tier 2 or Tier 3 when:

- the change affects experiment orchestration or runtime selection
- simulator or parity-sensitive logic changes
- world model / hybrid / RL / closed-loop behavior changes
- a prior fast loop passed, but the result still needs certification before release-level confidence

## Machine-Usable Source

The source-of-truth config for the planner is:

- `D:\MYFILES\BalatroAI\configs\runtime\gate_plan.yaml`

The planner and fast runner consume it via:

- `D:\MYFILES\BalatroAI\trainer\runtime\validation_planner.py`
- `D:\MYFILES\BalatroAI\trainer\runtime\run_fast_checks.py`
