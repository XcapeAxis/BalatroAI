# P27 SPEC: Status Publishing + Release Train

## RunP27.functional

Must satisfy:

- `RunP26` baseline passes.
- P27 additions do not regress existing P25/P26 docs/trend pipeline.

Artifacts:

- `gate_functional.json`
- `baseline_summary.json`
- `baseline_summary.md`

## RunP27.status_publishing

Must satisfy:

- `docs/artifacts/status/latest_status.json` exists
- `docs/artifacts/status/latest_badges.json` exists
- `docs/artifacts/status/latest_dashboard_data.json` exists
- `docs/artifacts/status/latest_status.md` exists
- README badges/status patch succeeds for dry-run and apply
- README marker blocks remain valid:
  - `<!-- BADGES:START --> ... <!-- BADGES:END -->`
  - `<!-- STATUS:START --> ... <!-- STATUS:END -->`

Artifacts:

- `gate_status_publishing.json`
- `status_publish_summary.json` (if emitted by status publish script)

## RunP27.docs_refresh

Must satisfy:

- `scripts/generate_readme_status.ps1` smoke succeeds
- `scripts/lint_readme_p25.ps1` still passes after README patching

Artifacts:

- `gate_docs_refresh.json`

## RunP27.release_ops

Must satisfy:

- Dashboard build smoke succeeds:
  - `docs/dashboard/index.html`
  - `docs/dashboard/data/latest.json`
  - `docs/dashboard/data/latest.js`
- Release train smoke succeeds and emits:
  - `rc_summary.md`
  - `rc_summary.json`
  - `benchmark_delta.csv`
  - `gate_snapshot.json`
  - `risk_snapshot.json`

Artifacts:

- `gate_release_ops.json`

## RunP27.workflow_ci

Must satisfy:

- `.github/workflows/ci-smoke.yml` exists
- `.github/workflows/nightly-orchestrator.yml` exists
- `scripts/lint_workflows.ps1` passes
- lint outputs are generated:
  - `workflow_lint_report.json`
  - `workflow_lint_report.md`

Artifacts:

- `gate_workflow_ci.json`

## RunP27 Overall

`report_p27_gate.json` is PASS only when all sections pass:

- `functional`
- `status_publishing`
- `docs_refresh`
- `release_ops`
- `workflow_ci`
