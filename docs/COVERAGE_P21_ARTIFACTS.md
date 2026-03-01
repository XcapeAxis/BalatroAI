# P21 Artifacts Summary

P21 artifacts are written under:

- `docs/artifacts/p21/<timestamp>/`

Required files:

- `baseline_summary.json`
- `baseline_summary.md`
- `branch_cleanup_before.json`
- `branch_cleanup_after.json`
- `branch_cleanup_summary.md`
- `git_mainline_status.json`
- `git_sync_dryrun_report.json`
- `gate_functional.json`
- `gate_git_workflow.json`
- `report_p21.json`
- `report_p21.md`

Notes:

- Artifacts are persisted on disk for audit and rollback lookup.
- Branch cleanup reports include deleted branch head SHA values for traceability.
