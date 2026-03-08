# Decision Policy

P57 defines three decision classes so the overnight runner knows when to continue and when to stop for a human.

## Auto-Approve

These actions are safe to execute automatically inside the repo:

- Update config sidecars after edits.
- Resume a campaign from a safe completed-stage boundary.
- Run training, evaluation, arena, triage, dashboard build, morning summary build, and cleanup.
- Refresh registry snapshots and promotion queues.
- Switch window mode only within already validated repo-managed modes.
- Run `doctor` / environment inspection and Python resolver checks.

## Auto-Suggest-But-Do-Not-Apply

These actions can be proposed automatically, but the system must not make them effective on its own:

- Promote a candidate or checkpoint from `promotion_review` to a real default.
- Change a training profile that would alter nightly cost/performance assumptions.
- Archive checkpoints or recommend artifact pruning beyond normal cleanup policy.
- Recommend deployment mode changes such as `rule -> canary` or `guarded -> learned`.
- Recommend a bootstrap path such as `cpu_safe`, `cuda_mainline`, or a specific `setup_windows.ps1` command.

These cases should usually create an attention item but can still allow the run to continue to final summaries.

## Must-Block-For-Human

These actions must stop the related branch of work and create a blocking attention item:

- Install dependencies, drivers, or system packages.
- Create or recreate local Python environments, install Python packages, or repair CUDA/runtime dependencies when no healthy project env exists.
- Rewrite git history, delete branches, or force-push.
- Switch a real champion / promoted checkpoint.
- Continue after an unexplained major regression or irreparable config provenance anomaly.
- Proceed with a route-changing decision when statistics are insufficient.

## Example Classification Table

| Action | Policy Class |
|---|---|
| `update_config_sidecars` | auto-approve |
| `resume_campaign` | auto-approve |
| `cleanup_artifacts` | auto-approve |
| `switch_window_mode` | auto-approve |
| `run_environment_doctor` | auto-approve |
| `promote_candidate` | auto-suggest-but-do-not-apply |
| `change_training_profile` | auto-suggest-but-do-not-apply |
| `archive_checkpoints` | auto-suggest-but-do-not-apply |
| `bootstrap_windows_env` | auto-suggest-but-do-not-apply |
| `install_dependencies` | must-block-for-human |
| `create_python_env` | must-block-for-human |
| `delete_branches` | must-block-for-human |
| `promote_checkpoint_live` | must-block-for-human |

## Stop Conditions

Nightly/campaign execution should stop the affected branch and queue human attention when any of the following hold:

- unresolved human gate from a prior blocked stage
- config provenance mismatch that cannot be repaired automatically
- environment / driver / install requirement
- doctor blocked the machine because no healthy project environment exists
- readiness failure with no safe fallback
- unexplained major regression
- insufficient statistics for a route-changing promotion

## Continue-With-Warning Conditions

Execution may continue, but the warning must be recorded in campaign state and summaries:

- promotion review generated but no live switch requested
- transient readiness retry recovered
- dashboard rebuild retried successfully
- open attention items that are advisory rather than blocking
- doctor found warnings but the machine is still safe for continuation
