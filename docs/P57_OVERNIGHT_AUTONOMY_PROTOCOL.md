# P57 Overnight Autonomy Protocol

## Goal

P57 makes the local BalatroAI stack safe for unattended overnight progress. The system should keep moving through low-risk work, stop only at explicit human gates, and leave a compact decision backlog plus a morning brief instead of raw logs as the primary handoff surface.

## Core Files

- `AGENTS.md`
- `docs/DECISION_POLICY.md`
- `configs/runtime/decision_policy.yaml`
- `trainer/autonomy/attention_item_schema.py`
- `trainer/autonomy/attention_queue.py`
- `trainer/autonomy/morning_summary.py`
- `trainer/autonomy/overnight_protocol.py`
- `trainer/campaigns/nightly_campaign.py`
- `trainer/campaigns/resume_state.py`
- `scripts/run_p22.ps1`

## `AGENTS.md` Role

The root `AGENTS.md` is the project-level autonomy contract. It tells Codex and local operators to:

- stay on `main`
- start with baseline gates
- use small commits
- leave a clean working tree at handoff
- treat campaign state, summary, attention queue, and morning summary as required long-run outputs

It also draws the line between safe automatic actions and actions that must block for a human.

## Decision Policy

Human-in-the-loop boundaries are defined twice on purpose:

- human-readable policy: `docs/DECISION_POLICY.md`
- machine-readable policy: `configs/runtime/decision_policy.yaml`

The policy groups actions into three buckets:

1. `auto_allow_actions`
   - code/config/docs updates
   - resume-safe campaign stages
   - dashboard rebuilds
   - registry / promotion queue refresh
   - cleanup inside the existing artifact retention policy
2. `auto_suggest_actions`
   - promotion recommendations
   - deployment-mode suggestions
   - config-sidecar repair suggestions
3. `human_required_actions`
   - live promotion / champion switching
   - destructive git/history operations
   - environment / driver / dependency changes
   - weak-statistics route changes
   - unexplained severe regressions or provenance anomalies

`trainer/runtime/decision_policy_check.py` provides a small smoke/diagnostic entrypoint for classifying an action or stop condition.

## Attention Queue

When the overnight runner reaches a human gate, it writes a structured attention item instead of only failing the stage.

Primary artifacts:

- `docs/artifacts/attention_required/attention_queue.json`
- `docs/artifacts/attention_required/attention_queue.md`
- `docs/artifacts/attention_required/<timestamp>_<slug>.md`

Each item includes:

- `attention_id`
- `created_at`
- `severity`
- `category`
- `title`
- `summary`
- `blocking_stage`
- `attempted_actions`
- `recommended_options`
- `recommended_default`
- `required_human_input`
- `artifact_refs`
- `status`

P59 extends the same queue surface with additional handoff fields:

- `blocking_scope`
- `related_campaign`
- `related_checkpoint_ids`
- `decision_deadline_hint`
- `suggested_commands`
- `summary_for_human`

Current high-value categories are promotion, config provenance, environment, regression, and ambiguity.

## Campaign Stop / Resume Semantics

The P57 overnight runner uses stage-level autonomy decisions:

- `continue`
- `continue_with_warning`
- `stop_and_queue_attention`

These decisions are written into `campaign_state.json` on each stage:

- `autonomy_decision`
- `autonomy_reason`
- `attention_item_ref`
- `continue_allowed`
- `human_gate_triggered`

Resume rules:

- completed safe stages are skipped on resume
- blocked stages are not crossed automatically
- unresolved attention items remain control signals, not passive logs
- finalizer stages such as dashboard/morning-summary can still run after a block so the handoff remains readable

## Morning Summary

Every overnight run can end with a compact summary built from the latest artifacts:

- latest campaign states
- registry snapshot
- promotion queue
- attention queue
- dashboard summary
- latest P22 summary rows

P59 keeps the same summary surface and adds a unified autonomy entry that refreshes morning summary even when a run stops on a human gate.

## P59 Alignment

P59 does not replace the overnight protocol. It standardizes the rule layer and adds a single entrypoint:

- `scripts/run_autonomy.ps1 -Quick`
- `scripts/run_autonomy.ps1 -Overnight`
- `scripts/run_autonomy.ps1 -ResumeLatest`

This entrypoint reuses:

- the same decision policy
- the same attention queue
- the same morning summary artifacts
- the same campaign stop/resume semantics

Outputs:

- `docs/artifacts/morning_summary/latest.md`
- `docs/artifacts/morning_summary/latest.json`
- `docs/artifacts/morning_summary/<timestamp>.md`
- `docs/artifacts/morning_summary/<timestamp>.json`

The summary answers four practical questions:

1. what ran
2. what completed or blocked
3. what new checkpoints or promotion candidates appeared
4. what the first human action should be

## P22 / Dashboard / Ops UI Integration

P22 now has a first-class P57 lane:

- `p57_overnight_smoke`
- `p57_overnight_nightly`

`scripts/run_p22.ps1` adds:

- `-RunP57`
- `-Overnight`
- `-ResumeLatestCampaign`

P22 summary rows now surface:

- `autonomy_mode`
- `decision_policy_path`
- `attention_queue_path`
- `morning_summary_path`
- `human_gate_triggered`
- `campaign_state_path`

P58 extends the same flow with environment-aware blocking:

- the overnight stage template now starts with `environment_doctor`
- blocked environment health can generate an attention item instead of failing silently
- morning summary and Ops UI now surface the latest doctor state together with blocked campaigns

Dashboard panels expose:

- latest autonomy decision
- blocked campaign count
- open attention items
- morning-summary excerpt

Ops UI pages expose:

- `Attention Queue`
- `Morning Summary`
- `Blocked Campaigns`
- resolve-only handling for attention items without auto-applying high-risk actions

## Smoke / Nightly Validation Pattern

Recommended local commands:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_p22.ps1 -RunP57
powershell -ExecutionPolicy Bypass -File scripts\run_p22.ps1 -Overnight
powershell -ExecutionPolicy Bypass -File scripts\run_p22.ps1 -ResumeLatestCampaign
python -m trainer.runtime.decision_policy_check --action promote_candidate
python -m trainer.autonomy.attention_queue --smoke
python -m trainer.autonomy.morning_summary
```

Typical artifact checkpoints:

- `docs/artifacts/p22/runs/<run_id>/summary_table.json`
- `docs/artifacts/p22/runs/<run_id>/p57_overnight_smoke/campaign_runs/seed_*/campaign_state.json`
- `docs/artifacts/attention_required/attention_queue.json`
- `docs/artifacts/morning_summary/latest.md`
- `docs/artifacts/dashboard/latest/index.html`
- `docs/artifacts/p58/latest_doctor.json`
- `docs/artifacts/p58/bootstrap/latest_bootstrap_state.json`

## Known Limitations

- the policy is intentionally conservative and will block rather than guess on high-risk actions
- attention-item resolution is a human acknowledgment flow, not a hidden approval engine
- stage resume is durable, but inner-loop trainer progress still depends on each subsystem's own checkpoints
- morning-summary prioritization is heuristic and may need refinement as more overnight flows are added
- environment repair remains a human-approved action even though doctor output is automatic
