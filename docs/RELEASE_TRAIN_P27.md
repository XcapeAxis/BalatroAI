# RELEASE TRAIN P27

P27 adds a local release candidate generator that packages trend deltas, gate snapshots, and risk context into an operator-facing RC summary.

## Wrappers

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_release_train.ps1 -DryRun:$false
```

Optional controls:

- `-SinceTag <tag>`
- `-SinceRun <run_id>`
- `-Candidate <model_or_strategy>`
- `-OutDir <path>`
- `-DryRun:$true`

## Python Entry

```powershell
.venv_trainer\Scripts\python.exe -m trainer.experiments.release_train --since-tag <tag> --since-run <run_id> --candidate <id> --out-dir docs\artifacts\p27\release_train\latest
```

## RC Package Outputs

- `rc_summary.md`
- `rc_summary.json`
- `benchmark_delta.csv`
- `gate_snapshot.json`
- `risk_snapshot.json`
- `release_notes_bridge.md`
- `release_notes_bridge.json` (if bridge generation succeeds)

## Recommendation Policy

RC summary includes one recommendation:

- `Promote candidate`
- `Hold`
- `Investigate`

The rationale must include:

- performance signal (benchmark deltas/improvements)
- stability signal (latest gate + gate changes)
- risk signal (hard/soft regression counts, candidate/release state)
- cost signal (runtime delta summary)
