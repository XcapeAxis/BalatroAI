# P55 — Config Loading Hardening

**Milestone**: P55
**Status**: COMPLETE
**Date**: 2026-03-07

---

## Summary

P55 addresses the config loading vulnerabilities discovered during P54:

- **Root cause**: The CUDA training env (`/.venv_trainer_cuda`) lacked PyYAML.
  When the orchestrator loaded `p22.yaml`, it silently fell back to a stale `p22.json`
  sidecar that did not contain the newly added P54 experiment definitions.
- **Symptom**: `run_p22.ps1 -DryRun -RunP54` returned "no experiments selected after filters".
- **Risk surface**: 7 YAML files had no JSON sidecar at all, and sidecar hashes were
  never verified — any YAML edit could cause silent stale reads with no diagnostic.

---

## Changes Made

### 1. Centralized Hardened Loader (`trainer/experiments/config_loader.py`)

New module implementing P55 load policy:

- **YAML is the sole source of truth**. JSON sidecars are generated cache artefacts.
- When PyYAML is available: always read YAML directly; returns `ConfigLoadResult`.
- When PyYAML is unavailable: fall back to JSON sidecar **only if** its embedded
  `__yaml_source_sha256__` matches the current YAML source hash.
  Stale or missing sidecars cause fast-fail `RuntimeError` with an explicit fix command.
- Returns `ConfigLoadResult` with full provenance:
  `config_source_path`, `config_source_type`, `config_hash`,
  `sidecar_path`, `sidecar_used`, `sidecar_in_sync`.

### 2. Sidecar Sync Tool (`trainer/experiments/config_sidecar_sync.py`)

New module + CLI with two modes:

- `--check`: scan `configs/experiments/**/*.yaml`, report drift/missing, exit 1 on any issue.
- `--sync`: regenerate all JSON sidecars from YAML, injecting `__yaml_source_sha256__`.
- Auto-downgrade: if `--sync` is requested but PyYAML is unavailable, runs `--check` instead
  (so CUDA env never silently fails the sync and always surfaces drift).
- Writes artefacts to `docs/artifacts/p55/config_sidecar_sync/<timestamp>/`.

```bash
# Check only (CI / pre-run gate):
python -m trainer.experiments.config_sidecar_sync --check

# Sync all sidecars (run from env with PyYAML):
python -m trainer.experiments.config_sidecar_sync --sync

# PowerShell wrapper:
powershell -ExecutionPolicy Bypass -File scripts/sync_config_sidecars.ps1
powershell -ExecutionPolicy Bypass -File scripts/sync_config_sidecars.ps1 -Check
```

### 3. Orchestrator Integration (`trainer/experiments/orchestrator.py`)

- `load_config()` now wraps the hardened loader; falls back to the old `_read_yaml_or_json` only if import fails.
- `load_config_with_provenance()` returns payload + `__config_provenance__` dict.
- `main()` runs sidecar sync/check **before** the experiment matrix is loaded.
  - With PyYAML: `--sync` mode (auto-refresh sidecars).
  - Without PyYAML: `--check` mode (drift → fast-fail with fix command).
- `run_plan.json` and `report_p23.json` both include `config_provenance` field.
- `summary_table.json` and `.csv` include per-run provenance columns.

### 4. Summary Tables (`trainer/experiments/report.py`)

New columns in all summary outputs:
`config_source_path`, `config_source_type`, `config_hash`,
`sidecar_used`, `sidecar_in_sync`, `config_sync_report_path`.

Markdown summary now includes a "Config Provenance (P55)" section.

### 5. Dashboard (`trainer/monitoring/dashboard_build.py`)

- `collect_dashboard_data()` now includes `config_provenance` and `config_sync_status` fields.
- HTML dashboard has a new "Config Provenance (P55)" panel showing source type, hash,
  sidecar sync status, and a link to the latest sync report.

### 6. Ops UI (`trainer/ops_ui/routes.py`)

- Overview page includes a "Config Provenance (P55)" section with the same fields.

### 7. PowerShell Scripts

- `scripts/sync_config_sidecars.ps1`: new wrapper script for the sidecar sync tool.
- `scripts/run_p22.ps1`: calls `sync_config_sidecars.ps1` before the orchestrator run.
- `scripts/run_regressions.ps1`: calls `sync_config_sidecars.ps1` before the P22 step.

### 8. PyYAML Dependency

- Added `pyyaml>=6.0` to `trainer/requirements.txt`.
- Installed in `.venv_trainer_cuda` (PyYAML 6.0.3).
- This eliminates the sidecar fallback path as the normal case for nightly runs.

---

## Sidecar Coverage After P55

Before P55: 7 YAML files had no JSON sidecar.
After P55 `--sync` run: all 65 YAML files have in-sync sidecars with `__yaml_source_sha256__`.

Missing files that now have sidecars:
- `configs/experiments/campaigns/p24_nightly.yaml`
- `configs/experiments/campaigns/p24_quick.yaml`
- `configs/experiments/p23.yaml`
- `configs/experiments/p29_targeted_data.yaml`
- `configs/experiments/ranking_p24.yaml`
- `configs/experiments/regression_alert_p26.yaml`
- `configs/experiments/seeds_p23.yaml`

---

## Validation Results

| Test | Result |
|---|---|
| `py_compile config_loader.py` | PASS |
| `py_compile config_sidecar_sync.py` | PASS |
| `py_compile orchestrator.py` | PASS |
| `py_compile report.py` | PASS |
| `py_compile dashboard_build.py` | PASS |
| `py_compile routes.py` (ops UI) | PASS |
| `config_sidecar_sync --sync` (65 files) | PASS: 65/65 in_sync |
| `config_sidecar_sync --check` | PASS: 65/65 clean |
| Drift injection + check | PASS: exit 1, explicit message |
| Drift injection + sync + re-check | PASS: restored in_sync |
| Config loader smoke (provenance) | PASS: yaml_direct / sidecar paths |
| P54 nightly dry-run (CUDA env) | PASS: json_sidecar + in_sync confirmed |
| P54 nightly full run | See `docs/artifacts/p55/p54_nightly_run_20260307.log` |
| PyYAML in CUDA env after install | PASS: PyYAML 6.0.3 |

---

## Policy Going Forward

1. YAML is the only source of truth for experiment configs.
2. JSON sidecars must be regenerated whenever a YAML is modified:
   `python -m trainer.experiments.config_sidecar_sync --sync`
3. The `__yaml_source_sha256__` key in every `.json` sidecar records the YAML hash at generation time.
4. Loading a YAML with a stale sidecar in a PyYAML-less env now causes an explicit `RuntimeError`
   (previously: silent wrong results).
5. All run summaries, dashboards, and ops UI surfaces record the actual config source used.

---

## Related Docs

- `docs/P54_LEARNED_ROUTER.md` — the P54 milestone that exposed this gap
- `docs/P49_GPU_MAINLINE_AND_DASHBOARD.md` — dashboard infrastructure
- `docs/P53_BACKGROUND_EXECUTION_AND_OPS_UI.md` — ops UI infrastructure
