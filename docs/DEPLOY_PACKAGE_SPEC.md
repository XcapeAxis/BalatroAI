# Deploy Package Specification (v1)

A deploy package is a self-contained directory that bundles everything needed to load and run a trained model for inference or evaluation.

## Directory Layout

```
<package_id>/
  model/
    best.pt              # Model weights
    manifest.json        # Source path, size
  config/
    inference_config.json  # Policy type, risk thresholds, search budget
    p19_risk_controller.yaml  # (if risk_aware)
  metrics/
    eval_summary.json    # Collected eval results from prior runs
  metadata.json          # Package identity and compatibility
  checksums.json         # File integrity (sha256 + size)
  README.md              # Loading and verification instructions
  package_verify_report.json  # (after verification)
```

## metadata.json Fields

| Field | Type | Description |
|-------|------|-------------|
| schema | string | Always `deploy_package_v1` |
| package_id | string | Unique package identifier |
| model_id | string | Model identifier |
| source_strategy | string | One of: pv, hybrid, rl, risk_aware, deploy_student |
| git_commit | string | Git commit hash at export time |
| created_at | string | ISO 8601 timestamp |
| compatibility.sim_version | string | Simulator version (e.g. p20) |
| compatibility.schema_version | string | Schema version (e.g. v1) |
| seed_files | list[string] | Reference seed file names |

## checksums.json Format

```json
{
  "model/best.pt": {"size_bytes": 123456, "sha256": "abcdef..."},
  "config/inference_config.json": {"size_bytes": 512, "sha256": "..."}
}
```

## Verification

Run `trainer/package/verify_model_package.py --package-dir <path>` to check:
1. All required entries exist
2. metadata.json has required keys
3. All checksums match
4. Optionally: `--once` runs a single inference smoke test

## Export

```bash
python trainer/package/export_model_package.py \
  --model trainer_runs/best.pt \
  --strategy risk_aware \
  --risk-config trainer/config/p19_risk_controller.yaml \
  --out-dir docs/artifacts/p20/packages/champion_rc
```
