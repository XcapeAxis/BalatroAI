# Configs Directory Rules

- YAML under `configs/` is the source of truth; generated sidecars are derived artifacts and must stay in sync.
- New experiment or runtime configs should fit the existing naming families and be consumable by P22/reporting without manual glue.
- Do not encode hidden defaults in code when they belong in config.
- Runtime validation tiers and trigger rules belong in config (for example `configs/runtime/gate_plan.yaml`) when the planner or autonomy stack needs to consume them.
- Changes that affect experiments, runtime policy, or device selection should preserve provenance and sidecar validation.
- If a config is intended for nightly or autonomy flows, ensure the corresponding scripts and reports can surface it.
