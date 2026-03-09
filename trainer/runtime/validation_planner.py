from __future__ import annotations

if __package__ is None or __package__ == "":
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from trainer.runtime.change_scope import build_change_scope, resolve_repo_root

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None


TIER_ORDER = ("tier0_instant", "tier1_targeted_smoke", "tier2_subsystem_gate", "tier3_certification")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def default_gate_plan_path(repo_root: Path | None = None) -> Path:
    root = resolve_repo_root(repo_root)
    return root / "configs" / "runtime" / "gate_plan.yaml"


def load_gate_plan(path: str | Path | None = None) -> dict[str, Any]:
    gate_path = Path(path).resolve() if path else default_gate_plan_path()
    text = gate_path.read_text(encoding="utf-8")
    if gate_path.suffix.lower() in {".yaml", ".yml"}:
        if yaml is None:
            raise RuntimeError("PyYAML is required to load gate_plan.yaml")
        payload = yaml.safe_load(text)
    else:
        payload = json.loads(text)
    if not isinstance(payload, dict):
        raise ValueError("gate plan must be a mapping")
    payload["source_path"] = str(gate_path)
    return payload


def _add_check(plan: dict[str, Any], gate_plan: dict[str, Any], check_id: str, *, reason: str) -> None:
    checks = gate_plan.get("checks") if isinstance(gate_plan.get("checks"), dict) else {}
    if check_id not in checks:
        return
    existing = plan["selected_checks"].setdefault(check_id, {"reasons": []})
    existing.update(dict(checks.get(check_id) or {}))
    existing["check_id"] = check_id
    if reason not in existing["reasons"]:
        existing["reasons"].append(reason)


def _defer_check(plan: dict[str, Any], gate_plan: dict[str, Any], check_id: str, *, reason: str) -> None:
    checks = gate_plan.get("checks") if isinstance(gate_plan.get("checks"), dict) else {}
    if check_id not in checks:
        return
    existing = plan["deferred_certification"].setdefault(check_id, {"reasons": []})
    existing.update(dict(checks.get(check_id) or {}))
    existing["check_id"] = check_id
    if reason not in existing["reasons"]:
        existing["reasons"].append(reason)


def build_validation_plan(
    *,
    repo_root: str | Path | None = None,
    changed_files: list[str] | None = None,
    gate_plan_path: str | Path | None = None,
) -> dict[str, Any]:
    root = resolve_repo_root(repo_root)
    scope = build_change_scope(repo_root=root, changed_files=changed_files)
    gate_plan = load_gate_plan(gate_plan_path)
    categories = set(scope.get("top_level_categories") or [])
    plan: dict[str, Any] = {
        "schema": "p61_validation_plan_v1",
        "generated_at": _now_iso(),
        "repo_root": str(root),
        "gate_plan_path": str(gate_plan.get("source_path") or ""),
        "change_scope": scope,
        "selected_checks": {},
        "deferred_certification": {},
        "summary_reasons": [],
    }

    if scope.get("changed_python_files"):
        _add_check(plan, gate_plan, "py_compile_changed_python", reason="python source changed")
    _add_check(plan, gate_plan, "agents_consistency", reason="repo rules and entrypoints must stay aligned")
    _add_check(plan, gate_plan, "decision_policy_smoke", reason="autonomy decision boundary must remain machine-readable")
    if "readme" in categories:
        _add_check(plan, gate_plan, "readme_lint", reason="README changed")
    if categories & {"configs", "scripts", "trainer_any", "sim_core"}:
        _add_check(plan, gate_plan, "config_sidecar_consistency", reason="runtime/config sources changed")
    if scope.get("requires_runtime_validation"):
        _add_check(plan, gate_plan, "doctor_precheck", reason="non-doc change requires environment precheck")

    if scope.get("docs_only"):
        plan["summary_reasons"].append("docs-only change: stay at Tier 0 unless README lint is required")
    else:
        if categories & {"scripts", "configs", "trainer_experiments", "trainer_runtime"} or scope.get("contains_p22_entrypoint_changes"):
            _add_check(plan, gate_plan, "p22_dry_run", reason="entrypoint/config change needs matrix planning smoke")
        if "trainer_hybrid" in categories:
            _add_check(plan, gate_plan, "router_smoke", reason="hybrid/router code changed")
            _add_check(plan, gate_plan, "p22_quick_gate", reason="hybrid changes affect P22 integration")
        if "trainer_world_model" in categories:
            _add_check(plan, gate_plan, "world_model_smoke", reason="world model code changed")
            _add_check(plan, gate_plan, "imagination_smoke", reason="world model changes affect imagination path")
            _add_check(plan, gate_plan, "p22_quick_gate", reason="world model changes affect mainline quick lane")
        if "trainer_rl" in categories:
            _add_check(plan, gate_plan, "rl_smoke", reason="RL code changed")
            _add_check(plan, gate_plan, "p22_quick_gate", reason="RL changes affect quick lane integration")
        if "trainer_closed_loop" in categories:
            _add_check(plan, gate_plan, "closed_loop_smoke", reason="closed-loop code changed")
            _add_check(plan, gate_plan, "p22_quick_gate", reason="closed-loop changes affect quick lane integration")
        if categories & {"trainer_experiments", "trainer_runtime"} or scope.get("contains_p22_entrypoint_changes"):
            _add_check(plan, gate_plan, "p22_quick_gate", reason="experiment/runtime changes require subsystem gate")
        if "sim_core" in categories:
            _add_check(plan, gate_plan, "sim_subsystem_gate", reason="sim/core changes are high risk for parity")

        if categories:
            _defer_check(plan, gate_plan, "run_regressions_p22", reason="full certification is deferred from the fast loop")
        if categories & {"trainer_hybrid", "trainer_world_model", "trainer_rl", "trainer_closed_loop", "trainer_experiments"}:
            _defer_check(plan, gate_plan, "p22_overnight_certification", reason="model/mainline changes should still hit nightly certification")

    selected_checks = list(plan["selected_checks"].values())
    selected_checks.sort(key=lambda item: (TIER_ORDER.index(str(item.get("tier") or "tier3_certification")), str(item.get("check_id") or "")))
    deferred_checks = list(plan["deferred_certification"].values())
    deferred_checks.sort(key=lambda item: (TIER_ORDER.index(str(item.get("tier") or "tier3_certification")), str(item.get("check_id") or "")))
    plan["selected_checks"] = selected_checks
    plan["deferred_certification"] = deferred_checks
    plan["validation_tiers_required"] = [
        tier
        for tier in TIER_ORDER
        if any(str(item.get("tier") or "") == tier for item in selected_checks)
    ]
    plan["pending_certification"] = bool(deferred_checks)
    plan["recommended_execution_mode"] = "docs_only" if scope.get("docs_only") else "fast_loop"
    plan["required_next_step"] = "certify" if deferred_checks else "can_merge"
    if any(str(item.get("tier") or "") == "tier2_subsystem_gate" for item in selected_checks):
        plan["recommended_execution_mode"] = "fast_loop_with_subsystem_gate"
    return plan


def render_validation_plan_md(payload: dict[str, Any]) -> str:
    scope = payload.get("change_scope") if isinstance(payload.get("change_scope"), dict) else {}
    lines = [
        "# P61 Validation Plan",
        "",
        f"- generated_at: `{payload.get('generated_at')}`",
        f"- repo_root: `{payload.get('repo_root')}`",
        f"- gate_plan_path: `{payload.get('gate_plan_path')}`",
        f"- changed_files: `{len(scope.get('changed_files') or [])}`",
        f"- top_level_categories: `{', '.join([str(item) for item in (scope.get('top_level_categories') or [])])}`",
        f"- docs_only: `{scope.get('docs_only')}`",
        f"- recommended_execution_mode: `{payload.get('recommended_execution_mode')}`",
        f"- pending_certification: `{payload.get('pending_certification')}`",
        f"- required_next_step: `{payload.get('required_next_step')}`",
        "",
        "## Selected Checks",
        "",
    ]
    selected = payload.get("selected_checks") if isinstance(payload.get("selected_checks"), list) else []
    if not selected:
        lines.append("- No checks selected.")
    for item in selected:
        if not isinstance(item, dict):
            continue
        lines.append(
            "- `{}` tier=`{}` blocking=`{}` reasons=`{}`".format(
                item.get("check_id") or "",
                item.get("tier") or "",
                item.get("blocking_behavior") or "",
                " | ".join([str(reason) for reason in (item.get("reasons") or [])]),
            )
        )
    lines += ["", "## Deferred Certification", ""]
    deferred = payload.get("deferred_certification") if isinstance(payload.get("deferred_certification"), list) else []
    if not deferred:
        lines.append("- None.")
    for item in deferred:
        if not isinstance(item, dict):
            continue
        lines.append(
            "- `{}` command=`{}` reasons=`{}`".format(
                item.get("check_id") or "",
                item.get("command_template") or "",
                " | ".join([str(reason) for reason in (item.get("reasons") or [])]),
            )
        )
    return "\n".join(lines).rstrip() + "\n"


def write_validation_plan_artifacts(payload: dict[str, Any], *, out_root: Path, stamp: str) -> dict[str, str]:
    out_root.mkdir(parents=True, exist_ok=True)
    json_path = out_root / f"validation_plan_{stamp}.json"
    md_path = out_root / f"validation_plan_{stamp}.md"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    md_path.write_text(render_validation_plan_md(payload), encoding="utf-8")
    return {"json_path": str(json_path.resolve()), "md_path": str(md_path.resolve())}


def main() -> int:
    parser = argparse.ArgumentParser(description="P61 validation planner")
    parser.add_argument("--repo-root", default="")
    parser.add_argument("--gate-plan", default="")
    parser.add_argument("--changed-file", action="append", default=[])
    parser.add_argument("--write-artifacts", action="store_true")
    args = parser.parse_args()
    payload = build_validation_plan(
        repo_root=args.repo_root or None,
        changed_files=list(args.changed_file or []),
        gate_plan_path=args.gate_plan or None,
    )
    if args.write_artifacts:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        out_root = resolve_repo_root(args.repo_root or None) / "docs" / "artifacts" / "p61"
        payload["artifacts"] = write_validation_plan_artifacts(payload, out_root=out_root, stamp=stamp)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
