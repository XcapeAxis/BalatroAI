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

from trainer.autonomy.decision_policy import default_policy_path


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def resolve_repo_root() -> Path:
    preferred = Path("D:/MYFILES/BalatroAI")
    if preferred.exists():
        return preferred.resolve()
    return Path(__file__).resolve().parents[2]


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return ""


def _check(status: str, name: str, detail: str, ref: str = "") -> dict[str, Any]:
    return {"name": name, "status": status, "detail": detail, "ref": ref}


def build_consistency_report(*, repo_root: str | Path | None = None) -> dict[str, Any]:
    root = Path(repo_root).resolve() if repo_root else resolve_repo_root()
    checks: list[dict[str, Any]] = []
    errors: list[str] = []
    warnings: list[str] = []

    agents_paths = {
        "root": root / "AGENTS.md",
        "trainer": root / "trainer" / "AGENTS.md",
        "sim": root / "sim" / "AGENTS.md",
        "scripts": root / "scripts" / "AGENTS.md",
        "docs": root / "docs" / "AGENTS.md",
        "configs": root / "configs" / "AGENTS.md",
    }
    for name, path in agents_paths.items():
        if path.exists():
            checks.append(_check("pass", f"{name}_agents_present", "present", str(path.resolve())))
        else:
            message = f"missing {name} AGENTS: {path}"
            errors.append(message)
            checks.append(_check("fail", f"{name}_agents_present", message, str(path)))

    root_text = _read_text(agents_paths["root"])
    if root_text:
        required_refs = [
            "scripts\\run_p22.ps1",
            "scripts\\run_regressions.ps1",
            "scripts\\run_autonomy.ps1",
            "docs/DECISION_POLICY.md",
            "docs/P57_OVERNIGHT_AUTONOMY_PROTOCOL.md",
            "docs/P51_CHECKPOINT_REGISTRY_AND_CAMPAIGNS.md",
            "docs/P58_WINDOWS_BOOTSTRAP.md",
        ]
        for token in required_refs:
            if token in root_text:
                checks.append(_check("pass", f"root_agents_ref::{token}", "referenced", token))
            else:
                warnings.append(f"root AGENTS does not reference {token}")
                checks.append(_check("warn", f"root_agents_ref::{token}", "missing reference", token))
        line_count = len(root_text.splitlines())
        if line_count > 80:
            warnings.append(f"root AGENTS is longer than expected ({line_count} lines)")
            checks.append(_check("warn", "root_agents_length", f"{line_count} lines", str(agents_paths["root"].resolve())))
        else:
            checks.append(_check("pass", "root_agents_length", f"{line_count} lines", str(agents_paths["root"].resolve())))

    policy_path = default_policy_path(root)
    if policy_path.exists():
        checks.append(_check("pass", "decision_policy_present", "present", str(policy_path.resolve())))
    else:
        errors.append(f"decision policy missing: {policy_path}")
        checks.append(_check("fail", "decision_policy_present", "missing", str(policy_path)))

    decision_doc = root / "docs" / "DECISION_POLICY.md"
    if decision_doc.exists():
        decision_text = _read_text(decision_doc).lower()
        if "auto-approve" in decision_text and "must-block-for-human" in decision_text:
            checks.append(_check("pass", "decision_policy_doc_classes", "decision classes documented", str(decision_doc.resolve())))
        else:
            warnings.append("decision policy doc does not clearly list all decision classes")
            checks.append(_check("warn", "decision_policy_doc_classes", "missing class wording", str(decision_doc.resolve())))
    else:
        errors.append(f"decision policy doc missing: {decision_doc}")
        checks.append(_check("fail", "decision_policy_doc_present", "missing", str(decision_doc)))

    readme_path = root / "README.md"
    readme_text = _read_text(readme_path)
    for token in ("AGENTS.md", "scripts\\run_autonomy.ps1", "docs/P59_AGENTS_STANDARDIZATION_AND_AUTONOMY_ENTRY.md"):
        if token in readme_text:
            checks.append(_check("pass", f"readme_ref::{token}", "referenced", str(readme_path.resolve())))
        else:
            warnings.append(f"README is missing {token}")
            checks.append(_check("warn", f"readme_ref::{token}", "missing reference", str(readme_path.resolve())))

    configs_agents_text = _read_text(agents_paths["configs"]).lower()
    if agents_paths["configs"].exists():
        if "source of truth" in configs_agents_text and "yaml" in configs_agents_text:
            checks.append(_check("pass", "configs_agents_source_of_truth", "YAML source-of-truth documented", str(agents_paths["configs"].resolve())))
        else:
            warnings.append("configs/AGENTS.md should state YAML source-of-truth more explicitly")
            checks.append(_check("warn", "configs_agents_source_of_truth", "source-of-truth wording weak", str(agents_paths["configs"].resolve())))

    status = "ok"
    if errors:
        status = "error"
    elif warnings:
        status = "warning"

    return {
        "schema": "p59_agents_consistency_v1",
        "generated_at": _now_iso(),
        "repo_root": str(root),
        "status": status,
        "errors": errors,
        "warnings": warnings,
        "checks": checks,
        "paths": {name: str(path.resolve()) for name, path in agents_paths.items()},
        "decision_policy_path": str(policy_path.resolve()),
        "readme_path": str(readme_path.resolve()),
    }


def render_report_md(payload: dict[str, Any]) -> str:
    lines = [
        "# P59 AGENTS Consistency",
        "",
        f"- generated_at: `{payload.get('generated_at')}`",
        f"- repo_root: `{payload.get('repo_root')}`",
        f"- status: `{payload.get('status')}`",
        f"- decision_policy_path: `{payload.get('decision_policy_path')}`",
        "",
        "## Errors",
        "",
    ]
    for item in payload.get("errors") or []:
        lines.append(f"- {item}")
    if not (payload.get("errors") or []):
        lines.append("- None.")
    lines += ["", "## Warnings", ""]
    for item in payload.get("warnings") or []:
        lines.append(f"- {item}")
    if not (payload.get("warnings") or []):
        lines.append("- None.")
    lines += ["", "## Checks", ""]
    for row in payload.get("checks") or []:
        if not isinstance(row, dict):
            continue
        lines.append(
            "- `{}` status=`{}` detail={} ref=`{}`".format(
                row.get("name") or "",
                row.get("status") or "",
                row.get("detail") or "",
                row.get("ref") or "",
            )
        )
    return "\n".join(lines).rstrip() + "\n"


def write_consistency_report(*, repo_root: str | Path | None = None, out_root: str | Path | None = None) -> dict[str, Any]:
    root = Path(repo_root).resolve() if repo_root else resolve_repo_root()
    out_dir = Path(out_root).resolve() if out_root else (root / "docs" / "artifacts" / "p59").resolve()
    payload = build_consistency_report(repo_root=root)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    json_path = out_dir / f"agents_consistency_{stamp}.json"
    md_path = out_dir / f"agents_consistency_{stamp}.md"
    latest_json = out_dir / "latest_agents_consistency.json"
    latest_md = out_dir / "latest_agents_consistency.md"
    out_dir.mkdir(parents=True, exist_ok=True)
    text = render_report_md(payload)
    json_text = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    json_path.write_text(json_text, encoding="utf-8")
    md_path.write_text(text, encoding="utf-8")
    latest_json.write_text(json_text, encoding="utf-8")
    latest_md.write_text(text, encoding="utf-8")
    payload["json_path"] = str(json_path.resolve())
    payload["md_path"] = str(md_path.resolve())
    payload["latest_json_path"] = str(latest_json.resolve())
    payload["latest_md_path"] = str(latest_md.resolve())
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="P59 AGENTS consistency check")
    parser.add_argument("--repo-root", default="")
    parser.add_argument("--out-root", default="")
    args = parser.parse_args()
    payload = write_consistency_report(repo_root=args.repo_root or None, out_root=args.out_root or None)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
