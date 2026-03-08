from __future__ import annotations

if __package__ is None or __package__ == "":
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

import argparse
import json
from pathlib import Path
from typing import Any

from trainer.autonomy.attention_item_schema import normalize_item, now_iso, slugify


def resolve_repo_root() -> Path:
    preferred = Path("D:/MYFILES/BalatroAI")
    if preferred.exists():
        return preferred.resolve()
    return Path(__file__).resolve().parents[2]


def default_attention_root(repo_root: Path | None = None) -> Path:
    root = repo_root.resolve() if isinstance(repo_root, Path) else resolve_repo_root()
    return (root / "docs" / "artifacts" / "attention_required").resolve()


def attention_root(repo_root: Path | None = None) -> Path:
    return default_attention_root(repo_root)


def queue_json_path(root: str | Path | None = None) -> Path:
    base = Path(root).resolve() if root else default_attention_root()
    return (base / "attention_queue.json").resolve()


def queue_md_path(root: str | Path | None = None) -> Path:
    base = Path(root).resolve() if root else default_attention_root()
    return (base / "attention_queue.md").resolve()


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_attention_queue(root: str | Path | None = None) -> dict[str, Any]:
    path = queue_json_path(root)
    payload = _read_json(path)
    if not isinstance(payload, dict):
        payload = {
            "schema": "p57_attention_queue_v1",
            "generated_at": now_iso(),
            "queue_path": str(path),
            "items": [],
        }
    payload.setdefault("schema", "p57_attention_queue_v1")
    payload["queue_path"] = str(path)
    payload["items"] = [normalize_item(item) for item in (payload.get("items") or []) if isinstance(item, dict)]
    return payload


def save_attention_queue(payload: dict[str, Any], root: str | Path | None = None) -> dict[str, Any]:
    path = queue_json_path(root)
    norm = load_attention_queue(root)
    for key, value in payload.items():
        if key == "items":
            continue
        norm[key] = value
    norm["generated_at"] = now_iso()
    norm["items"] = [normalize_item(item) for item in (payload.get("items") or []) if isinstance(item, dict)]
    _write_json(path, norm)
    queue_md_path(root).write_text(render_attention_queue_md(norm), encoding="utf-8")
    return norm


def render_attention_item_md(item: dict[str, Any]) -> str:
    lines = [
        f"# {item.get('title') or item.get('attention_id')}",
        "",
        f"- attention_id: `{item.get('attention_id')}`",
        f"- severity: `{item.get('severity')}`",
        f"- category: `{item.get('category')}`",
        f"- status: `{item.get('status')}`",
        f"- created_at: `{item.get('created_at')}`",
    ]
    if item.get("blocking_stage"):
        lines.append(f"- blocking_stage: `{item.get('blocking_stage')}`")
    if item.get("blocking_scope"):
        lines.append(f"- blocking_scope: `{item.get('blocking_scope')}`")
    if item.get("campaign_id"):
        lines.append(f"- campaign_id: `{item.get('campaign_id')}`")
    if item.get("related_campaign"):
        lines.append(f"- related_campaign: `{item.get('related_campaign')}`")
    if item.get("experiment_id"):
        lines.append(f"- experiment_id: `{item.get('experiment_id')}`")
    if item.get("seed"):
        lines.append(f"- seed: `{item.get('seed')}`")
    related_checkpoint_ids = item.get("related_checkpoint_ids") or []
    if related_checkpoint_ids:
        lines.append(f"- related_checkpoint_ids: `{', '.join([str(token) for token in related_checkpoint_ids])}`")
    if item.get("decision_deadline_hint"):
        lines.append(f"- decision_deadline_hint: `{item.get('decision_deadline_hint')}`")
    lines += [
        "",
        "## Summary",
        "",
        str(item.get("summary") or ""),
    ]
    summary_for_human = str(item.get("summary_for_human") or "").strip()
    if summary_for_human and summary_for_human != str(item.get("summary") or "").strip():
        lines += [
            "",
            "## Human Summary",
            "",
            summary_for_human,
        ]
    lines += [
        "",
        "## Attempted Actions",
        "",
    ]
    for action in item.get("attempted_actions") or []:
        lines.append(f"- {action}")
    if not (item.get("attempted_actions") or []):
        lines.append("- None recorded.")
    lines += [
        "",
        "## Recommended Options",
        "",
    ]
    for option in item.get("recommended_options") or []:
        if isinstance(option, dict):
            label = str(option.get("label") or "").strip()
            description = str(option.get("description") or "").strip()
            if label and description:
                lines.append(f"- `{label}`: {description}")
            elif label:
                lines.append(f"- `{label}`")
            elif description:
                lines.append(f"- {description}")
        else:
            lines.append(f"- {option}")
    if item.get("recommended_default"):
        lines.append(f"- default: `{item.get('recommended_default')}`")
    lines += [
        "",
        "## Required Human Input",
        "",
    ]
    required_human_input = item.get("required_human_input") or []
    if isinstance(required_human_input, list) and required_human_input:
        for prompt in required_human_input:
            lines.append(f"- {prompt}")
    elif str(required_human_input or "").strip():
        lines.append(str(required_human_input))
    else:
        lines.append("- None recorded.")
    lines += [
        "",
        "## Artifact Refs",
        "",
    ]
    for ref in item.get("artifact_refs") or []:
        lines.append(f"- `{ref}`")
    if not (item.get("artifact_refs") or []):
        lines.append("- None recorded.")
    lines += [
        "",
        "## Suggested Commands",
        "",
    ]
    for command in item.get("suggested_commands") or []:
        lines.append(f"- `{command}`")
    if not (item.get("suggested_commands") or []):
        lines.append("- None recorded.")
    if item.get("resolution_note"):
        lines += ["", "## Resolution", "", str(item.get("resolution_note") or "")]
    return "\n".join(lines).rstrip() + "\n"


def render_attention_queue_md(payload: dict[str, Any]) -> str:
    items = [item for item in (payload.get("items") or []) if isinstance(item, dict)]
    lines = [
        "# Attention Queue",
        "",
        f"- generated_at: `{payload.get('generated_at')}`",
        f"- queue_path: `{payload.get('queue_path')}`",
        f"- open_count: `{sum(1 for item in items if str(item.get('status') or '') == 'open')}`",
        f"- latest_morning_summary_path: `{payload.get('latest_morning_summary_path') or ''}`",
        "",
        "## Items",
        "",
    ]
    if not items:
        lines.append("- No attention items.")
    for item in items:
        lines.append(
            "- `{attention_id}` severity=`{severity}` status=`{status}` category=`{category}` stage=`{stage}` scope=`{scope}` title={title}".format(
                attention_id=item.get("attention_id") or "",
                severity=item.get("severity") or "",
                status=item.get("status") or "",
                category=item.get("category") or "",
                stage=item.get("blocking_stage") or "",
                scope=item.get("blocking_scope") or "",
                title=item.get("title") or "",
            )
        )
    return "\n".join(lines).rstrip() + "\n"


def _item_md_path(item: dict[str, Any], root: str | Path | None = None) -> Path:
    base = Path(root).resolve() if root else default_attention_root()
    title_slug = slugify(item.get("title") or item.get("attention_id") or "attention")
    stamp = str(item.get("created_at") or now_iso()).replace(":", "").replace("-", "").replace("T", "-").split(".")[0]
    return (base / f"{stamp}_{title_slug}.md").resolve()


def upsert_attention_item(item: dict[str, Any], root: str | Path | None = None) -> dict[str, Any]:
    payload = load_attention_queue(root)
    normalized = normalize_item(item)
    items = list(payload.get("items") or [])
    dedupe_key = str(normalized.get("dedupe_key") or "").strip()
    for index, existing in enumerate(items):
        same_id = str(existing.get("attention_id") or "") == str(normalized.get("attention_id") or "")
        same_dedupe = dedupe_key and str(existing.get("dedupe_key") or "") == dedupe_key
        if not same_id and not same_dedupe:
            continue
        merged = dict(existing)
        merged.update({key: value for key, value in normalized.items() if value not in (None, "", [], {}) or key == "status"})
        items[index] = normalize_item(merged)
        path = _item_md_path(items[index], root)
        items[index]["item_md_path"] = str(path)
        items[index]["attention_file"] = str(path)
        payload["items"] = items
        save_attention_queue(payload, root)
        path.write_text(render_attention_item_md(items[index]), encoding="utf-8")
        return items[index]
    path = _item_md_path(normalized, root)
    normalized["item_md_path"] = str(path)
    normalized["attention_file"] = str(path)
    items.append(normalized)
    payload["items"] = items
    save_attention_queue(payload, root)
    path.write_text(render_attention_item_md(normalized), encoding="utf-8")
    return normalized


def add_attention_item(item: dict[str, Any], root: str | Path | None = None) -> dict[str, Any]:
    return upsert_attention_item(item, root=root)


def open_attention_item(
    *,
    severity: str,
    category: str,
    title: str,
    summary: str,
    summary_for_human: str = "",
    blocking_stage: str = "",
    blocking_scope: str = "",
    attempted_actions: list[str] | None = None,
    recommended_options: list[dict[str, str]] | list[str] | None = None,
    recommended_default: str = "",
    required_human_input: list[str] | str = "",
    artifact_refs: list[str] | None = None,
    suggested_commands: list[str] | None = None,
    related_campaign: str = "",
    related_checkpoint_ids: list[str] | None = None,
    decision_deadline_hint: str = "",
    campaign_id: str = "",
    run_id: str = "",
    experiment_id: str = "",
    seed: str = "",
    dedupe_key: str = "",
    root: str | Path | None = None,
) -> dict[str, Any]:
    item = {
        "attention_id": "",
        "created_at": now_iso(),
        "severity": severity,
        "category": category,
        "title": title,
        "summary": summary,
        "summary_for_human": summary_for_human or summary,
        "blocking_stage": blocking_stage,
        "blocking_scope": blocking_scope,
        "attempted_actions": attempted_actions or [],
        "recommended_options": recommended_options or [],
        "recommended_default": recommended_default,
        "required_human_input": required_human_input,
        "artifact_refs": artifact_refs or [],
        "suggested_commands": suggested_commands or [],
        "related_campaign": related_campaign or campaign_id,
        "related_checkpoint_ids": related_checkpoint_ids or [],
        "decision_deadline_hint": decision_deadline_hint,
        "status": "open",
        "campaign_id": campaign_id,
        "run_id": run_id,
        "experiment_id": experiment_id,
        "seed": seed,
        "dedupe_key": dedupe_key,
    }
    return upsert_attention_item(item, root=root)


def resolve_attention_item(attention_id: str, *, resolution_note: str = "", root: str | Path | None = None) -> dict[str, Any]:
    payload = load_attention_queue(root)
    items = list(payload.get("items") or [])
    for index, item in enumerate(items):
        if str(item.get("attention_id") or "") != str(attention_id):
            continue
        updated = dict(item)
        updated["status"] = "resolved"
        updated["resolution_note"] = str(resolution_note or "resolved via ops ui")
        updated["resolved_at"] = now_iso()
        items[index] = normalize_item(updated)
        path = _item_md_path(items[index], root)
        items[index]["item_md_path"] = str(path)
        items[index]["attention_file"] = str(path)
        payload["items"] = items
        save_attention_queue(payload, root)
        path.write_text(render_attention_item_md(items[index]), encoding="utf-8")
        return items[index]
    raise KeyError(f"unknown attention_id: {attention_id}")


def update_attention_status(
    ref_or_id: str,
    *,
    status: str,
    resolution_note: str = "",
    root: str | Path | None = None,
) -> dict[str, Any]:
    token = str(status or "").strip().lower()
    if token in {"resolved", "closed"}:
        return resolve_attention_item(ref_or_id, resolution_note=resolution_note, root=root)
    payload = load_attention_queue(root)
    items = list(payload.get("items") or [])
    for index, item in enumerate(items):
        if not isinstance(item, dict):
            continue
        if str(ref_or_id) not in {str(item.get("attention_id") or ""), str(item.get("item_md_path") or "")}:
            continue
        updated = dict(item)
        updated["status"] = token or str(item.get("status") or "open")
        if resolution_note:
            updated["resolution_note"] = resolution_note
        items[index] = normalize_item(updated)
        path = _item_md_path(items[index], root)
        items[index]["item_md_path"] = str(path)
        items[index]["attention_file"] = str(path)
        payload["items"] = items
        save_attention_queue(payload, root)
        path.write_text(render_attention_item_md(items[index]), encoding="utf-8")
        return items[index]
    raise KeyError(f"unknown attention item: {ref_or_id}")


def get_attention_item_status(ref_or_id: str, root: str | Path | None = None) -> str:
    token = str(ref_or_id or "").strip()
    if not token:
        return ""
    path = Path(token)
    if path.suffix.lower() == ".md" and path.exists():
        text = path.read_text(encoding="utf-8", errors="replace")
        for line in text.splitlines():
            if line.startswith("- status:"):
                return line.split("`")[1] if "`" in line else line.split(":", 1)[-1].strip()
    payload = load_attention_queue(root)
    for item in payload.get("items") or []:
        if not isinstance(item, dict):
            continue
        if token in {str(item.get("attention_id") or ""), str(item.get("item_md_path") or "")}:
            return str(item.get("status") or "")
    return ""


def open_attention_items(root: str | Path | None = None) -> list[dict[str, Any]]:
    payload = load_attention_queue(root)
    return [
        dict(item)
        for item in (payload.get("items") or [])
        if isinstance(item, dict) and str(item.get("status") or "").strip().lower() == "open"
    ]


def _smoke(root: str | Path | None = None) -> dict[str, Any]:
    first = open_attention_item(
        severity="block",
        category="promotion",
        title="Promotion requires human approval",
        summary="A checkpoint reached promotion_review. The system can summarize evidence but cannot promote automatically.",
        summary_for_human="A candidate is waiting in promotion review. A person still needs to decide whether it stays in review or moves to a safer deployment mode.",
        blocking_stage="promotion_queue_scan",
        blocking_scope="deployment_decision",
        attempted_actions=["scan_promotion_queue", "summarize_registry_state"],
        recommended_options=["Keep candidate in review", "Promote as canary only after human sign-off"],
        recommended_default="Keep candidate in review",
        required_human_input="Choose whether to promote the candidate and in which deployment mode.",
        suggested_commands=[
            "powershell -ExecutionPolicy Bypass -File scripts\\run_p22.ps1 -RunP56",
            "powershell -ExecutionPolicy Bypass -File scripts\\run_ops_ui.ps1",
        ],
        decision_deadline_hint="before next overnight run",
        artifact_refs=[],
        dedupe_key="smoke-promotion-human-approval",
        root=root,
    )
    second = open_attention_item(
        severity="block",
        category="config",
        title="Config provenance anomaly",
        summary="Config sidecar drift or provenance mismatch needs confirmation before continuing an overnight branch.",
        summary_for_human="Config provenance is no longer trustworthy enough for unattended execution. Confirm the authoritative config before resuming.",
        blocking_stage="config_provenance_scan",
        blocking_scope="config_provenance",
        attempted_actions=["sync_config_sidecars", "inspect_summary_config_provenance"],
        recommended_options=["Re-run config sync", "Inspect config source and sidecar drift manually"],
        recommended_default="Re-run config sync",
        required_human_input="Confirm which config source is authoritative and whether the drift is expected.",
        suggested_commands=[
            "powershell -ExecutionPolicy Bypass -File scripts\\run_regressions.ps1 -RunP22",
            "powershell -ExecutionPolicy Bypass -File scripts\\doctor.ps1",
        ],
        decision_deadline_hint="before next nightly launch",
        artifact_refs=[],
        dedupe_key="smoke-config-provenance-anomaly",
        root=root,
    )
    queue = load_attention_queue(root)
    return {
        "schema": "p57_attention_queue_smoke_v1",
        "queue_path": str(queue_json_path(root)),
        "queue_md_path": str(queue_md_path(root)),
        "items": [first, second],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="P57 attention queue")
    parser.add_argument("--root", default="")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--resolve", default="")
    parser.add_argument("--resolution-note", default="")
    parser.add_argument("--title", default="")
    parser.add_argument("--summary", default="")
    parser.add_argument("--severity", default="info")
    parser.add_argument("--category", default="general")
    parser.add_argument("--blocking-stage", default="")
    parser.add_argument("--required-human-input", default="")
    parser.add_argument("--recommended-default", default="")
    parser.add_argument("--recommended-options", default="")
    parser.add_argument("--attempted-actions", default="")
    parser.add_argument("--artifact-refs", default="")
    args = parser.parse_args()

    root = args.root or None
    if args.smoke:
        payload = _smoke(root)
    elif str(args.resolve).strip():
        payload = resolve_attention_item(args.resolve, resolution_note=args.resolution_note or "", root=root)
    elif str(args.title).strip():
        payload = open_attention_item(
            severity=args.severity,
            category=args.category,
            title=args.title,
            summary=args.summary,
            blocking_stage=args.blocking_stage,
            attempted_actions=[item.strip() for item in str(args.attempted_actions or "").split(",") if item.strip()],
            recommended_options=[item.strip() for item in str(args.recommended_options or "").split(",") if item.strip()],
            recommended_default=args.recommended_default,
            required_human_input=[args.required_human_input] if str(args.required_human_input or "").strip() else [],
            artifact_refs=[item.strip() for item in str(args.artifact_refs or "").split(",") if item.strip()],
            root=root,
        )
    else:
        payload = load_attention_queue(root)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
