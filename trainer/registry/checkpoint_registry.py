from __future__ import annotations

if __package__ is None or __package__ == "":
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

import argparse
import json
from pathlib import Path
from typing import Any

from trainer.registry.checkpoint_query import (
    filter_entries,
    latest_by_family,
    promoted_by_family,
    sort_entries,
)
from trainer.registry.checkpoint_schema import normalize_entry, now_iso
from trainer.registry.checkpoint_state_machine import apply_transition


def resolve_repo_root() -> Path:
    preferred = Path("D:/MYFILES/BalatroAI")
    if preferred.exists():
        return preferred.resolve()
    return Path(__file__).resolve().parents[2]


def default_registry_path(repo_root: Path | None = None) -> Path:
    root = repo_root.resolve() if isinstance(repo_root, Path) else resolve_repo_root()
    return root / "docs" / "artifacts" / "registry" / "checkpoints_registry.json"


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


def _has_meaningful_value(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, dict, set, tuple)):
        return bool(value)
    return True


def load_registry(path: str | Path | None = None) -> dict[str, Any]:
    registry_path = Path(path).resolve() if path else default_registry_path()
    payload = _read_json(registry_path)
    if not isinstance(payload, dict):
        payload = {
            "schema": "p51_checkpoint_registry_v1",
            "generated_at": now_iso(),
            "registry_path": str(registry_path),
            "items": [],
        }
    payload["registry_path"] = str(registry_path)
    items = payload.get("items") if isinstance(payload.get("items"), list) else []
    payload["items"] = [normalize_entry(item) for item in items if isinstance(item, dict)]
    return payload


def save_registry(registry: dict[str, Any], path: str | Path | None = None) -> Path:
    registry_path = Path(path).resolve() if path else Path(str(registry.get("registry_path") or default_registry_path())).resolve()
    payload = dict(registry)
    payload["schema"] = "p51_checkpoint_registry_v1"
    payload["generated_at"] = now_iso()
    payload["registry_path"] = str(registry_path)
    payload["items"] = sort_entries([normalize_entry(item) for item in (payload.get("items") or []) if isinstance(item, dict)])
    _write_json(registry_path, payload)
    return registry_path


def list_entries(path: str | Path | None = None) -> list[dict[str, Any]]:
    return list(load_registry(path).get("items") or [])


def get_entry(checkpoint_id: str, *, path: str | Path | None = None) -> dict[str, Any] | None:
    for item in list_entries(path):
        if str(item.get("checkpoint_id") or "") == str(checkpoint_id):
            return item
    return None


def find_by_artifact_path(artifact_path: str | Path, *, path: str | Path | None = None) -> dict[str, Any] | None:
    token = str(Path(artifact_path).resolve())
    for item in list_entries(path):
        if str(item.get("artifact_path") or "").strip() == token:
            return item
    return None


def register_checkpoint(
    payload: dict[str, Any],
    *,
    path: str | Path | None = None,
    allow_update: bool = True,
) -> dict[str, Any]:
    registry = load_registry(path)
    item = normalize_entry(payload)
    artifact_path = str(Path(str(item.get("artifact_path") or "")).resolve()) if str(item.get("artifact_path") or "").strip() else ""
    if artifact_path:
        item["artifact_path"] = artifact_path
    items = list(registry.get("items") or [])

    for index, existing in enumerate(items):
        same_id = str(existing.get("checkpoint_id") or "") == str(item.get("checkpoint_id") or "")
        same_artifact = artifact_path and str(existing.get("artifact_path") or "") == artifact_path
        if not same_id and not same_artifact:
            continue
        if not allow_update:
            return existing
        merged = dict(existing)
        merged.update({key: value for key, value in item.items() if key == "status" or _has_meaningful_value(value)})
        merged["transitions"] = list(existing.get("transitions") or item.get("transitions") or [])
        items[index] = normalize_entry(merged)
        registry["items"] = items
        save_registry(registry)
        return items[index]

    items.append(item)
    registry["items"] = items
    save_registry(registry)
    return item


def update_checkpoint(checkpoint_id: str, updates: dict[str, Any], *, path: str | Path | None = None) -> dict[str, Any]:
    registry = load_registry(path)
    items = list(registry.get("items") or [])
    for index, existing in enumerate(items):
        if str(existing.get("checkpoint_id") or "") != str(checkpoint_id):
            continue
        merged = dict(existing)
        merged.update({key: value for key, value in dict(updates or {}).items() if value is not None})
        items[index] = normalize_entry(merged)
        registry["items"] = items
        save_registry(registry)
        return items[index]
    raise KeyError(f"unknown checkpoint_id: {checkpoint_id}")


def update_checkpoint_status(
    checkpoint_id: str,
    *,
    to_status: str,
    reason: str = "",
    operator: str = "system",
    refs: dict[str, Any] | None = None,
    path: str | Path | None = None,
) -> dict[str, Any]:
    registry = load_registry(path)
    items = list(registry.get("items") or [])
    for index, existing in enumerate(items):
        if str(existing.get("checkpoint_id") or "") != str(checkpoint_id):
            continue
        items[index] = apply_transition(existing, to_status=to_status, reason=reason, operator=operator, refs=refs)
        registry["items"] = items
        save_registry(registry)
        return items[index]
    raise KeyError(f"unknown checkpoint_id: {checkpoint_id}")


def query_registry(
    *,
    path: str | Path | None = None,
    family: str = "",
    status: str = "",
    source_run_id: str = "",
    source_experiment_id: str = "",
    checkpoint_id: str = "",
    latest: bool = False,
    promoted: bool = False,
) -> list[dict[str, Any]]:
    items = filter_entries(
        list_entries(path),
        family=family,
        status=status,
        source_run_id=source_run_id,
        source_experiment_id=source_experiment_id,
        checkpoint_id=checkpoint_id,
    )
    if promoted:
        items = promoted_by_family(items)
    elif latest:
        items = latest_by_family(items)
    else:
        items = sort_entries(items)
    return items


def snapshot_registry(*, out_path: str | Path, path: str | Path | None = None) -> Path:
    snapshot_path = Path(out_path).resolve()
    payload = load_registry(path)
    _write_json(snapshot_path, payload)
    return snapshot_path


def _guess_family_from_path(path: Path) -> str:
    token = str(path).lower()
    if "\\p45\\" in token or "/p45/" in token:
        return "world_model"
    if "\\p42\\" in token or "\\p44\\" in token or "/p42/" in token or "/p44/" in token:
        return "rl_policy"
    if "\\p48\\" in token or "/p48/" in token:
        return "hybrid"
    if (
        "\\p52\\" in token
        or "/p52/" in token
        or "\\p54\\" in token
        or "/p54/" in token
        or "\\router_train\\" in token
        or "/router_train/" in token
    ):
        return "learned_router"
    return "other"


def _guess_mode_from_path(path: Path) -> str:
    token = str(path).lower()
    if "\\p45\\" in token or "/p45/" in token:
        return "p45_world_model"
    if "\\p44\\" in token or "/p44/" in token:
        return "p44_distributed_rl"
    if "\\p42\\" in token or "/p42/" in token:
        return "p42_rl_candidate"
    if "\\p48\\" in token or "/p48/" in token:
        return "p48_hybrid_controller"
    if "\\p54\\" in token or "/p54/" in token:
        return "p54_learned_router"
    if "\\p52\\" in token or "/p52/" in token or "\\router_train\\" in token or "/router_train/" in token:
        return "p52_learned_router"
    return "unknown"


def discover_existing_checkpoints(repo_root: Path | None = None) -> list[dict[str, Any]]:
    root = repo_root.resolve() if isinstance(repo_root, Path) else resolve_repo_root()
    artifacts_root = root / "docs" / "artifacts"
    if not artifacts_root.exists():
        return []
    candidates: list[Path] = []
    for pattern in ("**/best.pt", "**/last.pt", "**/best_checkpoint_stub.json", "**/candidate_checkpoint.json"):
        candidates.extend(artifacts_root.glob(pattern))
    items: list[dict[str, Any]] = []
    for checkpoint_path in sorted({path.resolve() for path in candidates}):
        run_dir = checkpoint_path.parent
        metrics_path = run_dir / "metrics.json"
        manifest_candidates = [
            run_dir / "train_manifest.json",
            run_dir / "run_manifest.json",
            run_dir / "gpu_mainline_summary.json",
        ]
        manifest_path = next((candidate for candidate in manifest_candidates if candidate.exists()), None)
        source_run_id = run_dir.name
        item = normalize_entry(
            {
                "family": _guess_family_from_path(checkpoint_path),
                "training_mode": _guess_mode_from_path(checkpoint_path),
                "source_run_id": source_run_id,
                "seed_or_seed_group": run_dir.name if run_dir.name.startswith("seed_") else "",
                "artifact_path": str(checkpoint_path),
                "metrics_ref": str(metrics_path.resolve()) if metrics_path.exists() else "",
                "notes": "imported_from_existing_artifacts",
                "lineage_refs": {
                    "manifest_path": str(manifest_path.resolve()) if isinstance(manifest_path, Path) else "",
                },
            }
        )
        items.append(item)
    return items


def import_existing_checkpoints(*, path: str | Path | None = None, repo_root: Path | None = None) -> dict[str, Any]:
    imported: list[dict[str, Any]] = []
    for item in discover_existing_checkpoints(repo_root):
        imported.append(register_checkpoint(item, path=path, allow_update=False))
    return {
        "status": "ok",
        "imported_count": len(imported),
        "items": imported,
        "registry_path": str(default_registry_path(repo_root)),
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="P51 checkpoint registry")
    parser.add_argument("--registry", default="")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--family", default="")
    parser.add_argument("--status", default="")
    parser.add_argument("--source-run-id", default="")
    parser.add_argument("--source-experiment-id", default="")
    parser.add_argument("--checkpoint-id", default="")
    parser.add_argument("--latest", action="store_true")
    parser.add_argument("--promoted", action="store_true")
    parser.add_argument("--import-existing", action="store_true")
    parser.add_argument("--snapshot", default="")
    parser.add_argument("--json", action="store_true")
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    registry_path = Path(args.registry).resolve() if str(args.registry).strip() else None
    if args.import_existing:
        payload = import_existing_checkpoints(path=registry_path)
        print(json.dumps(payload, ensure_ascii=False, indent=2 if args.json else None))
        return 0
    if args.snapshot:
        snapshot_path = snapshot_registry(out_path=args.snapshot, path=registry_path)
        print(json.dumps({"status": "ok", "snapshot_path": str(snapshot_path)}, ensure_ascii=False, indent=2 if args.json else None))
        return 0
    items = query_registry(
        path=registry_path,
        family=str(args.family or ""),
        status=str(args.status or ""),
        source_run_id=str(args.source_run_id or ""),
        source_experiment_id=str(args.source_experiment_id or ""),
        checkpoint_id=str(args.checkpoint_id or ""),
        latest=bool(args.latest),
        promoted=bool(args.promoted),
    )
    payload = {
        "registry_path": str(registry_path or default_registry_path()),
        "count": len(items),
        "items": items,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2 if args.json or args.list else None))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
