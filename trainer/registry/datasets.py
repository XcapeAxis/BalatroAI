from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _registry_path(root: Path) -> Path:
    return root / "datasets_registry.jsonl"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except Exception:
            continue
    return rows


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="P17 dataset registry manager")
    p.add_argument("--registry-root", default="docs/artifacts/p17/registry")
    p.add_argument("--list", action="store_true")
    p.add_argument("--add-json", default="", help="Path to metadata json to append.")
    p.add_argument("--dataset-id", default="")
    p.add_argument("--source-type", default="")
    p.add_argument("--file-path", default="")
    p.add_argument("--hand-records", type=int, default=0)
    p.add_argument("--shop-records", type=int, default=0)
    p.add_argument("--invalid-rows", type=int, default=0)
    p.add_argument("--git-commit", default="")
    p.add_argument("--source-runs", default="", help="Comma-separated source run ids.")
    p.add_argument("--collection-params-json", default="")
    return p


def _entry_from_args(args: argparse.Namespace) -> dict[str, Any]:
    params: dict[str, Any] = {}
    if args.collection_params_json:
        try:
            params = json.loads(args.collection_params_json)
        except Exception:
            params = {"raw": args.collection_params_json}
    source_runs = [x.strip() for x in str(args.source_runs).split(",") if x.strip()]
    return {
        "schema": "dataset_registry_v1",
        "dataset_id": args.dataset_id or f"dataset_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        "source_type": args.source_type or "unknown",
        "source_runs": source_runs,
        "sample_counts": {
            "hand": int(args.hand_records),
            "shop": int(args.shop_records),
            "invalid": int(args.invalid_rows),
        },
        "collection_params": params,
        "file_path": args.file_path,
        "git_commit": args.git_commit,
        "created_at": _now_iso(),
    }


def main() -> int:
    args = _build_parser().parse_args()
    root = Path(args.registry_root)
    path = _registry_path(root)
    if args.list:
        rows = _read_jsonl(path)
        print(json.dumps({"count": len(rows), "items": rows[-20:]}, ensure_ascii=False, indent=2))
        return 0
    if args.add_json:
        payload = json.loads(Path(args.add_json).read_text(encoding="utf-8"))
        if "schema" not in payload:
            payload["schema"] = "dataset_registry_v1"
        payload.setdefault("created_at", _now_iso())
        _append_jsonl(path, payload)
        print(json.dumps({"status": "ok", "registry": str(path), "dataset_id": payload.get("dataset_id")}, ensure_ascii=False))
        return 0
    if args.dataset_id or args.file_path:
        payload = _entry_from_args(args)
        _append_jsonl(path, payload)
        print(json.dumps({"status": "ok", "registry": str(path), "dataset_id": payload.get("dataset_id")}, ensure_ascii=False))
        return 0
    print("No action. Use --list or --add-json or dataset fields.")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

