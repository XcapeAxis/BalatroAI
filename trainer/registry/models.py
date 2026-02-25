from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _registry_path(root: Path) -> Path:
    return root / "models_registry.jsonl"


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
    p = argparse.ArgumentParser(description="P17 model registry manager")
    p.add_argument("--registry-root", default="docs/artifacts/p17/registry")
    p.add_argument("--list", action="store_true")
    p.add_argument("--add-json", default="")
    p.add_argument("--model-id", default="")
    p.add_argument("--dataset-id", default="")
    p.add_argument("--parent-model-id", default="")
    p.add_argument("--model-path", default="")
    p.add_argument("--decision", default="candidate")
    p.add_argument("--git-commit", default="")
    p.add_argument("--offline-metrics-json", default="")
    p.add_argument("--eval100-json", default="")
    p.add_argument("--eval500-json", default="")
    p.add_argument("--train-params-json", default="")
    return p


def _load_json(path_or_json: str) -> dict[str, Any]:
    if not path_or_json:
        return {}
    p = Path(path_or_json)
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return {}
    try:
        return json.loads(path_or_json)
    except Exception:
        return {"raw": path_or_json}


def _entry_from_args(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "schema": "model_registry_v1",
        "model_id": args.model_id or f"model_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        "parent_model_id": args.parent_model_id or None,
        "dataset_id": args.dataset_id or None,
        "train_params": _load_json(args.train_params_json),
        "metrics_offline": _load_json(args.offline_metrics_json),
        "metrics_eval_100": _load_json(args.eval100_json),
        "metrics_eval_500": _load_json(args.eval500_json),
        "artifact_paths": {"model": args.model_path},
        "decision": args.decision,
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
        payload.setdefault("schema", "model_registry_v1")
        payload.setdefault("created_at", _now_iso())
        _append_jsonl(path, payload)
        print(json.dumps({"status": "ok", "registry": str(path), "model_id": payload.get("model_id")}, ensure_ascii=False))
        return 0
    if args.model_path or args.model_id:
        payload = _entry_from_args(args)
        _append_jsonl(path, payload)
        print(json.dumps({"status": "ok", "registry": str(path), "model_id": payload.get("model_id")}, ensure_ascii=False))
        return 0
    print("No action. Use --list or --add-json or model fields.")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

