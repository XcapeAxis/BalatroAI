from __future__ import annotations

if __package__ is None or __package__ == "":
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

import argparse
import json
import math
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:  # pragma: no cover
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None

from trainer.closed_loop.replay_manifest import write_json, write_markdown
from trainer.hybrid.learned_router_model import build_model_from_checkpoint, predict_router_distribution
from trainer.hybrid.router_schema import available_controller_mask, encode_routing_features
from trainer.registry.checkpoint_registry import list_entries, update_checkpoint


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except Exception:
        return float(default)
    if math.isnan(result) or math.isinf(result):
        return float(default)
    return result


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _resolve_repo_root() -> Path:
    preferred = Path("D:/MYFILES/BalatroAI")
    if preferred.exists():
        return preferred.resolve()
    return Path(__file__).resolve().parents[2]


def _read_yaml_or_json(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8-sig")
    if path.suffix.lower() in {".yaml", ".yml"}:
        if yaml is not None:
            payload = yaml.safe_load(text)
        else:
            sidecar = path.with_suffix(".json")
            if not sidecar.exists():
                raise RuntimeError(f"PyYAML unavailable and no sidecar JSON for: {path}")
            payload = json.loads(sidecar.read_text(encoding="utf-8-sig"))
    else:
        payload = json.loads(text)
    if not isinstance(payload, dict):
        raise ValueError(f"config must be mapping: {path}")
    return payload


def _deep_merge(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in (updates or {}).items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(dict(merged.get(key) or {}), value)
        else:
            merged[key] = value
    return merged


def _default_config() -> dict[str, Any]:
    return {
        "schema": "p56_router_calibration_config_v1",
        "calibration": {
            "artifacts_root": "docs/artifacts/p56/router_calibration",
            "bins": 10,
            "max_samples": 2048,
            "valid_only": True,
            "slice_keys": ["slice_stage", "slice_resource_pressure", "slice_action_type"],
            "device": "cpu",
        }
    }


def _merged_config(path: str | Path | None) -> dict[str, Any]:
    payload = _default_config()
    if not path:
        return payload
    cfg_path = Path(path)
    if not cfg_path.is_absolute():
        cfg_path = (_resolve_repo_root() / cfg_path).resolve()
    return _deep_merge(payload, _read_yaml_or_json(cfg_path))


def _read_json(path: Path) -> Any:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception:
        return None


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8", errors="replace") as fp:
        for line in fp:
            text = line.strip()
            if not text:
                continue
            try:
                payload = json.loads(text)
            except Exception:
                continue
            if isinstance(payload, dict):
                rows.append(payload)
    return rows


def _resolve_refs(
    *,
    checkpoint_path: str | Path | None,
    checkpoint_id: str = "",
    dataset_manifest_path: str | Path | None = None,
    train_manifest_path: str | Path | None = None,
) -> tuple[str, str, str, str]:
    repo_root = _resolve_repo_root()
    checkpoint_token = str(checkpoint_path or "").strip()
    dataset_token = str(dataset_manifest_path or "").strip()
    train_token = str(train_manifest_path or "").strip()
    checkpoint_id_token = str(checkpoint_id or "").strip()
    if checkpoint_token:
        resolved = Path(checkpoint_token)
        if not resolved.is_absolute():
            resolved = (repo_root / resolved).resolve()
        return str(resolved), checkpoint_id_token, dataset_token, train_token
    entries = [item for item in list_entries() if str(item.get("family") or "") == "learned_router"]
    entries.sort(key=lambda item: str(item.get("created_at") or ""), reverse=True)
    for entry in entries:
        artifact_path = str(entry.get("artifact_path") or "").strip()
        if not artifact_path:
            continue
        resolved = Path(artifact_path)
        if not resolved.exists():
            continue
        checkpoint_id_token = checkpoint_id_token or str(entry.get("checkpoint_id") or "")
        lineage_refs = entry.get("lineage_refs") if isinstance(entry.get("lineage_refs"), dict) else {}
        dataset_token = dataset_token or str(lineage_refs.get("dataset_manifest_json") or "")
        train_token = train_token or str(lineage_refs.get("train_manifest_json") or "")
        return str(resolved.resolve()), checkpoint_id_token, dataset_token, train_token
    return "", checkpoint_id_token, dataset_token, train_token


def _dataset_jsonl_from_manifest(dataset_manifest_path: str) -> Path:
    payload = _read_json(Path(dataset_manifest_path))
    if not isinstance(payload, dict):
        raise FileNotFoundError(f"invalid dataset manifest: {dataset_manifest_path}")
    dataset_jsonl = Path(str(payload.get("dataset_jsonl") or ""))
    if not dataset_jsonl.exists():
        raise FileNotFoundError(f"dataset_jsonl missing: {dataset_jsonl}")
    return dataset_jsonl


def _bias_label(*, avg_confidence: float, accuracy: float) -> str:
    gap = float(avg_confidence) - float(accuracy)
    if gap >= 0.05:
        return "optimistic"
    if gap <= -0.05:
        return "conservative"
    return "usable"


def run_router_calibration(
    *,
    config_path: str | Path | None = None,
    out_dir: str | Path | None = None,
    run_id: str = "",
    checkpoint_path: str | Path | None = None,
    checkpoint_id: str = "",
    dataset_manifest_path: str | Path | None = None,
    train_manifest_path: str | Path | None = None,
) -> dict[str, Any]:
    repo_root = _resolve_repo_root()
    cfg = _merged_config(config_path)
    calibration_cfg = cfg.get("calibration") if isinstance(cfg.get("calibration"), dict) else {}
    output_root = (
        (repo_root / str(calibration_cfg.get("artifacts_root") or "docs/artifacts/p56/router_calibration")).resolve()
        if out_dir is None
        else (Path(out_dir) if Path(out_dir).is_absolute() else (repo_root / out_dir).resolve())
    )
    chosen_run_id = str(run_id or _now_stamp())
    run_dir = output_root / chosen_run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    resolved_checkpoint, resolved_checkpoint_id, resolved_dataset_manifest, resolved_train_manifest = _resolve_refs(
        checkpoint_path=checkpoint_path,
        checkpoint_id=checkpoint_id,
        dataset_manifest_path=dataset_manifest_path,
        train_manifest_path=train_manifest_path,
    )
    if not resolved_checkpoint:
        raise FileNotFoundError("no learned-router checkpoint available for calibration")
    if not resolved_dataset_manifest:
        raise FileNotFoundError("router calibration requires dataset_manifest_path")

    model, checkpoint_payload = build_model_from_checkpoint(resolved_checkpoint, map_location=str(calibration_cfg.get("device") or "cpu"))
    feature_encoder = checkpoint_payload.get("feature_encoder") if isinstance(checkpoint_payload.get("feature_encoder"), dict) else {}
    calibration_meta = checkpoint_payload.get("calibration") if isinstance(checkpoint_payload.get("calibration"), dict) else {}
    temperature = _safe_float(calibration_meta.get("temperature"), 1.0)
    device = str(calibration_cfg.get("device") or "cpu")
    dataset_rows = _read_jsonl(_dataset_jsonl_from_manifest(resolved_dataset_manifest))
    if bool(calibration_cfg.get("valid_only", True)):
        dataset_rows = [row for row in dataset_rows if bool(row.get("valid_for_training", True))]
    max_samples = max(1, _safe_int(calibration_cfg.get("max_samples"), 2048))
    dataset_rows = dataset_rows[:max_samples]

    bins = max(2, _safe_int(calibration_cfg.get("bins"), 10))
    bucket_rows: list[dict[str, Any]] = []
    per_controller: dict[str, list[dict[str, Any]]] = defaultdict(list)
    slice_breakdown: dict[str, list[dict[str, Any]]] = defaultdict(list)
    confidences: list[float] = []
    correct_rows = 0
    bucket_state: list[dict[str, Any]] = [
        {"count": 0, "confidence_sum": 0.0, "correct_sum": 0.0}
        for _ in range(bins)
    ]

    for sample in dataset_rows:
        encoded = encode_routing_features(sample, feature_encoder)
        distribution = predict_router_distribution(
            model=model,
            feature_vector=list(encoded.get("vector") or []),
            available_mask=available_controller_mask(sample.get("available_controllers")),
            device=device,
            temperature=temperature,
        )
        predicted_controller = str(distribution.get("selected_controller") or "")
        confidence = _safe_float(distribution.get("confidence"), 0.0)
        target_controller = str(sample.get("target_controller_label") or "")
        correct = int(predicted_controller == target_controller and bool(target_controller))
        confidences.append(confidence)
        correct_rows += correct
        bucket_index = min(bins - 1, max(0, int(math.floor(confidence * bins))))
        bucket = bucket_state[bucket_index]
        bucket["count"] += 1
        bucket["confidence_sum"] += confidence
        bucket["correct_sum"] += correct
        row_payload = {
            "sample_id": str(sample.get("sample_id") or ""),
            "predicted_controller": predicted_controller,
            "target_controller": target_controller,
            "confidence": confidence,
            "correct": bool(correct),
            "slice_stage": str(sample.get("slice_stage") or "unknown"),
            "slice_resource_pressure": str(sample.get("slice_resource_pressure") or "unknown"),
            "slice_action_type": str(sample.get("slice_action_type") or "unknown"),
        }
        per_controller[predicted_controller].append(row_payload)
        for slice_key in calibration_cfg.get("slice_keys") or []:
            slice_breakdown[str(slice_key)].append({**row_payload, "slice_label": str(sample.get(slice_key) or sample.get("routing_features", {}).get(slice_key) or "unknown")})

    for index, bucket in enumerate(bucket_state):
        count = int(bucket.get("count") or 0)
        avg_confidence = float(bucket.get("confidence_sum") or 0.0) / max(1, count)
        accuracy = float(bucket.get("correct_sum") or 0.0) / max(1, count)
        bucket_rows.append(
            {
                "bucket_index": index,
                "bucket_start": float(index) / bins,
                "bucket_end": float(index + 1) / bins,
                "count": count,
                "avg_confidence": avg_confidence,
                "accuracy": accuracy,
                "gap": avg_confidence - accuracy,
            }
        )

    total = max(1, len(dataset_rows))
    accuracy = float(correct_rows) / total
    avg_confidence = sum(confidences) / max(1, len(confidences))
    ece = sum((float(row.get("count") or 0) / total) * abs(float(row.get("gap") or 0.0)) for row in bucket_rows)
    controller_rows = []
    for controller_id, rows in sorted(per_controller.items()):
        controller_rows.append(
            {
                "controller_id": controller_id,
                "count": len(rows),
                "accuracy": sum(1 for row in rows if bool(row.get("correct"))) / max(1, len(rows)),
                "avg_confidence": sum(_safe_float(row.get("confidence"), 0.0) for row in rows) / max(1, len(rows)),
            }
        )
    slice_rows = []
    for slice_key, rows in sorted(slice_breakdown.items()):
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            grouped[str(row.get("slice_label") or "unknown")].append(row)
        for slice_label, items in sorted(grouped.items()):
            slice_rows.append(
                {
                    "slice_key": slice_key,
                    "slice_label": slice_label,
                    "count": len(items),
                    "accuracy": sum(1 for row in items if bool(row.get("correct"))) / max(1, len(items)),
                    "avg_confidence": sum(_safe_float(row.get("confidence"), 0.0) for row in items) / max(1, len(items)),
                }
            )

    metrics = {
        "schema": "p56_router_calibration_metrics_v1",
        "generated_at": _now_iso(),
        "run_id": chosen_run_id,
        "checkpoint_path": resolved_checkpoint,
        "checkpoint_id": resolved_checkpoint_id,
        "dataset_manifest_path": resolved_dataset_manifest,
        "train_manifest_path": resolved_train_manifest,
        "sample_count": len(dataset_rows),
        "accuracy": accuracy,
        "avg_confidence": avg_confidence,
        "ece": ece,
        "calibration_bias": _bias_label(avg_confidence=avg_confidence, accuracy=accuracy),
        "temperature": temperature,
        "per_controller": controller_rows,
        "slice_breakdown": slice_rows,
    }
    write_json(run_dir / "calibration_metrics.json", metrics)
    write_json(run_dir / "reliability_bins.json", {"schema": "p56_reliability_bins_v1", "generated_at": _now_iso(), "bins": bucket_rows})
    write_markdown(
        run_dir / "calibration_report.md",
        [
            "# P56 Router Calibration",
            "",
            f"- checkpoint_id: `{resolved_checkpoint_id}`",
            f"- checkpoint_path: `{resolved_checkpoint}`",
            f"- sample_count: {len(dataset_rows)}",
            f"- accuracy: {accuracy:.4f}",
            f"- avg_confidence: {avg_confidence:.4f}",
            f"- ece: {ece:.4f}",
            f"- calibration_bias: `{metrics.get('calibration_bias')}`",
            "",
            "## Reliability Bins",
            *[
                "- [{start:.1f}, {end:.1f}): count={count} accuracy={acc:.4f} avg_confidence={conf:.4f}".format(
                    start=float(row.get("bucket_start") or 0.0),
                    end=float(row.get("bucket_end") or 0.0),
                    count=int(row.get("count") or 0),
                    acc=float(row.get("accuracy") or 0.0),
                    conf=float(row.get("avg_confidence") or 0.0),
                )
                for row in bucket_rows
            ],
        ],
    )
    if resolved_checkpoint_id:
        update_checkpoint(
            resolved_checkpoint_id,
            {
                "calibration_ref": str((run_dir / "calibration_metrics.json").resolve()),
            },
        )
    return {
        "status": "ok",
        "run_id": chosen_run_id,
        "run_dir": str(run_dir.resolve()),
        "checkpoint_id": resolved_checkpoint_id,
        "calibration_metrics_json": str((run_dir / "calibration_metrics.json").resolve()),
        "calibration_report_md": str((run_dir / "calibration_report.md").resolve()),
        "reliability_bins_json": str((run_dir / "reliability_bins.json").resolve()),
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="P56 learned-router calibration analysis")
    parser.add_argument("--config", default="")
    parser.add_argument("--out-dir", default="")
    parser.add_argument("--run-id", default="")
    parser.add_argument("--checkpoint-path", default="")
    parser.add_argument("--checkpoint-id", default="")
    parser.add_argument("--dataset-manifest", default="")
    parser.add_argument("--train-manifest", default="")
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    summary = run_router_calibration(
        config_path=(str(args.config).strip() or None),
        out_dir=(str(args.out_dir).strip() or None),
        run_id=str(args.run_id or ""),
        checkpoint_path=(str(args.checkpoint_path).strip() or None),
        checkpoint_id=str(args.checkpoint_id or ""),
        dataset_manifest_path=(str(args.dataset_manifest).strip() or None),
        train_manifest_path=(str(args.train_manifest).strip() or None),
    )
    print(json.dumps(summary, ensure_ascii=False))
    return 0 if str(summary.get("status") or "") == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
