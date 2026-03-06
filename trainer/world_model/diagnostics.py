from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_markdown(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def _pearson(xs: list[float], ys: list[float]) -> float:
    if len(xs) != len(ys) or len(xs) < 2:
        return 0.0
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    num = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    den_x = sum((x - mean_x) ** 2 for x in xs)
    den_y = sum((y - mean_y) ** 2 for y in ys)
    if den_x <= 0.0 or den_y <= 0.0:
        return 0.0
    return float(num / ((den_x * den_y) ** 0.5))


def build_diagnostics_report(
    prediction_rows: list[dict[str, Any]],
    *,
    topk: int = 12,
) -> tuple[dict[str, Any], list[str]]:
    rows = [row for row in prediction_rows if isinstance(row, dict)]
    errors = [_safe_float(row.get("combined_error"), 0.0) for row in rows]
    uncertainties = [_safe_float(row.get("uncertainty_pred"), 0.0) for row in rows]
    pearson = _pearson(errors, uncertainties)

    paired = list(zip(rows, errors, uncertainties))
    error_sorted = sorted(paired, key=lambda item: item[1], reverse=True)
    uncertainty_sorted = sorted(paired, key=lambda item: item[2], reverse=True)
    cut = max(1, len(rows) // 4) if rows else 0
    error_top_ids = {str(item[0].get("sample_id") or "") for item in error_sorted[:cut]}
    uncertainty_top_ids = {str(item[0].get("sample_id") or "") for item in uncertainty_sorted[:cut]}
    overlap_ratio = float(len(error_top_ids & uncertainty_top_ids) / max(1, cut)) if cut > 0 else 0.0
    top_error_mean_uncertainty = (
        sum(item[2] for item in error_sorted[:cut]) / max(1, cut)
        if cut > 0
        else 0.0
    )
    base_mean_uncertainty = (
        sum(item[2] for item in error_sorted[cut:]) / max(1, len(error_sorted[cut:]))
        if len(error_sorted) > cut
        else top_error_mean_uncertainty
    )

    diagnostics = {
        "schema": "p45_world_model_diagnostics_v1",
        "sample_count": len(rows),
        "uncertainty_error_pearson": pearson,
        "top_quartile_overlap_ratio": overlap_ratio,
        "top_quartile_error_mean_uncertainty": top_error_mean_uncertainty,
        "rest_mean_uncertainty": base_mean_uncertainty,
        "worst_samples": [
            {
                "sample_id": str(item[0].get("sample_id") or ""),
                "source_type": str(item[0].get("source_type") or ""),
                "phase_t": str(item[0].get("phase_t") or ""),
                "reward_t": _safe_float(item[0].get("reward_t"), 0.0),
                "reward_pred": _safe_float(item[0].get("reward_pred"), 0.0),
                "score_delta_t": _safe_float(item[0].get("score_delta_t"), 0.0),
                "score_pred": _safe_float(item[0].get("score_pred"), 0.0),
                "combined_error": _safe_float(item[1], 0.0),
                "uncertainty_pred": _safe_float(item[2], 0.0),
                "slice_labels": item[0].get("slice_labels") if isinstance(item[0].get("slice_labels"), dict) else {},
            }
            for item in error_sorted[: max(1, int(topk))]
        ],
    }

    lines = [
        "# P45 World Model Diagnostics",
        "",
        f"- sample_count: {int(diagnostics.get('sample_count') or 0)}",
        f"- uncertainty_error_pearson: {float(diagnostics.get('uncertainty_error_pearson') or 0.0):.4f}",
        f"- top_quartile_overlap_ratio: {float(diagnostics.get('top_quartile_overlap_ratio') or 0.0):.4f}",
        f"- top_quartile_error_mean_uncertainty: {float(diagnostics.get('top_quartile_error_mean_uncertainty') or 0.0):.6f}",
        f"- rest_mean_uncertainty: {float(diagnostics.get('rest_mean_uncertainty') or 0.0):.6f}",
        "",
        "## Worst Samples",
    ]
    for row in diagnostics.get("worst_samples") if isinstance(diagnostics.get("worst_samples"), list) else []:
        if not isinstance(row, dict):
            continue
        lines.append(
            "- sample={sample_id} source={source_type} phase={phase_t} error={combined_error:.6f} uncertainty={uncertainty_pred:.6f}".format(
                sample_id=row.get("sample_id"),
                source_type=row.get("source_type"),
                phase_t=row.get("phase_t"),
                combined_error=_safe_float(row.get("combined_error"), 0.0),
                uncertainty_pred=_safe_float(row.get("uncertainty_pred"), 0.0),
            )
        )
    return diagnostics, lines


def write_diagnostics(
    *,
    out_dir: str | Path,
    prediction_rows: list[dict[str, Any]],
    topk: int = 12,
) -> dict[str, Any]:
    target = Path(out_dir)
    target.mkdir(parents=True, exist_ok=True)
    payload, lines = build_diagnostics_report(prediction_rows, topk=topk)
    json_path = target / "diagnostics.json"
    md_path = target / "diagnostics_report.md"
    _write_json(json_path, payload)
    _write_markdown(md_path, lines)
    return {
        "diagnostics_json": str(json_path),
        "diagnostics_md": str(md_path),
        "payload": payload,
    }
