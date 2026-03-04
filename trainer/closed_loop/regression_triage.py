from __future__ import annotations

if __package__ is None or __package__ == "":
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

import argparse
import json
from pathlib import Path
from typing import Any

from trainer.closed_loop.replay_manifest import now_iso, write_json, write_markdown


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _read_json(path: Path) -> dict[str, Any] | list[Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception:
        return None


def _pick_baseline_run(current_run_dir: Path) -> Path | None:
    parent = current_run_dir.parent
    if not parent.exists():
        return None
    dirs = sorted([p for p in parent.iterdir() if p.is_dir() and p.name != current_run_dir.name], key=lambda p: p.name)
    if not dirs:
        return None
    return dirs[-1]


def _load_slice_breakdown(decision_payload: dict[str, Any]) -> dict[str, Any]:
    path_raw = str(decision_payload.get("slice_decision_breakdown_json") or "").strip()
    if path_raw:
        path = Path(path_raw)
        if path.exists():
            payload = _read_json(path)
            if isinstance(payload, dict):
                return payload
    champion_rules = decision_payload.get("champion_rules_payload")
    if isinstance(champion_rules, dict):
        path_raw = str(champion_rules.get("slice_decision_breakdown_json") or "").strip()
        if path_raw:
            path = Path(path_raw)
            if path.exists():
                payload = _read_json(path)
                if isinstance(payload, dict):
                    return payload
    return {"rows": []}


def _curriculum_signature(candidate_manifest: dict[str, Any]) -> dict[str, Any]:
    plan = candidate_manifest.get("curriculum_plan") if isinstance(candidate_manifest.get("curriculum_plan"), dict) else {}
    phases = plan.get("phases") if isinstance(plan.get("phases"), list) else []
    rows: list[dict[str, Any]] = []
    for phase in phases:
        if not isinstance(phase, dict):
            continue
        rows.append(
            {
                "name": str(phase.get("name") or ""),
                "source_weights": phase.get("source_weights") if isinstance(phase.get("source_weights"), dict) else {},
                "slice_weights": phase.get("slice_weights") if isinstance(phase.get("slice_weights"), dict) else {},
            }
        )
    return {"phase_count": len(rows), "phases": rows}


def _source_attribution(replay_manifest: dict[str, Any], degraded_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    entries = replay_manifest.get("selected_entries") if isinstance(replay_manifest.get("selected_entries"), list) else []
    source_totals: dict[str, int] = {}
    source_degraded: dict[str, int] = {}
    degrade_matchers = []
    for row in degraded_rows:
        if not isinstance(row, dict):
            continue
        degrade_matchers.append((str(row.get("slice_key") or ""), str(row.get("slice_label") or "")))

    for entry in entries:
        if not isinstance(entry, dict):
            continue
        source_type = str(entry.get("source_type") or "unknown")
        sample_count = int(entry.get("sample_count") or 0)
        source_totals[source_type] = int(source_totals.get(source_type, 0)) + sample_count
        for slice_key, slice_label in degrade_matchers:
            if slice_key and str(entry.get(slice_key) if slice_key in entry else ((entry.get("slice_labels") or {}).get(slice_key) if isinstance(entry.get("slice_labels"), dict) else "")) == slice_label:
                source_degraded[source_type] = int(source_degraded.get(source_type, 0)) + sample_count
                break

    rows: list[dict[str, Any]] = []
    for source_type in sorted(source_totals.keys()):
        total = int(source_totals[source_type])
        deg = int(source_degraded.get(source_type, 0))
        rows.append(
            {
                "source_type": source_type,
                "sample_count": total,
                "degraded_slice_sample_count": deg,
                "degraded_slice_ratio": (float(deg) / total) if total > 0 else 0.0,
            }
        )
    rows.sort(key=lambda r: (-float(r.get("degraded_slice_sample_count") or 0.0), str(r.get("source_type") or "")))
    return rows


def _lineage_quality(replay_manifest: dict[str, Any]) -> dict[str, Any]:
    entries = replay_manifest.get("selected_entries") if isinstance(replay_manifest.get("selected_entries"), list) else []
    if not entries:
        return {"entry_count": 0, "invalid_ratio": 0.0}
    invalid = 0
    for entry in entries:
        if isinstance(entry, dict) and not bool(entry.get("valid_for_training", True)):
            invalid += 1
    return {"entry_count": len(entries), "invalid_ratio": float(invalid) / max(1, len(entries))}


def run_regression_triage(
    *,
    current_run_dir: str | Path,
    out_dir: str | Path | None = None,
    baseline_run_dir: str | Path | None = None,
) -> dict[str, Any]:
    current_dir = Path(current_run_dir).resolve()
    if not current_dir.exists():
        raise FileNotFoundError(f"current_run_dir not found: {current_dir}")

    baseline_dir: Path | None = None
    if baseline_run_dir:
        baseline_dir = Path(baseline_run_dir).resolve()
        if not baseline_dir.exists():
            baseline_dir = None
    if baseline_dir is None:
        baseline_dir = _pick_baseline_run(current_dir)

    current_decision = _read_json(current_dir / "promotion_decision.json")
    current_replay_ref = _read_json(current_dir / "replay_mix_manifest_ref.json")
    current_candidate_ref = _read_json(current_dir / "candidate_train_ref.json")
    current_manifest = _read_json(current_dir / "run_manifest.json")

    current_replay_manifest_path = ""
    if isinstance(current_replay_ref, dict):
        summary = current_replay_ref.get("summary") if isinstance(current_replay_ref.get("summary"), dict) else {}
        current_replay_manifest_path = str(summary.get("replay_mix_manifest") or "")
    current_replay_manifest = _read_json(Path(current_replay_manifest_path)) if current_replay_manifest_path else {}
    if not isinstance(current_replay_manifest, dict):
        current_replay_manifest = {}

    current_candidate_manifest_path = ""
    if isinstance(current_candidate_ref, dict):
        summary = current_candidate_ref.get("summary") if isinstance(current_candidate_ref.get("summary"), dict) else {}
        current_candidate_manifest_path = str(summary.get("candidate_train_manifest") or "")
    current_candidate_manifest = _read_json(Path(current_candidate_manifest_path)) if current_candidate_manifest_path else {}
    if not isinstance(current_candidate_manifest, dict):
        current_candidate_manifest = {}

    baseline_missing = baseline_dir is None
    baseline_decision: dict[str, Any] = {}
    baseline_candidate_manifest: dict[str, Any] = {}
    baseline_replay_manifest: dict[str, Any] = {}
    if baseline_dir is not None:
        payload = _read_json(baseline_dir / "promotion_decision.json")
        if isinstance(payload, dict):
            baseline_decision = payload
        replay_ref = _read_json(baseline_dir / "replay_mix_manifest_ref.json")
        if isinstance(replay_ref, dict):
            summary = replay_ref.get("summary") if isinstance(replay_ref.get("summary"), dict) else {}
            replay_path = str(summary.get("replay_mix_manifest") or "")
            replay_payload = _read_json(Path(replay_path)) if replay_path else None
            if isinstance(replay_payload, dict):
                baseline_replay_manifest = replay_payload
        cand_ref = _read_json(baseline_dir / "candidate_train_ref.json")
        if isinstance(cand_ref, dict):
            summary = cand_ref.get("summary") if isinstance(cand_ref.get("summary"), dict) else {}
            cand_path = str(summary.get("candidate_train_manifest") or "")
            cand_payload = _read_json(Path(cand_path)) if cand_path else None
            if isinstance(cand_payload, dict):
                baseline_candidate_manifest = cand_payload

    current_decision_dict = current_decision if isinstance(current_decision, dict) else {}
    baseline_decision_dict = baseline_decision if isinstance(baseline_decision, dict) else {}
    current_slice = _load_slice_breakdown(current_decision_dict)
    degraded_rows = [
        row
        for row in (current_slice.get("rows") if isinstance(current_slice.get("rows"), list) else [])
        if isinstance(row, dict) and bool((row.get("signals") or {}).get("degraded_significant"))
    ]
    degraded_rows = degraded_rows[:5]

    current_source_attr = _source_attribution(current_replay_manifest, degraded_rows)
    curr_signature = _curriculum_signature(current_candidate_manifest)
    prev_signature = _curriculum_signature(baseline_candidate_manifest)

    curriculum_change = {
        "current_phase_count": int(curr_signature.get("phase_count") or 0),
        "baseline_phase_count": int(prev_signature.get("phase_count") or 0),
        "phase_count_delta": int(curr_signature.get("phase_count") or 0) - int(prev_signature.get("phase_count") or 0),
        "current_phases": curr_signature.get("phases"),
        "baseline_phases": prev_signature.get("phases"),
    }

    current_quality = _lineage_quality(current_replay_manifest)
    baseline_quality = _lineage_quality(baseline_replay_manifest)
    quality_delta = _safe_float(current_quality.get("invalid_ratio")) - _safe_float(baseline_quality.get("invalid_ratio"))

    overall = {
        "current_candidate_score": _safe_float(current_decision_dict.get("candidate_score"), 0.0),
        "current_champion_score": _safe_float(current_decision_dict.get("champion_score"), 0.0),
        "current_score_delta": _safe_float(current_decision_dict.get("score_delta"), 0.0),
        "baseline_score_delta": _safe_float(baseline_decision_dict.get("score_delta"), 0.0),
        "score_delta_change_vs_baseline": _safe_float(current_decision_dict.get("score_delta"), 0.0)
        - _safe_float(baseline_decision_dict.get("score_delta"), 0.0),
        "recommendation": str(current_decision_dict.get("recommendation") or ""),
    }

    payload = {
        "schema": "p41_regression_triage_v1",
        "generated_at": now_iso(),
        "current_run_dir": str(current_dir),
        "baseline_run_dir": str(baseline_dir) if baseline_dir else "",
        "baseline_missing": baseline_missing,
        "overall": overall,
        "degraded_slices_topk": degraded_rows,
        "source_attribution": current_source_attr,
        "curriculum_change": curriculum_change,
        "data_quality": {
            "current": current_quality,
            "baseline": baseline_quality,
            "invalid_ratio_delta": quality_delta,
        },
        "refs": {
            "current_run_manifest": str(current_dir / "run_manifest.json"),
            "current_promotion_decision": str(current_dir / "promotion_decision.json"),
            "current_slice_breakdown": str(current_decision_dict.get("slice_decision_breakdown_json") or ""),
        },
    }

    output_dir = Path(out_dir).resolve() if out_dir else current_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "triage_report.json"
    md_path = output_dir / "triage_report.md"
    write_json(json_path, payload)

    md_lines = [
        "# P41 Regression Triage Report",
        "",
        f"- current_run_dir: `{payload.get('current_run_dir')}`",
        f"- baseline_run_dir: `{payload.get('baseline_run_dir')}`",
        f"- baseline_missing: `{payload.get('baseline_missing')}`",
        "",
        "## Overall",
        f"- current_score_delta: {float((overall or {}).get('current_score_delta') or 0.0):.6f}",
        f"- baseline_score_delta: {float((overall or {}).get('baseline_score_delta') or 0.0):.6f}",
        f"- score_delta_change_vs_baseline: {float((overall or {}).get('score_delta_change_vs_baseline') or 0.0):.6f}",
        f"- recommendation: `{(overall or {}).get('recommendation')}`",
        "",
        "## Degraded Slices (Top-K)",
    ]
    if degraded_rows:
        for row in degraded_rows:
            if not isinstance(row, dict):
                continue
            metrics = row.get("metrics") if isinstance(row.get("metrics"), dict) else {}
            md_lines.append(
                "- {slice_key}:{slice_label} score_delta={score:.6f} win_delta={win:.6f}".format(
                    slice_key=row.get("slice_key"),
                    slice_label=row.get("slice_label"),
                    score=float(metrics.get("mean_total_score_delta") or 0.0),
                    win=float(metrics.get("win_rate_delta") or 0.0),
                )
            )
    else:
        md_lines.append("- none")

    md_lines.extend(["", "## Source Attribution"])
    if current_source_attr:
        for row in current_source_attr:
            if not isinstance(row, dict):
                continue
            md_lines.append(
                "- {source}: samples={samples}, degraded_slice_samples={deg}, ratio={ratio:.3f}".format(
                    source=row.get("source_type"),
                    samples=int(row.get("sample_count") or 0),
                    deg=int(row.get("degraded_slice_sample_count") or 0),
                    ratio=float(row.get("degraded_slice_ratio") or 0.0),
                )
            )
    else:
        md_lines.append("- none")
    md_lines.extend(
        [
            "",
            "## Curriculum Change",
            f"- current_phase_count: {int(curriculum_change.get('current_phase_count') or 0)}",
            f"- baseline_phase_count: {int(curriculum_change.get('baseline_phase_count') or 0)}",
            f"- phase_count_delta: {int(curriculum_change.get('phase_count_delta') or 0)}",
            "",
            "## Data Quality",
            f"- current_invalid_ratio: {float((current_quality or {}).get('invalid_ratio') or 0.0):.6f}",
            f"- baseline_invalid_ratio: {float((baseline_quality or {}).get('invalid_ratio') or 0.0):.6f}",
            f"- invalid_ratio_delta: {float(quality_delta):.6f}",
        ]
    )
    write_markdown(md_path, md_lines)
    return {
        "status": "ok",
        "triage_report_json": str(json_path),
        "triage_report_md": str(md_path),
        "baseline_missing": bool(baseline_missing),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="P41 regression triage for closed-loop candidate degradation.")
    parser.add_argument("--current-run-dir", required=True)
    parser.add_argument("--out-dir", default="")
    parser.add_argument("--baseline-run-dir", default="")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = run_regression_triage(
        current_run_dir=args.current_run_dir,
        out_dir=(args.out_dir if str(args.out_dir).strip() else None),
        baseline_run_dir=(args.baseline_run_dir if str(args.baseline_run_dir).strip() else None),
    )
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

