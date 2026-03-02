from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def update_champion_candidate(
    out_root: Path,
    run_id: str,
    ranked_rows: list[dict[str, Any]],
    primary_metric: str,
    evaluation_cfg: dict[str, Any],
) -> dict[str, Any]:
    champion_path = out_root / "champion.json"
    candidate_path = out_root / "candidate.json"
    changelog_path = out_root / "CHANGELOG_P22.md"

    old_champion = _read_json(champion_path)
    winner = ranked_rows[0] if ranked_rows else None
    threshold = float((evaluation_cfg.get("promotion") or {}).get("min_delta") or 0.0)

    if winner is None:
        decision = "hold"
        reason = "no_experiment_results"
        candidate_payload = {
            "schema": "p22_candidate_v1",
            "run_id": run_id,
            "decision": decision,
            "reason": reason,
            "primary_metric": primary_metric,
        }
        _write_json(candidate_path, candidate_payload)
        return {
            "decision": decision,
            "reason": reason,
            "champion_path": str(champion_path),
            "candidate_path": str(candidate_path),
        }

    winner_metric = float(winner.get("mean") or 0.0)
    winner_success = str(winner.get("status")) == "success"
    old_metric = float(old_champion.get("mean")) if old_champion and isinstance(old_champion.get("mean"), (int, float)) else None

    if old_champion is None and winner_success:
        decision = "promote"
        reason = "bootstrap_no_previous_champion"
    elif (not winner_success):
        decision = "hold"
        reason = "winner_failed_gate"
    else:
        delta = winner_metric - (old_metric if old_metric is not None else 0.0)
        if old_metric is None or delta >= threshold:
            decision = "promote"
            reason = f"delta={delta:.6f} meets threshold={threshold:.6f}"
        else:
            decision = "hold"
            reason = f"delta={delta:.6f} below threshold={threshold:.6f}"

    candidate_payload = {
        "schema": "p22_candidate_v1",
        "run_id": run_id,
        "decision": decision,
        "reason": reason,
        "primary_metric": primary_metric,
        "winner": winner,
        "previous_champion": old_champion,
    }
    _write_json(candidate_path, candidate_payload)

    if decision == "promote":
        champion_payload = {
            "schema": "p22_champion_v1",
            "updated_by_run": run_id,
            "primary_metric": primary_metric,
            "mean": winner_metric,
            "exp_id": winner.get("exp_id"),
            "run_dir": winner.get("run_dir"),
            "status": "champion",
            "reason": reason,
        }
        _write_json(champion_path, champion_payload)

    log_lines = []
    if changelog_path.exists():
        log_lines = changelog_path.read_text(encoding="utf-8").splitlines()
    if not log_lines:
        log_lines = ["# P22 Champion/Candidate Changelog", ""]
    log_lines.append(f"## {run_id}")
    log_lines.append(f"- decision: {decision}")
    log_lines.append(f"- reason: {reason}")
    log_lines.append(f"- winner: {winner.get('exp_id') if winner else 'N/A'}")
    log_lines.append("")
    changelog_path.write_text("\n".join(log_lines).rstrip() + "\n", encoding="utf-8")

    return {
        "decision": decision,
        "reason": reason,
        "champion_path": str(champion_path),
        "candidate_path": str(candidate_path),
    }


def update_nightly_decision(
    out_root: Path,
    *,
    run_id: str,
    ranked_rows: list[dict[str, Any]],
    primary_metric: str,
    evaluation_cfg: dict[str, Any],
) -> dict[str, Any]:
    champion_path = out_root / "champion.json"
    candidate_path = out_root / "candidate.json"
    decision_path = out_root / "nightly_decision.json"
    decision_md_path = out_root / "nightly_decision.md"
    changelog_path = out_root / "CHANGELOG_P23.md"

    gate_thresholds = evaluation_cfg.get("gate_thresholds") if isinstance(evaluation_cfg.get("gate_thresholds"), dict) else {}
    min_avg = float(gate_thresholds.get("min_avg_ante") or 0.0)
    min_median = float(gate_thresholds.get("min_median_ante") or 0.0)
    min_win = float(gate_thresholds.get("min_win_rate") or 0.0)
    min_delta = float((evaluation_cfg.get("promotion") or {}).get("min_delta") or 0.0)

    old_champion = _read_json(champion_path)

    valid_rows: list[dict[str, Any]] = []
    for row in ranked_rows:
        if str(row.get("status")) not in {"passed", "success", "dry_run"}:
            continue
        avg_ante = float(row.get("avg_ante_reached") or row.get("mean") or 0.0)
        median_ante = float(row.get("median_ante") or 0.0)
        win_rate = float(row.get("win_rate") or 0.0)
        gate_pass = (avg_ante >= min_avg) and (median_ante >= min_median) and (win_rate >= min_win)
        enriched = dict(row)
        enriched["gate_pass"] = gate_pass
        valid_rows.append(enriched)

    candidate = None
    for row in valid_rows:
        if bool(row.get("gate_pass")):
            candidate = row
            break

    if candidate is None:
        decision = "hold"
        reason = "no_valid_candidate"
        champion_changed = False
    else:
        old_metric = float(old_champion.get("avg_ante_reached") or old_champion.get("mean") or 0.0) if old_champion else None
        candidate_metric = float(candidate.get("avg_ante_reached") or candidate.get("mean") or 0.0)
        if old_metric is None:
            decision = "promote"
            reason = "bootstrap_no_previous_champion"
            champion_changed = True
        else:
            delta = candidate_metric - old_metric
            if delta >= min_delta:
                decision = "promote"
                reason = f"delta_avg_ante={delta:.6f} meets threshold={min_delta:.6f}"
                champion_changed = True
            else:
                decision = "hold"
                reason = f"delta_avg_ante={delta:.6f} below threshold={min_delta:.6f}"
                champion_changed = False

    candidate_payload = {
        "schema": "p23_candidate_v1",
        "generated_at": _now_iso(),
        "run_id": run_id,
        "primary_metric": primary_metric,
        "candidate": candidate,
        "decision": decision,
        "reason": reason,
        "gate_thresholds": {
            "min_avg_ante": min_avg,
            "min_median_ante": min_median,
            "min_win_rate": min_win,
        },
    }
    _write_json(candidate_path, candidate_payload)

    champion_payload = old_champion if old_champion is not None else {}
    if candidate is not None and champion_changed:
        champion_payload = {
            "schema": "p23_champion_v1",
            "updated_at": _now_iso(),
            "updated_by_run": run_id,
            "primary_metric": primary_metric,
            "exp_id": candidate.get("exp_id"),
            "run_dir": candidate.get("run_dir"),
            "avg_ante_reached": float(candidate.get("avg_ante_reached") or candidate.get("mean") or 0.0),
            "median_ante": float(candidate.get("median_ante") or 0.0),
            "win_rate": float(candidate.get("win_rate") or 0.0),
            "mean": float(candidate.get("mean") or 0.0),
            "std": float(candidate.get("std") or 0.0),
            "status": "champion",
            "reason": reason,
        }
        _write_json(champion_path, champion_payload)
    elif old_champion is not None:
        champion_payload = old_champion

    decision_payload = {
        "schema": "p23_nightly_decision_v1",
        "generated_at": _now_iso(),
        "run_id": run_id,
        "decision": decision,
        "reason": reason,
        "primary_metric": primary_metric,
        "champion_changed": champion_changed,
        "candidate": candidate,
        "champion_before": old_champion,
        "champion_after": champion_payload if champion_payload else old_champion,
    }
    _write_json(decision_path, decision_payload)

    md_lines = [
        "# P23 Nightly Decision",
        "",
        f"- run_id: `{run_id}`",
        f"- decision: `{decision}`",
        f"- reason: {reason}",
        f"- champion_changed: `{champion_changed}`",
        f"- candidate: `{candidate.get('exp_id') if candidate else 'N/A'}`",
    ]
    decision_md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    changelog_lines = []
    if changelog_path.exists():
        changelog_lines = changelog_path.read_text(encoding="utf-8").splitlines()
    if not changelog_lines:
        changelog_lines = ["# P23 Champion/Candidate Changelog", ""]
    changelog_lines.append(f"## {run_id}")
    changelog_lines.append(f"- decision: {decision}")
    changelog_lines.append(f"- reason: {reason}")
    changelog_lines.append(f"- candidate: {candidate.get('exp_id') if candidate else 'N/A'}")
    changelog_lines.append("")
    changelog_path.write_text("\n".join(changelog_lines).rstrip() + "\n", encoding="utf-8")

    return {
        "decision": decision,
        "reason": reason,
        "candidate_path": str(candidate_path),
        "champion_path": str(champion_path),
        "nightly_decision_path": str(decision_path),
    }


def update_p24_from_ranking(
    *,
    out_root: Path,
    run_id: str,
    ranking_summary: dict[str, Any],
) -> dict[str, Any]:
    champion_path = out_root / "champion.json"
    candidate_path = out_root / "candidate.json"
    decision_path = out_root / "nightly_decision.json"
    decision_md_path = out_root / "nightly_decision.md"
    changelog_path = out_root / "CHANGELOG_P24.md"

    old_champion = _read_json(champion_path)
    recs = ranking_summary.get("recommendations") if isinstance(ranking_summary.get("recommendations"), dict) else {}
    top_candidate = recs.get("top_candidate") if isinstance(recs.get("top_candidate"), dict) else None
    conservative = recs.get("conservative_candidate") if isinstance(recs.get("conservative_candidate"), dict) else None
    exploratory = recs.get("exploratory_candidate") if isinstance(recs.get("exploratory_candidate"), dict) else None

    if top_candidate is None:
        decision = "hold"
        reason = "no_ranked_candidate"
        champion_changed = False
    else:
        top_score = float(top_candidate.get("weighted_score") or 0.0)
        old_score = 0.0
        if old_champion and isinstance(old_champion.get("weighted_score"), (int, float)):
            old_score = float(old_champion.get("weighted_score") or 0.0)
        if old_champion is None:
            decision = "promote"
            reason = "bootstrap_from_ranking"
            champion_changed = True
        elif top_score > old_score + 0.01:
            decision = "promote"
            reason = f"weighted_score delta={(top_score - old_score):.6f} > 0.01"
            champion_changed = True
        else:
            decision = "hold"
            reason = f"weighted_score delta={(top_score - old_score):.6f} <= 0.01"
            champion_changed = False

    candidate_payload = {
        "schema": "p24_candidate_v1",
        "generated_at": _now_iso(),
        "run_id": run_id,
        "decision": decision,
        "reason": reason,
        "top_candidate": top_candidate,
        "conservative_candidate": conservative,
        "exploratory_candidate": exploratory,
    }
    _write_json(candidate_path, candidate_payload)

    champion_payload = old_champion if old_champion else {}
    if champion_changed and top_candidate is not None:
        champion_payload = {
            "schema": "p24_champion_v1",
            "updated_at": _now_iso(),
            "updated_by_run": run_id,
            "exp_id": top_candidate.get("exp_id"),
            "run_dir": top_candidate.get("run_dir"),
            "weighted_score": float(top_candidate.get("weighted_score") or 0.0),
            "avg_ante_reached": float(top_candidate.get("avg_ante_reached") or 0.0),
            "median_ante": float(top_candidate.get("median_ante") or 0.0),
            "win_rate": float(top_candidate.get("win_rate") or 0.0),
            "reason": reason,
            "status": "champion",
        }
        _write_json(champion_path, champion_payload)

    decision_payload = {
        "schema": "p24_nightly_decision_v1",
        "generated_at": _now_iso(),
        "run_id": run_id,
        "decision": decision,
        "reason": reason,
        "champion_changed": champion_changed,
        "candidate": top_candidate,
        "conservative_candidate": conservative,
        "exploratory_candidate": exploratory,
        "champion_before": old_champion,
        "champion_after": champion_payload if champion_payload else old_champion,
    }
    _write_json(decision_path, decision_payload)

    md_lines = [
        "# P24 Nightly Recommendation Decision",
        "",
        f"- run_id: `{run_id}`",
        f"- decision: `{decision}`",
        f"- reason: {reason}",
        f"- champion_changed: `{champion_changed}`",
        f"- top_candidate: `{top_candidate.get('exp_id') if isinstance(top_candidate, dict) else 'N/A'}`",
    ]
    decision_md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    changelog_lines = []
    if changelog_path.exists():
        changelog_lines = changelog_path.read_text(encoding="utf-8").splitlines()
    if not changelog_lines:
        changelog_lines = ["# P24 Champion/Candidate Changelog", ""]
    changelog_lines.append(f"## {run_id}")
    changelog_lines.append(f"- decision: {decision}")
    changelog_lines.append(f"- reason: {reason}")
    changelog_lines.append(
        f"- top_candidate: {top_candidate.get('exp_id') if isinstance(top_candidate, dict) else 'N/A'}"
    )
    changelog_lines.append("")
    changelog_path.write_text("\n".join(changelog_lines).rstrip() + "\n", encoding="utf-8")

    return {
        "decision": decision,
        "reason": reason,
        "champion_path": str(champion_path),
        "candidate_path": str(candidate_path),
        "nightly_decision_path": str(decision_path),
        "changelog_path": str(changelog_path),
    }
