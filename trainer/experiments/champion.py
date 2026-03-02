from __future__ import annotations

import json
from pathlib import Path
from typing import Any


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

