"""P29 candidate comparison and recommendation synthesis."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _candidate_key(row: dict[str, Any]) -> str:
    exp_id = str(row.get("exp_id") or "").strip()
    if exp_id:
        return exp_id
    return str(row.get("strategy") or "").strip()


def _extract_metrics(row: dict[str, Any]) -> dict[str, float]:
    return {
        "avg_ante_reached": float(row.get("avg_ante_reached") or 0.0),
        "median_ante_reached": float(row.get("median_ante_reached") or row.get("median_ante") or 0.0),
        "win_rate": float(row.get("win_rate") or 0.0),
        "runtime_seconds": float(row.get("runtime_seconds") or row.get("elapsed_sec") or 0.0),
    }


def _load_eval_summary(root: Path) -> dict[str, Any]:
    summary_path = root / "summary.json"
    payload = read_json(summary_path)
    rows = payload.get("rows") if isinstance(payload.get("rows"), list) else []
    parsed_rows: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        key = _candidate_key(row)
        if not key:
            continue
        parsed_rows.append({
            "key": key,
            "strategy": str(row.get("strategy") or ""),
            "exp_id": str(row.get("exp_id") or key),
            "status": str(row.get("status") or "passed"),
            "metrics": _extract_metrics(row),
            "row": row,
        })
    return {
        "path": str(summary_path),
        "episodes": int(payload.get("episodes") or 0),
        "baseline": str(payload.get("baseline") or ""),
        "rows": parsed_rows,
    }


def _find_champion_key(rows: list[dict[str, Any]]) -> str:
    for row in rows:
        key = row.get("key")
        sk = str(key).lower()
        if "champion" in sk:
            return str(key)
    for row in rows:
        if str(row.get("strategy") or "") == "champion":
            return str(row.get("key"))
    return str(rows[0].get("key")) if rows else ""


def _rank_by_scale(scale_payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows = scale_payload.get("rows") if isinstance(scale_payload.get("rows"), list) else []
    if not rows:
        return []
    champion_key = _find_champion_key(rows)
    champion = next((r for r in rows if str(r.get("key")) == champion_key), rows[0])
    base = champion.get("metrics") if isinstance(champion.get("metrics"), dict) else {}

    ranked: list[dict[str, Any]] = []
    for row in rows:
        if str(row.get("key")) == champion_key:
            continue
        m = row.get("metrics") if isinstance(row.get("metrics"), dict) else {}
        d_avg = float(m.get("avg_ante_reached") or 0.0) - float(base.get("avg_ante_reached") or 0.0)
        d_med = float(m.get("median_ante_reached") or 0.0) - float(base.get("median_ante_reached") or 0.0)
        d_win = float(m.get("win_rate") or 0.0) - float(base.get("win_rate") or 0.0)
        d_rt = float(base.get("runtime_seconds") or 0.0) - float(m.get("runtime_seconds") or 0.0)
        score = (0.45 * d_avg) + (0.30 * d_med) + (0.20 * d_win) + (0.05 * d_rt)
        ranked.append(
            {
                "exp_id": row.get("exp_id"),
                "key": row.get("key"),
                "strategy": row.get("strategy"),
                "deltas_vs_champion": {
                    "avg_ante_reached": d_avg,
                    "median_ante_reached": d_med,
                    "win_rate": d_win,
                    "runtime_seconds": -d_rt,
                },
                "score_vs_champion": score,
                "metrics": m,
            }
        )
    return sorted(ranked, key=lambda x: float(x.get("score_vs_champion") or 0.0), reverse=True)


def _pick_recommendations(
    *,
    ranking_summary: dict[str, Any],
    scale_ranked: list[dict[str, Any]],
    flake_pass: bool,
) -> dict[str, Any]:
    recs = ranking_summary.get("recommendations") if isinstance(ranking_summary.get("recommendations"), dict) else {}

    default = recs.get("top_candidate") if isinstance(recs.get("top_candidate"), dict) else {}
    conservative = recs.get("conservative_candidate") if isinstance(recs.get("conservative_candidate"), dict) else {}
    exploration = recs.get("exploratory_candidate") if isinstance(recs.get("exploratory_candidate"), dict) else {}

    if not default and scale_ranked:
        default = scale_ranked[0]
    if not conservative and scale_ranked:
        conservative = sorted(scale_ranked, key=lambda r: float((r.get("deltas_vs_champion") or {}).get("runtime_seconds") or 0.0), reverse=True)[0]
    if not exploration and scale_ranked:
        exploration = sorted(scale_ranked, key=lambda r: float((r.get("deltas_vs_champion") or {}).get("avg_ante_reached") or 0.0), reverse=True)[0]

    default_exp = str(default.get("exp_id") or default.get("key") or "")
    default_delta_avg = 0.0
    for row in scale_ranked:
        if str(row.get("exp_id") or row.get("key")) == default_exp:
            default_delta_avg = float((row.get("deltas_vs_champion") or {}).get("avg_ante_reached") or 0.0)
            break

    if default_delta_avg > 0.05 and flake_pass:
        action = "promote"
    elif default_delta_avg > 0.0:
        action = "investigate"
    else:
        action = "hold"

    return {
        "recommended_default_candidate": default,
        "recommended_conservative_candidate": conservative,
        "recommended_exploration_candidate": exploration,
        "release_suggestion": action,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="P29 candidate compare")
    p.add_argument("--eval-root", required=True)
    p.add_argument("--ranking-root", required=True)
    p.add_argument("--out-dir", required=True)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    eval_root = Path(args.eval_root).resolve()
    ranking_root = Path(args.ranking_root).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    scale_dirs = {
        "ablation_100": eval_root / "ablation_100",
        "ablation_500": eval_root / "ablation_500",
        "ablation_1000": eval_root / "ablation_1000",
    }
    scales: dict[str, Any] = {}
    for key, path in scale_dirs.items():
        if path.exists():
            scales[key] = _load_eval_summary(path)

    ranking_summary = read_json(ranking_root / "ranking_summary.json")

    best_scale_key = ""
    if "ablation_1000" in scales:
        best_scale_key = "ablation_1000"
    elif "ablation_500" in scales:
        best_scale_key = "ablation_500"
    elif "ablation_100" in scales:
        best_scale_key = "ablation_100"

    ranked = _rank_by_scale(scales.get(best_scale_key, {})) if best_scale_key else []

    flake_report = read_json(eval_root.parent / "flake" / "best_candidate_latest" / "best_candidate_flake_report.json")
    flake_pass = str(flake_report.get("status") or "PASS").upper() == "PASS"

    recs = _pick_recommendations(ranking_summary=ranking_summary, scale_ranked=ranked, flake_pass=flake_pass)
    suggestion = str(recs.get("release_suggestion") or "hold")

    report = {
        "schema": "p29_candidate_compare_summary_v1",
        "generated_at": now_iso(),
        "eval_root": str(eval_root),
        "ranking_root": str(ranking_root),
        "scales_available": sorted(scales.keys()),
        "best_scale_used": best_scale_key,
        "best_scale_ranked": ranked,
        "flake_pass": flake_pass,
        "recommendations": recs,
        "decision": {
            "action": suggestion,
            "reason": {
                "performance": "top candidate deltas from largest completed scale",
                "stability": "flake gate consulted when available",
                "risk": "fallback to hold/investigate if flake fails or weak deltas",
                "cost": "runtime delta incorporated in score_vs_champion",
            },
        },
    }

    json_path = out_dir / "candidate_compare_summary.json"
    md_path = out_dir / "candidate_compare_summary.md"
    write_json(json_path, report)

    lines = [
        "# P29 Candidate Compare Summary",
        "",
        f"- best_scale_used: `{best_scale_key}`",
        f"- flake_pass: `{flake_pass}`",
        f"- release_suggestion: `{suggestion}`",
        "",
        "## Recommendations",
        f"- default: `{((recs.get('recommended_default_candidate') or {}).get('exp_id') or (recs.get('recommended_default_candidate') or {}).get('key') or 'N/A')}`",
        f"- conservative: `{((recs.get('recommended_conservative_candidate') or {}).get('exp_id') or (recs.get('recommended_conservative_candidate') or {}).get('key') or 'N/A')}`",
        f"- exploration: `{((recs.get('recommended_exploration_candidate') or {}).get('exp_id') or (recs.get('recommended_exploration_candidate') or {}).get('key') or 'N/A')}`",
        "",
        "## Ranked Candidates",
    ]
    for row in ranked[:10]:
        d = row.get("deltas_vs_champion") if isinstance(row.get("deltas_vs_champion"), dict) else {}
        lines.append(
            f"- {row.get('exp_id') or row.get('key')}: score={round(float(row.get('score_vs_champion') or 0.0), 6)} | d_avg={round(float(d.get('avg_ante_reached') or 0.0), 6)} | d_median={round(float(d.get('median_ante_reached') or 0.0), 6)} | d_win={round(float(d.get('win_rate') or 0.0), 6)}"
        )

    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(json.dumps({"status": "PASS", "out_json": str(json_path), "out_md": str(md_path), "release_suggestion": suggestion}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
