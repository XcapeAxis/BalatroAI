from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:  # pragma: no cover
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def read_list_json(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            return [x for x in payload if isinstance(x, dict)]
        return []
    except Exception:
        return []


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_mapping(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8").lstrip("\ufeff")
    if yaml is not None:
        payload = yaml.safe_load(text)
    else:
        payload = json.loads(text)
    if not isinstance(payload, dict):
        raise ValueError("ranking config must be mapping")
    return payload


def _minmax(values: list[float], value: float, *, invert: bool = False) -> float:
    if not values:
        return 0.0
    lo = min(values)
    hi = max(values)
    if hi - lo <= 1e-9:
        return 0.5
    norm = (value - lo) / (hi - lo)
    if invert:
        norm = 1.0 - norm
    return max(0.0, min(1.0, norm))


def _load_rows(run_root: Path) -> list[dict[str, Any]]:
    table_path = run_root / "summary_table.json"
    if table_path.exists():
        rows = read_list_json(table_path)
        if rows:
            return rows

    summary_path = run_root / "summary.json"
    payload = read_json(summary_path)
    rows = payload.get("rows") if isinstance(payload.get("rows"), list) else []
    if rows:
        return [x for x in rows if isinstance(x, dict)]
    return []


def _pareto_front(rows: list[dict[str, Any]]) -> set[str]:
    def dominates(a: dict[str, Any], b: dict[str, Any]) -> bool:
        ge_perf = (
            float(a.get("avg_ante_reached") or 0.0) >= float(b.get("avg_ante_reached") or 0.0)
            and float(a.get("median_ante") or 0.0) >= float(b.get("median_ante") or 0.0)
            and float(a.get("win_rate") or 0.0) >= float(b.get("win_rate") or 0.0)
        )
        le_cost = float(a.get("elapsed_sec") or 0.0) <= float(b.get("elapsed_sec") or 0.0)
        le_risk = float(a.get("risk_score") or 0.0) <= float(b.get("risk_score") or 0.0)
        strict = (
            float(a.get("avg_ante_reached") or 0.0) > float(b.get("avg_ante_reached") or 0.0)
            or float(a.get("median_ante") or 0.0) > float(b.get("median_ante") or 0.0)
            or float(a.get("win_rate") or 0.0) > float(b.get("win_rate") or 0.0)
            or float(a.get("elapsed_sec") or 0.0) < float(b.get("elapsed_sec") or 0.0)
            or float(a.get("risk_score") or 0.0) < float(b.get("risk_score") or 0.0)
        )
        return ge_perf and le_cost and le_risk and strict

    front: set[str] = set()
    for r in rows:
        dominated = False
        for other in rows:
            if str(other.get("exp_id")) == str(r.get("exp_id")):
                continue
            if dominates(other, r):
                dominated = True
                break
        if not dominated:
            front.add(str(r.get("exp_id")))
    return front


def build_ranking(run_root: Path, cfg: dict[str, Any]) -> dict[str, Any]:
    rows = _load_rows(run_root)
    gate = cfg.get("hard_filters") if isinstance(cfg.get("hard_filters"), dict) else {}

    min_avg = float(gate.get("min_avg_ante") or 0.0)
    min_median = float(gate.get("min_median_ante") or 0.0)
    min_win = float(gate.get("min_win_rate") or 0.0)

    prepared: list[dict[str, Any]] = []
    for row in rows:
        exp_id = str(row.get("exp_id") or row.get("strategy") or "")
        if not exp_id:
            continue
        avg_ante = float(row.get("avg_ante_reached") or row.get("mean") or 0.0)
        median_ante = float(row.get("median_ante") or row.get("median_ante_reached") or 0.0)
        win_rate = float(row.get("win_rate") or 0.0)
        elapsed = float(row.get("elapsed_sec") or row.get("runtime_seconds") or 0.0)
        status = str(row.get("status") or "passed")
        gate_pass = (
            status in {"passed", "success", "dry_run", "ok"}
            and avg_ante >= min_avg
            and median_ante >= min_median
            and win_rate >= min_win
        )
        prepared.append(
            {
                "exp_id": exp_id,
                "strategy": str(row.get("strategy") or ""),
                "status": status,
                "avg_ante_reached": avg_ante,
                "median_ante": median_ante,
                "win_rate": win_rate,
                "elapsed_sec": elapsed,
                "std": float(row.get("std") or 0.0),
                "risk_score": float(row.get("risk_score") or 0.0),
                "gate_pass": gate_pass,
                "model": str(row.get("model") or row.get("model_path") or ""),
                "rl_model": str(row.get("rl_model") or row.get("rl_model_path") or ""),
                "risk_config": str(row.get("risk_config") or ""),
                "run_dir": row.get("run_dir") or row.get("eval_json") or "",
            }
        )

    filtered = [r for r in prepared if bool(r.get("gate_pass"))]
    score_cfg = cfg.get("weighted_score") if isinstance(cfg.get("weighted_score"), dict) else {}
    weights = score_cfg.get("weights") if isinstance(score_cfg.get("weights"), dict) else {}
    w_perf = float(weights.get("performance", 0.45))
    w_stability = float(weights.get("stability", 0.20))
    w_cost = float(weights.get("cost", 0.20))
    w_risk = float(weights.get("risk", 0.15))

    avg_vals = [float(r["avg_ante_reached"]) for r in filtered] or [0.0]
    med_vals = [float(r["median_ante"]) for r in filtered] or [0.0]
    win_vals = [float(r["win_rate"]) for r in filtered] or [0.0]
    std_vals = [float(r["std"]) for r in filtered] or [0.0]
    elapsed_vals = [float(r["elapsed_sec"]) for r in filtered] or [0.0]
    risk_vals = [float(r["risk_score"]) for r in filtered] or [0.0]

    for row in prepared:
        if not row["gate_pass"]:
            row["weighted_score"] = -1.0
            continue
        perf = (
            _minmax(avg_vals, float(row["avg_ante_reached"]))
            + _minmax(med_vals, float(row["median_ante"]))
            + _minmax(win_vals, float(row["win_rate"]))
        ) / 3.0
        stability = _minmax(std_vals, float(row["std"]), invert=True)
        cost = _minmax(elapsed_vals, float(row["elapsed_sec"]), invert=True)
        risk = _minmax(risk_vals, float(row["risk_score"]), invert=True)
        row["weighted_score"] = (w_perf * perf) + (w_stability * stability) + (w_cost * cost) + (w_risk * risk)

    filtered_sorted = sorted(filtered, key=lambda r: float(r.get("weighted_score") or 0.0), reverse=True)
    fallback_sorted = sorted(
        [r for r in prepared if str(r.get("status") or "") in {"passed", "success", "dry_run", "ok"}],
        key=lambda r: (
            float(r.get("avg_ante_reached") or 0.0),
            float(r.get("median_ante") or 0.0),
            float(r.get("win_rate") or 0.0),
            -float(r.get("elapsed_sec") or 0.0),
        ),
        reverse=True,
    )
    if not filtered_sorted and fallback_sorted:
        # Degraded fallback: still provide candidate recommendations even when hard filters reject all rows.
        filtered_sorted = fallback_sorted
    pareto_set = _pareto_front(filtered)
    for row in prepared:
        row["pareto_front"] = str(row.get("exp_id")) in pareto_set

    top_candidate = filtered_sorted[0] if filtered_sorted else None
    conservative_candidate = None
    exploratory_candidate = None
    if filtered_sorted:
        conservative_candidate = sorted(
            filtered_sorted,
            key=lambda r: (
                float(r.get("risk_score") or 0.0),
                float(r.get("elapsed_sec") or 0.0),
                -float(r.get("weighted_score") or 0.0),
            ),
        )[0]
        exploratory_candidate = sorted(
            filtered_sorted,
            key=lambda r: (
                -(float(r.get("avg_ante_reached") or 0.0) + 0.4 * float(r.get("win_rate") or 0.0)),
                float(r.get("elapsed_sec") or 0.0),
            ),
        )[0]

    release_suggestion = "hold"
    if top_candidate is not None:
        if float(top_candidate.get("weighted_score") or 0.0) >= 0.55:
            release_suggestion = "investigate"
        if float(top_candidate.get("weighted_score") or 0.0) >= 0.70:
            release_suggestion = "promote"

    return {
        "schema": "p29_ranking_summary_v2",
        "generated_at": now_iso(),
        "run_root": str(run_root),
        "input_rows": len(rows),
        "filtered_rows": len(filtered),
        "weights": {
            "performance": w_perf,
            "stability": w_stability,
            "cost": w_cost,
            "risk": w_risk,
        },
        "rows": prepared,
        "recommendations": {
            "top_candidate": top_candidate,
            "conservative_candidate": conservative_candidate,
            "exploratory_candidate": exploratory_candidate,
            "pareto_front_exp_ids": sorted(pareto_set),
            "recommended_default_candidate": top_candidate,
            "recommended_conservative_candidate": conservative_candidate,
            "recommended_exploration_candidate": exploratory_candidate,
            "release_suggestion": release_suggestion,
            "rationale": {
                "performance": "weighted score combines avg/median ante and win_rate",
                "stability": "std and risk score penalize unstable candidates",
                "cost": "elapsed/runtime preferred when scores are close",
            },
        },
    }


def write_outputs(summary: dict[str, Any], out_dir: Path) -> dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "ranking_summary.json"
    md_path = out_dir / "ranking_summary.md"
    csv_path = out_dir / "ranking_table.csv"

    write_json(json_path, summary)

    fieldnames = [
        "exp_id",
        "strategy",
        "status",
        "gate_pass",
        "weighted_score",
        "pareto_front",
        "avg_ante_reached",
        "median_ante",
        "win_rate",
        "std",
        "elapsed_sec",
        "risk_score",
        "model",
        "rl_model",
        "risk_config",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary.get("rows") or []:
            writer.writerow({k: row.get(k) for k in fieldnames})

    recs = summary.get("recommendations") or {}
    lines = [
        "# P29 Ranking Summary",
        "",
        f"- input_rows: `{summary.get('input_rows')}`",
        f"- filtered_rows: `{summary.get('filtered_rows')}`",
        f"- release_suggestion: `{recs.get('release_suggestion')}`",
        "",
        "## Recommendations",
        f"- default: `{(recs.get('recommended_default_candidate') or {}).get('exp_id') if isinstance(recs.get('recommended_default_candidate'), dict) else 'N/A'}`",
        f"- conservative: `{(recs.get('recommended_conservative_candidate') or {}).get('exp_id') if isinstance(recs.get('recommended_conservative_candidate'), dict) else 'N/A'}`",
        f"- exploration: `{(recs.get('recommended_exploration_candidate') or {}).get('exp_id') if isinstance(recs.get('recommended_exploration_candidate'), dict) else 'N/A'}`",
        f"- pareto_front_exp_ids: `{recs.get('pareto_front_exp_ids')}`",
    ]
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"json": str(json_path), "md": str(md_path), "csv": str(csv_path)}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="P24/P29 multi-objective ranking")
    p.add_argument("--run-root", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--out-dir", required=True)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    run_root = Path(args.run_root).resolve()
    cfg = load_mapping(Path(args.config).resolve())
    summary = build_ranking(run_root, cfg)
    paths = write_outputs(summary, Path(args.out_dir).resolve())
    print(json.dumps({"status": "PASS", **paths}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

