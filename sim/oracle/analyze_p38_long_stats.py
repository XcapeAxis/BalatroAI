from __future__ import annotations

if __package__ is None or __package__ == "":
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

import argparse
import csv
import json
import math
import statistics
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from scipy.stats import chi2 as scipy_chi2  # type: ignore
except Exception:  # pragma: no cover
    scipy_chi2 = None


NUMERIC_METRICS = [
    "total_score",
    "rounds_survived",
    "money_earned",
    "rerolls_count",
    "packs_opened",
    "consumables_used",
]

CATEGORICAL_METRICS = [
    "shop_offer_keys",
    "pack_types",
    "joker_keys",
]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze P38 long-episode aggregate stats.")
    parser.add_argument("--fixtures-dir", required=True)
    parser.add_argument("--out-dir", default="")
    parser.add_argument("--warn-relative-pct", type=float, default=5.0)
    parser.add_argument("--warn-pvalue", type=float, default=0.01)
    return parser.parse_args()


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        obj = json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception:
        return None
    return obj if isinstance(obj, dict) else None


def _safe_float(value: Any) -> float | None:
    try:
        return float(value)
    except Exception:
        return None


def _stats(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"count": 0, "mean": None, "std": None, "min": None, "max": None}
    return {
        "count": len(values),
        "mean": float(statistics.mean(values)),
        "std": float(statistics.pstdev(values)) if len(values) >= 2 else 0.0,
        "min": float(min(values)),
        "max": float(max(values)),
    }


def _sum_counter(episodes: list[dict[str, Any]], side: str, key: str) -> Counter[str]:
    counter: Counter[str] = Counter()
    for ep in episodes:
        side_block = ep.get(side) if isinstance(ep.get(side), dict) else {}
        freq = side_block.get("frequencies") if isinstance(side_block.get("frequencies"), dict) else {}
        items = freq.get(key) if isinstance(freq.get(key), dict) else {}
        for raw_k, raw_v in items.items():
            k = str(raw_k)
            try:
                v = int(raw_v)
            except Exception:
                v = 0
            if v > 0:
                counter[k] += v
    return counter


def _counter_probs(counter: Counter[str]) -> dict[str, float]:
    total = float(sum(counter.values()))
    if total <= 0:
        return {}
    return {k: float(v) / total for k, v in sorted(counter.items()) if v > 0}


def _l1_distance(p: dict[str, float], q: dict[str, float]) -> float:
    keys = sorted(set(p.keys()) | set(q.keys()))
    if not keys:
        return 0.0
    return float(sum(abs(float(p.get(k, 0.0)) - float(q.get(k, 0.0))) for k in keys))


def _chi_square_stat(obs: Counter[str], exp: Counter[str]) -> tuple[float, int]:
    obs_total = float(sum(obs.values()))
    exp_total = float(sum(exp.values()))
    if obs_total <= 0.0 or exp_total <= 0.0:
        return 0.0, 0
    chi2 = 0.0
    nonzero_bins = 0
    keys = sorted(set(obs.keys()) | set(exp.keys()))
    for key in keys:
        expected = (float(exp.get(key, 0)) / exp_total) * obs_total
        if expected <= 1e-12:
            continue
        observed = float(obs.get(key, 0))
        chi2 += ((observed - expected) ** 2) / expected
        nonzero_bins += 1
    dof = max(0, nonzero_bins - 1)
    return float(chi2), int(dof)


def _chi_square_pvalue(chi2_stat: float, dof: int) -> float | None:
    if dof <= 0:
        return None
    if scipy_chi2 is None:
        return None
    try:
        return float(scipy_chi2.sf(chi2_stat, dof))
    except Exception:
        return None


def _numeric_metric_rows(episodes: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[str]]:
    rows: list[dict[str, Any]] = []
    warnings: list[str] = []
    for metric in NUMERIC_METRICS:
        oracle_values: list[float] = []
        sim_values: list[float] = []
        pair_diffs: list[float] = []
        for ep in episodes:
            oracle_block = ep.get("oracle") if isinstance(ep.get("oracle"), dict) else {}
            sim_block = ep.get("sim") if isinstance(ep.get("sim"), dict) else {}
            oracle_metrics = oracle_block.get("metrics") if isinstance(oracle_block.get("metrics"), dict) else {}
            sim_metrics = sim_block.get("metrics") if isinstance(sim_block.get("metrics"), dict) else {}
            ov = _safe_float(oracle_metrics.get(metric))
            sv = _safe_float(sim_metrics.get(metric))
            if ov is not None:
                oracle_values.append(float(ov))
            if sv is not None:
                sim_values.append(float(sv))
            if ov is not None and sv is not None:
                pair_diffs.append(float(sv - ov))

        oracle_stats = _stats(oracle_values)
        sim_stats = _stats(sim_values)
        oracle_mean = oracle_stats.get("mean")
        sim_mean = sim_stats.get("mean")
        mean_diff = None
        relative_diff_pct = None
        if isinstance(oracle_mean, (int, float)) and isinstance(sim_mean, (int, float)):
            mean_diff = float(sim_mean - oracle_mean)
            denom = abs(float(oracle_mean))
            if denom > 1e-9:
                relative_diff_pct = float((mean_diff / denom) * 100.0)

        pair_abs_mean = float(statistics.mean(abs(x) for x in pair_diffs)) if pair_diffs else None
        row = {
            "metric": metric,
            "oracle": oracle_stats,
            "sim": sim_stats,
            "mean_diff": mean_diff,
            "relative_diff_pct": relative_diff_pct,
            "pair_abs_diff_mean": pair_abs_mean,
            "pair_count": len(pair_diffs),
        }
        rows.append(row)

    return rows, warnings


def _categorical_rows(episodes: list[dict[str, Any]], warn_pvalue: float) -> tuple[list[dict[str, Any]], list[str]]:
    rows: list[dict[str, Any]] = []
    warnings: list[str] = []
    for metric in CATEGORICAL_METRICS:
        oracle_counter = _sum_counter(episodes, "oracle", metric)
        sim_counter = _sum_counter(episodes, "sim", metric)
        oracle_probs = _counter_probs(oracle_counter)
        sim_probs = _counter_probs(sim_counter)
        chi2_stat, dof = _chi_square_stat(sim_counter, oracle_counter)
        p_value = _chi_square_pvalue(chi2_stat, dof)
        if p_value is not None and p_value < float(warn_pvalue):
            warnings.append(
                f"{metric}: categorical chi-square p-value={p_value:.6g} < {float(warn_pvalue):.6g}"
            )
        rows.append(
            {
                "metric": metric,
                "oracle_total": int(sum(oracle_counter.values())),
                "sim_total": int(sum(sim_counter.values())),
                "oracle_unique": int(len(oracle_counter)),
                "sim_unique": int(len(sim_counter)),
                "l1_distance": float(_l1_distance(oracle_probs, sim_probs)),
                "chi_square_stat": float(chi2_stat),
                "chi_square_dof": int(dof),
                "p_value": p_value,
                "top_oracle": [{"key": k, "count": int(v)} for k, v in oracle_counter.most_common(12)],
                "top_sim": [{"key": k, "count": int(v)} for k, v in sim_counter.most_common(12)],
            }
        )
    return rows, warnings


def _distribution_rows(episodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for ep in episodes:
        oracle_metrics = ((ep.get("oracle") or {}).get("metrics") or {}) if isinstance(ep.get("oracle"), dict) else {}
        sim_metrics = ((ep.get("sim") or {}).get("metrics") or {}) if isinstance(ep.get("sim"), dict) else {}
        rows.append(
            {
                "episode_index": int(ep.get("episode_index") or 0),
                "seed": str(ep.get("seed") or ""),
                "status": str(ep.get("status") or "unknown"),
                "mismatch_count": int(ep.get("mismatch_count") or 0),
                "oracle_total_score": _safe_float(oracle_metrics.get("total_score")),
                "sim_total_score": _safe_float(sim_metrics.get("total_score")),
                "oracle_rounds_survived": _safe_float(oracle_metrics.get("rounds_survived")),
                "sim_rounds_survived": _safe_float(sim_metrics.get("rounds_survived")),
                "oracle_money_earned": _safe_float(oracle_metrics.get("money_earned")),
                "sim_money_earned": _safe_float(sim_metrics.get("money_earned")),
                "oracle_rerolls_count": _safe_float(oracle_metrics.get("rerolls_count")),
                "sim_rerolls_count": _safe_float(sim_metrics.get("rerolls_count")),
                "oracle_packs_opened": _safe_float(oracle_metrics.get("packs_opened")),
                "sim_packs_opened": _safe_float(sim_metrics.get("packs_opened")),
                "oracle_consumables_used": _safe_float(oracle_metrics.get("consumables_used")),
                "sim_consumables_used": _safe_float(sim_metrics.get("consumables_used")),
            }
        )
    return rows


def _write_distribution_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "episode_index",
        "seed",
        "status",
        "mismatch_count",
        "oracle_total_score",
        "sim_total_score",
        "oracle_rounds_survived",
        "sim_rounds_survived",
        "oracle_money_earned",
        "sim_money_earned",
        "oracle_rerolls_count",
        "sim_rerolls_count",
        "oracle_packs_opened",
        "sim_packs_opened",
        "oracle_consumables_used",
        "sim_consumables_used",
    ]
    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_md(
    path: Path,
    *,
    summary: dict[str, Any],
    numeric_rows: list[dict[str, Any]],
    categorical_rows: list[dict[str, Any]],
    warnings: list[str],
) -> None:
    lines: list[str] = []
    lines.append("# P38 Long-Horizon Statistical Summary")
    lines.append("")
    lines.append(f"- generated_at: {summary.get('generated_at')}")
    lines.append(f"- fixtures_dir: {summary.get('fixtures_dir')}")
    lines.append(f"- episodes_total: {summary.get('episodes_total')}")
    lines.append(f"- hard_fail_count: {summary.get('hard_fail_count')}")
    lines.append(f"- soft_warn_count: {summary.get('soft_warn_count')}")
    lines.append(f"- status: {summary.get('status')}")
    lines.append("")
    lines.append("## Numeric Metrics")
    lines.append("")
    lines.append("| metric | oracle_mean | sim_mean | mean_diff | relative_diff_pct |")
    lines.append("|---|---:|---:|---:|---:|")
    for row in numeric_rows:
        oracle_mean = row["oracle"].get("mean")
        sim_mean = row["sim"].get("mean")
        mean_diff = row.get("mean_diff")
        rel = row.get("relative_diff_pct")
        lines.append(
            "| {metric} | {om} | {sm} | {md} | {rd} |".format(
                metric=row.get("metric"),
                om=f"{float(oracle_mean):.6f}" if isinstance(oracle_mean, (int, float)) else "NA",
                sm=f"{float(sim_mean):.6f}" if isinstance(sim_mean, (int, float)) else "NA",
                md=f"{float(mean_diff):.6f}" if isinstance(mean_diff, (int, float)) else "NA",
                rd=f"{float(rel):.3f}%" if isinstance(rel, (int, float)) else "NA",
            )
        )
    lines.append("")
    lines.append("## Categorical Metrics")
    lines.append("")
    lines.append("| metric | oracle_total | sim_total | l1_distance | chi_square_stat | p_value |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for row in categorical_rows:
        p_value = row.get("p_value")
        lines.append(
            "| {metric} | {ot} | {st} | {l1:.6f} | {chi:.6f} | {pv} |".format(
                metric=row.get("metric"),
                ot=int(row.get("oracle_total") or 0),
                st=int(row.get("sim_total") or 0),
                l1=float(row.get("l1_distance") or 0.0),
                chi=float(row.get("chi_square_stat") or 0.0),
                pv=(f"{float(p_value):.6g}" if isinstance(p_value, (int, float)) else "NA"),
            )
        )
    lines.append("")
    lines.append("## Soft Warnings")
    if warnings:
        lines.extend([f"- {msg}" for msg in warnings])
    else:
        lines.append("- none")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = _parse_args()
    repo_root = Path(__file__).resolve().parents[2]

    fixtures_dir = Path(args.fixtures_dir)
    if not fixtures_dir.is_absolute():
        fixtures_dir = (repo_root / fixtures_dir).resolve()
    if not fixtures_dir.exists():
        raise SystemExit(f"fixtures dir not found: {fixtures_dir}")

    if args.out_dir:
        out_dir = Path(args.out_dir)
        if not out_dir.is_absolute():
            out_dir = (repo_root / out_dir).resolve()
    else:
        out_dir = (repo_root / "docs" / "artifacts" / "p38" / f"analysis_{_now_stamp()}").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    episode_files = sorted(fixtures_dir.glob("episode_*.json"))
    episodes: list[dict[str, Any]] = []
    for path in episode_files:
        obj = _load_json(path)
        if obj is None:
            continue
        episodes.append(obj)
    if not episodes:
        raise SystemExit(f"no episode_*.json found in {fixtures_dir}")

    numeric_rows, numeric_warnings = _numeric_metric_rows(episodes)
    warnings = list(numeric_warnings)
    for row in numeric_rows:
        rel = row.get("relative_diff_pct")
        if isinstance(rel, (int, float)) and abs(float(rel)) > float(args.warn_relative_pct):
            warnings.append(
                f"{row.get('metric')}: relative_diff_pct={float(rel):.3f}% > {float(args.warn_relative_pct):.3f}%"
            )

    categorical_rows, cat_warnings = _categorical_rows(episodes, warn_pvalue=float(args.warn_pvalue))
    warnings.extend(cat_warnings)

    hard_fail_count = 0
    for ep in episodes:
        mismatch = int(ep.get("mismatch_count") or 0)
        status = str(ep.get("status") or "").lower()
        if mismatch > 0 or status not in {"pass"}:
            hard_fail_count += 1

    distribution_rows = _distribution_rows(episodes)
    distribution_csv = out_dir / "distribution_table.csv"
    _write_distribution_csv(distribution_csv, distribution_rows)

    summary = {
        "schema": "p38_long_stats_summary_v1",
        "generated_at": _now_iso(),
        "fixtures_dir": str(fixtures_dir),
        "out_dir": str(out_dir),
        "episodes_total": int(len(episodes)),
        "hard_fail_count": int(hard_fail_count),
        "soft_warn_count": int(len(warnings)),
        "warn_thresholds": {
            "relative_diff_pct": float(args.warn_relative_pct),
            "categorical_p_value": float(args.warn_pvalue),
        },
        "numeric_metrics": numeric_rows,
        "categorical_metrics": categorical_rows,
        "soft_warnings": warnings,
        "distribution_table_csv": str(distribution_csv),
        "status": "PASS" if hard_fail_count == 0 else "FAIL",
    }

    summary_json = out_dir / "summary_stats.json"
    summary_md = out_dir / "summary_stats.md"
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _write_md(summary_md, summary=summary, numeric_rows=numeric_rows, categorical_rows=categorical_rows, warnings=warnings)

    print(
        json.dumps(
            {
                "status": summary["status"],
                "episodes_total": summary["episodes_total"],
                "hard_fail_count": summary["hard_fail_count"],
                "soft_warn_count": summary["soft_warn_count"],
                "summary_json": str(summary_json),
                "summary_md": str(summary_md),
                "distribution_csv": str(distribution_csv),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
