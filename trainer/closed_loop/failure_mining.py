from __future__ import annotations

if __package__ is None or __package__ == "":
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

try:  # pragma: no cover - optional dependency
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None

from trainer.closed_loop.replay_manifest import now_iso, now_stamp, read_json, write_json, write_markdown


def _read_yaml_or_json(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8-sig")
    if path.suffix.lower() in {".yml", ".yaml"}:
        if yaml is not None:
            payload = yaml.safe_load(text)
        else:
            try:
                payload = json.loads(text)
            except Exception:
                sidecar = path.with_suffix(".json")
                if sidecar.exists():
                    payload = json.loads(sidecar.read_text(encoding="utf-8-sig"))
                else:
                    raise RuntimeError(f"PyYAML unavailable and no sidecar JSON for: {path}")
    else:
        payload = json.loads(text)
    if not isinstance(payload, dict):
        raise ValueError(f"config must be mapping: {path}")
    return payload


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _quantile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    qq = min(1.0, max(0.0, float(q)))
    pos = (len(ordered) - 1) * qq
    lo = int(pos)
    hi = min(len(ordered) - 1, lo + 1)
    frac = pos - lo
    return float(ordered[lo] * (1.0 - frac) + ordered[hi] * frac)


def _read_jsonl_with_lineno(path: Path, *, max_rows: int = 0) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if not path.exists():
        return out
    with path.open("r", encoding="utf-8", errors="replace") as fp:
        for idx, line in enumerate(fp, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                obj = json.loads(text)
            except Exception:
                continue
            if not isinstance(obj, dict):
                continue
            obj["_source_line"] = idx
            out.append(obj)
            if max_rows > 0 and len(out) >= max_rows:
                break
    return out


def _latest_dir(path: Path) -> Path | None:
    if not path.exists():
        return None
    dirs = sorted([p for p in path.iterdir() if p.is_dir()], key=lambda p: p.name)
    if not dirs:
        return None
    return dirs[-1]


def _pick_policy_row(summary_rows: list[dict[str, Any]], policy_id: str | None) -> dict[str, Any] | None:
    if not policy_id:
        return None
    token = str(policy_id).strip().lower()
    for row in summary_rows:
        if str(row.get("policy_id") or "").strip().lower() == token:
            return row
    return None


def _load_summary_rows(path: Path) -> list[dict[str, Any]]:
    payload = read_json(path)
    if not isinstance(payload, list):
        return []
    return [row for row in payload if isinstance(row, dict)]


def _build_markdown(
    *,
    run_id: str,
    status: str,
    arena_run_dir: Path | None,
    selected_total: int,
    counters_by_type: dict[str, int],
    counters_by_policy: dict[str, int],
    counters_by_seed: dict[str, int],
    warnings: list[str],
) -> list[str]:
    lines = [
        f"# P40 Failure Mining ({run_id})",
        "",
        f"- status: `{status}`",
        f"- arena_run_dir: `{str(arena_run_dir) if arena_run_dir else ''}`",
        f"- selected_failures: `{selected_total}`",
        "",
        "## Failure Type Distribution",
    ]
    if counters_by_type:
        lines.extend([f"- {k}: {v}" for k, v in sorted(counters_by_type.items(), key=lambda kv: (-kv[1], kv[0]))])
    else:
        lines.append("- none")
    lines.extend(["", "## Policy Distribution"])
    if counters_by_policy:
        lines.extend([f"- {k}: {v}" for k, v in sorted(counters_by_policy.items(), key=lambda kv: (-kv[1], kv[0]))])
    else:
        lines.append("- none")
    lines.extend(["", "## Seed Distribution"])
    if counters_by_seed:
        lines.extend([f"- {k}: {v}" for k, v in sorted(counters_by_seed.items(), key=lambda kv: (-kv[1], kv[0]))])
    else:
        lines.append("- none")
    if warnings:
        lines.extend(["", "## Warnings"])
        lines.extend([f"- {w}" for w in warnings])
    return lines


def run_failure_mining(
    *,
    config_path: str | Path | None = None,
    out_dir: str | Path | None = None,
    run_id: str = "",
    quick: bool = False,
    dry_run: bool = False,
    arena_run_dir_override: str | Path | None = None,
) -> dict[str, Any]:
    repo_root = Path(__file__).resolve().parents[2]
    cfg: dict[str, Any] = {}
    cfg_path: Path | None = None
    if config_path:
        cfg_path = Path(config_path)
        if not cfg_path.is_absolute():
            cfg_path = (repo_root / cfg_path).resolve()
        cfg = _read_yaml_or_json(cfg_path)

    output_cfg = cfg.get("output") if isinstance(cfg.get("output"), dict) else {}
    artifacts_root = str(output_cfg.get("artifacts_root") or "docs/artifacts/p40/failure_mining")
    chosen_run_id = str(run_id or output_cfg.get("run_id") or now_stamp())
    if out_dir:
        run_dir = Path(out_dir)
        if not run_dir.is_absolute():
            run_dir = (repo_root / run_dir).resolve()
    else:
        run_dir = (repo_root / artifacts_root).resolve() / chosen_run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    input_cfg = cfg.get("input") if isinstance(cfg.get("input"), dict) else {}
    p39_root_raw = str(input_cfg.get("p39_root") or "docs/artifacts/p39")
    p39_root = Path(p39_root_raw)
    if not p39_root.is_absolute():
        p39_root = (repo_root / p39_root).resolve()

    arena_run_dir: Path | None = None
    if arena_run_dir_override:
        arena_run_dir = Path(arena_run_dir_override)
        if not arena_run_dir.is_absolute():
            arena_run_dir = (repo_root / arena_run_dir).resolve()
    else:
        arena_cfg = str(input_cfg.get("arena_run_dir") or "").strip()
        if arena_cfg:
            arena_run_dir = Path(arena_cfg)
            if not arena_run_dir.is_absolute():
                arena_run_dir = (repo_root / arena_run_dir).resolve()
        else:
            arena_run_dir = _latest_dir(p39_root / "arena_runs")

    criteria_cfg = cfg.get("criteria") if isinstance(cfg.get("criteria"), dict) else {}
    bottom_q = float(criteria_cfg.get("bottom_quantile") or 0.2)
    score_reg_threshold = float(criteria_cfg.get("champion_score_regression_ratio") or 0.05)
    high_risk_round_threshold = int(criteria_cfg.get("high_risk_round_threshold") or 2)
    max_failures = int(criteria_cfg.get("max_failures") or 1200)

    quick_cfg = cfg.get("quick") if isinstance(cfg.get("quick"), dict) else {}
    max_episode_scan = int(quick_cfg.get("max_episode_scan") or 240) if quick else 0

    warnings: list[str] = []

    if arena_run_dir is None or not arena_run_dir.exists():
        manifest = {
            "schema": "p40_failure_pack_manifest_v1",
            "generated_at": now_iso(),
            "run_id": chosen_run_id,
            "status": "stub",
            "reason": "p39 arena run directory unavailable",
            "paths": {},
            "failures": [],
            "warnings": warnings,
            "replay_jsonl_path": str(run_dir / "hard_failure_replay.jsonl"),
        }
        stats = {
            "schema": "p40_failure_pack_stats_v1",
            "generated_at": now_iso(),
            "run_id": chosen_run_id,
            "status": "stub",
            "selected_failures": 0,
            "by_type": {},
            "by_policy": {},
            "by_seed": {},
        }
        write_json(run_dir / "failure_pack_manifest.json", manifest)
        write_json(run_dir / "failure_pack_stats.json", stats)
        write_markdown(
            run_dir / "failure_pack_stats.md",
            _build_markdown(
                run_id=chosen_run_id,
                status="stub",
                arena_run_dir=arena_run_dir,
                selected_total=0,
                counters_by_type={},
                counters_by_policy={},
                counters_by_seed={},
                warnings=warnings,
            ),
        )
        return {
            "status": "stub",
            "run_id": chosen_run_id,
            "run_dir": str(run_dir),
            "failure_pack_manifest": str(run_dir / "failure_pack_manifest.json"),
            "failure_pack_stats": str(run_dir / "failure_pack_stats.json"),
            "selected_failures": 0,
        }

    episode_records_path = arena_run_dir / "episode_records.jsonl"
    summary_path = arena_run_dir / "summary_table.json"
    bucket_path = arena_run_dir / "bucket_metrics.json"
    candidate_decision_path = None
    explicit_decision_path = str(input_cfg.get("candidate_decision_json") or "").strip()
    if explicit_decision_path:
        candidate_decision_path = Path(explicit_decision_path)
        if not candidate_decision_path.is_absolute():
            candidate_decision_path = (repo_root / candidate_decision_path).resolve()
    else:
        latest_eval = _latest_dir(p39_root)
        if latest_eval and latest_eval.name.startswith("champion_eval_"):
            path = latest_eval / "candidate_decision.json"
            if path.exists():
                candidate_decision_path = path

    rows = _read_jsonl_with_lineno(episode_records_path, max_rows=max_episode_scan)
    summary_rows = _load_summary_rows(summary_path)
    bucket_metrics = read_json(bucket_path) if bucket_path.exists() else None
    candidate_decision = (
        read_json(candidate_decision_path)
        if candidate_decision_path is not None and candidate_decision_path.exists()
        else None
    )

    if not rows:
        warnings.append(f"episode records missing or empty: {episode_records_path}")

    candidate_policy = str(criteria_cfg.get("candidate_policy") or "").strip()
    champion_policy = str(criteria_cfg.get("champion_policy") or "").strip()
    if not candidate_policy and isinstance(candidate_decision, dict):
        candidate_policy = str(candidate_decision.get("candidate_policy_id") or "").strip()
    if not champion_policy and isinstance(candidate_decision, dict):
        champion_policy = str(candidate_decision.get("champion_policy_id") or "").strip()
    if not candidate_policy and summary_rows:
        candidate_policy = str(summary_rows[0].get("policy_id") or "").strip()

    candidate_row = _pick_policy_row(summary_rows, candidate_policy) if candidate_policy else None
    champion_row = _pick_policy_row(summary_rows, champion_policy) if champion_policy else None

    candidate_rows = [r for r in rows if str(r.get("policy_id") or "") == candidate_policy] if candidate_policy else list(rows)
    working_rows = candidate_rows if candidate_rows else list(rows)
    if candidate_policy and not candidate_rows:
        warnings.append(f"candidate policy not present in episode records: {candidate_policy}")

    scores = [_safe_float(r.get("total_score")) for r in working_rows]
    low_threshold = _quantile(scores, bottom_q) if scores else 0.0
    candidate_mean = _safe_float((candidate_row or {}).get("mean_total_score"), 0.0)
    champion_mean = _safe_float((champion_row or {}).get("mean_total_score"), 0.0)
    champion_regression = False
    if candidate_row is not None and champion_row is not None and champion_mean > 0.0:
        champion_regression = candidate_mean < champion_mean * (1.0 - score_reg_threshold)
    elif isinstance(candidate_decision, dict):
        decision_token = str(candidate_decision.get("decision") or "").lower()
        if decision_token in {"hold", "reject"}:
            champion_regression = True

    selected: list[dict[str, Any]] = []
    by_type: Counter[str] = Counter()
    by_policy: Counter[str] = Counter()
    by_seed: Counter[str] = Counter()
    bucket_fail_counter: Counter[str] = Counter()

    for row in working_rows:
        failure_types: set[str] = set()
        status = str(row.get("status") or "unknown").lower()
        error = str(row.get("error") or "").strip().lower()
        score = _safe_float(row.get("total_score"), 0.0)
        rounds_survived = _safe_int(row.get("rounds_survived"), 0)
        invalid_rate = _safe_float(row.get("invalid_action_rate"), 0.0)
        timeout_rate = _safe_float(row.get("timeout_rate"), 0.0)

        if status != "ok":
            failure_types.add("episode_failure_status")
        if invalid_rate > 0.0:
            failure_types.add("invalid_action")
        if timeout_rate > 0.0:
            failure_types.add("timeout")
        if "exception" in error or "failed" in error or "timeout" in error:
            failure_types.add("execution_error")
        if score <= low_threshold:
            failure_types.add("low_score_quantile")

        risk_counts = {}
        if isinstance(row.get("bucket_counts"), dict):
            raw_risk = (row.get("bucket_counts") or {}).get("risk")
            if isinstance(raw_risk, dict):
                risk_counts = {str(k): _safe_int(v) for k, v in raw_risk.items()}
        if risk_counts.get("resource_tight", 0) > 0 and (rounds_survived <= high_risk_round_threshold or score <= low_threshold):
            failure_types.add("high_risk_bucket_failure")
            bucket_fail_counter["resource_tight"] += 1

        if champion_regression and score <= candidate_mean:
            failure_types.add("champion_regression_segment")

        if not failure_types:
            continue

        episode_id = "{policy}|{seed}|{ep}".format(
            policy=str(row.get("policy_id") or ""),
            seed=str(row.get("seed") or ""),
            ep=_safe_int(row.get("episode_index"), 0),
        )
        payload = {
            "episode_id": episode_id,
            "policy_id": str(row.get("policy_id") or ""),
            "seed": str(row.get("seed") or ""),
            "episode_index": _safe_int(row.get("episode_index"), 0),
            "status": str(row.get("status") or ""),
            "error": str(row.get("error") or ""),
            "total_score": score,
            "rounds_survived": rounds_survived,
            "invalid_action_rate": invalid_rate,
            "timeout_rate": timeout_rate,
            "failure_types": sorted(failure_types),
            "source": {
                "episode_records": str(episode_records_path),
                "line": _safe_int(row.get("_source_line"), 0),
            },
            "raw_episode": row,
        }
        selected.append(payload)
        for token in payload["failure_types"]:
            by_type[token] += 1
        by_policy[payload["policy_id"]] += 1
        by_seed[payload["seed"]] += 1

        if len(selected) >= max(1, max_failures):
            warnings.append(f"failure cap reached: {max_failures}")
            break

    replay_jsonl_path = run_dir / "hard_failure_replay.jsonl"
    if not dry_run:
        with replay_jsonl_path.open("w", encoding="utf-8", newline="\n") as fp:
            for item in selected:
                # Keep replay pack concise but still reusable for downstream mixers.
                replay_row = {
                    "schema": "p40_hard_failure_row_v1",
                    "episode_id": item["episode_id"],
                    "policy_id": item["policy_id"],
                    "seed": item["seed"],
                    "episode_index": item["episode_index"],
                    "failure_types": item["failure_types"],
                    "status": item["status"],
                    "error": item["error"],
                    "total_score": item["total_score"],
                    "rounds_survived": item["rounds_survived"],
                    "invalid_action_rate": item["invalid_action_rate"],
                    "timeout_rate": item["timeout_rate"],
                    "source": item["source"],
                    "episode_record": item["raw_episode"],
                }
                fp.write(json.dumps(replay_row, ensure_ascii=False) + "\n")

    status = "ok" if selected else "stub"
    manifest = {
        "schema": "p40_failure_pack_manifest_v1",
        "generated_at": now_iso(),
        "run_id": chosen_run_id,
        "status": status,
        "config_path": str(cfg_path) if cfg_path else "",
        "arena_run_dir": str(arena_run_dir),
        "criteria": {
            "bottom_quantile": bottom_q,
            "champion_score_regression_ratio": score_reg_threshold,
            "high_risk_round_threshold": high_risk_round_threshold,
            "max_failures": max_failures,
        },
        "inputs": {
            "episode_records": str(episode_records_path),
            "summary_table": str(summary_path),
            "bucket_metrics": str(bucket_path),
            "candidate_decision": str(candidate_decision_path) if candidate_decision_path else "",
        },
        "candidate_policy": candidate_policy,
        "champion_policy": champion_policy,
        "candidate_mean_total_score": candidate_mean,
        "champion_mean_total_score": champion_mean,
        "champion_regression_detected": champion_regression,
        "low_score_threshold": low_threshold,
        "selected_count": len(selected),
        "failures": [
            {
                "episode_id": item["episode_id"],
                "policy_id": item["policy_id"],
                "seed": item["seed"],
                "episode_index": item["episode_index"],
                "failure_types": item["failure_types"],
                "total_score": item["total_score"],
                "status": item["status"],
                "error": item["error"],
                "source": item["source"],
            }
            for item in selected
        ],
        "replay_jsonl_path": str(replay_jsonl_path),
        "warnings": warnings,
    }
    stats = {
        "schema": "p40_failure_pack_stats_v1",
        "generated_at": now_iso(),
        "run_id": chosen_run_id,
        "status": status,
        "selected_failures": len(selected),
        "by_type": dict(sorted(by_type.items())),
        "by_policy": dict(sorted(by_policy.items())),
        "by_seed": dict(sorted(by_seed.items())),
        "risk_bucket_failures": dict(sorted(bucket_fail_counter.items())),
        "scan_rows": len(rows),
        "working_rows": len(working_rows),
        "candidate_policy": candidate_policy,
        "champion_policy": champion_policy,
    }
    md_lines = _build_markdown(
        run_id=chosen_run_id,
        status=status,
        arena_run_dir=arena_run_dir,
        selected_total=len(selected),
        counters_by_type=dict(by_type),
        counters_by_policy=dict(by_policy),
        counters_by_seed=dict(by_seed),
        warnings=warnings,
    )

    write_json(run_dir / "failure_pack_manifest.json", manifest)
    write_json(run_dir / "failure_pack_stats.json", stats)
    write_markdown(run_dir / "failure_pack_stats.md", md_lines)

    return {
        "status": status,
        "run_id": chosen_run_id,
        "run_dir": str(run_dir),
        "failure_pack_manifest": str(run_dir / "failure_pack_manifest.json"),
        "failure_pack_stats": str(run_dir / "failure_pack_stats.json"),
        "replay_jsonl_path": str(replay_jsonl_path),
        "selected_failures": len(selected),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="P40 failure mining from P39 arena artifacts.")
    parser.add_argument("--config", default="", help="Optional YAML/JSON config path")
    parser.add_argument("--out-dir", default="", help="Optional explicit output directory")
    parser.add_argument("--run-id", default="", help="Optional explicit run_id")
    parser.add_argument("--arena-run-dir", default="", help="Optional explicit arena run directory")
    parser.add_argument("--quick", action="store_true", help="Use small scan budget")
    parser.add_argument("--dry-run", action="store_true", help="Plan-only mode")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = run_failure_mining(
        config_path=(args.config if str(args.config).strip() else None),
        out_dir=(args.out_dir if str(args.out_dir).strip() else None),
        run_id=str(args.run_id or ""),
        quick=bool(args.quick),
        dry_run=bool(args.dry_run),
        arena_run_dir_override=(args.arena_run_dir if str(args.arena_run_dir).strip() else None),
    )
    print(json.dumps(summary, ensure_ascii=False))
    return 0 if str(summary.get("status")) in {"ok", "stub"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
