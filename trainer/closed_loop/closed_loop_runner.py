from __future__ import annotations

if __package__ is None or __package__ == "":
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

try:  # pragma: no cover - optional dependency
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None

from trainer.closed_loop.candidate_train import run_candidate_training
from trainer.closed_loop.failure_mining import run_failure_mining
from trainer.closed_loop.replay_manifest import now_iso, now_stamp, to_abs_path, write_json, write_markdown
from trainer.closed_loop.replay_mixer import run_replay_mixer


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


def _run_process(command: list[str], *, cwd: Path, timeout_sec: int) -> dict[str, Any]:
    started = time.time()
    try:
        proc = subprocess.run(
            command,
            cwd=str(cwd),
            text=True,
            capture_output=True,
            timeout=max(1, int(timeout_sec)),
        )
        return {
            "returncode": int(proc.returncode),
            "stdout": str(proc.stdout or ""),
            "stderr": str(proc.stderr or ""),
            "elapsed_sec": time.time() - started,
            "timed_out": False,
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "returncode": 124,
            "stdout": str(exc.stdout or ""),
            "stderr": str(exc.stderr or ""),
            "elapsed_sec": time.time() - started,
            "timed_out": True,
        }


def _normalize_seeds(cfg: dict[str, Any], *, quick: bool) -> list[str]:
    raw = cfg.get("seeds")
    seeds: list[str] = []
    if isinstance(raw, list):
        seeds = [str(s).strip() for s in raw if str(s).strip()]
    if not seeds:
        seeds = ["AAAAAAA", "BBBBBBB"]
    if quick and len(seeds) > 2:
        seeds = seeds[:2]
    return seeds


def _summary_row_to_metrics(summary_path: Path, candidate_policy: str, champion_policy: str) -> dict[str, Any]:
    payload: Any = {}
    if summary_path.exists():
        try:
            payload = json.loads(summary_path.read_text(encoding="utf-8-sig"))
        except Exception:
            payload = {}
    rows = payload if isinstance(payload, list) else []
    candidate_row: dict[str, Any] | None = None
    champion_row: dict[str, Any] | None = None
    for row in rows:
        if not isinstance(row, dict):
            continue
        pid = str(row.get("policy_id") or "")
        if pid == candidate_policy:
            candidate_row = row
        if pid == champion_policy:
            champion_row = row

    candidate_score = float((candidate_row or {}).get("mean_total_score") or 0.0)
    champion_score = float((champion_row or {}).get("mean_total_score") or 0.0)
    candidate_win = float((candidate_row or {}).get("win_rate") or 0.0)
    candidate_invalid = float((candidate_row or {}).get("invalid_action_rate") or 0.0)
    return {
        "candidate_row": candidate_row or {},
        "champion_row": champion_row or {},
        "candidate_score": candidate_score,
        "champion_score": champion_score,
        "score_delta": candidate_score - champion_score,
        "candidate_win_rate": candidate_win,
        "candidate_invalid_action_rate": candidate_invalid,
    }


def _write_summary_tables(run_dir: Path, row: dict[str, Any]) -> dict[str, str]:
    fields = [
        "run_id",
        "status",
        "replay_status",
        "failure_status",
        "candidate_status",
        "arena_status",
        "recommendation",
        "recommend_promotion",
        "candidate_policy",
        "champion_policy",
        "candidate_score",
        "champion_score",
        "score_delta",
        "candidate_win_rate",
        "candidate_invalid_action_rate",
    ]
    csv_path = run_dir / "summary_table.csv"
    json_path = run_dir / "summary_table.json"
    md_path = run_dir / "summary_table.md"

    with csv_path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fields)
        writer.writeheader()
        writer.writerow({k: row.get(k) for k in fields})

    write_json(json_path, [row])
    lines = [
        f"# P40 Closed-Loop Summary ({row.get('run_id')})",
        "",
        "| run_id | status | replay | failure | candidate | arena | recommendation | promote | cand_score | champ_score | delta | win_rate | invalid |",
        "|---|---|---|---|---|---|---|---:|---:|---:|---:|---:|---:|",
        "| {run_id} | {status} | {replay_status} | {failure_status} | {candidate_status} | {arena_status} | {recommendation} | {recommend_promotion} | {candidate_score:.6f} | {champion_score:.6f} | {score_delta:.6f} | {candidate_win_rate:.6f} | {candidate_invalid_action_rate:.6f} |".format(
            run_id=row.get("run_id"),
            status=row.get("status"),
            replay_status=row.get("replay_status"),
            failure_status=row.get("failure_status"),
            candidate_status=row.get("candidate_status"),
            arena_status=row.get("arena_status"),
            recommendation=row.get("recommendation"),
            recommend_promotion=str(bool(row.get("recommend_promotion"))).lower(),
            candidate_score=float(row.get("candidate_score") or 0.0),
            champion_score=float(row.get("champion_score") or 0.0),
            score_delta=float(row.get("score_delta") or 0.0),
            candidate_win_rate=float(row.get("candidate_win_rate") or 0.0),
            candidate_invalid_action_rate=float(row.get("candidate_invalid_action_rate") or 0.0),
        ),
    ]
    write_markdown(md_path, lines)
    return {"csv": str(csv_path), "json": str(json_path), "md": str(md_path)}


def _build_promotion_md(payload: dict[str, Any]) -> list[str]:
    lines = [
        "# P40 Promotion Decision",
        "",
        f"- generated_at: {payload.get('generated_at')}",
        f"- recommendation: `{payload.get('recommendation')}`",
        f"- recommend_promotion: `{payload.get('recommend_promotion')}`",
        f"- arena_status: `{payload.get('arena_status')}`",
        f"- candidate_policy: `{payload.get('candidate_policy')}`",
        f"- champion_policy: `{payload.get('champion_policy')}`",
        "",
        "## Metrics",
        "",
        f"- candidate_score: {float(payload.get('candidate_score') or 0.0):.6f}",
        f"- champion_score: {float(payload.get('champion_score') or 0.0):.6f}",
        f"- score_delta: {float(payload.get('score_delta') or 0.0):.6f}",
        f"- candidate_win_rate: {float(payload.get('candidate_win_rate') or 0.0):.6f}",
        f"- candidate_invalid_action_rate: {float(payload.get('candidate_invalid_action_rate') or 0.0):.6f}",
        "",
        "## Reasons",
    ]
    reasons = payload.get("reasons") if isinstance(payload.get("reasons"), list) else []
    if reasons:
        lines.extend([f"- {str(item)}" for item in reasons])
    else:
        lines.append("- none")
    return lines


def run_closed_loop(
    *,
    config_path: str | Path,
    out_dir: str | Path | None = None,
    run_id: str = "",
    quick: bool = False,
    dry_run: bool = False,
    seeds_override: list[str] | None = None,
) -> dict[str, Any]:
    repo_root = Path(__file__).resolve().parents[2]
    cfg_path = to_abs_path(repo_root, config_path)
    cfg = _read_yaml_or_json(cfg_path)

    output_cfg = cfg.get("output") if isinstance(cfg.get("output"), dict) else {}
    artifacts_root = str(output_cfg.get("artifacts_root") or "docs/artifacts/p40/closed_loop_runs")
    chosen_run_id = str(run_id or output_cfg.get("run_id") or now_stamp())
    if out_dir:
        run_dir = to_abs_path(repo_root, out_dir)
    else:
        run_dir = to_abs_path(repo_root, artifacts_root) / chosen_run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    seeds = seeds_override if seeds_override else _normalize_seeds(cfg, quick=quick)
    if quick and len(seeds) > 2:
        seeds = seeds[:2]

    replay_cfg = cfg.get("replay_mixer") if isinstance(cfg.get("replay_mixer"), dict) else {}
    replay_config_path = str(replay_cfg.get("config") or "configs/experiments/p40_replay_mix_smoke.yaml")
    replay_summary = run_replay_mixer(
        config_path=replay_config_path,
        out_dir=run_dir / "replay_mixer",
        run_id=f"{chosen_run_id}-replay",
        quick=bool(quick or replay_cfg.get("quick")),
        dry_run=bool(dry_run),
    )
    write_json(
        run_dir / "replay_mix_manifest_ref.json",
        {
            "schema": "p40_ref_v1",
            "generated_at": now_iso(),
            "kind": "replay_mixer",
            "summary": replay_summary,
        },
    )

    failure_cfg = cfg.get("failure_mining") if isinstance(cfg.get("failure_mining"), dict) else {}
    failure_enabled = bool(failure_cfg.get("enabled", True))
    if failure_enabled:
        failure_summary = run_failure_mining(
            config_path=(str(failure_cfg.get("config")) if str(failure_cfg.get("config") or "").strip() else None),
            out_dir=run_dir / "failure_mining",
            run_id=f"{chosen_run_id}-failure",
            quick=bool(quick or failure_cfg.get("quick")),
            dry_run=bool(dry_run),
            arena_run_dir_override=(
                str(failure_cfg.get("arena_run_dir")) if str(failure_cfg.get("arena_run_dir") or "").strip() else None
            ),
        )
    else:
        failure_summary = {
            "status": "skipped",
            "run_id": f"{chosen_run_id}-failure",
            "run_dir": str(run_dir / "failure_mining"),
            "failure_pack_manifest": "",
        }
    write_json(
        run_dir / "failure_mining_ref.json",
        {
            "schema": "p40_ref_v1",
            "generated_at": now_iso(),
            "kind": "failure_mining",
            "summary": failure_summary,
        },
    )

    candidate_cfg = cfg.get("candidate_train") if isinstance(cfg.get("candidate_train"), dict) else {}
    candidate_config_path = to_abs_path(
        repo_root,
        str(candidate_cfg.get("config") or "configs/experiments/p40_candidate_smoke.yaml"),
    )
    candidate_cfg_payload = _read_yaml_or_json(candidate_config_path)
    candidate_cfg_payload["replay_mix_manifest"] = str(replay_summary.get("replay_mix_manifest") or "")
    candidate_cfg_payload["seeds"] = list(seeds)
    generated_candidate_cfg = run_dir / "candidate_config.generated.json"
    write_json(generated_candidate_cfg, candidate_cfg_payload)

    candidate_summary = run_candidate_training(
        config_path=generated_candidate_cfg,
        out_dir=run_dir / "candidate_train",
        run_id=f"{chosen_run_id}-candidate",
        quick=bool(quick or candidate_cfg.get("quick")),
        dry_run=bool(dry_run),
    )
    write_json(
        run_dir / "candidate_train_ref.json",
        {
            "schema": "p40_ref_v1",
            "generated_at": now_iso(),
            "kind": "candidate_train",
            "summary": candidate_summary,
        },
    )

    arena_cfg = cfg.get("arena_eval") if isinstance(cfg.get("arena_eval"), dict) else {}
    arena_enabled = bool(arena_cfg.get("enabled", True))
    arena_status = "skipped"
    arena_reason = ""
    arena_summary_ref: dict[str, Any] = {}
    decision_payload: dict[str, Any] = {}
    candidate_policy = str(arena_cfg.get("candidate_policy") or "model_policy")
    champion_policy = str(arena_cfg.get("champion_policy") or "heuristic_baseline")

    candidate_checkpoint = str(candidate_summary.get("best_checkpoint") or "")
    if arena_enabled and not dry_run:
        arena_out = run_dir / "arena_eval"
        arena_root = arena_out / "arena_runs"
        arena_root.mkdir(parents=True, exist_ok=True)
        arena_run_id = f"{chosen_run_id}-arena"
        policies_cfg = arena_cfg.get("policies")
        if isinstance(policies_cfg, list) and policies_cfg:
            policies = [str(p).strip() for p in policies_cfg if str(p).strip()]
        else:
            policies = [champion_policy, candidate_policy]
        if champion_policy not in policies:
            policies.append(champion_policy)
        if candidate_policy not in policies:
            policies.append(candidate_policy)

        max_steps = int(arena_cfg.get("max_steps") or (120 if quick else 180))
        episodes_per_seed = int(arena_cfg.get("episodes_per_seed") or (1 if quick else 2))
        timeout_sec = int(arena_cfg.get("timeout_sec") or max(600, len(policies) * len(seeds) * max_steps * episodes_per_seed // 3))

        arena_cmd = [
            str(sys.executable),
            "-B",
            "-m",
            "trainer.policy_arena.arena_runner",
            "--out-dir",
            str(arena_root),
            "--run-id",
            arena_run_id,
            "--backend",
            str(arena_cfg.get("backend") or "sim"),
            "--mode",
            str(arena_cfg.get("mode") or "long_episode"),
            "--policies",
            ",".join(policies),
            "--seeds",
            ",".join(seeds),
            "--episodes-per-seed",
            str(max(1, episodes_per_seed)),
            "--max-steps",
            str(max(1, max_steps)),
            "--skip-unavailable",
        ]
        if candidate_policy == "model_policy" and candidate_checkpoint:
            arena_cmd.extend(["--model-path", candidate_checkpoint])
        if quick:
            arena_cmd.append("--quick")

        arena_result = _run_process(arena_cmd, cwd=repo_root, timeout_sec=timeout_sec)
        arena_run_dir = arena_root / arena_run_id
        summary_path = arena_run_dir / "summary_table.json"
        arena_status = "ok" if arena_result.get("returncode") == 0 and summary_path.exists() else "failed"
        arena_reason = ""
        if arena_status != "ok":
            arena_reason = "arena_runner_failed_or_summary_missing"

        arena_summary_ref = {
            "schema": "p40_arena_summary_ref_v1",
            "generated_at": now_iso(),
            "arena_status": arena_status,
            "arena_reason": arena_reason,
            "arena_command": arena_cmd,
            "arena_returncode": int(arena_result.get("returncode") or 0),
            "arena_elapsed_sec": float(arena_result.get("elapsed_sec") or 0.0),
            "arena_run_dir": str(arena_run_dir),
            "summary_json": str(summary_path),
        }

        champion_cfg = cfg.get("champion_rules") if isinstance(cfg.get("champion_rules"), dict) else {}
        if arena_status == "ok" and bool(champion_cfg.get("enabled", True)):
            champion_out = arena_out / "champion_eval"
            champion_cmd = [
                str(sys.executable),
                "-B",
                "-m",
                "trainer.policy_arena.champion_rules",
                "--summary-json",
                str(summary_path),
                "--out-dir",
                str(champion_out),
                "--candidate-policy",
                candidate_policy,
                "--champion-policy",
                champion_policy,
            ]
            champion_json = str(champion_cfg.get("champion_json") or "docs/artifacts/p22/champion.json")
            champion_cmd.extend(["--champion-json", champion_json])
            for arg_key, cli_key in [
                ("min_seeds", "--min-seeds"),
                ("max_invalid_increase", "--max-invalid-increase"),
                ("max_timeout_increase", "--max-timeout-increase"),
                ("min_score_improvement", "--min-score-improvement"),
                ("max_score_regression", "--max-score-regression"),
                ("min_win_improvement", "--min-win-improvement"),
            ]:
                if champion_cfg.get(arg_key) is not None:
                    champion_cmd.extend([cli_key, str(champion_cfg.get(arg_key))])
            champion_result = _run_process(champion_cmd, cwd=repo_root, timeout_sec=300)
            arena_summary_ref["champion_rules_command"] = champion_cmd
            arena_summary_ref["champion_rules_returncode"] = int(champion_result.get("returncode") or 0)

            decision_json_path: Path | None = None
            for line in str(champion_result.get("stdout") or "").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                if isinstance(obj, dict):
                    raw = str(obj.get("json") or "").strip()
                    if raw:
                        candidate_path = Path(raw)
                        if candidate_path.exists():
                            decision_json_path = candidate_path
            if decision_json_path is not None and decision_json_path.exists():
                loaded = json.loads(decision_json_path.read_text(encoding="utf-8-sig"))
                if isinstance(loaded, dict):
                    decision_payload = loaded
                    arena_summary_ref["champion_decision_json"] = str(decision_json_path)

        if not decision_payload:
            metrics = _summary_row_to_metrics(summary_path, candidate_policy, champion_policy)
            decision_payload = {
                "schema": "p40_promotion_decision_v1",
                "generated_at": now_iso(),
                "decision": "observe" if arena_status == "ok" else "observe",
                "recommend_promotion": False,
                "candidate_policy_id": candidate_policy,
                "champion_policy_id": champion_policy,
                "candidate_row": metrics.get("candidate_row"),
                "champion_row": metrics.get("champion_row"),
                "reasons": [
                    "champion_rules_unavailable_or_not_emitted"
                    if arena_status == "ok"
                    else "arena_status_not_ok"
                ],
            }
    else:
        arena_status = "skipped"
        arena_reason = "arena_disabled_or_dry_run"
        arena_summary_ref = {
            "schema": "p40_arena_summary_ref_v1",
            "generated_at": now_iso(),
            "arena_status": arena_status,
            "arena_reason": arena_reason,
        }
        decision_payload = {
            "schema": "p40_promotion_decision_v1",
            "generated_at": now_iso(),
            "decision": "observe",
            "recommend_promotion": False,
            "candidate_policy_id": candidate_policy,
            "champion_policy_id": champion_policy,
            "reasons": [arena_reason],
        }

    write_json(run_dir / "arena_summary_ref.json", arena_summary_ref)

    decision_token = str(decision_payload.get("decision") or "").lower()
    recommendation = "reject"
    if decision_token == "promote":
        recommendation = "promote"
    elif decision_token in {"observe", "hold"}:
        recommendation = "observe"

    candidate_row_payload = decision_payload.get("candidate_row") if isinstance(decision_payload.get("candidate_row"), dict) else {}
    champion_row_payload = decision_payload.get("champion_row") if isinstance(decision_payload.get("champion_row"), dict) else {}
    delta_payload = decision_payload.get("deltas") if isinstance(decision_payload.get("deltas"), dict) else {}
    threshold_payload = decision_payload.get("thresholds") if isinstance(decision_payload.get("thresholds"), dict) else {}

    candidate_score = float((candidate_row_payload.get("mean_total_score")) or 0.0)
    champion_score = float((champion_row_payload.get("mean_total_score")) or 0.0)
    candidate_win_rate = float((candidate_row_payload.get("win_rate")) or 0.0)
    candidate_invalid = float((candidate_row_payload.get("invalid_action_rate")) or 0.0)

    conservative_reasons: list[str] = []
    min_seeds_required = int(threshold_payload.get("min_seeds") or 0)
    seed_count = int(candidate_row_payload.get("seed_count") or 0)
    if min_seeds_required > 0 and seed_count < min_seeds_required:
        conservative_reasons.append(
            f"insufficient_seed_count:{seed_count}<{min_seeds_required}"
        )

    std_score = float(candidate_row_payload.get("std_total_score") or 0.0)
    mean_score_abs = abs(float(candidate_row_payload.get("mean_total_score") or 0.0))
    if std_score > max(1.0, mean_score_abs * 0.6):
        conservative_reasons.append("candidate_variance_high")

    delta_score = float(delta_payload.get("mean_total_score") or 0.0)
    delta_win = float(delta_payload.get("win_rate") or 0.0)
    if delta_score <= 0.0 and delta_win <= 0.0:
        conservative_reasons.append("no_clear_uplift_vs_champion")

    if arena_status != "ok":
        conservative_reasons.append("arena_status_not_ok")

    recommend_promotion = bool(decision_payload.get("recommend_promotion", False)) and recommendation == "promote"
    if conservative_reasons and recommendation == "promote":
        recommendation = "observe"
        recommend_promotion = False

    promotion_decision = {
        "schema": "p40_promotion_decision_v1",
        "generated_at": now_iso(),
        "run_id": chosen_run_id,
        "recommendation": recommendation,
        "recommend_promotion": recommend_promotion,
        "arena_status": arena_status,
        "arena_reason": arena_reason,
        "candidate_policy": candidate_policy,
        "champion_policy": champion_policy,
        "candidate_score": candidate_score,
        "champion_score": champion_score,
        "score_delta": candidate_score - champion_score,
        "candidate_win_rate": candidate_win_rate,
        "candidate_invalid_action_rate": candidate_invalid,
        "reasons": (
            (decision_payload.get("reasons") if isinstance(decision_payload.get("reasons"), list) else [])
            + conservative_reasons
        ),
        "champion_rules_payload": decision_payload,
    }
    write_json(run_dir / "promotion_decision.json", promotion_decision)
    write_markdown(run_dir / "promotion_decision.md", _build_promotion_md(promotion_decision))

    row = {
        "run_id": chosen_run_id,
        "status": "ok" if recommendation in {"promote", "observe", "reject"} else "stub",
        "replay_status": str(replay_summary.get("status") or "stub"),
        "failure_status": str(failure_summary.get("status") or "skipped"),
        "candidate_status": str(candidate_summary.get("status") or "stub"),
        "arena_status": arena_status,
        "recommendation": recommendation,
        "recommend_promotion": bool(promotion_decision.get("recommend_promotion")),
        "candidate_policy": candidate_policy,
        "champion_policy": champion_policy,
        "candidate_score": promotion_decision.get("candidate_score"),
        "champion_score": promotion_decision.get("champion_score"),
        "score_delta": promotion_decision.get("score_delta"),
        "candidate_win_rate": promotion_decision.get("candidate_win_rate"),
        "candidate_invalid_action_rate": promotion_decision.get("candidate_invalid_action_rate"),
    }
    summary_paths = _write_summary_tables(run_dir, row)

    run_manifest = {
        "schema": "p40_closed_loop_run_manifest_v1",
        "generated_at": now_iso(),
        "run_id": chosen_run_id,
        "status": row["status"],
        "config_path": str(cfg_path),
        "run_dir": str(run_dir),
        "quick": bool(quick),
        "dry_run": bool(dry_run),
        "seeds": seeds,
        "steps": {
            "replay_mixer": replay_summary,
            "failure_mining": failure_summary,
            "candidate_train": candidate_summary,
            "arena_eval": arena_summary_ref,
            "promotion_decision": promotion_decision,
        },
        "summary_table_paths": summary_paths,
    }
    write_json(run_dir / "run_manifest.json", run_manifest)

    return {
        "status": row["status"],
        "run_id": chosen_run_id,
        "run_dir": str(run_dir),
        "run_manifest": str(run_dir / "run_manifest.json"),
        "promotion_decision": str(run_dir / "promotion_decision.json"),
        "summary_table_json": summary_paths["json"],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="P40 closed-loop runner (replay mix -> failure mining -> train -> arena).")
    parser.add_argument("--config", default="configs/experiments/p40_closed_loop_smoke.yaml")
    parser.add_argument("--out-dir", default="")
    parser.add_argument("--run-id", default="")
    parser.add_argument("--seeds", default="", help="Optional comma-separated seed override")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    seeds_override = [s.strip() for s in str(args.seeds or "").split(",") if s.strip()]
    summary = run_closed_loop(
        config_path=args.config,
        out_dir=(args.out_dir if str(args.out_dir).strip() else None),
        run_id=str(args.run_id or ""),
        quick=bool(args.quick),
        dry_run=bool(args.dry_run),
        seeds_override=seeds_override or None,
    )
    print(json.dumps(summary, ensure_ascii=False))
    return 0 if str(summary.get("status")) in {"ok", "stub"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
