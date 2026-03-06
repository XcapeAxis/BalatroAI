from __future__ import annotations

if __package__ is None or __package__ == "":
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

import argparse
import copy
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:  # pragma: no cover - optional dependency in local venv
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None

from trainer.experiments.champion import update_nightly_decision
from trainer.experiments.matrix import build_matrix
from trainer.experiments.metrics import aggregate_seed_metrics, is_success
from trainer.experiments.report import write_comparison_report, write_summary_tables
from trainer.experiments.seed_policy import (
    materialize_nightly_seed_set,
    materialize_seed_set,
    read_seed_policy,
    validate_seed_policy,
)
from trainer.experiments.training_modes import (
    MODE_CATEGORY_EXPERIMENTAL,
    MODE_CATEGORY_LEGACY_BASELINE,
    MODE_CATEGORY_MAINLINE,
)


MODE_CATEGORY_REQUIRED_VALIDATION = "required_validation"


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as fp:
        fp.write(json.dumps(payload, ensure_ascii=False) + "\n")


def run_process(
    command: str | list[str],
    *,
    cwd: Path,
    timeout_sec: int | None = None,
) -> dict[str, Any]:
    start = time.time()
    shell = isinstance(command, str)
    try:
        proc = subprocess.run(
            command,
            cwd=str(cwd),
            shell=shell,
            text=True,
            capture_output=True,
            timeout=timeout_sec,
        )
        elapsed = time.time() - start
        return {
            "returncode": int(proc.returncode),
            "stdout": proc.stdout,
            "stderr": proc.stderr,
            "elapsed_sec": elapsed,
            "command": command,
            "timed_out": False,
        }
    except subprocess.TimeoutExpired as exc:
        elapsed = time.time() - start
        return {
            "returncode": 124,
            "stdout": str(exc.stdout or ""),
            "stderr": str(exc.stderr or ""),
            "elapsed_sec": elapsed,
            "command": command,
            "timed_out": True,
        }


def parse_csv_list(text: str) -> set[str]:
    if not text:
        return set()
    return {part.strip() for part in text.split(",") if part.strip()}


def normalize_experiment_category(exp: dict[str, Any]) -> str:
    raw = str(exp.get("category") or "").strip().lower()
    if raw:
        return raw

    exp_id = str(exp.get("id") or "").strip().lower()
    exp_type = str(exp.get("experiment_type") or "standard").strip().lower()
    policy = str(exp.get("policy") or "").strip().lower()
    joined = " ".join([exp_id, exp_type, policy])

    if exp_id in {"quick_baseline", "quick_candidate"}:
        return MODE_CATEGORY_REQUIRED_VALIDATION
    if any(token in joined for token in ("bc", "dagger", "legacy")):
        return MODE_CATEGORY_LEGACY_BASELINE
    if exp_type in {
        "selfsup_pretrain",
        "selfsup_p33",
        "pretrain_repr",
        "self_supervised",
        "selfsup_stub",
        "selfsup_future_value",
        "selfsup_action_type",
        "ssl_pretrain",
        "ssl_probe",
        "rl_selfplay",
        "rl_selfplay_v1",
        "long_consistency",
        "long_horizon_consistency",
        "policy_arena",
        "policy_arena_v1",
        "arena",
        "closed_loop_improvement",
        "closed_loop",
        "p40_closed_loop",
        "closed_loop_improvement_v2",
        "p41_closed_loop_v2",
        "closed_loop_v2",
        "closed_loop_rl_candidate",
        "rl_candidate_pipeline",
        "p42_rl_candidate",
        "p42_rl_candidate_pipeline",
        "p44_distributed_rl",
        "world_model_train",
        "world_model_eval",
        "world_model_assist_compare",
        "p45_world_model",
        "imagination_augmented_candidate",
        "p46_imagination",
        "world_model_rerank_eval",
        "p47_wm_search",
    }:
        return MODE_CATEGORY_MAINLINE
    if exp_type == "standard" and policy in {"baseline", "candidate"}:
        return MODE_CATEGORY_REQUIRED_VALIDATION
    return MODE_CATEGORY_EXPERIMENTAL


def resolve_experiment_default_enabled(exp: dict[str, Any], *, category: str) -> bool:
    raw = exp.get("default_enabled")
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, (int, float)):
        return bool(raw)
    if isinstance(raw, str) and raw.strip():
        token = raw.strip().lower()
        if token in {"1", "true", "yes", "on"}:
            return True
        if token in {"0", "false", "no", "off"}:
            return False
    return category != MODE_CATEGORY_LEGACY_BASELINE


def attach_experiment_mode_metadata(experiments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for exp in experiments:
        copied = copy.deepcopy(exp)
        category = normalize_experiment_category(copied)
        copied["category"] = category
        copied["default_enabled"] = resolve_experiment_default_enabled(copied, category=category)
        out.append(copied)
    return out


def current_git_commit(repo_root: Path) -> str:
    try:
        res = run_process(["git", "rev-parse", "HEAD"], cwd=repo_root, timeout_sec=10)
        if res["returncode"] == 0:
            return str(res["stdout"]).strip()
    except Exception:
        pass
    return "unknown"


def detect_python_version() -> str:
    return sys.version.replace("\n", " ").strip()


def _deterministic_letter_seed(source: str, width: int = 7) -> str:
    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    h = hashlib.sha256(source.encode("utf-8")).hexdigest()
    num = int(h, 16)
    out = []
    for _ in range(width):
        out.append(alphabet[num % len(alphabet)])
        num //= len(alphabet)
    return "".join(out)


def build_extra_random_seeds(
    *,
    base_seed: str,
    git_commit: str,
    date_bucket: str,
    count: int,
    existing: set[str],
) -> list[str]:
    seeds: list[str] = []
    idx = 0
    while len(seeds) < count:
        candidate = _deterministic_letter_seed(f"{base_seed}|{git_commit}|{date_bucket}|{idx}")
        idx += 1
        if candidate in existing:
            continue
        existing.add(candidate)
        seeds.append(candidate)
    return seeds


def _normalize_seed_list(values: Any) -> list[str]:
    if not isinstance(values, list):
        return []
    out: list[str] = []
    for v in values:
        token = str(v).strip()
        if token:
            out.append(token)
    return out


def _legacy_seed_policy_block(cfg: dict[str, Any]) -> dict[str, Any]:
    # P34: local seed-policy block for quick reproducibility without external policy file.
    raw = cfg.get("seed_policy")
    if not isinstance(raw, dict):
        return {}
    return raw


def _select_seeds_legacy(
    *,
    exp: dict[str, Any],
    cfg: dict[str, Any],
    git_commit: str,
    nightly: bool,
    seed_limit: int | None,
) -> list[str]:
    policy_block = _legacy_seed_policy_block(cfg)
    seed_set_name = str(exp.get("seed_set_name") or "").strip().lower()
    mode = str(exp.get("seed_mode") or "regression_fixed").strip().lower()

    regression_smoke = _normalize_seed_list(policy_block.get("regression_smoke"))
    train_default = _normalize_seed_list(policy_block.get("train_default"))
    eval_default = _normalize_seed_list(policy_block.get("eval_default"))
    base_seed = str(policy_block.get("base_seed") or "P22BASE")
    extra_count = int(policy_block.get("nightly_extra_random") or 0)

    # Backward-compatible fallback.
    if not regression_smoke and not train_default and not eval_default:
        seeds_cfg = cfg.get("seeds") or {}
        fixed = [str(s) for s in (seeds_cfg.get("regression_fixed") or [])]
        if not fixed:
            raise ValueError("config.seeds.regression_fixed must not be empty")
        regression_smoke = list(fixed)
        train_default = list(fixed)
        eval_default = list(fixed)
        base_seed = str(seeds_cfg.get("base_seed") or base_seed)
        extra_count = int(seeds_cfg.get("extra_random") or extra_count)

    if not regression_smoke:
        raise ValueError("seed policy regression_smoke must not be empty")
    if not train_default:
        train_default = list(regression_smoke)
    if not eval_default:
        eval_default = list(regression_smoke)

    if seed_set_name in {"regression_smoke", "contract_regression", "regression_fixed"}:
        chosen = list(regression_smoke)
    elif seed_set_name in {"train_default", "train"}:
        chosen = list(train_default)
    elif seed_set_name in {"eval_default", "eval"}:
        chosen = list(eval_default)
    elif mode in {"train", "train_default"}:
        chosen = list(train_default)
    elif mode in {"eval", "eval_default", "nightly"}:
        chosen = list(eval_default)
    else:
        chosen = list(regression_smoke)

    use_extra = nightly or mode == "nightly"
    if use_extra and extra_count > 0:
        extras = build_extra_random_seeds(
            base_seed=base_seed,
            git_commit=git_commit,
            date_bucket=datetime.now().strftime("%Y%m%d"),
            count=extra_count,
            existing=set(chosen),
        )
        chosen.extend(extras)

    budget = cfg.get("budget") or {}
    max_seeds = int(budget.get("max_seeds") or len(chosen))
    if seed_limit is not None:
        max_seeds = min(max_seeds, int(seed_limit))
    return chosen[: max(1, max_seeds)]


def select_seeds(
    *,
    exp: dict[str, Any],
    cfg: dict[str, Any],
    git_commit: str,
    nightly: bool,
    seed_limit: int | None,
    run_id: str,
    seed_policy: dict[str, Any] | None,
) -> dict[str, Any]:
    if seed_policy is None:
        local_policy = _legacy_seed_policy_block(cfg)
        local_policy_version = str(local_policy.get("version") or "legacy.p22")
        legacy = _select_seeds_legacy(
            exp=exp,
            cfg=cfg,
            git_commit=git_commit,
            nightly=nightly,
            seed_limit=seed_limit,
        )
        hash_v = hashlib.sha256(
            json.dumps(legacy, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        return {
            "schema": "p23_seeds_used_v1",
            "generated_at": now_iso(),
            "seed_policy_version": local_policy_version,
            "seed_set_name": str(exp.get("seed_set_name") or exp.get("seed_mode") or "regression_fixed"),
            "seed_hash": hash_v,
            "seed_count": len(legacy),
            "seeds": legacy,
            "metadata": {"source": "legacy_p22", "seed_policy_present": bool(local_policy)},
        }

    explicit_single_seed_override = bool(exp.get("allow_single_seed_override"))
    seed_set_name = str(exp.get("seed_set_name") or "contract_regression")
    if nightly or str(exp.get("seed_mode") or "").lower() == "nightly":
        seeds_payload = materialize_nightly_seed_set(
            seed_policy,
            git_commit=git_commit,
            date_bucket=datetime.now().strftime("%Y%m%d"),
            run_id=run_id,
            extra_count_override=None,
            explicit_single_seed_override=explicit_single_seed_override,
        )
    else:
        seeds_payload = materialize_seed_set(
            seed_policy,
            seed_set_name,
            explicit_single_seed_override=explicit_single_seed_override,
        )

    seeds = [str(s) for s in (seeds_payload.get("seeds") or [])]
    if seed_limit is not None:
        seeds = seeds[: max(1, int(seed_limit))]
    if bool(seed_policy.get("disallow_single_seed_default")) and len(seeds) == 1 and not explicit_single_seed_override:
        raise ValueError(
            f"single seed default disallowed by seed policy (exp={exp.get('id')} set={seed_set_name})"
        )
    seeds_payload = dict(seeds_payload)
    seeds_payload["seeds"] = seeds
    seeds_payload["seed_count"] = len(seeds)
    seeds_payload["seed_hash"] = hashlib.sha256(
        json.dumps(seeds, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return seeds_payload


def resolve_experiment_seeds(
    *,
    exp: dict[str, Any],
    cfg: dict[str, Any],
    git_commit: str,
    nightly: bool,
    seed_limit: int | None,
    run_id: str,
    seed_policy: dict[str, Any] | None,
    cli_seeds: list[str] | None,
) -> dict[str, Any]:
    if cli_seeds:
        seeds = [str(s).strip() for s in cli_seeds if str(s).strip()]
        if seed_limit is not None:
            seeds = seeds[: max(1, int(seed_limit))]
        if not seeds:
            raise ValueError("CLI seeds override resolved to empty set")
        return {
            "schema": "p34_seeds_used_v1",
            "generated_at": now_iso(),
            "seed_policy_version": "explicit.cli_override",
            "seed_set_name": str(exp.get("seed_set_name") or "cli_override"),
            "seed_hash": hashlib.sha256(
                json.dumps(seeds, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
            ).hexdigest(),
            "seed_count": len(seeds),
            "seeds": seeds,
            "metadata": {"source": "cli.seeds_override"},
        }

    explicit_seeds = exp.get("seeds") if isinstance(exp.get("seeds"), list) else []
    if explicit_seeds:
        seeds = [str(s).strip() for s in explicit_seeds if str(s).strip()]
        if seed_limit is not None:
            seeds = seeds[: max(1, int(seed_limit))]
        if not seeds:
            raise ValueError(f"exp={exp.get('id')} has empty explicit seeds list after filtering")
        return {
            "schema": "p31_seeds_used_v1",
            "generated_at": now_iso(),
            "seed_policy_version": "explicit.experiment",
            "seed_set_name": str(exp.get("seed_set_name") or "explicit"),
            "seed_hash": hashlib.sha256(
                json.dumps(seeds, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
            ).hexdigest(),
            "seed_count": len(seeds),
            "seeds": seeds,
            "metadata": {"source": "experiment.seeds"},
        }

    return select_seeds(
        exp=exp,
        cfg=cfg,
        git_commit=git_commit,
        nightly=nightly,
        seed_limit=seed_limit,
        run_id=run_id,
        seed_policy=seed_policy,
    )


def format_progress(
    *,
    run_id: str,
    exp_id: str,
    stage: str,
    status: str,
    elapsed_sec: float,
    eta_sec: float | None = None,
    seed_index: int | None = None,
    seed_total: int | None = None,
    metric_snapshot: str = "",
) -> str:
    seed_part = "-"
    if seed_index is not None and seed_total is not None:
        seed_part = f"{seed_index}/{seed_total}"
    eta_part = "-"
    if eta_sec is not None:
        eta_part = f"{max(0.0, eta_sec):.1f}s"
    metric_text = metric_snapshot if metric_snapshot else "-"
    return (
        f"{run_id:<15} | {exp_id:<24} | {stage:<10} | {seed_part:<8} | "
        f"{elapsed_sec:>8.1f}s | {eta_part:>8} | {metric_text:<24} | {status}"
    )


def _update_live_snapshot(ctx: "RunContext") -> None:
    status_counts: dict[str, int] = {}
    for row in ctx.queue_state.values():
        token = str(row.get("status") or "unknown")
        status_counts[token] = status_counts.get(token, 0) + 1
    payload = {
        "schema": "p23_live_summary_v1",
        "generated_at": now_iso(),
        "run_id": ctx.run_id,
        "mode": ctx.mode,
        "elapsed_sec": time.time() - ctx.run_started_ts,
        "status_counts": status_counts,
        "rows": list(ctx.queue_state.values()),
    }
    write_json(ctx.live_summary_path, payload)


def emit_progress(
    ctx: "RunContext",
    *,
    exp_id: str,
    stage: str,
    status: str,
    elapsed_sec: float,
    phase: str = "orchestrator",
    seed: str | None = None,
    step_or_epoch: int | str | None = None,
    metrics: dict[str, Any] | None = None,
    wall_time_sec: float | None = None,
    message: str = "",
    seed_index: int | None = None,
    seed_total: int | None = None,
    metric_snapshot: str = "",
    eta_sec: float | None = None,
) -> None:
    event = {
        "schema": "p34_telemetry_event_v1",
        "ts": now_iso(),
        "run_id": ctx.run_id,
        "mode": ctx.mode,
        "exp_id": exp_id,
        "seed": seed,
        "phase": phase,
        "stage": stage,
        "status": status,
        "step_or_epoch": step_or_epoch,
        "metrics": metrics or {},
        "wall_time_sec": wall_time_sec,
        "seed_index": seed_index,
        "seed_total": seed_total,
        "elapsed_sec": elapsed_sec,
        "eta_sec": eta_sec,
        "metric_snapshot": metric_snapshot,
        "message": message,
    }
    append_jsonl(ctx.telemetry_path, event)
    row = ctx.queue_state.get(exp_id) or {"exp_id": exp_id}
    row.update(
        {
            "exp_id": exp_id,
            "stage": stage,
            "status": status,
            "seed": (f"{seed_index}/{seed_total}" if seed_index and seed_total else "-"),
            "elapsed_sec": elapsed_sec,
            "eta_sec": eta_sec,
            "metric_snapshot": metric_snapshot,
            "updated_at": event["ts"],
        }
    )
    ctx.queue_state[exp_id] = row
    _update_live_snapshot(ctx)
    print(
        format_progress(
            run_id=ctx.run_id,
            exp_id=exp_id,
            stage=stage,
            status=status,
            elapsed_sec=elapsed_sec,
            eta_sec=eta_sec,
            seed_index=seed_index,
            seed_total=seed_total,
            metric_snapshot=metric_snapshot,
        )
    )


def append_experiment_progress_event(
    progress_path: Path,
    *,
    run_id: str,
    exp_id: str,
    phase: str,
    stage: str,
    status: str,
    seed: str | None = None,
    step_or_epoch: int | str | None = None,
    metrics: dict[str, Any] | None = None,
    elapsed_sec: float | None = None,
    wall_time_sec: float | None = None,
    message: str = "",
    extra: dict[str, Any] | None = None,
) -> None:
    payload: dict[str, Any] = {
        "schema": "p34_progress_event_v1",
        "ts": now_iso(),
        "run_id": run_id,
        "exp_id": exp_id,
        "seed": seed,
        "phase": phase,
        "stage": stage,
        "status": status,
        "step_or_epoch": step_or_epoch,
        "metrics": metrics or {},
        "elapsed_sec": elapsed_sec,
        "wall_time_sec": wall_time_sec,
        "message": message,
    }
    if extra:
        payload.update(extra)
    append_jsonl(progress_path, payload)


@dataclass
class RunContext:
    repo_root: Path
    out_root: Path
    run_root: Path
    run_id: str
    config: dict[str, Any]
    nightly: bool
    dry_run: bool
    keep_intermediate: bool
    git_commit: str
    max_parallel: int
    resume: bool
    seed_limit: int | None
    seed_policy: dict[str, Any] | None
    seed_policy_config: str
    mode: str
    verbose: bool
    telemetry_path: Path
    live_summary_path: Path
    run_started_ts: float
    queue_state: dict[str, dict[str, Any]]
    cli_seeds: list[str] | None


def _expand_command(command_tmpl: Any, context: dict[str, Any]) -> str | list[str]:
    if command_tmpl is None:
        return ""
    if isinstance(command_tmpl, list):
        return [str(x).format(**context) for x in command_tmpl]
    return str(command_tmpl).format(**context)


def _run_stage_command(
    *,
    ctx: RunContext,
    exp_id: str,
    stage: str,
    command_tmpl: Any,
    context: dict[str, Any],
    repo_root: Path,
    progress_path: Path,
    timeout_sec: int | None,
    max_retries: int,
    started_ts: float,
) -> dict[str, Any]:
    command = _expand_command(command_tmpl, context)
    if not command:
        payload = {"status": "skipped", "reason": "empty_command"}
        append_experiment_progress_event(
            progress_path,
            run_id=ctx.run_id,
            exp_id=exp_id,
            phase="stage",
            stage=stage,
            status="skipped",
            elapsed_sec=time.time() - started_ts,
            message="empty_command",
        )
        emit_progress(
            ctx,
            exp_id=exp_id,
            stage=stage,
            status="skipped",
            elapsed_sec=time.time() - started_ts,
            phase="stage",
            message="empty_command",
        )
        return payload
    attempts = 0
    result: dict[str, Any] = {}
    while attempts < max(1, max_retries):
        attempts += 1
        if ctx.verbose:
            print(
                "[P22][verbose] exp={exp} stage={stage} attempt={attempt} cmd={cmd}".format(
                    exp=exp_id,
                    stage=stage,
                    attempt=attempts,
                    cmd=command,
                )
            )
        result = run_process(command, cwd=repo_root, timeout_sec=timeout_sec)
        timed_out = bool(result.get("timed_out"))
        status = "timed_out" if timed_out else ("ok" if result["returncode"] == 0 else "failed")
        append_experiment_progress_event(
            progress_path,
            run_id=ctx.run_id,
            exp_id=exp_id,
            phase="stage",
            stage=stage,
            status=status,
            step_or_epoch=attempts,
            elapsed_sec=time.time() - started_ts,
            wall_time_sec=float(result.get("elapsed_sec") or 0.0),
            metrics={
                "returncode": result.get("returncode"),
                "timed_out": bool(result.get("timed_out")),
            },
            extra={
                "attempt": attempts,
                "returncode": result["returncode"],
                "timeout_sec": timeout_sec,
                "command": command,
            },
        )
        emit_progress(
            ctx,
            exp_id=exp_id,
            stage=stage,
            status=status,
            elapsed_sec=time.time() - started_ts,
            phase="stage",
            step_or_epoch=attempts,
            metrics={
                "returncode": result.get("returncode"),
                "timed_out": bool(result.get("timed_out")),
            },
            wall_time_sec=float(result.get("elapsed_sec") or 0.0),
        )
        if status == "ok":
            break
        if attempts >= max(1, max_retries):
            break
    timed_out = bool(result.get("timed_out"))
    final_status = "timed_out" if timed_out else ("ok" if result.get("returncode", 1) == 0 else "failed")
    return {
        "status": final_status,
        "attempts": attempts,
        "returncode": result.get("returncode"),
        "elapsed_sec": result.get("elapsed_sec"),
        "stdout_tail": str(result.get("stdout") or "")[-4000:],
        "stderr_tail": str(result.get("stderr") or "")[-4000:],
        "command": command,
    }


def _synthetic_eval_metrics(exp_id: str, seed: str, bias: float) -> dict[str, float]:
    digest = hashlib.sha256(f"{exp_id}|{seed}".encode("utf-8")).hexdigest()
    frac_a = (int(digest[:8], 16) % 1000) / 1000.0
    frac_b = (int(digest[8:16], 16) % 1000) / 1000.0
    frac_c = (int(digest[16:24], 16) % 1000) / 1000.0

    avg_ante = 3.0 + bias + (frac_a * 1.2)
    median_ante = round(3.0 + bias + (frac_b * 1.0), 1)
    win_rate = max(0.0, min(1.0, 0.25 + (frac_c * 0.45) + (bias * 0.05)))
    score = avg_ante + (win_rate * 0.4)
    hand_top1 = max(0.0, min(1.0, 0.42 + (frac_a * 0.45) + (bias * 0.06)))
    hand_top3 = max(hand_top1, min(1.0, hand_top1 + 0.10 + (frac_b * 0.10)))
    shop_top1 = max(0.0, min(1.0, 0.38 + (frac_b * 0.50) + (bias * 0.04)))
    illegal_action_rate = max(0.0, min(0.20, 0.02 + ((1.0 - frac_c) * 0.01) - (bias * 0.003)))
    return {
        "score": score,
        "avg_ante_reached": avg_ante,
        "median_ante": median_ante,
        "win_rate": win_rate,
        "hand_top1": hand_top1,
        "hand_top3": hand_top3,
        "shop_top1": shop_top1,
        "illegal_action_rate": illegal_action_rate,
    }


def _as_number(value: Any) -> float | None:
    try:
        v = float(value)
    except Exception:
        return None
    if v != v:
        return None
    return v


def _fmt_num(value: Any, digits: int = 4) -> str:
    v = _as_number(value)
    if v is None:
        return "n/a"
    return f"{v:.{digits}f}"


def _fmt_pct(value: Any) -> str:
    v = _as_number(value)
    if v is None:
        return "n/a"
    return f"{(v * 100.0):.2f}%"


def _experiment_summary_line(metric_summary: dict[str, Any]) -> str:
    return "avg_ante={avg} median_ante={median} win_rate={win} hand_top1={h1} hand_top3={h3} shop_top1={s1} illegal={illegal}".format(
        avg=_fmt_num(metric_summary.get("avg_ante_reached")),
        median=_fmt_num(metric_summary.get("median_ante"), digits=3),
        win=_fmt_pct(metric_summary.get("win_rate")),
        h1=_fmt_pct(metric_summary.get("hand_top1")),
        h3=_fmt_pct(metric_summary.get("hand_top3")),
        s1=_fmt_pct(metric_summary.get("shop_top1")),
        illegal=_fmt_pct(metric_summary.get("illegal_action_rate")),
    )


def _read_yaml_or_json(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    suffix = path.suffix.lower()
    if suffix in {".yaml", ".yml"}:
        if yaml is not None:
            payload = yaml.safe_load(text)
        else:
            try:
                # Some repo YAML files are JSON-formatted text with .yaml suffix.
                payload = json.loads(text)
            except Exception:
                sidecar = path.with_suffix(".json")
                if sidecar.exists():
                    payload = json.loads(sidecar.read_text(encoding="utf-8"))
                else:
                    raise RuntimeError(f"PyYAML unavailable and no JSON sidecar found for {path}")
    else:
        payload = json.loads(text)
    if not isinstance(payload, dict):
        raise ValueError(f"config payload must be mapping: {path}")
    return payload


def _seed_to_int(seed: str) -> int:
    digest = hashlib.sha256(str(seed).encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def _run_selfsup_seed_experiment(
    *,
    ctx: RunContext,
    exp: dict[str, Any],
    exp_dir: Path,
    seed: str,
    seed_idx: int,
    seed_total: int,
) -> dict[str, Any]:
    from trainer.experiments.selfsup_train import run_selfsup_training

    eval_cfg = exp.get("eval") if isinstance(exp.get("eval"), dict) else {}
    selfsup_cfg_rel = str(
        eval_cfg.get("config")
        or exp.get("selfsup_config")
        or "configs/experiments/p31_selfsup.yaml"
    )
    selfsup_cfg = (ctx.repo_root / selfsup_cfg_rel).resolve()
    max_steps = int(eval_cfg.get("max_steps") or 0)
    out_dir = exp_dir / "selfsup_runs" / f"seed_{seed_idx:03d}_{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = run_selfsup_training(
        config_path=selfsup_cfg,
        out_dir=out_dir,
        seed_override=_seed_to_int(seed),
        max_steps_override=(max_steps if max_steps > 0 else None),
        run_name=f"{ctx.run_id}_{exp.get('id')}_{seed}",
        quiet=(not ctx.verbose),
    )
    final = summary.get("final_metrics") if isinstance(summary.get("final_metrics"), dict) else {}
    val_loss = float(final.get("val_loss") or 0.0)
    val_mae = float(
        final.get("val_score_delta_mae")
        or final.get("val_next_delta_mae")
        or 0.0
    )
    hand_acc = float(
        final.get("val_hand_type_acc")
        or final.get("val_mask_acc")
        or 0.0
    )

    # Keep orchestrator metric schema compatible while carrying selfsup signals.
    avg_ante = max(0.0, 3.0 + (hand_acc * 1.6) + (max(0.0, 1.0 - val_mae) * 0.4))
    win_rate = max(0.0, min(1.0, hand_acc))
    score = avg_ante + (win_rate * 0.3) - (val_loss * 0.05)
    metrics = {
        "score": score,
        "avg_ante_reached": avg_ante,
        "median_ante": avg_ante,
        "win_rate": win_rate,
        "hand_top1": hand_acc,
        "hand_top3": min(1.0, hand_acc + 0.12),
        "shop_top1": max(0.0, min(1.0, 1.0 - val_mae)),
        "illegal_action_rate": max(0.0, min(0.20, val_loss * 0.01)),
        "selfsup_val_loss": val_loss,
        "selfsup_score_delta_mae": val_mae,
        "selfsup_hand_type_acc": hand_acc,
        "selfsup_run_dir": str(summary.get("run_dir") or out_dir),
    }
    return {
        "status": "ok" if str(summary.get("status") or "") == "ok" else "failed",
        "metrics": metrics,
        "summary": summary,
    }


def _run_selfsup_p33_seed_experiment(
    *,
    ctx: RunContext,
    exp: dict[str, Any],
    exp_dir: Path,
    seed: str,
    seed_idx: int,
    seed_total: int,
) -> dict[str, Any]:
    from trainer.self_supervised.train import run_p33_selfsup_training

    eval_cfg = exp.get("eval") if isinstance(exp.get("eval"), dict) else {}
    cfg_rel = str(eval_cfg.get("config") or exp.get("selfsup_config") or "configs/experiments/p33_selfsup.yaml")
    cfg_path = (ctx.repo_root / cfg_rel).resolve()
    max_samples = int(eval_cfg.get("max_samples") or eval_cfg.get("max_steps") or 0)
    out_dir = exp_dir / "selfsup_p33_runs" / f"seed_{seed_idx:03d}_{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = run_p33_selfsup_training(
        config_path=cfg_path,
        out_dir=out_dir,
        seed_override=_seed_to_int(seed),
        max_samples_override=(max_samples if max_samples > 0 else None),
        quiet=(not ctx.verbose),
    )
    final = summary.get("final_metrics") if isinstance(summary.get("final_metrics"), dict) else {}
    val_loss = float(final.get("val_loss") or 0.0)
    val_acc = float(final.get("val_acc") or 0.0)
    score = max(0.0, 2.8 + (val_acc * 1.8) - (val_loss * 0.5))
    metrics = {
        "score": score,
        "avg_ante_reached": max(0.0, 2.5 + (val_acc * 1.2)),
        "median_ante": max(0.0, 2.5 + (val_acc * 1.2)),
        "win_rate": max(0.0, min(1.0, val_acc * 0.95)),
        "hand_top1": max(0.0, min(1.0, val_acc)),
        "hand_top3": max(0.0, min(1.0, val_acc + 0.15)),
        "shop_top1": max(0.0, min(1.0, 1.0 - min(1.0, val_loss))),
        "illegal_action_rate": max(0.0, min(0.25, val_loss * 0.05)),
        "selfsup_p33_val_loss": val_loss,
        "selfsup_p33_val_acc": val_acc,
        "selfsup_p33_run_dir": str(summary.get("run_dir") or out_dir),
    }
    return {
        "status": "ok" if str(summary.get("status") or "") == "ok" else "failed",
        "metrics": metrics,
        "summary": summary,
    }


def _run_selfsup_p32_seed_experiment(
    *,
    ctx: RunContext,
    exp: dict[str, Any],
    exp_dir: Path,
    seed: str,
    seed_idx: int,
    seed_total: int,
) -> dict[str, Any]:
    from trainer.self_supervised.run_pretrain import run_p32_pretrain_stub

    eval_cfg = exp.get("eval") if isinstance(exp.get("eval"), dict) else {}
    cfg_rel = str(
        eval_cfg.get("config")
        or exp.get("selfsup_config")
        or "configs/experiments/p32_self_supervised.yaml"
    )
    cfg_path = (ctx.repo_root / cfg_rel).resolve()
    max_samples = int(eval_cfg.get("max_samples") or eval_cfg.get("max_steps") or 0)
    out_dir = exp_dir / "selfsup_p32_runs" / f"seed_{seed_idx:03d}_{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = run_p32_pretrain_stub(
        config_path=cfg_path,
        out_dir=out_dir,
        seed_override=_seed_to_int(seed),
        max_samples_override=(max_samples if max_samples > 0 else None),
        quiet=(not ctx.verbose),
    )
    final = summary.get("final_metrics") if isinstance(summary.get("final_metrics"), dict) else {}
    val_loss = float(final.get("val_loss") or 0.0)
    val_acc = float(final.get("val_acc") or 0.0)
    score = max(0.0, 2.4 + (val_acc * 1.7) - (val_loss * 0.3))
    metrics = {
        "score": score,
        "avg_ante_reached": max(0.0, 2.2 + (val_acc * 1.2)),
        "median_ante": max(0.0, 2.2 + (val_acc * 1.2)),
        "win_rate": max(0.0, min(1.0, val_acc * 0.9)),
        "hand_top1": max(0.0, min(1.0, val_acc)),
        "hand_top3": max(0.0, min(1.0, val_acc + 0.10)),
        "shop_top1": max(0.0, min(1.0, 1.0 - min(1.0, val_loss))),
        "illegal_action_rate": max(0.0, min(0.25, val_loss * 0.04)),
        "selfsup_p32_val_loss": val_loss,
        "selfsup_p32_val_acc": val_acc,
        "selfsup_p32_run_dir": str(summary.get("run_dir") or out_dir),
    }
    return {
        "status": "ok" if str(summary.get("status") or "") == "ok" else "failed",
        "metrics": metrics,
        "summary": summary,
    }


def _run_selfsup_future_value_seed_experiment(
    *,
    ctx: RunContext,
    exp: dict[str, Any],
    exp_dir: Path,
    seed: str,
    seed_idx: int,
    seed_total: int,
) -> dict[str, Any]:
    from trainer.selfsup.train_future_value import run_train_future_value

    eval_cfg = exp.get("eval") if isinstance(exp.get("eval"), dict) else {}
    cfg_rel = str(
        eval_cfg.get("config")
        or exp.get("selfsup_config")
        or "configs/experiments/p36_selfsup_future_value.yaml"
    )
    cfg_path = (ctx.repo_root / cfg_rel).resolve()
    max_samples = int(eval_cfg.get("max_samples") or eval_cfg.get("max_steps") or 0)
    out_dir = exp_dir / "selfsup_p36_future_runs" / f"seed_{seed_idx:03d}_{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = run_train_future_value(
        config_path=cfg_path,
        out_dir=out_dir,
        seed_override=_seed_to_int(seed),
        max_samples_override=(max_samples if max_samples > 0 else None),
        quiet=(not ctx.verbose),
    )
    final = summary.get("final_metrics") if isinstance(summary.get("final_metrics"), dict) else {}
    val_loss = float(final.get("val_loss") or 0.0)
    val_mae = float(final.get("val_mae") or 0.0)
    score = max(0.0, 2.5 + (1.0 / (1.0 + max(0.0, val_mae))) * 1.7 - (val_loss * 0.04))
    hand_top1 = max(0.0, min(1.0, 1.0 / (1.0 + max(0.0, val_mae))))
    metrics = {
        "score": score,
        "avg_ante_reached": max(0.0, 2.1 + hand_top1),
        "median_ante": max(0.0, 2.1 + hand_top1),
        "win_rate": max(0.0, min(1.0, hand_top1 * 0.85)),
        "hand_top1": hand_top1,
        "hand_top3": max(0.0, min(1.0, hand_top1 + 0.12)),
        "shop_top1": max(0.0, min(1.0, 1.0 - min(1.0, val_loss * 0.1))),
        "illegal_action_rate": max(0.0, min(0.30, val_loss * 0.02)),
        "selfsup_p36_future_val_loss": val_loss,
        "selfsup_p36_future_val_mae": val_mae,
        "selfsup_p36_future_run_dir": str(summary.get("run_dir") or out_dir),
    }
    return {
        "status": "ok" if str(summary.get("status") or "") == "ok" else "failed",
        "metrics": metrics,
        "summary": summary,
    }


def _run_selfsup_action_type_seed_experiment(
    *,
    ctx: RunContext,
    exp: dict[str, Any],
    exp_dir: Path,
    seed: str,
    seed_idx: int,
    seed_total: int,
) -> dict[str, Any]:
    from trainer.selfsup.train_action_type import run_train_action_type

    eval_cfg = exp.get("eval") if isinstance(exp.get("eval"), dict) else {}
    cfg_rel = str(
        eval_cfg.get("config")
        or exp.get("selfsup_config")
        or "configs/experiments/p36_selfsup_action_type.yaml"
    )
    cfg_path = (ctx.repo_root / cfg_rel).resolve()
    max_samples = int(eval_cfg.get("max_samples") or eval_cfg.get("max_steps") or 0)
    out_dir = exp_dir / "selfsup_p36_action_runs" / f"seed_{seed_idx:03d}_{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = run_train_action_type(
        config_path=cfg_path,
        out_dir=out_dir,
        seed_override=_seed_to_int(seed),
        max_samples_override=(max_samples if max_samples > 0 else None),
        quiet=(not ctx.verbose),
    )
    final = summary.get("final_metrics") if isinstance(summary.get("final_metrics"), dict) else {}
    val_loss = float(final.get("val_loss") or 0.0)
    val_acc = float(final.get("val_acc") or 0.0)
    score = max(0.0, 2.4 + (val_acc * 1.9) - (val_loss * 0.25))
    metrics = {
        "score": score,
        "avg_ante_reached": max(0.0, 2.2 + (val_acc * 1.2)),
        "median_ante": max(0.0, 2.2 + (val_acc * 1.2)),
        "win_rate": max(0.0, min(1.0, val_acc * 0.9)),
        "hand_top1": max(0.0, min(1.0, val_acc)),
        "hand_top3": max(0.0, min(1.0, val_acc + 0.15)),
        "shop_top1": max(0.0, min(1.0, 1.0 - min(1.0, val_loss * 0.1))),
        "illegal_action_rate": max(0.0, min(0.30, val_loss * 0.02)),
        "selfsup_p36_action_val_loss": val_loss,
        "selfsup_p36_action_val_acc": val_acc,
        "selfsup_p36_action_run_dir": str(summary.get("run_dir") or out_dir),
    }
    return {
        "status": "ok" if str(summary.get("status") or "") == "ok" else "failed",
        "metrics": metrics,
        "summary": summary,
    }


def _run_ssl_pretrain_seed_experiment(
    *,
    ctx: RunContext,
    exp: dict[str, Any],
    exp_dir: Path,
    seed: str,
    seed_idx: int,
    seed_total: int,
) -> dict[str, Any]:
    from trainer.experiments.ssl_trainer import run_ssl_pretrain

    eval_cfg = exp.get("eval") if isinstance(exp.get("eval"), dict) else {}
    cfg_rel = str(
        eval_cfg.get("config")
        or exp.get("ssl_config")
        or "configs/experiments/p37_ssl_pretrain.yaml"
    )
    cfg_path = (ctx.repo_root / cfg_rel).resolve()
    max_samples = int(eval_cfg.get("max_samples") or eval_cfg.get("max_steps") or 0)
    out_dir = exp_dir / "ssl_pretrain_runs" / f"seed_{seed_idx:03d}_{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = run_ssl_pretrain(
        config_path=cfg_path,
        out_dir=out_dir,
        seed_override=_seed_to_int(seed),
        max_samples_override=(max_samples if max_samples > 0 else None),
        quiet=(not ctx.verbose),
    )
    final = summary.get("final_metrics") if isinstance(summary.get("final_metrics"), dict) else {}
    val_loss = float(final.get("val_loss") or 0.0)
    val_pos_cos = float(final.get("val_pos_cos") or 0.0)
    emb_std = float(final.get("val_embedding_std") or 0.0)
    win_rate = max(0.0, min(1.0, (val_pos_cos + 1.0) * 0.5))
    score = max(0.0, 2.2 + (val_pos_cos * 1.8) - (val_loss * 0.35))
    metrics = {
        "score": score,
        "avg_reward": score,
        "avg_ante_reached": score,
        "median_ante": score,
        "win_rate": win_rate,
        "hand_top1": win_rate,
        "hand_top3": max(0.0, min(1.0, win_rate + 0.10)),
        "shop_top1": max(0.0, min(1.0, 1.0 / (1.0 + max(0.0, val_loss)))),
        "illegal_action_rate": max(0.0, min(0.30, val_loss * 0.05)),
        "ssl_val_loss": val_loss,
        "ssl_val_pos_cos": val_pos_cos,
        "ssl_val_embedding_std": emb_std,
        "ssl_run_dir": str(summary.get("run_dir") or out_dir),
    }
    return {
        "status": "ok" if str(summary.get("status") or "") == "ok" else "failed",
        "metrics": metrics,
        "summary": summary,
    }


def _run_ssl_probe_seed_experiment(
    *,
    ctx: RunContext,
    exp: dict[str, Any],
    exp_dir: Path,
    seed: str,
    seed_idx: int,
    seed_total: int,
) -> dict[str, Any]:
    from trainer.experiments.ssl_probe import run_ssl_probe

    eval_cfg = exp.get("eval") if isinstance(exp.get("eval"), dict) else {}
    cfg_rel = str(
        eval_cfg.get("config")
        or exp.get("ssl_config")
        or "configs/experiments/p37_ssl_probe.yaml"
    )
    cfg_path = (ctx.repo_root / cfg_rel).resolve()
    max_samples = int(eval_cfg.get("max_samples") or eval_cfg.get("max_steps") or 0)
    out_dir = exp_dir / "ssl_probe_runs" / f"seed_{seed_idx:03d}_{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = run_ssl_probe(
        config_path=cfg_path,
        out_dir=out_dir,
        seed_override=_seed_to_int(seed),
        max_samples_override=(max_samples if max_samples > 0 else None),
        quiet=(not ctx.verbose),
    )
    final = summary.get("final_metrics") if isinstance(summary.get("final_metrics"), dict) else {}
    ssl_acc = float(final.get("ssl_val_acc") or 0.0)
    baseline_acc = float(final.get("baseline_val_acc") or 0.0)
    delta_acc = float(final.get("delta_val_acc") or 0.0)
    ssl_val_loss = float(final.get("ssl_val_loss") or 0.0)
    score = max(0.0, 2.0 + (ssl_acc * 2.2) + (delta_acc * 1.5) - (ssl_val_loss * 0.2))
    metrics = {
        "score": score,
        "avg_reward": score,
        "avg_ante_reached": score,
        "median_ante": score,
        "win_rate": max(0.0, min(1.0, ssl_acc)),
        "hand_top1": max(0.0, min(1.0, ssl_acc)),
        "hand_top3": max(0.0, min(1.0, ssl_acc + 0.12)),
        "shop_top1": max(0.0, min(1.0, 1.0 - min(1.0, ssl_val_loss))),
        "illegal_action_rate": max(0.0, min(0.20, ssl_val_loss * 0.05)),
        "ssl_probe_baseline_acc": baseline_acc,
        "ssl_probe_ssl_acc": ssl_acc,
        "ssl_probe_delta_acc": delta_acc,
        "ssl_probe_val_loss": ssl_val_loss,
        "ssl_probe_run_dir": str(summary.get("run_dir") or out_dir),
    }
    return {
        "status": "ok" if str(summary.get("status") or "") == "ok" else "failed",
        "metrics": metrics,
        "summary": summary,
    }


def _run_rl_selfplay_seed_experiment(
    *,
    ctx: RunContext,
    exp: dict[str, Any],
    exp_dir: Path,
    seed: str,
    seed_idx: int,
    seed_total: int,
    progress_path: Path,
    started_ts: float,
    run_id: str,
    exp_id: str,
) -> dict[str, Any]:
    from trainer.rl.ppo_skeleton import run_ppo_skeleton

    eval_cfg = exp.get("eval") if isinstance(exp.get("eval"), dict) else {}
    algo = str(exp.get("algo") or eval_cfg.get("algo") or "ppo").strip().lower()
    if algo not in {"ppo", "ppo_skeleton"}:
        raise ValueError(f"unsupported RL algo for p37 skeleton: {algo}")

    episodes = int(eval_cfg.get("episodes") or exp.get("episodes") or 10)
    max_steps = int(eval_cfg.get("max_steps_per_episode") or exp.get("max_steps_per_episode") or 320)
    lr = float(eval_cfg.get("lr") or exp.get("lr") or 1e-3)
    gamma = float(eval_cfg.get("gamma") or exp.get("gamma") or 0.99)
    entropy_coef = float(eval_cfg.get("entropy_coef") or exp.get("entropy_coef") or 0.01)
    value_coef = float(eval_cfg.get("value_coef") or exp.get("value_coef") or 0.5)
    reward_mode = str(eval_cfg.get("reward_mode") or exp.get("reward_mode") or "score_delta")

    out_dir = exp_dir / "rl_selfplay_runs" / f"seed_{seed_idx:03d}_{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = run_ppo_skeleton(
        episodes=max(1, episodes),
        seed=seed,
        gamma=gamma,
        lr=lr,
        entropy_coef=entropy_coef,
        value_coef=value_coef,
        backend=str(exp.get("backend") or "sim"),
        max_steps_per_episode=max(1, max_steps),
        reward_mode=reward_mode,
        out_dir=out_dir,
        run_id=f"{run_id}_{exp_id}_{seed}",
        quiet=(not ctx.verbose),
    )
    metrics = summary.get("metrics") if isinstance(summary.get("metrics"), dict) else {}
    episode_rows = summary.get("episodes") if isinstance(summary.get("episodes"), list) else []

    for row in episode_rows:
        if not isinstance(row, dict):
            continue
        ep_idx = int(row.get("episode_idx") or 0)
        ep_reward = float(row.get("total_reward") or 0.0)
        ep_len = int(row.get("length") or 0)
        ep_wall = float(row.get("wall_time") or 0.0)
        append_experiment_progress_event(
            progress_path,
            run_id=run_id,
            exp_id=exp_id,
            phase="eval",
            stage="episode",
            status="ok",
            seed=seed,
            step_or_epoch=ep_idx,
            elapsed_sec=time.time() - started_ts,
            wall_time_sec=ep_wall,
            metrics={
                "total_reward": ep_reward,
                "episode_length": ep_len,
            },
            message="rl_selfplay_episode",
            extra={
                "episode_idx": ep_idx,
                "seed_index": seed_idx,
                "seed_total": seed_total,
            },
        )

    avg_reward = float(metrics.get("avg_reward") or 0.0)
    std_reward = float(metrics.get("std_reward") or 0.0)
    best_reward = float(metrics.get("best_episode_reward") or 0.0)
    avg_len = float(metrics.get("episode_length") or 0.0)
    loss = float(metrics.get("loss") or 0.0)
    win_rate = 1.0 if avg_reward > 0 else 0.0

    converted = {
        "score": avg_reward,
        "avg_reward": avg_reward,
        "std_reward": std_reward,
        "best_episode_reward": best_reward,
        "episode_length": avg_len,
        "avg_ante_reached": avg_reward,
        "median_ante": avg_reward,
        "win_rate": win_rate,
        "hand_top1": 0.0,
        "hand_top3": 0.0,
        "shop_top1": 0.0,
        "illegal_action_rate": 0.0,
        "loss": loss,
        "rl_run_dir": str(summary.get("run_dir") or out_dir),
        "rl_algo": algo,
    }
    return {
        "status": "ok" if str(summary.get("status") or "") == "ok" else "failed",
        "metrics": converted,
        "summary": summary,
    }


def _numeric_mean_from_p38(summary: dict[str, Any], metric: str, side: str) -> float:
    rows = summary.get("numeric_metrics") if isinstance(summary.get("numeric_metrics"), list) else []
    for row in rows:
        if not isinstance(row, dict):
            continue
        if str(row.get("metric") or "") != metric:
            continue
        side_block = row.get(side) if isinstance(row.get(side), dict) else {}
        value = side_block.get("mean")
        try:
            return float(value)
        except Exception:
            return 0.0
    return 0.0


def _avg_abs_relative_diff(summary: dict[str, Any]) -> float:
    rows = summary.get("numeric_metrics") if isinstance(summary.get("numeric_metrics"), list) else []
    vals: list[float] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        rel = row.get("relative_diff_pct")
        try:
            vals.append(abs(float(rel)))
        except Exception:
            continue
    if not vals:
        return 0.0
    return float(sum(vals) / len(vals))


def _append_p38_orchestrator_summary(ctx: RunContext, payload: dict[str, Any]) -> None:
    path = ctx.run_root / "p38_summary.json"
    existing = read_json(path)
    rows: list[dict[str, Any]] = []
    if isinstance(existing, dict):
        raw_rows = existing.get("rows")
        if isinstance(raw_rows, list):
            for item in raw_rows:
                if isinstance(item, dict):
                    rows.append(item)
    rows.append(payload)
    write_json(
        path,
        {
            "schema": "p38_orchestrator_summary_v1",
            "generated_at": now_iso(),
            "run_id": ctx.run_id,
            "rows": rows,
        },
    )


def _normalize_text_list(raw: Any) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    if isinstance(raw, str):
        values = [x.strip() for x in raw.split(",")]
    elif isinstance(raw, list):
        values = [str(x).strip() for x in raw]
    else:
        values = []
    for value in values:
        if not value or value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def _read_json_list(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []
    if not isinstance(obj, list):
        return []
    out: list[dict[str, Any]] = []
    for item in obj:
        if isinstance(item, dict):
            out.append(item)
    return out


def _append_p39_orchestrator_summary(ctx: RunContext, payload: dict[str, Any]) -> None:
    path = ctx.run_root / "p39_summary.json"
    existing = read_json(path)
    rows: list[dict[str, Any]] = []
    if isinstance(existing, dict):
        raw_rows = existing.get("rows")
        if isinstance(raw_rows, list):
            for item in raw_rows:
                if isinstance(item, dict):
                    rows.append(item)
    rows.append(payload)
    write_json(
        path,
        {
            "schema": "p39_orchestrator_summary_v1",
            "generated_at": now_iso(),
            "run_id": ctx.run_id,
            "rows": rows,
        },
    )


def _append_p40_orchestrator_summary(ctx: RunContext, payload: dict[str, Any]) -> None:
    path = ctx.run_root / "p40_summary.json"
    existing = read_json(path)
    rows: list[dict[str, Any]] = []
    if isinstance(existing, dict):
        raw_rows = existing.get("rows")
        if isinstance(raw_rows, list):
            for item in raw_rows:
                if isinstance(item, dict):
                    rows.append(item)
    rows.append(payload)
    write_json(
        path,
        {
            "schema": "p40_orchestrator_summary_v1",
            "generated_at": now_iso(),
            "run_id": ctx.run_id,
            "rows": rows,
        },
    )


def _append_p41_orchestrator_summary(ctx: RunContext, payload: dict[str, Any]) -> None:
    path = ctx.run_root / "p41_summary.json"
    existing = read_json(path)
    rows: list[dict[str, Any]] = []
    if isinstance(existing, dict):
        raw_rows = existing.get("rows")
        if isinstance(raw_rows, list):
            for item in raw_rows:
                if isinstance(item, dict):
                    rows.append(item)
    rows.append(payload)
    write_json(
        path,
        {
            "schema": "p41_orchestrator_summary_v1",
            "generated_at": now_iso(),
            "run_id": ctx.run_id,
            "rows": rows,
        },
    )


def _append_p42_orchestrator_summary(ctx: RunContext, payload: dict[str, Any]) -> None:
    path = ctx.run_root / "p42_summary.json"
    existing = read_json(path)
    rows: list[dict[str, Any]] = []
    if isinstance(existing, dict):
        raw_rows = existing.get("rows")
        if isinstance(raw_rows, list):
            for item in raw_rows:
                if isinstance(item, dict):
                    rows.append(item)
    rows.append(payload)
    write_json(
        path,
        {
            "schema": "p42_orchestrator_summary_v1",
            "generated_at": now_iso(),
            "run_id": ctx.run_id,
            "rows": rows,
        },
    )


def _pick_arena_focus_row(summary_rows: list[dict[str, Any]], focus_policy: str) -> dict[str, Any] | None:
    token = str(focus_policy or "").strip().lower()
    if token:
        for row in summary_rows:
            if str(row.get("policy_id") or "").strip().lower() == token:
                return row
    if summary_rows:
        return summary_rows[0]
    return None


def _run_policy_arena_seed_experiment(
    *,
    ctx: RunContext,
    exp: dict[str, Any],
    exp_dir: Path,
    seed: str,
    seed_idx: int,
    seed_total: int,
) -> dict[str, Any]:
    eval_cfg = exp.get("eval") if isinstance(exp.get("eval"), dict) else {}
    mode = str(eval_cfg.get("mode") or exp.get("mode") or "long_episode")
    backend = str(eval_cfg.get("backend") or exp.get("backend") or "sim")
    episodes = int(eval_cfg.get("episodes_per_seed") or exp.get("episodes_per_seed") or exp.get("episodes") or 1)
    max_steps = int(eval_cfg.get("max_steps") or exp.get("max_steps") or 160)
    model_path = str(eval_cfg.get("model_path") or exp.get("model_path") or "")
    skip_unavailable = bool(eval_cfg.get("skip_unavailable") or exp.get("skip_unavailable") or False)
    policies = _normalize_text_list(eval_cfg.get("policies") or exp.get("policies") or "heuristic_baseline,search_expert,model_policy")
    if not policies:
        policies = ["heuristic_baseline", "search_expert", "model_policy"]

    out_dir = exp_dir / "policy_arena_runs" / f"seed_{seed_idx:03d}_{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    arena_root = out_dir / "arena_runs"
    arena_run_id = f"seed_{seed_idx:03d}_{seed}"
    arena_cmd = [
        sys.executable,
        "-B",
        "-m",
        "trainer.policy_arena.arena_runner",
        "--out-dir",
        str(arena_root),
        "--run-id",
        arena_run_id,
        "--backend",
        backend,
        "--mode",
        mode,
        "--policies",
        ",".join(policies),
        "--seeds",
        seed,
        "--episodes-per-seed",
        str(max(1, episodes)),
        "--max-steps",
        str(max(1, max_steps)),
    ]
    if model_path:
        arena_cmd.extend(["--model-path", model_path])
    if skip_unavailable:
        arena_cmd.append("--skip-unavailable")

    timeout_sec = max(600, max_steps * max(1, episodes) * max(1, len(policies)) * 2)
    arena_result = run_process(arena_cmd, cwd=ctx.repo_root, timeout_sec=timeout_sec)

    run_dir = arena_root / arena_run_id
    manifest_path = run_dir / "run_manifest.json"
    summary_path = run_dir / "summary_table.json"
    bucket_path = run_dir / "bucket_metrics.json"
    warnings_path = run_dir / "warnings.log"

    manifest = read_json(manifest_path) or {}
    summary_rows = _read_json_list(summary_path)
    focus_policy = str(eval_cfg.get("focus_policy") or eval_cfg.get("candidate_policy") or policies[0])
    focus_row = _pick_arena_focus_row(summary_rows, focus_policy)

    champion_rules_enabled = bool(eval_cfg.get("enable_champion_rules") or exp.get("enable_champion_rules") or False)
    champion_result = {"returncode": 0, "stdout": "", "stderr": "", "command": []}
    champion_payload: dict[str, Any] | None = None
    if champion_rules_enabled and summary_path.exists():
        champion_out = out_dir / "champion_eval"
        champion_cmd = [
            sys.executable,
            "-B",
            "-m",
            "trainer.policy_arena.champion_rules",
            "--summary-json",
            str(summary_path),
            "--out-dir",
            str(champion_out),
        ]
        champion_json_path = str(eval_cfg.get("champion_json") or exp.get("champion_json") or "").strip()
        if champion_json_path:
            champion_cmd.extend(["--champion-json", champion_json_path])
        candidate_policy = str(eval_cfg.get("candidate_policy") or "").strip()
        if candidate_policy:
            champion_cmd.extend(["--candidate-policy", candidate_policy])
        champion_policy = str(eval_cfg.get("champion_policy") or "").strip()
        if champion_policy:
            champion_cmd.extend(["--champion-policy", champion_policy])
        champion_result = run_process(champion_cmd, cwd=ctx.repo_root, timeout_sec=300)
        try:
            lines = [ln for ln in str(champion_result.get("stdout") or "").splitlines() if ln.strip()]
            if lines:
                last = json.loads(lines[-1])
                if isinstance(last, dict):
                    json_path = Path(str(last.get("json") or ""))
                    if json_path.exists():
                        champion_payload = read_json(json_path) or {}
        except Exception:
            champion_payload = None

    mean_score = 0.0
    mean_rounds = 0.0
    win_rate = 0.0
    invalid_rate = 0.0
    p90_score = 0.0
    episodes_total = int(manifest.get("episode_total") or 0)
    if isinstance(focus_row, dict):
        mean_score = float(focus_row.get("mean_total_score") or 0.0)
        mean_rounds = float(focus_row.get("mean_rounds_survived") or 0.0)
        win_rate = float(focus_row.get("win_rate") or 0.0)
        invalid_rate = float(focus_row.get("invalid_action_rate") or 0.0)
        p90_score = float(focus_row.get("p90_total_score") or mean_score)
        episodes_total = int(focus_row.get("episodes") or episodes_total)

    metrics = {
        "score": mean_score,
        "avg_reward": mean_score,
        "best_episode_reward": p90_score,
        "avg_ante_reached": mean_rounds,
        "median_ante": mean_rounds,
        "win_rate": win_rate,
        "hand_top1": 0.0,
        "hand_top3": 0.0,
        "shop_top1": max(0.0, 1.0 - min(1.0, invalid_rate)),
        "illegal_action_rate": invalid_rate,
        "p39_mean_total_score": mean_score,
        "p39_mean_rounds_survived": mean_rounds,
        "p39_invalid_action_rate": invalid_rate,
        "p39_episodes": episodes_total,
        "p39_focus_policy": str(focus_row.get("policy_id")) if isinstance(focus_row, dict) else "",
        "p39_run_dir": str(run_dir),
        "p39_summary_path": str(summary_path),
        "p39_bucket_path": str(bucket_path),
        "p39_warnings_path": str(warnings_path),
    }

    summary_payload = {
        "schema": "p39_orchestrator_seed_summary_v1",
        "generated_at": now_iso(),
        "exp_id": str(exp.get("id") or ""),
        "seed": seed,
        "seed_index": seed_idx,
        "seed_total": seed_total,
        "policies": policies,
        "focus_policy": focus_policy,
        "arena_returncode": int(arena_result.get("returncode") or 0),
        "champion_rules_returncode": int(champion_result.get("returncode") or 0),
        "run_dir": str(run_dir),
        "summary_path": str(summary_path),
        "bucket_path": str(bucket_path),
        "warnings_path": str(warnings_path),
        "metrics": metrics,
        "champion_decision": champion_payload if isinstance(champion_payload, dict) else {},
    }
    _append_p39_orchestrator_summary(ctx, summary_payload)

    status = "ok"
    if int(arena_result.get("returncode") or 0) != 0:
        status = "failed"
    if not isinstance(focus_row, dict):
        status = "failed"

    return {
        "status": status,
        "metrics": metrics,
        "summary": {
            "arena": arena_result,
            "champion_rules": champion_result,
            "payload": summary_payload,
        },
    }


def _run_world_model_seed_experiment(
    *,
    ctx: RunContext,
    exp: dict[str, Any],
    exp_dir: Path,
    seed: str,
    seed_idx: int,
    seed_total: int,
) -> dict[str, Any]:
    from trainer.world_model.eval import run_world_model_eval
    from trainer.world_model.planning_hook import run_world_model_assist_compare
    from trainer.world_model.train import run_world_model_train

    eval_cfg = exp.get("eval") if isinstance(exp.get("eval"), dict) else {}
    exp_type = str(exp.get("experiment_type") or "world_model_train").strip().lower()
    cfg_rel = str(
        eval_cfg.get("config")
        or exp.get("world_model_config")
        or (
            "configs/experiments/p45_world_model_smoke.yaml"
            if ctx.mode == "quick"
            else "configs/experiments/p45_world_model_nightly.yaml"
        )
    )
    cfg_path = (ctx.repo_root / cfg_rel).resolve()
    world_model_root = exp_dir / "world_model_runs" / f"seed_{seed_idx:03d}_{seed}"
    world_model_root.mkdir(parents=True, exist_ok=True)
    run_name = f"seed_{seed_idx:03d}_{seed}"
    quick_flag = bool(eval_cfg.get("quick", False) or ctx.mode == "quick")

    summary: dict[str, Any]
    if exp_type == "world_model_eval":
        checkpoint_path = str(eval_cfg.get("checkpoint") or exp.get("checkpoint") or "").strip()
        if not checkpoint_path:
            raise ValueError("world_model_eval requires checkpoint")
        summary = run_world_model_eval(
            config_path=cfg_path,
            checkpoint_path=checkpoint_path,
            dataset_manifest_path=(str(eval_cfg.get("dataset_manifest") or "").strip() or None),
            out_dir=world_model_root / "eval",
            run_id=run_name,
            quick=quick_flag,
        )
    elif exp_type == "world_model_assist_compare":
        checkpoint_path = str(eval_cfg.get("checkpoint") or exp.get("checkpoint") or "").strip()
        if not checkpoint_path:
            raise ValueError("world_model_assist_compare requires checkpoint")
        summary = run_world_model_assist_compare(
            config_path=cfg_path,
            checkpoint_path=checkpoint_path,
            out_dir=world_model_root / "assist_compare",
            run_id=run_name,
            quick=quick_flag,
        )
    else:
        summary = run_world_model_train(
            config_path=cfg_path,
            out_dir=world_model_root,
            run_id=run_name,
            quick=quick_flag,
            seed_override=_seed_to_int(seed),
        )

    eval_summary = summary.get("eval_summary") if isinstance(summary.get("eval_summary"), dict) else {}
    eval_metrics = eval_summary.get("metrics") if isinstance(eval_summary.get("metrics"), dict) else {}
    assist_summary = summary.get("assist_summary") if isinstance(summary.get("assist_summary"), dict) else {}
    assist_payload = assist_summary.get("summary") if isinstance(assist_summary.get("summary"), dict) else {}
    candidate_policy = str(
        (eval_cfg.get("candidate_policy") or exp.get("candidate_policy") or "heuristic_wm_assist")
    )
    summary_table_json = str(
        ((assist_payload.get("arena_summary") or {}).get("summary_table_json"))
        if isinstance(assist_payload.get("arena_summary"), dict)
        else ""
    )
    arena_rows = _read_json_list(Path(summary_table_json)) if summary_table_json else []
    arena_row = _pick_arena_focus_row(arena_rows, candidate_policy)

    candidate_score = float((arena_row or {}).get("mean_total_score") or 0.0)
    candidate_rounds = float((arena_row or {}).get("mean_rounds_survived") or 0.0)
    candidate_win = float((arena_row or {}).get("win_rate") or 0.0)
    candidate_invalid = float((arena_row or {}).get("invalid_action_rate") or 0.0)
    reward_error = float(eval_metrics.get("reward_prediction_error") or 0.0)
    latent_error = float(eval_metrics.get("latent_transition_error") or 0.0)
    uncertainty_corr = float(
        ((eval_metrics.get("uncertainty_calibration") or {}).get("uncertainty_error_pearson"))
        if isinstance(eval_metrics.get("uncertainty_calibration"), dict)
        else 0.0
    )
    fallback_score = max(0.0, 5.0 - min(4.0, reward_error * 0.02) - min(1.0, latent_error * 10.0))
    metrics = {
        "score": (candidate_score if candidate_score > 0.0 else fallback_score),
        "avg_reward": (candidate_score if candidate_score > 0.0 else fallback_score),
        "best_episode_reward": max(candidate_score, fallback_score),
        "avg_ante_reached": (candidate_rounds if candidate_rounds > 0.0 else max(0.0, 3.0 - latent_error * 10.0)),
        "median_ante": (candidate_rounds if candidate_rounds > 0.0 else max(0.0, 3.0 - latent_error * 10.0)),
        "win_rate": (candidate_win if candidate_win > 0.0 else max(0.0, min(1.0, 0.5 + uncertainty_corr * 0.25))),
        "hand_top1": max(0.0, min(1.0, 1.0 / (1.0 + max(0.0, reward_error)))),
        "hand_top3": max(0.0, min(1.0, 1.15 / (1.0 + max(0.0, reward_error)))),
        "shop_top1": max(0.0, min(1.0, 1.0 - candidate_invalid)),
        "illegal_action_rate": candidate_invalid,
        "world_model_reward_prediction_error": reward_error,
        "world_model_latent_transition_error": latent_error,
        "world_model_uncertainty_pearson": uncertainty_corr,
        "world_model_run_dir": str(summary.get("run_dir") or world_model_root),
        "world_model_eval_metrics_json": str(eval_summary.get("eval_metrics_json") or ""),
        "world_model_assist_compare_json": str(assist_summary.get("assist_compare_summary_json") or ""),
        "world_model_best_checkpoint": str(summary.get("best_checkpoint") or ""),
    }
    return {
        "status": "ok" if str(summary.get("status") or "") == "ok" else "failed",
        "metrics": metrics,
        "summary": summary,
    }


def _run_imagination_seed_experiment(
    *,
    ctx: RunContext,
    exp: dict[str, Any],
    exp_dir: Path,
    seed: str,
    seed_idx: int,
    seed_total: int,
) -> dict[str, Any]:
    from trainer.world_model.imagination_pipeline import run_imagination_pipeline

    eval_cfg = exp.get("eval") if isinstance(exp.get("eval"), dict) else {}
    cfg_rel = str(
        eval_cfg.get("config")
        or exp.get("imagination_config")
        or (
            "configs/experiments/p46_imagination_smoke.yaml"
            if ctx.mode == "quick"
            else "configs/experiments/p46_imagination_nightly.yaml"
        )
    )
    cfg_path = (ctx.repo_root / cfg_rel).resolve()
    imag_root = exp_dir / "imagination_runs" / f"seed_{seed_idx:03d}_{seed}"
    imag_root.mkdir(parents=True, exist_ok=True)
    run_name = f"seed_{seed_idx:03d}_{seed}"

    summary = run_imagination_pipeline(
        config_path=cfg_path,
        out_dir=imag_root,
        run_id=run_name,
        quick=bool(eval_cfg.get("quick", False) or ctx.mode == "quick"),
        dry_run=bool(ctx.dry_run),
        seeds_override=[seed],
    )
    metrics = summary.get("metrics") if isinstance(summary.get("metrics"), dict) else {}
    filtered_score = float(metrics.get("p46_filtered_score") or metrics.get("score") or 0.0)
    real_score = float(metrics.get("p46_real_only_score") or 0.0)
    acceptance_rate = float(metrics.get("p46_imagined_acceptance_rate") or 0.0)
    result_metrics = {
        "score": filtered_score if filtered_score > 0.0 else real_score,
        "avg_reward": filtered_score if filtered_score > 0.0 else real_score,
        "best_episode_reward": max(filtered_score, real_score),
        "avg_ante_reached": filtered_score if filtered_score > 0.0 else real_score,
        "median_ante": filtered_score if filtered_score > 0.0 else real_score,
        "win_rate": float(metrics.get("win_rate") or 0.0),
        "hand_top1": 0.0,
        "hand_top3": 0.0,
        "shop_top1": max(0.0, 1.0 - float(metrics.get("illegal_action_rate") or 0.0)),
        "illegal_action_rate": float(metrics.get("illegal_action_rate") or 0.0),
        "p46_real_only_score": real_score,
        "p46_filtered_score": filtered_score,
        "p46_filtered_delta_vs_real_only": float(metrics.get("p46_filtered_delta_vs_real_only") or 0.0),
        "p46_imagined_acceptance_rate": acceptance_rate,
        "p46_imagined_sample_count": float(metrics.get("p46_imagined_sample_count") or 0.0),
        "p46_pipeline_summary_json": str(summary.get("pipeline_summary_json") or ""),
        "p46_promotion_decision_json": str(summary.get("promotion_decision_json") or ""),
        "p46_triage_report_json": str(summary.get("triage_report_json") or ""),
    }
    return {
        "status": "ok" if str(summary.get("status") or "") == "ok" else "failed",
        "metrics": result_metrics,
        "summary": summary,
    }


def _run_model_based_search_seed_experiment(
    *,
    ctx: RunContext,
    exp: dict[str, Any],
    exp_dir: Path,
    seed: str,
    seed_idx: int,
    seed_total: int,
) -> dict[str, Any]:
    from trainer.world_model.model_based_search import run_model_based_search_pipeline

    eval_cfg = exp.get("eval") if isinstance(exp.get("eval"), dict) else {}
    cfg_rel = str(
        eval_cfg.get("config")
        or exp.get("model_based_search_config")
        or (
            "configs/experiments/p47_wm_search_smoke.yaml"
            if ctx.mode == "quick"
            else "configs/experiments/p47_wm_search_nightly.yaml"
        )
    )
    cfg_path = (ctx.repo_root / cfg_rel).resolve()
    run_root = exp_dir / "model_based_search_runs" / f"seed_{seed_idx:03d}_{seed}"
    run_root.mkdir(parents=True, exist_ok=True)
    run_name = f"seed_{seed_idx:03d}_{seed}"

    summary = run_model_based_search_pipeline(
        config_path=cfg_path,
        out_dir=run_root,
        run_id=run_name,
        quick=bool(eval_cfg.get("quick", False) or ctx.mode == "quick"),
        dry_run=bool(ctx.dry_run),
        seeds_override=[seed],
    )
    metrics = summary.get("metrics") if isinstance(summary.get("metrics"), dict) else {}
    candidate_score = float(metrics.get("p47_candidate_score") or metrics.get("score") or 0.0)
    baseline_score = float(metrics.get("p47_baseline_score") or 0.0)
    best_score = float(metrics.get("p47_best_variant_score") or candidate_score)
    result_metrics = {
        "score": candidate_score if candidate_score > 0.0 else best_score,
        "avg_reward": candidate_score if candidate_score > 0.0 else best_score,
        "best_episode_reward": best_score,
        "avg_ante_reached": candidate_score if candidate_score > 0.0 else best_score,
        "median_ante": candidate_score if candidate_score > 0.0 else best_score,
        "win_rate": float(metrics.get("win_rate") or 0.0),
        "hand_top1": 0.0,
        "hand_top3": 0.0,
        "shop_top1": max(0.0, 1.0 - float(metrics.get("illegal_action_rate") or 0.0)),
        "illegal_action_rate": float(metrics.get("illegal_action_rate") or 0.0),
        "p47_baseline_score": baseline_score,
        "p47_candidate_score": candidate_score,
        "p47_best_variant_score": best_score,
        "p47_candidate_delta_vs_baseline": float(metrics.get("p47_candidate_delta_vs_baseline") or (candidate_score - baseline_score)),
        "p47_best_variant_delta_vs_baseline": float(metrics.get("p47_best_variant_delta_vs_baseline") or (best_score - baseline_score)),
        "p47_pipeline_summary_json": str(summary.get("pipeline_summary_json") or ""),
        "p47_promotion_decision_json": str(summary.get("promotion_decision_json") or ""),
        "p47_triage_report_json": str(summary.get("triage_report_json") or ""),
    }
    return {
        "status": "ok" if str(summary.get("status") or "") == "ok" else "failed",
        "metrics": result_metrics,
        "summary": summary,
    }


def _run_closed_loop_seed_experiment(
    *,
    ctx: RunContext,
    exp: dict[str, Any],
    exp_dir: Path,
    seed: str,
    seed_idx: int,
    seed_total: int,
) -> dict[str, Any]:
    eval_cfg = exp.get("eval") if isinstance(exp.get("eval"), dict) else {}
    exp_type = str(exp.get("experiment_type") or "").strip().lower()
    is_p44 = exp_type in {"p44_distributed_rl"}
    is_p42 = exp_type in {"closed_loop_rl_candidate", "rl_candidate_pipeline", "p42_rl_candidate", "p42_rl_candidate_pipeline"}
    is_v2 = is_p44 or is_p42 or exp_type in {"closed_loop_improvement_v2", "p41_closed_loop_v2", "closed_loop_v2"}
    cfg_rel = str(
        eval_cfg.get("config")
        or exp.get("closed_loop_config")
        or (
            "configs/experiments/p44_closed_loop_rl_smoke.yaml"
            if is_p44
            else (
                "configs/experiments/p42_closed_loop_rl_smoke.yaml"
                if is_p42
            else ("configs/experiments/p41_closed_loop_v2_smoke.yaml" if is_v2 else "configs/experiments/p40_closed_loop_smoke.yaml")
            )
        )
    )
    cfg_path = (ctx.repo_root / cfg_rel).resolve()
    out_dir = exp_dir / "closed_loop_runs" / f"seed_{seed_idx:03d}_{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    run_id = f"seed_{seed_idx:03d}_{seed}"

    cmd = [
        sys.executable,
        "-B",
        "-m",
        "trainer.closed_loop.closed_loop_runner",
        "--config",
        str(cfg_path),
        "--out-dir",
        str(out_dir),
        "--run-id",
        run_id,
        "--seeds",
        seed,
    ]
    if bool(eval_cfg.get("quick", False)) or ctx.mode == "quick":
        cmd.append("--quick")

    timeout_sec = int(eval_cfg.get("timeout_sec") or 3600)
    result = run_process(cmd, cwd=ctx.repo_root, timeout_sec=max(600, timeout_sec))

    run_manifest_path = out_dir / "run_manifest.json"
    summary_table_path = out_dir / "summary_table.json"
    decision_path = out_dir / "promotion_decision.json"

    run_manifest = read_json(run_manifest_path) or {}
    decision_payload = read_json(decision_path) or {}
    summary_rows = _read_json_list(summary_table_path)
    summary_row = summary_rows[0] if summary_rows else {}

    closed_loop_ok = int(result.get("returncode") or 0) == 0 and run_manifest_path.exists()
    if not closed_loop_ok:
        payload = {
            "schema": (
                "p42_orchestrator_seed_summary_v1"
                if is_p42
                else ("p41_orchestrator_seed_summary_v1" if is_v2 else "p40_orchestrator_seed_summary_v1")
            ),
            "generated_at": now_iso(),
            "exp_id": str(exp.get("id") or ""),
            "seed": seed,
            "seed_index": seed_idx,
            "seed_total": seed_total,
            "closed_loop_config": str(cfg_path),
            "closed_loop_returncode": int(result.get("returncode") or 0),
            "run_dir": str(out_dir),
            "run_manifest_path": str(run_manifest_path),
            "summary_table_path": str(summary_table_path),
            "promotion_decision_path": str(decision_path),
            "summary_row": summary_row if isinstance(summary_row, dict) else {},
            "pipeline_type": (
                "p44_distributed_rl"
                if is_p44
                else ("p42_rl_candidate" if is_p42 else ("p41_closed_loop_v2" if is_v2 else "p40_closed_loop_v1"))
            ),
            "metrics": {},
        }
        if is_p44 or is_p42:
            _append_p42_orchestrator_summary(ctx, payload)
        elif is_v2:
            _append_p41_orchestrator_summary(ctx, payload)
        else:
            _append_p40_orchestrator_summary(ctx, payload)
        return {
            "status": "failed",
            "metrics": {},
            "summary": {
                "closed_loop": result,
                "payload": payload,
                "run_manifest": run_manifest,
                "decision": decision_payload,
            },
        }

    candidate_score = _as_number(decision_payload.get("candidate_score"))
    champion_score = _as_number(decision_payload.get("champion_score"))
    score_delta = _as_number(decision_payload.get("score_delta"))
    candidate_win = _as_number(decision_payload.get("candidate_win_rate"))
    invalid_rate = _as_number(decision_payload.get("candidate_invalid_action_rate"))
    candidate_score_num = candidate_score if candidate_score is not None else 0.0
    champion_score_num = champion_score if champion_score is not None else 0.0
    score_delta_num = score_delta if score_delta is not None else (candidate_score_num - champion_score_num)
    candidate_win_num = candidate_win if candidate_win is not None else 0.0
    invalid_rate_num = invalid_rate if invalid_rate is not None else 0.0
    recommendation = str(decision_payload.get("recommendation") or "")
    recommend_promotion = bool(decision_payload.get("recommend_promotion", False))
    key_prefix = "p44" if is_p44 else ("p42" if is_p42 else ("p41" if is_v2 else "p40"))

    metrics = {
        "score": candidate_score_num,
        "avg_reward": candidate_score_num,
        "best_episode_reward": max(candidate_score_num, champion_score_num),
        "avg_ante_reached": candidate_score_num,
        "median_ante": candidate_score_num,
        "win_rate": candidate_win_num,
        "hand_top1": 0.0,
        "hand_top3": 0.0,
        "shop_top1": max(0.0, 1.0 - min(1.0, invalid_rate_num)),
        "illegal_action_rate": invalid_rate_num,
        (f"{key_prefix}_candidate_score"): candidate_score_num,
        (f"{key_prefix}_champion_score"): champion_score_num,
        (f"{key_prefix}_score_delta"): score_delta_num,
        (f"{key_prefix}_recommendation"): recommendation,
        (f"{key_prefix}_recommend_promotion"): 1.0 if recommend_promotion else 0.0,
        (f"{key_prefix}_run_dir"): str(out_dir),
        (f"{key_prefix}_run_manifest"): str(run_manifest_path),
        (f"{key_prefix}_promotion_decision"): str(decision_path),
    }
    if (is_v2 or is_p42) and isinstance(run_manifest, dict):
        steps = run_manifest.get("steps") if isinstance(run_manifest.get("steps"), dict) else {}
        replay = steps.get("replay_mixer") if isinstance(steps.get("replay_mixer"), dict) else {}
        triage = steps.get("regression_triage") if isinstance(steps.get("regression_triage"), dict) else {}
        candidate_step = steps.get("candidate_train") if isinstance(steps.get("candidate_train"), dict) else {}
        metrics.update(
            {
                f"{key_prefix}_lineage_summary_json": str(replay.get("lineage_summary_json") or ""),
                f"{key_prefix}_curriculum_plan_json": str(
                    (candidate_step.get("curriculum_plan") or "")
                ),
                f"{key_prefix}_triage_report_json": str(triage.get("triage_report_json") or ""),
                f"{key_prefix}_reward_config_json": str(candidate_step.get("reward_config") or ""),
                f"{key_prefix}_warnings_log": str(candidate_step.get("warnings_log") or ""),
                f"{key_prefix}_multi_seed_eval_json": str(candidate_step.get("multi_seed_eval") or ""),
                f"{key_prefix}_diagnostics_json": str(candidate_step.get("diagnostics_json") or ""),
            }
        )

    payload = {
        "schema": (
            "p42_orchestrator_seed_summary_v1"
            if is_p42
            else ("p41_orchestrator_seed_summary_v1" if is_v2 else "p40_orchestrator_seed_summary_v1")
        ),
        "generated_at": now_iso(),
        "exp_id": str(exp.get("id") or ""),
        "seed": seed,
        "seed_index": seed_idx,
        "seed_total": seed_total,
        "closed_loop_config": str(cfg_path),
        "closed_loop_returncode": int(result.get("returncode") or 0),
        "run_dir": str(out_dir),
        "run_manifest_path": str(run_manifest_path),
        "summary_table_path": str(summary_table_path),
        "promotion_decision_path": str(decision_path),
        "summary_row": summary_row if isinstance(summary_row, dict) else {},
        "pipeline_type": (
            "p44_distributed_rl"
            if is_p44
            else ("p42_rl_candidate" if is_p42 else ("p41_closed_loop_v2" if is_v2 else "p40_closed_loop_v1"))
        ),
        "metrics": metrics,
    }
    if is_p44 or is_p42:
        _append_p42_orchestrator_summary(ctx, payload)
    elif is_v2:
        _append_p41_orchestrator_summary(ctx, payload)
    else:
        _append_p40_orchestrator_summary(ctx, payload)

    status = "ok"
    return {
        "status": status,
        "metrics": metrics,
        "summary": {
            "closed_loop": result,
            "payload": payload,
            "run_manifest": run_manifest,
            "decision": decision_payload,
        },
    }


def _run_long_consistency_seed_experiment(
    *,
    ctx: RunContext,
    exp: dict[str, Any],
    exp_dir: Path,
    seed: str,
    seed_idx: int,
    seed_total: int,
) -> dict[str, Any]:
    eval_cfg = exp.get("eval") if isinstance(exp.get("eval"), dict) else {}
    base_url = str(eval_cfg.get("base_url") or exp.get("base_url") or "http://127.0.0.1:12346")
    episodes = int(eval_cfg.get("episodes") or exp.get("episodes") or 5)
    max_steps = int(eval_cfg.get("max_steps") or exp.get("max_steps") or 240)
    scope = str(eval_cfg.get("scope") or exp.get("scope") or "p37_action_fidelity_core")

    out_dir = exp_dir / "long_consistency_runs" / f"seed_{seed_idx:03d}_{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)

    batch_root = out_dir / "long_episode"
    batch_run_id = f"seed_{seed_idx:03d}_{seed}"
    batch_cmd = [
        sys.executable,
        "-B",
        "sim/oracle/batch_build_p38_long_episode.py",
        "--base-url",
        base_url,
        "--out-dir",
        str(batch_root),
        "--run-id",
        batch_run_id,
        "--episodes",
        str(max(1, episodes)),
        "--max-steps",
        str(max(1, max_steps)),
        "--seeds",
        seed,
        "--scope",
        scope,
    ]
    batch_result = run_process(batch_cmd, cwd=ctx.repo_root, timeout_sec=max(600, max_steps * episodes * 2))
    fixtures_dir = batch_root / batch_run_id
    report_path = fixtures_dir / "report_p38_long_episode.json"
    report_obj = read_json(report_path) or {}

    analyze_out = out_dir / "analysis"
    analyze_cmd = [
        sys.executable,
        "-B",
        "sim/oracle/analyze_p38_long_stats.py",
        "--fixtures-dir",
        str(fixtures_dir),
        "--out-dir",
        str(analyze_out),
    ]
    analyze_result = run_process(analyze_cmd, cwd=ctx.repo_root, timeout_sec=600)
    summary_path = analyze_out / "summary_stats.json"
    summary_obj = read_json(summary_path) or {}

    plot_cmd = [
        sys.executable,
        "-B",
        "sim/oracle/plot_p38_stats.py",
        "--fixtures-dir",
        str(fixtures_dir),
        "--out-dir",
        str(out_dir / "plots"),
    ]
    plot_result = run_process(plot_cmd, cwd=ctx.repo_root, timeout_sec=300)

    hard_fail = int(report_obj.get("hard_fail_count") or summary_obj.get("hard_fail_count") or 0)
    soft_warn = int(summary_obj.get("soft_warn_count") or 0)
    episodes_total = int(report_obj.get("episodes_total") or summary_obj.get("episodes_total") or episodes)

    sim_score = _numeric_mean_from_p38(summary_obj, "total_score", "sim")
    sim_rounds = _numeric_mean_from_p38(summary_obj, "rounds_survived", "sim")
    sim_money = _numeric_mean_from_p38(summary_obj, "money_earned", "sim")
    sim_rerolls = _numeric_mean_from_p38(summary_obj, "rerolls_count", "sim")
    sim_packs = _numeric_mean_from_p38(summary_obj, "packs_opened", "sim")
    sim_consumables = _numeric_mean_from_p38(summary_obj, "consumables_used", "sim")
    rel_diff_avg = _avg_abs_relative_diff(summary_obj)

    metrics = {
        "score": sim_score,
        "avg_reward": sim_score,
        "best_episode_reward": sim_score,
        "avg_ante_reached": sim_rounds,
        "median_ante": sim_rounds,
        "win_rate": (1.0 if hard_fail == 0 else 0.0),
        "hand_top1": 0.0,
        "hand_top3": 0.0,
        "shop_top1": max(0.0, 1.0 - min(1.0, rel_diff_avg / 100.0)),
        "illegal_action_rate": (float(hard_fail) / max(1.0, float(episodes_total))),
        "p38_mean_money_earned": sim_money,
        "p38_mean_rerolls": sim_rerolls,
        "p38_mean_packs_opened": sim_packs,
        "p38_mean_consumables_used": sim_consumables,
        "p38_hard_fail_count": hard_fail,
        "p38_soft_warn_count": soft_warn,
        "p38_mean_relative_diff_pct": rel_diff_avg,
        "p38_report_path": str(report_path),
        "p38_summary_path": str(summary_path),
        "p38_fixtures_dir": str(fixtures_dir),
    }

    summary_payload = {
        "schema": "p38_orchestrator_seed_summary_v1",
        "generated_at": now_iso(),
        "exp_id": str(exp.get("id") or ""),
        "seed": seed,
        "seed_index": seed_idx,
        "seed_total": seed_total,
        "episodes": episodes_total,
        "hard_fail_count": hard_fail,
        "soft_warn_count": soft_warn,
        "batch_returncode": int(batch_result.get("returncode") or 0),
        "analyze_returncode": int(analyze_result.get("returncode") or 0),
        "plot_returncode": int(plot_result.get("returncode") or 0),
        "report_path": str(report_path),
        "summary_path": str(summary_path),
        "fixtures_dir": str(fixtures_dir),
        "metrics": metrics,
    }
    _append_p38_orchestrator_summary(ctx, summary_payload)

    status = "ok"
    if int(batch_result.get("returncode") or 0) != 0:
        status = "failed"
    if int(analyze_result.get("returncode") or 0) != 0:
        status = "failed"
    if hard_fail > 0:
        status = "failed"

    return {
        "status": status,
        "metrics": metrics,
        "summary": {
            "batch": batch_result,
            "analyze": analyze_result,
            "plot": plot_result,
            "payload": summary_payload,
        },
    }


def run_single_experiment(ctx: RunContext, exp: dict[str, Any], exp_index: int, exp_total: int) -> dict[str, Any]:
    exp_id = str(exp["id"])
    exp_type = str(exp.get("experiment_type") or "standard").strip().lower()
    exp_category = str(exp.get("category") or normalize_experiment_category(exp)).strip().lower()
    exp_default_enabled = resolve_experiment_default_enabled(exp, category=exp_category)
    exp_dir = ctx.run_root / exp_id
    exp_dir.mkdir(parents=True, exist_ok=True)
    status_path = exp_dir / "status.json"
    progress_path = exp_dir / "progress.jsonl"
    stage_results: dict[str, Any] = {}
    started = time.time()
    seeds: list[str] = []
    seeds_payload: dict[str, Any] = {
        "seed_set_name": str(exp.get("seed_set_name") or ""),
        "seed_hash": "",
        "seed_count": 0,
        "seeds": [],
    }

    budget = ctx.config.get("budget") or {}
    max_wall_time_min = int(budget.get("max_wall_time_minutes") or 60)
    max_exp_time_min = int(budget.get("max_per_experiment_minutes") or max_wall_time_min)
    max_exp_time_sec = max(60, max_exp_time_min * 60)
    max_stage_retries = max(1, int(budget.get("max_retries_per_stage") or 1))

    def elapsed() -> float:
        return time.time() - started

    def over_budget() -> bool:
        return elapsed() > max_exp_time_sec

    def fail_row(*, status: str, reason: str, stage: str) -> dict[str, Any]:
        payload = {
            "status": status,
            "reason": reason,
            "stage": stage,
            "elapsed_sec": elapsed(),
            "seed_count": len(seeds),
            "catastrophic_failure_count": 1,
            "mean": 0.0,
            "std": 0.0,
            "avg_reward": 0.0,
            "reward_std": 0.0,
            "best_episode_reward": 0.0,
            "avg_ante_reached": 0.0,
            "median_ante": 0.0,
            "win_rate": 0.0,
            "hand_top1": 0.0,
            "hand_top3": 0.0,
            "shop_top1": 0.0,
            "illegal_action_rate": 1.0,
            "seed_set_name": seeds_payload.get("seed_set_name"),
            "seed_hash": seeds_payload.get("seed_hash"),
            "seeds_used": list(seeds),
            "final_win_rate": 0.0,
            "final_loss": 0.0,
            "category": exp_category,
            "default_enabled": exp_default_enabled,
        }
        write_json(status_path, payload)
        emit_progress(
            ctx,
            exp_id=exp_id,
            stage=stage,
            status=status,
            elapsed_sec=elapsed(),
            phase="stage",
            metric_snapshot=reason,
            message=reason,
        )
        return {
            "exp_id": exp_id,
            "status": status,
            "mean": payload["mean"],
            "std": payload["std"],
            "avg_reward": payload["avg_reward"],
            "reward_std": payload["reward_std"],
            "best_episode_reward": payload["best_episode_reward"],
            "avg_ante_reached": payload["avg_ante_reached"],
            "median_ante": payload["median_ante"],
            "win_rate": payload["win_rate"],
            "hand_top1": payload["hand_top1"],
            "hand_top3": payload["hand_top3"],
            "shop_top1": payload["shop_top1"],
            "illegal_action_rate": payload["illegal_action_rate"],
            "seed_count": payload["seed_count"],
            "catastrophic_failure_count": payload["catastrophic_failure_count"],
            "elapsed_sec": payload["elapsed_sec"],
            "run_dir": str(exp_dir),
            "seed_set_name": payload["seed_set_name"],
            "seed_hash": payload["seed_hash"],
            "seeds_used": payload["seeds_used"],
            "final_win_rate": payload["final_win_rate"],
            "final_loss": payload["final_loss"],
            "category": exp_category,
            "default_enabled": exp_default_enabled,
        }

    if ctx.resume:
        old = read_json(status_path)
        if old and str(old.get("status")) in {"success", "passed", "dry_run"}:
            emit_progress(
                ctx,
                exp_id=exp_id,
                stage="resume",
                status="skipped",
                elapsed_sec=elapsed(),
                metric_snapshot="already successful",
            )
            return {
                "exp_id": exp_id,
                "status": "passed",
                "mean": old.get("mean"),
                "std": old.get("std"),
                "avg_reward": old.get("avg_reward", old.get("mean")),
                "reward_std": old.get("reward_std", old.get("std")),
                "best_episode_reward": old.get("best_episode_reward"),
                "avg_ante_reached": old.get("avg_ante_reached", old.get("mean")),
                "median_ante": old.get("median_ante"),
                "win_rate": old.get("win_rate"),
                "hand_top1": old.get("hand_top1"),
                "hand_top3": old.get("hand_top3"),
                "shop_top1": old.get("shop_top1"),
                "illegal_action_rate": old.get("illegal_action_rate"),
                "seed_count": old.get("seed_count"),
                "catastrophic_failure_count": old.get("catastrophic_failure_count", 0),
                "elapsed_sec": old.get("elapsed_sec", 0.0),
                "run_dir": str(exp_dir),
                "seed_set_name": old.get("seed_set_name"),
                "seed_hash": old.get("seed_hash"),
                "seeds_used": old.get("seeds_used", []),
                "final_win_rate": old.get("final_win_rate"),
                "final_loss": old.get("final_loss"),
                "resumed": True,
                "category": exp_category,
                "default_enabled": exp_default_enabled,
            }

    seeds_payload = resolve_experiment_seeds(
        exp=exp,
        cfg=ctx.config,
        git_commit=ctx.git_commit,
        nightly=ctx.nightly,
        seed_limit=ctx.seed_limit,
        run_id=ctx.run_id,
        seed_policy=ctx.seed_policy,
        cli_seeds=ctx.cli_seeds,
    )
    seeds = [str(s) for s in (seeds_payload.get("seeds") or [])]
    write_json(exp_dir / "seeds_used.json", seeds_payload)

    manifest = {
        "schema": "p23_run_manifest_v1",
        "generated_at": now_iso(),
        "run_id": ctx.run_id,
        "exp_id": exp_id,
        "git_commit": ctx.git_commit,
        "python_version": detect_python_version(),
        "platform": platform.platform(),
        "budget": copy.deepcopy(budget),
        "mode": ctx.mode,
        "seed_mode": exp.get("seed_mode", "regression_fixed"),
        "seed_set_name": seeds_payload.get("seed_set_name"),
        "seeds_used": seeds,
        "seed_policy_version": seeds_payload.get("seed_policy_version"),
        "seed_hash": seeds_payload.get("seed_hash"),
        "seed_policy_config": ctx.seed_policy_config,
        "experiment_type": exp_type,
        "experiment_category": exp_category,
        "default_enabled": exp_default_enabled,
        "experiment": exp,
    }
    if exp_type in {
        "selfsup_pretrain",
        "selfsup_p33",
        "pretrain_repr",
        "self_supervised",
        "selfsup_stub",
        "selfsup_future_value",
        "selfsup_action_type",
    }:
        eval_cfg = exp.get("eval") if isinstance(exp.get("eval"), dict) else {}
        default_cfg_map = {
            "selfsup_pretrain": "configs/experiments/p31_selfsup.yaml",
            "selfsup_p33": "configs/experiments/p33_selfsup.yaml",
            "pretrain_repr": "configs/experiments/p32_self_supervised.yaml",
            "self_supervised": "configs/experiments/p32_self_supervised.yaml",
            "selfsup_stub": "configs/experiments/p32_self_supervised.yaml",
            "selfsup_future_value": "configs/experiments/p36_selfsup_future_value.yaml",
            "selfsup_action_type": "configs/experiments/p36_selfsup_action_type.yaml",
        }
        default_cfg = default_cfg_map.get(exp_type, "configs/experiments/p32_self_supervised.yaml")
        cfg_rel = str(eval_cfg.get("config") or exp.get("selfsup_config") or default_cfg)
        cfg_path = (ctx.repo_root / cfg_rel).resolve()
        selfsup_cfg = _read_yaml_or_json(cfg_path) if cfg_path.exists() else {}
        data_cfg = selfsup_cfg.get("data") if isinstance(selfsup_cfg.get("data"), dict) else {}
        manifest["selfsup"] = {
            "selfsup_type": exp_type,
            "config_path": str(cfg_path),
            "data_sources": data_cfg.get("sources") if isinstance(data_cfg.get("sources"), list) else [],
            "losses": selfsup_cfg.get("losses") if isinstance(selfsup_cfg.get("losses"), dict) else {},
        }
    elif exp_type in {"ssl_pretrain", "ssl_probe"}:
        eval_cfg = exp.get("eval") if isinstance(exp.get("eval"), dict) else {}
        default_cfg_map = {
            "ssl_pretrain": "configs/experiments/p37_ssl_pretrain.yaml",
            "ssl_probe": "configs/experiments/p37_ssl_probe.yaml",
        }
        default_cfg = default_cfg_map.get(exp_type, "configs/experiments/p37_ssl_pretrain.yaml")
        cfg_rel = str(eval_cfg.get("config") or exp.get("ssl_config") or default_cfg)
        cfg_path = (ctx.repo_root / cfg_rel).resolve()
        ssl_cfg = _read_yaml_or_json(cfg_path) if cfg_path.exists() else {}
        data_cfg = ssl_cfg.get("data") if isinstance(ssl_cfg.get("data"), dict) else {}
        manifest["ssl"] = {
            "ssl_type": exp_type,
            "config_path": str(cfg_path),
            "data_sources": data_cfg.get("sources") if isinstance(data_cfg.get("sources"), list) else [],
            "dataset_path": str(data_cfg.get("dataset_path") or ""),
            "probe": ssl_cfg.get("probe") if isinstance(ssl_cfg.get("probe"), dict) else {},
        }
    elif exp_type in {"rl_selfplay", "rl_selfplay_v1"}:
        eval_cfg = exp.get("eval") if isinstance(exp.get("eval"), dict) else {}
        manifest["rl"] = {
            "algo": str(exp.get("algo") or eval_cfg.get("algo") or "ppo"),
            "episodes": int(eval_cfg.get("episodes") or exp.get("episodes") or 10),
            "gamma": float(eval_cfg.get("gamma") or exp.get("gamma") or 0.99),
            "lr": float(eval_cfg.get("lr") or exp.get("lr") or 1e-3),
            "max_steps_per_episode": int(
                eval_cfg.get("max_steps_per_episode") or exp.get("max_steps_per_episode") or 320
            ),
            "reward_mode": str(eval_cfg.get("reward_mode") or exp.get("reward_mode") or "score_delta"),
        }
    elif exp_type in {"long_consistency", "long_horizon_consistency"}:
        eval_cfg = exp.get("eval") if isinstance(exp.get("eval"), dict) else {}
        manifest["long_consistency"] = {
            "base_url": str(eval_cfg.get("base_url") or exp.get("base_url") or "http://127.0.0.1:12346"),
            "episodes": int(eval_cfg.get("episodes") or exp.get("episodes") or 5),
            "max_steps": int(eval_cfg.get("max_steps") or exp.get("max_steps") or 240),
            "scope": str(eval_cfg.get("scope") or exp.get("scope") or "p37_action_fidelity_core"),
        }
    elif exp_type in {"policy_arena", "policy_arena_v1", "arena"}:
        eval_cfg = exp.get("eval") if isinstance(exp.get("eval"), dict) else {}
        manifest["policy_arena"] = {
            "backend": str(eval_cfg.get("backend") or exp.get("backend") or "sim"),
            "mode": str(eval_cfg.get("mode") or exp.get("mode") or "long_episode"),
            "policies": _normalize_text_list(
                eval_cfg.get("policies") or exp.get("policies") or "heuristic_baseline,search_expert,model_policy"
            ),
            "episodes_per_seed": int(
                eval_cfg.get("episodes_per_seed") or exp.get("episodes_per_seed") or exp.get("episodes") or 1
            ),
            "max_steps": int(eval_cfg.get("max_steps") or exp.get("max_steps") or 160),
            "enable_champion_rules": bool(
                eval_cfg.get("enable_champion_rules") or exp.get("enable_champion_rules") or False
            ),
            "candidate_policy": str(eval_cfg.get("candidate_policy") or ""),
            "champion_policy": str(eval_cfg.get("champion_policy") or ""),
            "focus_policy": str(eval_cfg.get("focus_policy") or eval_cfg.get("candidate_policy") or ""),
        }
    elif exp_type in {"closed_loop_improvement", "closed_loop", "p40_closed_loop", "closed_loop_improvement_v2", "p41_closed_loop_v2", "closed_loop_v2", "closed_loop_rl_candidate", "rl_candidate_pipeline", "p42_rl_candidate", "p42_rl_candidate_pipeline", "p44_distributed_rl"}:
        eval_cfg = exp.get("eval") if isinstance(exp.get("eval"), dict) else {}
        is_p44 = exp_type in {"p44_distributed_rl"}
        is_p42 = exp_type in {"closed_loop_rl_candidate", "rl_candidate_pipeline", "p42_rl_candidate", "p42_rl_candidate_pipeline"}
        is_v2 = is_p44 or is_p42 or exp_type in {"closed_loop_improvement_v2", "p41_closed_loop_v2", "closed_loop_v2"}
        manifest["closed_loop"] = {
            "config": str(
                eval_cfg.get("config")
                or exp.get("closed_loop_config")
                or (
                    "configs/experiments/p44_closed_loop_rl_smoke.yaml"
                    if is_p44
                    else (
                        "configs/experiments/p42_closed_loop_rl_smoke.yaml"
                        if is_p42
                    else ("configs/experiments/p41_closed_loop_v2_smoke.yaml" if is_v2 else "configs/experiments/p40_closed_loop_smoke.yaml")
                    )
                )
            ),
            "quick": bool(eval_cfg.get("quick") or False),
            "timeout_sec": int(eval_cfg.get("timeout_sec") or 3600),
            "candidate_policy": str(eval_cfg.get("candidate_policy") or "model_policy"),
            "champion_policy": str(eval_cfg.get("champion_policy") or "heuristic_baseline"),
            "pipeline_type": ("p44_distributed_rl" if is_p44 else ("rl_candidate_v1" if is_p42 else ("closed_loop_v2" if is_v2 else "closed_loop_v1"))),
        }
    elif exp_type in {"world_model_train", "world_model_eval", "world_model_assist_compare", "p45_world_model"}:
        eval_cfg = exp.get("eval") if isinstance(exp.get("eval"), dict) else {}
        manifest["world_model"] = {
            "config": str(
                eval_cfg.get("config")
                or exp.get("world_model_config")
                or (
                    "configs/experiments/p45_world_model_smoke.yaml"
                    if ctx.mode == "quick"
                    else "configs/experiments/p45_world_model_nightly.yaml"
                )
            ),
            "type": exp_type,
            "quick": bool(eval_cfg.get("quick") or False),
            "candidate_policy": str(eval_cfg.get("candidate_policy") or "heuristic_wm_assist"),
            "checkpoint": str(eval_cfg.get("checkpoint") or exp.get("checkpoint") or ""),
        }
    elif exp_type in {"imagination_augmented_candidate", "p46_imagination"}:
        eval_cfg = exp.get("eval") if isinstance(exp.get("eval"), dict) else {}
        manifest["imagination"] = {
            "config": str(
                eval_cfg.get("config")
                or exp.get("imagination_config")
                or (
                    "configs/experiments/p46_imagination_smoke.yaml"
                    if ctx.mode == "quick"
                    else "configs/experiments/p46_imagination_nightly.yaml"
                )
            ),
            "quick": bool(eval_cfg.get("quick") or False),
            "candidate_policy": str(eval_cfg.get("candidate_policy") or "candidate_real_plus_imagined_filtered"),
            "champion_policy": str(eval_cfg.get("champion_policy") or "heuristic_baseline"),
        }
    elif exp_type in {"world_model_rerank_eval", "p47_wm_search"}:
        eval_cfg = exp.get("eval") if isinstance(exp.get("eval"), dict) else {}
        manifest["model_based_search"] = {
            "config": str(
                eval_cfg.get("config")
                or exp.get("model_based_search_config")
                or (
                    "configs/experiments/p47_wm_search_smoke.yaml"
                    if ctx.mode == "quick"
                    else "configs/experiments/p47_wm_search_nightly.yaml"
                )
            ),
            "quick": bool(eval_cfg.get("quick") or False),
            "candidate_policy": str(eval_cfg.get("candidate_policy") or "heuristic_wm_rerank_h1"),
            "champion_policy": str(eval_cfg.get("champion_policy") or "heuristic_baseline"),
        }
    write_json(exp_dir / "run_manifest.json", manifest)
    print(
        "[P22] Experiment {idx}/{total}: {exp_id} (seeds {seed_count}, mode={mode})".format(
            idx=exp_index,
            total=exp_total,
            exp_id=exp_id,
            seed_count=len(seeds),
            mode=ctx.mode,
        )
    )

    append_experiment_progress_event(
        progress_path,
        run_id=ctx.run_id,
        exp_id=exp_id,
        phase="orchestrator",
        stage="init",
        status="start",
        elapsed_sec=0.0,
        message="experiment_start",
        extra={"seed_set_name": seeds_payload.get("seed_set_name"), "seeds": seeds},
    )
    emit_progress(
        ctx,
        exp_id=exp_id,
        stage="init",
        status="running",
        elapsed_sec=0.0,
        phase="orchestrator",
        message="experiment_start",
    )

    if ctx.dry_run:
        payload = {
            "status": "dry_run",
            "mean": 0.0,
            "std": 0.0,
            "avg_reward": 0.0,
            "reward_std": 0.0,
            "best_episode_reward": 0.0,
            "avg_ante_reached": 0.0,
            "median_ante": 0.0,
            "win_rate": 0.0,
            "hand_top1": 0.0,
            "hand_top3": 0.0,
            "shop_top1": 0.0,
            "illegal_action_rate": 0.0,
            "seed_count": len(seeds),
            "catastrophic_failure_count": 0,
            "elapsed_sec": 0.0,
            "seed_set_name": seeds_payload.get("seed_set_name"),
            "seed_hash": seeds_payload.get("seed_hash"),
            "seeds_used": list(seeds),
            "final_metrics": {},
            "final_win_rate": 0.0,
            "final_loss": 0.0,
            "category": exp_category,
            "default_enabled": exp_default_enabled,
        }
        write_json(status_path, payload)
        emit_progress(
            ctx,
            exp_id=exp_id,
            stage="done",
            status="passed",
            elapsed_sec=0.0,
            phase="orchestrator",
            metric_snapshot="dry_run",
            message="dry_run",
        )
        return {
            "exp_id": exp_id,
            "status": "passed",
            "mean": 0.0,
            "std": 0.0,
            "avg_reward": 0.0,
            "reward_std": 0.0,
            "best_episode_reward": 0.0,
            "avg_ante_reached": 0.0,
            "median_ante": 0.0,
            "win_rate": 0.0,
            "hand_top1": 0.0,
            "hand_top3": 0.0,
            "shop_top1": 0.0,
            "illegal_action_rate": 0.0,
            "seed_count": len(seeds),
            "catastrophic_failure_count": 0,
            "elapsed_sec": 0.0,
            "run_dir": str(exp_dir),
            "seed_set_name": seeds_payload.get("seed_set_name"),
            "seed_hash": seeds_payload.get("seed_hash"),
            "seeds_used": list(seeds),
            "final_win_rate": 0.0,
            "final_loss": 0.0,
            "category": exp_category,
            "default_enabled": exp_default_enabled,
        }

    stages = exp.get("stages") or {}
    gate_flag = str(exp.get("gate_flag") or "RunFast")
    context = {
        "exp_id": exp_id,
        "run_dir": str(exp_dir),
        "repo_root": str(ctx.repo_root),
    }

    stage_order = [
        ("sanity", bool(stages.get("sanity", True))),
        ("gate", bool(stages.get("gate", True))),
        ("dataset", bool(stages.get("dataset"))),
        ("train", bool(stages.get("train"))),
    ]
    for stage, enabled in stage_order:
        if not enabled:
            continue
        if over_budget():
            return fail_row(status="budget_cut", reason="per_experiment_budget_exceeded", stage=stage)
        if stage == "sanity":
            compile_files = exp.get("sanity_compile") or ["trainer/train_rl.py", "trainer/eval_long_horizon.py"]
            cmd = ["python", "-B", "-m", "py_compile"] + [str(x) for x in compile_files]
        elif stage == "gate":
            cmd = [
                "powershell",
                "-ExecutionPolicy",
                "Bypass",
                "-File",
                "scripts/run_regressions.ps1",
                f"-{gate_flag}",
            ]
        elif stage == "dataset":
            cmd = exp.get("dataset_command")
        else:
            cmd = exp.get("train_command")

        stage_results[stage] = _run_stage_command(
            ctx=ctx,
            exp_id=exp_id,
            stage=stage,
            command_tmpl=cmd,
            context=context,
            repo_root=ctx.repo_root,
            progress_path=progress_path,
            timeout_sec=max_exp_time_sec,
            max_retries=max_stage_retries,
            started_ts=started,
        )
        token = str(stage_results[stage].get("status"))
        if token not in {"ok", "skipped"}:
            failure_status = "timed_out" if token == "timed_out" else "failed"
            return fail_row(status=failure_status, reason=f"{stage}_failed", stage=stage)

    primary_metric = str((ctx.config.get("evaluation") or {}).get("primary_metric") or "avg_ante_reached")
    eval_cfg = exp.get("eval") or {}
    bias = float(eval_cfg.get("primary_metric_bias") or 0.0)
    seed_results: list[dict[str, Any]] = []

    if bool(stages.get("eval", True)):
        eval_cmd_tmpl = exp.get("eval_command")
        for seed_idx, seed in enumerate(seeds, start=1):
            if over_budget():
                seed_results.append(
                    {
                        "seed": seed,
                        "status": "failed",
                        "stage": "eval",
                        "error": "budget_exceeded",
                        "elapsed_sec": elapsed(),
                        "metrics": {},
                    }
                )
                emit_progress(
                    ctx,
                    exp_id=exp_id,
                    stage="eval",
                    status="budget_cut",
                    elapsed_sec=elapsed(),
                    phase="eval",
                    seed=seed,
                    step_or_epoch=seed_idx,
                    seed_index=seed_idx,
                    seed_total=len(seeds),
                    metric_snapshot="budget_exceeded",
                    message="per_experiment_budget_exceeded",
                )
                break

            seed_ctx = dict(context)
            seed_ctx["seed"] = seed
            seed_ctx["seed_index"] = seed_idx
            seed_ctx["seed_total"] = len(seeds)
            if ctx.verbose:
                print(
                    "[P22][verbose] Experiment {idx}/{total}: {exp_id} (seed {seed_idx}/{seed_total}: {seed})".format(
                        idx=exp_index,
                        total=exp_total,
                        exp_id=exp_id,
                        seed_idx=seed_idx,
                        seed_total=len(seeds),
                        seed=seed,
                    )
                )
            if exp_type in {
                "selfsup_pretrain",
                "selfsup_p33",
                "pretrain_repr",
                "self_supervised",
                "selfsup_stub",
                "selfsup_future_value",
                "selfsup_action_type",
            }:
                try:
                    if exp_type == "selfsup_pretrain":
                        selfsup_result = _run_selfsup_seed_experiment(
                            ctx=ctx,
                            exp=exp,
                            exp_dir=exp_dir,
                            seed=seed,
                            seed_idx=seed_idx,
                            seed_total=len(seeds),
                        )
                    elif exp_type in {"pretrain_repr", "self_supervised", "selfsup_stub"}:
                        selfsup_result = _run_selfsup_p32_seed_experiment(
                            ctx=ctx,
                            exp=exp,
                            exp_dir=exp_dir,
                            seed=seed,
                            seed_idx=seed_idx,
                            seed_total=len(seeds),
                        )
                    elif exp_type == "selfsup_future_value":
                        selfsup_result = _run_selfsup_future_value_seed_experiment(
                            ctx=ctx,
                            exp=exp,
                            exp_dir=exp_dir,
                            seed=seed,
                            seed_idx=seed_idx,
                            seed_total=len(seeds),
                        )
                    elif exp_type == "selfsup_action_type":
                        selfsup_result = _run_selfsup_action_type_seed_experiment(
                            ctx=ctx,
                            exp=exp,
                            exp_dir=exp_dir,
                            seed=seed,
                            seed_idx=seed_idx,
                            seed_total=len(seeds),
                        )
                    else:
                        selfsup_result = _run_selfsup_p33_seed_experiment(
                            ctx=ctx,
                            exp=exp,
                            exp_dir=exp_dir,
                            seed=seed,
                            seed_idx=seed_idx,
                            seed_total=len(seeds),
                        )
                except Exception as exc:
                    seed_results.append(
                        {
                            "seed": seed,
                            "status": "failed",
                            "stage": "eval",
                            "error": f"selfsup_exception: {exc}",
                            "elapsed_sec": elapsed(),
                            "metrics": {},
                        }
                    )
                else:
                    seed_results.append(
                        {
                            "seed": seed,
                            "status": "ok" if selfsup_result["status"] == "ok" else "failed",
                            "stage": "eval",
                            "elapsed_sec": elapsed(),
                            "metrics": dict(selfsup_result.get("metrics") or {}),
                            "selfsup_summary": selfsup_result.get("summary") or {},
                        }
                    )
                append_experiment_progress_event(
                    progress_path,
                    run_id=ctx.run_id,
                    exp_id=exp_id,
                    phase="eval",
                    stage="eval",
                    status=str(seed_results[-1].get("status")),
                    seed=seed,
                    step_or_epoch=seed_idx,
                    elapsed_sec=elapsed(),
                    metrics=seed_results[-1].get("metrics") or {},
                    message=exp_type,
                    extra={"mode": exp_type, "seed_index": seed_idx, "seed_total": len(seeds)},
                )
            elif exp_type in {"ssl_pretrain", "ssl_probe"}:
                try:
                    if exp_type == "ssl_probe":
                        ssl_result = _run_ssl_probe_seed_experiment(
                            ctx=ctx,
                            exp=exp,
                            exp_dir=exp_dir,
                            seed=seed,
                            seed_idx=seed_idx,
                            seed_total=len(seeds),
                        )
                    else:
                        ssl_result = _run_ssl_pretrain_seed_experiment(
                            ctx=ctx,
                            exp=exp,
                            exp_dir=exp_dir,
                            seed=seed,
                            seed_idx=seed_idx,
                            seed_total=len(seeds),
                        )
                except Exception as exc:
                    seed_results.append(
                        {
                            "seed": seed,
                            "status": "failed",
                            "stage": "eval",
                            "error": f"ssl_exception: {exc}",
                            "elapsed_sec": elapsed(),
                            "metrics": {},
                        }
                    )
                else:
                    seed_results.append(
                        {
                            "seed": seed,
                            "status": "ok" if ssl_result["status"] == "ok" else "failed",
                            "stage": "eval",
                            "elapsed_sec": elapsed(),
                            "metrics": dict(ssl_result.get("metrics") or {}),
                            "ssl_summary": ssl_result.get("summary") or {},
                        }
                    )
                append_experiment_progress_event(
                    progress_path,
                    run_id=ctx.run_id,
                    exp_id=exp_id,
                    phase="eval",
                    stage="eval",
                    status=str(seed_results[-1].get("status")),
                    seed=seed,
                    step_or_epoch=seed_idx,
                    elapsed_sec=elapsed(),
                    metrics=seed_results[-1].get("metrics") or {},
                    message=exp_type,
                    extra={"mode": exp_type, "seed_index": seed_idx, "seed_total": len(seeds)},
                )
            elif exp_type in {"rl_selfplay", "rl_selfplay_v1"}:
                try:
                    rl_result = _run_rl_selfplay_seed_experiment(
                        ctx=ctx,
                        exp=exp,
                        exp_dir=exp_dir,
                        seed=seed,
                        seed_idx=seed_idx,
                        seed_total=len(seeds),
                        progress_path=progress_path,
                        started_ts=started,
                        run_id=ctx.run_id,
                        exp_id=exp_id,
                    )
                except Exception as exc:
                    seed_results.append(
                        {
                            "seed": seed,
                            "status": "failed",
                            "stage": "eval",
                            "error": f"rl_selfplay_exception: {exc}",
                            "elapsed_sec": elapsed(),
                            "metrics": {},
                        }
                    )
                else:
                    seed_results.append(
                        {
                            "seed": seed,
                            "status": "ok" if rl_result["status"] == "ok" else "failed",
                            "stage": "eval",
                            "elapsed_sec": elapsed(),
                            "metrics": dict(rl_result.get("metrics") or {}),
                            "rl_summary": rl_result.get("summary") or {},
                        }
                    )
                append_experiment_progress_event(
                    progress_path,
                    run_id=ctx.run_id,
                    exp_id=exp_id,
                    phase="eval",
                    stage="eval",
                    status=str(seed_results[-1].get("status")),
                    seed=seed,
                    step_or_epoch=seed_idx,
                    elapsed_sec=elapsed(),
                    metrics=seed_results[-1].get("metrics") or {},
                    message=exp_type,
                    extra={"mode": exp_type, "seed_index": seed_idx, "seed_total": len(seeds)},
                )
            elif exp_type in {"long_consistency", "long_horizon_consistency"}:
                try:
                    long_result = _run_long_consistency_seed_experiment(
                        ctx=ctx,
                        exp=exp,
                        exp_dir=exp_dir,
                        seed=seed,
                        seed_idx=seed_idx,
                        seed_total=len(seeds),
                    )
                except Exception as exc:
                    seed_results.append(
                        {
                            "seed": seed,
                            "status": "failed",
                            "stage": "eval",
                            "error": f"long_consistency_exception: {exc}",
                            "elapsed_sec": elapsed(),
                            "metrics": {},
                        }
                    )
                else:
                    seed_results.append(
                        {
                            "seed": seed,
                            "status": "ok" if long_result["status"] == "ok" else "failed",
                            "stage": "eval",
                            "elapsed_sec": elapsed(),
                            "metrics": dict(long_result.get("metrics") or {}),
                            "long_summary": long_result.get("summary") or {},
                        }
                    )
                append_experiment_progress_event(
                    progress_path,
                    run_id=ctx.run_id,
                    exp_id=exp_id,
                    phase="eval",
                    stage="eval",
                    status=str(seed_results[-1].get("status")),
                    seed=seed,
                    step_or_epoch=seed_idx,
                    elapsed_sec=elapsed(),
                    metrics=seed_results[-1].get("metrics") or {},
                    message=exp_type,
                    extra={"mode": exp_type, "seed_index": seed_idx, "seed_total": len(seeds)},
                )
            elif exp_type in {"policy_arena", "policy_arena_v1", "arena"}:
                try:
                    arena_result = _run_policy_arena_seed_experiment(
                        ctx=ctx,
                        exp=exp,
                        exp_dir=exp_dir,
                        seed=seed,
                        seed_idx=seed_idx,
                        seed_total=len(seeds),
                    )
                except Exception as exc:
                    seed_results.append(
                        {
                            "seed": seed,
                            "status": "failed",
                            "stage": "eval",
                            "error": f"policy_arena_exception: {exc}",
                            "elapsed_sec": elapsed(),
                            "metrics": {},
                        }
                    )
                else:
                    seed_results.append(
                        {
                            "seed": seed,
                            "status": "ok" if arena_result["status"] == "ok" else "failed",
                            "stage": "eval",
                            "elapsed_sec": elapsed(),
                            "metrics": dict(arena_result.get("metrics") or {}),
                            "policy_arena_summary": arena_result.get("summary") or {},
                        }
                    )
                append_experiment_progress_event(
                    progress_path,
                    run_id=ctx.run_id,
                    exp_id=exp_id,
                    phase="eval",
                    stage="eval",
                    status=str(seed_results[-1].get("status")),
                    seed=seed,
                    step_or_epoch=seed_idx,
                    elapsed_sec=elapsed(),
                    metrics=seed_results[-1].get("metrics") or {},
                        message=exp_type,
                        extra={"mode": exp_type, "seed_index": seed_idx, "seed_total": len(seeds)},
                    )
            elif exp_type in {"world_model_train", "world_model_eval", "world_model_assist_compare", "p45_world_model"}:
                try:
                    world_model_result = _run_world_model_seed_experiment(
                        ctx=ctx,
                        exp=exp,
                        exp_dir=exp_dir,
                        seed=seed,
                        seed_idx=seed_idx,
                        seed_total=len(seeds),
                    )
                except Exception as exc:
                    seed_results.append(
                        {
                            "seed": seed,
                            "status": "failed",
                            "stage": "eval",
                            "error": f"world_model_exception: {exc}",
                            "elapsed_sec": elapsed(),
                            "metrics": {},
                        }
                    )
                else:
                    seed_results.append(
                        {
                            "seed": seed,
                            "status": "ok" if world_model_result["status"] == "ok" else "failed",
                            "stage": "eval",
                            "elapsed_sec": elapsed(),
                            "metrics": dict(world_model_result.get("metrics") or {}),
                            "world_model_summary": world_model_result.get("summary") or {},
                        }
                    )
                append_experiment_progress_event(
                    progress_path,
                    run_id=ctx.run_id,
                    exp_id=exp_id,
                    phase="eval",
                    stage="eval",
                    status=str(seed_results[-1].get("status")),
                    seed=seed,
                    step_or_epoch=seed_idx,
                    elapsed_sec=elapsed(),
                    metrics=seed_results[-1].get("metrics") or {},
                    message=exp_type,
                    extra={"mode": exp_type, "seed_index": seed_idx, "seed_total": len(seeds)},
                )
            elif exp_type in {"imagination_augmented_candidate", "p46_imagination"}:
                try:
                    imagination_result = _run_imagination_seed_experiment(
                        ctx=ctx,
                        exp=exp,
                        exp_dir=exp_dir,
                        seed=seed,
                        seed_idx=seed_idx,
                        seed_total=len(seeds),
                    )
                except Exception as exc:
                    seed_results.append(
                        {
                            "seed": seed,
                            "status": "failed",
                            "stage": "eval",
                            "error": f"imagination_exception: {exc}",
                            "elapsed_sec": elapsed(),
                            "metrics": {},
                        }
                    )
                else:
                    seed_results.append(
                        {
                            "seed": seed,
                            "status": "ok" if imagination_result["status"] == "ok" else "failed",
                            "stage": "eval",
                            "elapsed_sec": elapsed(),
                            "metrics": dict(imagination_result.get("metrics") or {}),
                            "imagination_summary": imagination_result.get("summary") or {},
                        }
                    )
                append_experiment_progress_event(
                    progress_path,
                    run_id=ctx.run_id,
                    exp_id=exp_id,
                    phase="eval",
                    stage="eval",
                    status=str(seed_results[-1].get("status")),
                    seed=seed,
                    step_or_epoch=seed_idx,
                    elapsed_sec=elapsed(),
                    metrics=seed_results[-1].get("metrics") or {},
                    message=exp_type,
                    extra={"mode": exp_type, "seed_index": seed_idx, "seed_total": len(seeds)},
                )
            elif exp_type in {"world_model_rerank_eval", "p47_wm_search"}:
                try:
                    p47_result = _run_model_based_search_seed_experiment(
                        ctx=ctx,
                        exp=exp,
                        exp_dir=exp_dir,
                        seed=seed,
                        seed_idx=seed_idx,
                        seed_total=len(seeds),
                    )
                except Exception as exc:
                    seed_results.append(
                        {
                            "seed": seed,
                            "status": "failed",
                            "stage": "eval",
                            "error": f"model_based_search_exception: {exc}",
                            "elapsed_sec": elapsed(),
                            "metrics": {},
                        }
                    )
                else:
                    seed_results.append(
                        {
                            "seed": seed,
                            "status": "ok" if p47_result["status"] == "ok" else "failed",
                            "stage": "eval",
                            "elapsed_sec": elapsed(),
                            "metrics": dict(p47_result.get("metrics") or {}),
                            "model_based_search_summary": p47_result.get("summary") or {},
                        }
                    )
                append_experiment_progress_event(
                    progress_path,
                    run_id=ctx.run_id,
                    exp_id=exp_id,
                    phase="eval",
                    stage="eval",
                    status=str(seed_results[-1].get("status")),
                    seed=seed,
                    step_or_epoch=seed_idx,
                    elapsed_sec=elapsed(),
                    metrics=seed_results[-1].get("metrics") or {},
                    message=exp_type,
                    extra={"mode": exp_type, "seed_index": seed_idx, "seed_total": len(seeds)},
                )
            elif exp_type in {"closed_loop_improvement", "closed_loop", "p40_closed_loop", "closed_loop_improvement_v2", "p41_closed_loop_v2", "closed_loop_v2", "closed_loop_rl_candidate", "rl_candidate_pipeline", "p42_rl_candidate", "p42_rl_candidate_pipeline", "p44_distributed_rl"}:
                try:
                    closed_loop_result = _run_closed_loop_seed_experiment(
                        ctx=ctx,
                        exp=exp,
                        exp_dir=exp_dir,
                        seed=seed,
                        seed_idx=seed_idx,
                        seed_total=len(seeds),
                    )
                except Exception as exc:
                    seed_results.append(
                        {
                            "seed": seed,
                            "status": "failed",
                            "stage": "eval",
                            "error": f"closed_loop_exception: {exc}",
                            "elapsed_sec": elapsed(),
                            "metrics": {},
                        }
                    )
                else:
                    seed_results.append(
                        {
                            "seed": seed,
                            "status": "ok" if closed_loop_result["status"] == "ok" else "failed",
                            "stage": "eval",
                            "elapsed_sec": elapsed(),
                            "metrics": dict(closed_loop_result.get("metrics") or {}),
                            "closed_loop_summary": closed_loop_result.get("summary") or {},
                        }
                    )
                append_experiment_progress_event(
                    progress_path,
                    run_id=ctx.run_id,
                    exp_id=exp_id,
                    phase="eval",
                    stage="eval",
                    status=str(seed_results[-1].get("status")),
                    seed=seed,
                    step_or_epoch=seed_idx,
                    elapsed_sec=elapsed(),
                    metrics=seed_results[-1].get("metrics") or {},
                    message=exp_type,
                    extra={"mode": exp_type, "seed_index": seed_idx, "seed_total": len(seeds)},
                )
            else:
                metrics = _synthetic_eval_metrics(exp_id, seed, bias)
                if eval_cmd_tmpl:
                    result = _run_stage_command(
                        ctx=ctx,
                        exp_id=exp_id,
                        stage="eval",
                        command_tmpl=eval_cmd_tmpl,
                        context=seed_ctx,
                        repo_root=ctx.repo_root,
                        progress_path=progress_path,
                        timeout_sec=max_exp_time_sec,
                        max_retries=max_stage_retries,
                        started_ts=started,
                    )
                    if result["status"] == "ok":
                        seed_results.append(
                            {
                                "seed": seed,
                                "status": "ok",
                                "stage": "eval",
                                "elapsed_sec": result["elapsed_sec"],
                                "metrics": metrics,
                            }
                        )
                    else:
                        seed_results.append(
                            {
                                "seed": seed,
                                "status": "failed",
                                "stage": "eval",
                                "error": result.get("stderr_tail") or "eval_command_failed",
                                "elapsed_sec": result["elapsed_sec"],
                                "metrics": {},
                            }
                        )
                else:
                    seed_results.append(
                        {
                            "seed": seed,
                            "status": "ok",
                            "stage": "eval",
                            "elapsed_sec": 0.0,
                            "metrics": metrics,
                        }
                    )
                    append_experiment_progress_event(
                        progress_path,
                        run_id=ctx.run_id,
                        exp_id=exp_id,
                        phase="eval",
                        stage="eval",
                        status="ok",
                        seed=seed,
                        step_or_epoch=seed_idx,
                        elapsed_sec=elapsed(),
                        metrics=metrics,
                        message="synthetic_eval",
                        extra={"seed_index": seed_idx, "seed_total": len(seeds)},
                    )
            latest_metrics = seed_results[-1].get("metrics") or {}
            snap = latest_metrics.get(primary_metric)
            eta_sec = None
            if seed_idx > 0:
                eta_sec = ((elapsed() / seed_idx) * (len(seeds) - seed_idx))
            emit_progress(
                ctx,
                exp_id=exp_id,
                stage="eval",
                status=str(seed_results[-1].get("status")),
                elapsed_sec=elapsed(),
                phase="eval",
                seed=seed,
                step_or_epoch=seed_idx,
                metrics=latest_metrics,
                seed_index=seed_idx,
                seed_total=len(seeds),
                eta_sec=eta_sec,
                metric_snapshot=f"{primary_metric}={snap}",
            )
            print(
                "[P22] exp={exp} seed={seed} ({idx}/{total}) status={status} score={score} win_rate={win} loss={loss} elapsed={elapsed:.1f}s".format(
                    exp=exp_id,
                    seed=seed,
                    idx=seed_idx,
                    total=len(seeds),
                    status=str(seed_results[-1].get("status")),
                    score=_fmt_num(latest_metrics.get("score")),
                    win=_fmt_pct(latest_metrics.get("win_rate")),
                    loss=_fmt_num(
                        latest_metrics.get("ssl_probe_val_loss")
                        if latest_metrics.get("ssl_probe_val_loss") is not None
                        else (
                            latest_metrics.get("ssl_val_loss")
                            if latest_metrics.get("ssl_val_loss") is not None
                            else (
                                latest_metrics.get("selfsup_val_loss")
                                if latest_metrics.get("selfsup_val_loss") is not None
                                else (
                                    latest_metrics.get("selfsup_p33_val_loss")
                                    if latest_metrics.get("selfsup_p33_val_loss") is not None
                                    else (
                                        latest_metrics.get("selfsup_p32_val_loss")
                                        if latest_metrics.get("selfsup_p32_val_loss") is not None
                                        else (
                                            latest_metrics.get("selfsup_p36_future_val_loss")
                                            if latest_metrics.get("selfsup_p36_future_val_loss") is not None
                                            else (
                                                latest_metrics.get("selfsup_p36_action_val_loss")
                                                if latest_metrics.get("selfsup_p36_action_val_loss") is not None
                                                else latest_metrics.get("loss")
                                            )
                                        )
                                    )
                                )
                            )
                        )
                    ),
                    elapsed=elapsed(),
                )
            )

    metric_summary = aggregate_seed_metrics(seed_results, primary_metric=primary_metric)
    success = is_success(metric_summary)
    elapsed_total = elapsed()
    final_seed_metrics = {}
    if seed_results:
        final_seed_metrics = dict(seed_results[-1].get("metrics") or {})
    final_win_rate = _as_number(final_seed_metrics.get("win_rate"))
    final_loss = _as_number(
        final_seed_metrics.get("ssl_probe_val_loss")
        if final_seed_metrics.get("ssl_probe_val_loss") is not None
        else (
            final_seed_metrics.get("ssl_val_loss")
            if final_seed_metrics.get("ssl_val_loss") is not None
            else (
                final_seed_metrics.get("selfsup_val_loss")
                if final_seed_metrics.get("selfsup_val_loss") is not None
                else (
                    final_seed_metrics.get("selfsup_p33_val_loss")
                    if final_seed_metrics.get("selfsup_p33_val_loss") is not None
                    else (
                        final_seed_metrics.get("selfsup_p32_val_loss")
                        if final_seed_metrics.get("selfsup_p32_val_loss") is not None
                        else (
                            final_seed_metrics.get("selfsup_p36_future_val_loss")
                            if final_seed_metrics.get("selfsup_p36_future_val_loss") is not None
                            else (
                                final_seed_metrics.get("selfsup_p36_action_val_loss")
                                if final_seed_metrics.get("selfsup_p36_action_val_loss") is not None
                                else final_seed_metrics.get("loss")
                            )
                        )
                    )
                )
            )
        )
    )

    exp_summary = {
        "schema": "p23_experiment_summary_v1",
        "generated_at": now_iso(),
        "run_id": ctx.run_id,
        "exp_id": exp_id,
        "status": "success" if success else "failed",
        "stages": stage_results,
        "seed_metrics": metric_summary,
        "seed_set_name": seeds_payload.get("seed_set_name"),
        "seed_hash": seeds_payload.get("seed_hash"),
        "seeds_used": list(seeds),
        "final_metrics": final_seed_metrics,
        "elapsed_sec": elapsed_total,
        "run_dir": str(exp_dir),
        "category": exp_category,
        "default_enabled": exp_default_enabled,
    }
    write_json(exp_dir / "stage_results.json", stage_results)
    write_json(exp_dir / "exp_summary.json", exp_summary)
    write_json(
        status_path,
        {
            "status": "success" if success else "failed",
            "mean": metric_summary.get("mean"),
            "std": metric_summary.get("std"),
            "avg_reward": metric_summary.get("avg_reward", metric_summary.get("mean")),
            "reward_std": metric_summary.get("reward_std", metric_summary.get("std")),
            "best_episode_reward": metric_summary.get("best_episode_reward"),
            "avg_ante_reached": metric_summary.get("avg_ante_reached"),
            "median_ante": metric_summary.get("median_ante"),
            "win_rate": metric_summary.get("win_rate"),
            "hand_top1": metric_summary.get("hand_top1"),
            "hand_top3": metric_summary.get("hand_top3"),
            "shop_top1": metric_summary.get("shop_top1"),
            "illegal_action_rate": metric_summary.get("illegal_action_rate"),
            "seed_count": metric_summary.get("count"),
            "catastrophic_failure_count": metric_summary.get("catastrophic_failure_count"),
            "elapsed_sec": elapsed_total,
            "seed_set_name": seeds_payload.get("seed_set_name"),
            "seed_hash": seeds_payload.get("seed_hash"),
            "seeds_used": list(seeds),
            "final_metrics": final_seed_metrics,
            "final_win_rate": final_win_rate,
            "final_loss": final_loss,
            "category": exp_category,
            "default_enabled": exp_default_enabled,
        },
    )
    write_json(
        exp_dir / "metrics.json",
        {
            "schema": "p37_orchestrator_metrics_v1",
            "generated_at": now_iso(),
            "run_id": ctx.run_id,
            "exp_id": exp_id,
            "status": "success" if success else "failed",
            "avg_reward": metric_summary.get("avg_reward", metric_summary.get("mean")),
            "reward_std": metric_summary.get("reward_std", metric_summary.get("std")),
            "best_episode_reward": metric_summary.get("best_episode_reward"),
            "seed_count": metric_summary.get("count"),
            "seeds": list(seeds),
            "final_metrics": final_seed_metrics,
            "final_loss": final_loss,
            "category": exp_category,
            "default_enabled": exp_default_enabled,
        },
    )

    if not ctx.keep_intermediate:
        for pattern in ("*.pt", "*.ckpt", "*.bin", "*.parquet"):
            for file_path in exp_dir.rglob(pattern):
                try:
                    file_path.unlink(missing_ok=True)
                except Exception:
                    pass

    final_status = "passed" if success else "failed"
    emit_progress(
        ctx,
        exp_id=exp_id,
        stage="done",
        status=final_status,
        elapsed_sec=elapsed_total,
        phase="orchestrator",
        metrics={
            "mean": metric_summary.get("mean"),
            "win_rate": metric_summary.get("win_rate"),
            "final_win_rate": final_win_rate,
            "final_loss": final_loss,
        },
        metric_snapshot=f"{primary_metric}={metric_summary.get('mean')}",
    )
    print(
        "[P22] Completed {idx}/{total}: {exp_id} status={status} | {summary}".format(
            idx=exp_index,
            total=exp_total,
            exp_id=exp_id,
            status=final_status,
            summary=_experiment_summary_line(metric_summary),
        )
    )
    return {
        "exp_id": exp_id,
        "status": final_status,
        "mean": metric_summary.get("mean"),
        "std": metric_summary.get("std"),
        "avg_reward": metric_summary.get("avg_reward", metric_summary.get("mean")),
        "reward_std": metric_summary.get("reward_std", metric_summary.get("std")),
        "best_episode_reward": metric_summary.get("best_episode_reward"),
        "avg_ante_reached": metric_summary.get("avg_ante_reached"),
        "median_ante": metric_summary.get("median_ante"),
        "win_rate": metric_summary.get("win_rate"),
        "hand_top1": metric_summary.get("hand_top1"),
        "hand_top3": metric_summary.get("hand_top3"),
        "shop_top1": metric_summary.get("shop_top1"),
        "illegal_action_rate": metric_summary.get("illegal_action_rate"),
        "seed_count": metric_summary.get("count"),
        "catastrophic_failure_count": metric_summary.get("catastrophic_failure_count"),
        "elapsed_sec": elapsed_total,
        "run_dir": str(exp_dir),
        "seed_set_name": seeds_payload.get("seed_set_name"),
        "seed_hash": seeds_payload.get("seed_hash"),
        "seeds_used": list(seeds),
        "final_win_rate": final_win_rate,
        "final_loss": final_loss,
        "category": exp_category,
        "default_enabled": exp_default_enabled,
    }


def load_config(path: Path) -> dict[str, Any]:
    return _read_yaml_or_json(path)


def resolve_run_root(out_root: Path, resume: bool) -> tuple[str, Path]:
    runs_root = out_root / "runs"
    runs_root.mkdir(parents=True, exist_ok=True)
    if resume:
        latest = sorted([p for p in runs_root.iterdir() if p.is_dir()], key=lambda p: p.name)
        if latest:
            run_id = latest[-1].name
            return run_id, latest[-1]
    run_id = now_stamp()
    run_root = runs_root / run_id
    run_root.mkdir(parents=True, exist_ok=True)
    return run_id, run_root


def select_experiments_for_mode(
    experiments: list[dict[str, Any]],
    mode: str,
    *,
    include_legacy: bool = False,
    legacy_only: bool = False,
) -> list[dict[str, Any]]:
    annotated = attach_experiment_mode_metadata(experiments)
    mode = str(mode or "gate").lower()
    if mode == "quick":
        picked = [e for e in annotated if bool((e.get("modes") or {}).get("quick", False))]
        if not picked:
            picked = annotated[:3]
    elif mode == "milestone":
        picked = [e for e in annotated if bool((e.get("modes") or {}).get("milestone", False))]
        if not picked:
            picked = annotated
    elif mode == "nightly":
        picked = [e for e in annotated if bool((e.get("modes") or {}).get("nightly", True))]
        if not picked:
            picked = annotated
    else:
        picked = [e for e in annotated if bool((e.get("modes") or {}).get("gate", True))]
        if not picked:
            picked = annotated

    filtered: list[dict[str, Any]] = []
    for exp in picked:
        category = str(exp.get("category") or MODE_CATEGORY_EXPERIMENTAL).strip().lower()
        enabled = bool(exp.get("default_enabled"))
        if legacy_only and category != MODE_CATEGORY_LEGACY_BASELINE:
            continue
        if not legacy_only and category == MODE_CATEGORY_LEGACY_BASELINE and not include_legacy:
            continue
        if not enabled and category != MODE_CATEGORY_LEGACY_BASELINE:
            continue
        filtered.append(exp)

    return filtered


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="P23 experiment orchestrator.")
    p.add_argument("--config", default="configs/experiments/p22.yaml")
    p.add_argument("--out-root", default="docs/artifacts/p22")
    p.add_argument("--mode", default="gate", choices=["quick", "gate", "nightly", "milestone"])
    p.add_argument("--resume", action="store_true")
    p.add_argument("--only", default="")
    p.add_argument("--exclude", default="")
    p.add_argument("--nightly", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--keep-intermediate", action="store_true")
    p.add_argument("--max-parallel", type=int, default=1)
    p.add_argument("--max-experiments", type=int, default=0)
    p.add_argument("--max-wall-time-minutes", type=int, default=0)
    p.add_argument("--seed-limit", type=int, default=0)
    p.add_argument("--seeds", default="")
    p.add_argument("--seed-policy-config", default="")
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--include-legacy", action="store_true", help="Include legacy_baseline experiments.")
    p.add_argument("--legacy-only", action="store_true", help="Run only legacy_baseline experiments.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    os.chdir(repo_root)

    mode = str(args.mode or "gate").lower()
    if bool(args.nightly):
        mode = "nightly"
    include_legacy = bool(args.include_legacy)
    legacy_only = bool(args.legacy_only)
    if legacy_only:
        include_legacy = True

    config_path = (repo_root / args.config).resolve()
    out_root = (repo_root / args.out_root).resolve()
    cfg = load_config(config_path)
    seed_policy_config = str(args.seed_policy_config or cfg.get("seed_policy_config") or "").strip()
    seed_policy: dict[str, Any] | None = None
    if seed_policy_config:
        seed_policy_path = (repo_root / seed_policy_config).resolve()
        seed_policy = read_seed_policy(seed_policy_path)
        validation = validate_seed_policy(seed_policy)
        write_json(out_root / "seed_policy_validation.json", validation)
        if not bool(validation.get("ok")):
            raise SystemExit("seed policy validation failed")

    experiments = build_matrix(cfg)
    experiments = select_experiments_for_mode(
        experiments,
        mode=mode,
        include_legacy=include_legacy,
        legacy_only=legacy_only,
    )
    only_set = parse_csv_list(args.only)
    exclude_set = parse_csv_list(args.exclude)

    if only_set:
        experiments = [e for e in experiments if str(e["id"]) in only_set]
    if exclude_set:
        experiments = [e for e in experiments if str(e["id"]) not in exclude_set]
    if not experiments:
        raise SystemExit("no experiments selected after filters")

    budget_cfg = cfg.get("budget") if isinstance(cfg.get("budget"), dict) else {}
    max_experiments = int(args.max_experiments) if int(args.max_experiments) > 0 else int(budget_cfg.get("max_experiments") or len(experiments))
    experiments = experiments[: max(1, max_experiments)]

    run_id, run_root = resolve_run_root(out_root, resume=bool(args.resume))
    git_commit = current_git_commit(repo_root)

    if seed_policy is not None:
        write_json(run_root / "seed_policy.json", seed_policy)

    seed_limit = int(args.seed_limit) if int(args.seed_limit) > 0 else None
    cli_seed_override = _normalize_seed_list((args.seeds or "").split(",")) if str(args.seeds or "").strip() else None
    if cli_seed_override and seed_limit is not None:
        cli_seed_override = cli_seed_override[: max(1, int(seed_limit))]
    queue_state = {
        str(exp["id"]): {
            "exp_id": str(exp["id"]),
            "category": str(exp.get("category") or ""),
            "default_enabled": bool(exp.get("default_enabled")),
            "stage": "queued",
            "status": "queued",
            "seed": "-",
            "elapsed_sec": 0.0,
            "eta_sec": None,
            "metric_snapshot": "",
            "updated_at": now_iso(),
        }
        for exp in experiments
    }
    ctx = RunContext(
        repo_root=repo_root,
        out_root=out_root,
        run_root=run_root,
        run_id=run_id,
        config=cfg,
        nightly=(mode == "nightly"),
        dry_run=bool(args.dry_run),
        keep_intermediate=bool(args.keep_intermediate),
        git_commit=git_commit,
        max_parallel=max(1, int(args.max_parallel)),
        resume=bool(args.resume),
        seed_limit=seed_limit,
        seed_policy=seed_policy,
        seed_policy_config=seed_policy_config,
        mode=mode,
        verbose=bool(args.verbose),
        telemetry_path=run_root / "telemetry.jsonl",
        live_summary_path=run_root / "live_summary_snapshot.json",
        run_started_ts=time.time(),
        queue_state=queue_state,
        cli_seeds=cli_seed_override,
    )
    _update_live_snapshot(ctx)

    plan_experiments: list[dict[str, Any]] = []
    for exp in experiments:
        seeds_payload = resolve_experiment_seeds(
            exp=exp,
            cfg=cfg,
            git_commit=git_commit,
            nightly=(mode == "nightly"),
            seed_limit=seed_limit,
            run_id=run_id,
            seed_policy=seed_policy,
            cli_seeds=cli_seed_override,
        )
        plan_experiments.append(
            {
                "exp_id": str(exp.get("id")),
                "category": str(exp.get("category") or ""),
                "default_enabled": bool(exp.get("default_enabled")),
                "seed_mode": str(exp.get("seed_mode") or "regression_fixed"),
                "seed_set_name": seeds_payload.get("seed_set_name"),
                "seed_policy_version": seeds_payload.get("seed_policy_version"),
                "seed_hash": seeds_payload.get("seed_hash"),
                "seed_count": seeds_payload.get("seed_count"),
                "seeds": seeds_payload.get("seeds") or [],
            }
        )

    plan_payload = {
        "schema": "p34_run_plan_v1",
        "generated_at": now_iso(),
        "run_id": run_id,
        "mode": mode,
        "include_legacy": include_legacy,
        "legacy_only": legacy_only,
        "config_path": str(config_path),
        "out_root": str(out_root),
        "run_root": str(run_root),
        "git_commit": git_commit,
        "nightly": ctx.nightly,
        "dry_run": ctx.dry_run,
        "resume": ctx.resume,
        "verbose": ctx.verbose,
        "max_parallel": ctx.max_parallel,
        "max_experiments": max_experiments,
        "seed_policy_config": seed_policy_config,
        "seed_policy_version": (
            (seed_policy or {}).get("seed_policy_version")
            if seed_policy
            else str((_legacy_seed_policy_block(cfg).get("version") or "legacy.p22"))
        ),
        "cli_seed_override": cli_seed_override or [],
        "selected_categories": sorted(
            {
                str(exp.get("category") or MODE_CATEGORY_EXPERIMENTAL)
                for exp in experiments
            }
        ),
        "experiments": [e["id"] for e in experiments],
        "experiments_with_seeds": plan_experiments,
        "budget": budget_cfg,
    }
    write_json(run_root / "run_plan.json", plan_payload)

    rows: list[dict[str, Any]] = []
    max_wall_minutes = int(args.max_wall_time_minutes) if int(args.max_wall_time_minutes) > 0 else int(budget_cfg.get("max_wall_time_minutes") or 120)
    max_wall_sec = max(60, max_wall_minutes * 60)
    exp_total = len(experiments)
    for idx, exp in enumerate(experiments):
        if (time.time() - ctx.run_started_ts) > max_wall_sec:
            exp_id = str(exp["id"])
            emit_progress(
                ctx,
                exp_id=exp_id,
                stage="budget",
                status="budget_cut",
                elapsed_sec=time.time() - ctx.run_started_ts,
                phase="orchestrator",
                metric_snapshot="run_budget_exceeded",
                message="run_budget_exceeded",
            )
            rows.append(
                {
                    "exp_id": exp_id,
                    "category": str(exp.get("category") or MODE_CATEGORY_EXPERIMENTAL),
                    "default_enabled": bool(exp.get("default_enabled")),
                    "status": "budget_cut",
                    "mean": 0.0,
                    "std": 0.0,
                    "avg_reward": 0.0,
                    "reward_std": 0.0,
                    "best_episode_reward": 0.0,
                    "avg_ante_reached": 0.0,
                    "median_ante": 0.0,
                    "win_rate": 0.0,
                    "hand_top1": 0.0,
                    "hand_top3": 0.0,
                    "shop_top1": 0.0,
                    "illegal_action_rate": 1.0,
                    "seed_count": 0,
                    "catastrophic_failure_count": 1,
                    "elapsed_sec": 0.0,
                    "run_dir": str(run_root / exp_id),
                    "seed_set_name": "",
                    "seed_hash": "",
                    "seeds_used": [],
                    "final_win_rate": 0.0,
                    "final_loss": 0.0,
                }
            )
            continue
        rows.append(run_single_experiment(ctx, exp, exp_index=idx + 1, exp_total=exp_total))

    primary_metric = str((cfg.get("evaluation") or {}).get("primary_metric") or "avg_ante_reached")
    rows_sorted = sorted(rows, key=lambda r: (str(r.get("status")) != "passed", -(float(r.get("mean") or 0.0))))

    summary_paths = write_summary_tables(run_root, rows_sorted, primary_metric=primary_metric, run_id=run_id)
    champion_update = update_nightly_decision(
        out_root,
        run_id=run_id,
        ranked_rows=rows_sorted,
        primary_metric=primary_metric,
        evaluation_cfg=(cfg.get("evaluation") or {}),
    )
    report_path = out_root / f"report_p23_{run_id}.md"
    write_comparison_report(report_path, rows_sorted, primary_metric=primary_metric, champion_update=champion_update)

    final_payload = {
        "schema": "p23_report_v1",
        "generated_at": now_iso(),
        "run_id": run_id,
        "mode": mode,
        "status": "PASS" if all(str(r.get("status")) in {"passed", "dry_run", "skipped"} for r in rows_sorted) else "FAIL",
        "rows": rows_sorted,
        "summary_paths": summary_paths,
        "comparison_report": str(report_path),
        "champion_update": champion_update,
    }
    write_json(run_root / "report_p23.json", final_payload)
    _update_live_snapshot(ctx)

    print(f"[P23] run_id={run_id} mode={mode} status={final_payload['status']}")
    print(f"[P23] telemetry={ctx.telemetry_path}")
    print(f"[P23] live_snapshot={ctx.live_summary_path}")
    print(f"[P23] summary_json={summary_paths['json']}")
    print(f"[P23] report_md={report_path}")
    return 0 if final_payload["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
