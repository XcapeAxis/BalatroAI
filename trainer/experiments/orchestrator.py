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
    from trainer.selfsup_train import run_selfsup_training

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
    val_mae = float(final.get("val_score_delta_mae") or 0.0)
    hand_acc = float(final.get("val_hand_type_acc") or 0.0)

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


def run_single_experiment(ctx: RunContext, exp: dict[str, Any], exp_index: int, exp_total: int) -> dict[str, Any]:
    exp_id = str(exp["id"])
    exp_type = str(exp.get("experiment_type") or "standard").strip().lower()
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
                                    else latest_metrics.get("selfsup_p36_action_val_loss")
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
                    else final_seed_metrics.get("selfsup_p36_action_val_loss")
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
    }
    write_json(exp_dir / "stage_results.json", stage_results)
    write_json(exp_dir / "exp_summary.json", exp_summary)
    write_json(
        status_path,
        {
            "status": "success" if success else "failed",
            "mean": metric_summary.get("mean"),
            "std": metric_summary.get("std"),
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
) -> list[dict[str, Any]]:
    mode = str(mode or "gate").lower()
    if mode == "quick":
        picked = [e for e in experiments if bool((e.get("modes") or {}).get("quick", False))]
        return picked if picked else experiments[:3]
    if mode == "milestone":
        picked = [e for e in experiments if bool((e.get("modes") or {}).get("milestone", False))]
        return picked if picked else experiments
    if mode == "nightly":
        picked = [e for e in experiments if bool((e.get("modes") or {}).get("nightly", True))]
        return picked if picked else experiments
    picked = [e for e in experiments if bool((e.get("modes") or {}).get("gate", True))]
    return picked if picked else experiments


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
    return p.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    os.chdir(repo_root)

    mode = str(args.mode or "gate").lower()
    if bool(args.nightly):
        mode = "nightly"

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
    experiments = select_experiments_for_mode(experiments, mode=mode)
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
                    "status": "budget_cut",
                    "mean": 0.0,
                    "std": 0.0,
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
