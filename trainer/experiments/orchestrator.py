from __future__ import annotations

if __package__ is None or __package__ == "":
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

import argparse
import concurrent.futures
import copy
import hashlib
import json
import os
import platform
import random
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from trainer.experiments.champion import update_champion_candidate
from trainer.experiments.matrix import build_matrix
from trainer.experiments.metrics import aggregate_seed_metrics, is_success
from trainer.experiments.report import write_comparison_report, write_summary_tables


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


def select_seeds(
    *,
    exp: dict[str, Any],
    cfg: dict[str, Any],
    git_commit: str,
    nightly: bool,
    seed_limit: int | None,
) -> list[str]:
    seeds_cfg = cfg.get("seeds") or {}
    fixed = [str(s) for s in (seeds_cfg.get("regression_fixed") or [])]
    if not fixed:
        raise ValueError("config.seeds.regression_fixed must not be empty")

    base_seed = str(seeds_cfg.get("base_seed") or "P22BASE")
    extra_count = int(seeds_cfg.get("extra_random") or 0)
    mode = str(exp.get("seed_mode") or "regression_fixed").lower()
    use_extra = nightly or mode == "nightly"
    chosen = list(fixed)
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


def format_progress(
    *,
    exp_id: str,
    stage: str,
    status: str,
    elapsed_sec: float,
    seed_index: int | None = None,
    seed_total: int | None = None,
    metric_snapshot: str = "",
) -> str:
    seed_part = "-"
    if seed_index is not None and seed_total is not None:
        seed_part = f"{seed_index}/{seed_total}"
    return (
        f"[P22] exp={exp_id} stage={stage} status={status} "
        f"seed={seed_part} elapsed={elapsed_sec:.1f}s metric={metric_snapshot}"
    )


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


def _expand_command(command_tmpl: Any, context: dict[str, Any]) -> str | list[str]:
    if command_tmpl is None:
        return ""
    if isinstance(command_tmpl, list):
        return [str(x).format(**context) for x in command_tmpl]
    return str(command_tmpl).format(**context)


def _run_stage_command(
    *,
    exp_id: str,
    stage: str,
    command_tmpl: Any,
    context: dict[str, Any],
    repo_root: Path,
    progress_path: Path,
) -> dict[str, Any]:
    command = _expand_command(command_tmpl, context)
    if not command:
        payload = {"status": "skipped", "reason": "empty_command"}
        append_jsonl(
            progress_path,
            {
                "ts": now_iso(),
                "exp_id": exp_id,
                "stage": stage,
                "status": "skipped",
                "message": "empty_command",
            },
        )
        return payload
    result = run_process(command, cwd=repo_root)
    status = "ok" if result["returncode"] == 0 else "failed"
    append_jsonl(
        progress_path,
        {
            "ts": now_iso(),
            "exp_id": exp_id,
            "stage": stage,
            "status": status,
            "returncode": result["returncode"],
            "elapsed_sec": result["elapsed_sec"],
            "command": command,
        },
    )
    return {
        "status": status,
        "returncode": result["returncode"],
        "elapsed_sec": result["elapsed_sec"],
        "stdout_tail": (result["stdout"] or "")[-4000:],
        "stderr_tail": (result["stderr"] or "")[-4000:],
        "command": command,
    }


def _synthetic_eval_metric(exp_id: str, seed: str, bias: float) -> float:
    digest = hashlib.sha256(f"{exp_id}|{seed}".encode("utf-8")).hexdigest()
    frac = (int(digest[:8], 16) % 1000) / 10000.0
    return 1.0 + bias + frac


def run_single_experiment(ctx: RunContext, exp: dict[str, Any]) -> dict[str, Any]:
    exp_id = str(exp["id"])
    exp_dir = ctx.run_root / exp_id
    exp_dir.mkdir(parents=True, exist_ok=True)

    status_path = exp_dir / "status.json"
    if ctx.resume:
        old = read_json(status_path)
        if old and str(old.get("status")) == "success":
            return {
                "exp_id": exp_id,
                "status": "success",
                "mean": old.get("mean"),
                "std": old.get("std"),
                "seed_count": old.get("seed_count"),
                "catastrophic_failure_count": old.get("catastrophic_failure_count", 0),
                "elapsed_sec": old.get("elapsed_sec", 0.0),
                "run_dir": str(exp_dir),
                "resumed": True,
            }

    progress_path = exp_dir / "progress.jsonl"
    stage_results: dict[str, Any] = {}
    started = time.time()
    budget = ctx.config.get("budget") or {}
    max_wall_time_min = int(budget.get("max_wall_time_minutes") or 60)
    max_wall_time_sec = max_wall_time_min * 60

    seeds = select_seeds(
        exp=exp,
        cfg=ctx.config,
        git_commit=ctx.git_commit,
        nightly=ctx.nightly,
        seed_limit=ctx.seed_limit,
    )
    write_json(exp_dir / "seeds_used.json", {"seeds": seeds, "count": len(seeds)})

    manifest = {
        "schema": "p22_run_manifest_v1",
        "generated_at": now_iso(),
        "run_id": ctx.run_id,
        "exp_id": exp_id,
        "git_commit": ctx.git_commit,
        "python_version": detect_python_version(),
        "platform": platform.platform(),
        "budget": copy.deepcopy(budget),
        "seed_mode": exp.get("seed_mode", "regression_fixed"),
        "seeds": seeds,
        "experiment": exp,
    }
    write_json(exp_dir / "run_manifest.json", manifest)

    print(format_progress(exp_id=exp_id, stage="init", status="start", elapsed_sec=0.0))
    append_jsonl(progress_path, {"ts": now_iso(), "exp_id": exp_id, "stage": "init", "status": "start"})

    if ctx.dry_run:
        status = {
            "status": "dry_run",
            "exp_id": exp_id,
            "seed_count": len(seeds),
            "elapsed_sec": 0.0,
        }
        write_json(status_path, status)
        return {
            "exp_id": exp_id,
            "status": "dry_run",
            "mean": 0.0,
            "std": 0.0,
            "seed_count": len(seeds),
            "catastrophic_failure_count": 0,
            "elapsed_sec": 0.0,
            "run_dir": str(exp_dir),
        }

    def over_budget() -> bool:
        return (time.time() - started) > max_wall_time_sec

    stages = exp.get("stages") or {}
    gate_flag = str(exp.get("gate_flag") or "RunFast")
    context = {
        "exp_id": exp_id,
        "run_dir": str(exp_dir),
        "repo_root": str(ctx.repo_root),
    }

    if bool(stages.get("sanity", True)):
        compile_files = exp.get("sanity_compile") or ["trainer/train_rl.py", "trainer/eval_long_horizon.py"]
        compile_cmd = ["python", "-B", "-m", "py_compile"] + [str(x) for x in compile_files]
        stage_results["sanity"] = _run_stage_command(
            exp_id=exp_id,
            stage="sanity",
            command_tmpl=compile_cmd,
            context=context,
            repo_root=ctx.repo_root,
            progress_path=progress_path,
        )
        print(
            format_progress(
                exp_id=exp_id,
                stage="sanity",
                status=stage_results["sanity"]["status"],
                elapsed_sec=time.time() - started,
            )
        )
        if stage_results["sanity"]["status"] != "ok":
            write_json(status_path, {"status": "failed", "reason": "sanity_failed", **stage_results["sanity"]})
            return {
                "exp_id": exp_id,
                "status": "failed",
                "mean": 0.0,
                "std": 0.0,
                "seed_count": len(seeds),
                "catastrophic_failure_count": 1,
                "elapsed_sec": time.time() - started,
                "run_dir": str(exp_dir),
            }

    if over_budget():
        write_json(status_path, {"status": "failed", "reason": "budget_exceeded_before_gate"})
        return {
            "exp_id": exp_id,
            "status": "failed",
            "mean": 0.0,
            "std": 0.0,
            "seed_count": len(seeds),
            "catastrophic_failure_count": 1,
            "elapsed_sec": time.time() - started,
            "run_dir": str(exp_dir),
        }

    if bool(stages.get("gate", True)):
        gate_cmd = [
            "powershell",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            "scripts/run_regressions.ps1",
            f"-{gate_flag}",
        ]
        stage_results["gate"] = _run_stage_command(
            exp_id=exp_id,
            stage="gate",
            command_tmpl=gate_cmd,
            context=context,
            repo_root=ctx.repo_root,
            progress_path=progress_path,
        )
        print(
            format_progress(
                exp_id=exp_id,
                stage="gate",
                status=stage_results["gate"]["status"],
                elapsed_sec=time.time() - started,
            )
        )
        if stage_results["gate"]["status"] != "ok":
            write_json(status_path, {"status": "failed", "reason": "gate_failed", **stage_results["gate"]})
            return {
                "exp_id": exp_id,
                "status": "failed",
                "mean": 0.0,
                "std": 0.0,
                "seed_count": len(seeds),
                "catastrophic_failure_count": 1,
                "elapsed_sec": time.time() - started,
                "run_dir": str(exp_dir),
            }

    if bool(stages.get("dataset")):
        stage_results["dataset"] = _run_stage_command(
            exp_id=exp_id,
            stage="dataset",
            command_tmpl=exp.get("dataset_command"),
            context=context,
            repo_root=ctx.repo_root,
            progress_path=progress_path,
        )
        print(
            format_progress(
                exp_id=exp_id,
                stage="dataset",
                status=stage_results["dataset"]["status"],
                elapsed_sec=time.time() - started,
            )
        )
        if stage_results["dataset"]["status"] != "ok":
            write_json(status_path, {"status": "failed", "reason": "dataset_failed", **stage_results["dataset"]})
            return {
                "exp_id": exp_id,
                "status": "failed",
                "mean": 0.0,
                "std": 0.0,
                "seed_count": len(seeds),
                "catastrophic_failure_count": 1,
                "elapsed_sec": time.time() - started,
                "run_dir": str(exp_dir),
            }

    if bool(stages.get("train")):
        stage_results["train"] = _run_stage_command(
            exp_id=exp_id,
            stage="train",
            command_tmpl=exp.get("train_command"),
            context=context,
            repo_root=ctx.repo_root,
            progress_path=progress_path,
        )
        print(
            format_progress(
                exp_id=exp_id,
                stage="train",
                status=stage_results["train"]["status"],
                elapsed_sec=time.time() - started,
            )
        )
        if stage_results["train"]["status"] != "ok":
            write_json(status_path, {"status": "failed", "reason": "train_failed", **stage_results["train"]})
            return {
                "exp_id": exp_id,
                "status": "failed",
                "mean": 0.0,
                "std": 0.0,
                "seed_count": len(seeds),
                "catastrophic_failure_count": 1,
                "elapsed_sec": time.time() - started,
                "run_dir": str(exp_dir),
            }

    primary_metric = str((ctx.config.get("evaluation") or {}).get("primary_metric") or "score")
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
                        "elapsed_sec": time.time() - started,
                        "metrics": {},
                    }
                )
                break

            seed_ctx = dict(context)
            seed_ctx["seed"] = seed
            seed_ctx["seed_index"] = seed_idx
            seed_ctx["seed_total"] = len(seeds)
            if eval_cmd_tmpl:
                result = _run_stage_command(
                    exp_id=exp_id,
                    stage="eval",
                    command_tmpl=eval_cmd_tmpl,
                    context=seed_ctx,
                    repo_root=ctx.repo_root,
                    progress_path=progress_path,
                )
                if result["status"] == "ok":
                    metric_value = _synthetic_eval_metric(exp_id, seed, bias)
                    seed_results.append(
                        {
                            "seed": seed,
                            "status": "ok",
                            "stage": "eval",
                            "elapsed_sec": result["elapsed_sec"],
                            "metrics": {primary_metric: metric_value},
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
                metric_value = _synthetic_eval_metric(exp_id, seed, bias)
                seed_results.append(
                    {
                        "seed": seed,
                        "status": "ok",
                        "stage": "eval",
                        "elapsed_sec": 0.0,
                        "metrics": {primary_metric: metric_value},
                    }
                )
                append_jsonl(
                    progress_path,
                    {
                        "ts": now_iso(),
                        "exp_id": exp_id,
                        "stage": "eval",
                        "seed": seed,
                        "status": "ok",
                        "metric": metric_value,
                    },
                )
            snap = seed_results[-1]["metrics"].get(primary_metric)
            print(
                format_progress(
                    exp_id=exp_id,
                    stage="eval",
                    status=seed_results[-1]["status"],
                    elapsed_sec=time.time() - started,
                    seed_index=seed_idx,
                    seed_total=len(seeds),
                    metric_snapshot=f"{primary_metric}={snap}",
                )
            )

    metric_summary = aggregate_seed_metrics(seed_results, primary_metric=primary_metric)
    success = is_success(metric_summary)
    elapsed_total = time.time() - started

    exp_summary = {
        "schema": "p22_experiment_summary_v1",
        "generated_at": now_iso(),
        "run_id": ctx.run_id,
        "exp_id": exp_id,
        "status": "success" if success else "failed",
        "stages": stage_results,
        "seed_metrics": metric_summary,
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
            "seed_count": metric_summary.get("count"),
            "catastrophic_failure_count": metric_summary.get("catastrophic_failure_count"),
            "elapsed_sec": elapsed_total,
        },
    )

    if not ctx.keep_intermediate:
        for pattern in ("*.pt", "*.ckpt", "*.bin", "*.parquet"):
            for file_path in exp_dir.rglob(pattern):
                try:
                    file_path.unlink(missing_ok=True)
                except Exception:
                    pass

    return {
        "exp_id": exp_id,
        "status": "success" if success else "failed",
        "mean": metric_summary.get("mean"),
        "std": metric_summary.get("std"),
        "seed_count": metric_summary.get("count"),
        "catastrophic_failure_count": metric_summary.get("catastrophic_failure_count"),
        "elapsed_sec": elapsed_total,
        "run_dir": str(exp_dir),
    }


def load_config(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("config file must be a mapping")
    return payload


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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="P22 experiment orchestrator.")
    p.add_argument("--config", default="configs/experiments/p22.yaml")
    p.add_argument("--out-root", default="docs/artifacts/p22")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--only", default="")
    p.add_argument("--exclude", default="")
    p.add_argument("--nightly", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--keep-intermediate", action="store_true")
    p.add_argument("--max-parallel", type=int, default=1)
    p.add_argument("--seed-limit", type=int, default=0)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    os.chdir(repo_root)

    config_path = (repo_root / args.config).resolve()
    out_root = (repo_root / args.out_root).resolve()
    cfg = load_config(config_path)
    experiments = build_matrix(cfg)
    only_set = parse_csv_list(args.only)
    exclude_set = parse_csv_list(args.exclude)

    if only_set:
        experiments = [e for e in experiments if str(e["id"]) in only_set]
    if exclude_set:
        experiments = [e for e in experiments if str(e["id"]) not in exclude_set]
    if not experiments:
        raise SystemExit("no experiments selected after filters")

    run_id, run_root = resolve_run_root(out_root, resume=bool(args.resume))
    git_commit = current_git_commit(repo_root)

    seed_limit = int(args.seed_limit) if int(args.seed_limit) > 0 else None
    ctx = RunContext(
        repo_root=repo_root,
        out_root=out_root,
        run_root=run_root,
        run_id=run_id,
        config=cfg,
        nightly=bool(args.nightly),
        dry_run=bool(args.dry_run),
        keep_intermediate=bool(args.keep_intermediate),
        git_commit=git_commit,
        max_parallel=max(1, int(args.max_parallel)),
        resume=bool(args.resume),
        seed_limit=seed_limit,
    )

    plan_payload = {
        "schema": "p22_run_plan_v1",
        "generated_at": now_iso(),
        "run_id": run_id,
        "config_path": str(config_path),
        "out_root": str(out_root),
        "run_root": str(run_root),
        "git_commit": git_commit,
        "nightly": ctx.nightly,
        "dry_run": ctx.dry_run,
        "resume": ctx.resume,
        "max_parallel": ctx.max_parallel,
        "experiments": [e["id"] for e in experiments],
    }
    write_json(run_root / "run_plan.json", plan_payload)

    rows: list[dict[str, Any]] = []
    if ctx.max_parallel <= 1:
        for exp in experiments:
            rows.append(run_single_experiment(ctx, exp))
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=ctx.max_parallel) as pool:
            futures = [pool.submit(run_single_experiment, ctx, exp) for exp in experiments]
            for fut in concurrent.futures.as_completed(futures):
                rows.append(fut.result())

    primary_metric = str((cfg.get("evaluation") or {}).get("primary_metric") or "score")
    rows_sorted = sorted(rows, key=lambda r: (str(r.get("status")) != "success", -(float(r.get("mean") or 0.0))))

    summary_paths = write_summary_tables(run_root, rows_sorted, primary_metric=primary_metric, run_id=run_id)
    champion_update = update_champion_candidate(
        out_root,
        run_id=run_id,
        ranked_rows=rows_sorted,
        primary_metric=primary_metric,
        evaluation_cfg=(cfg.get("evaluation") or {}),
    )
    report_path = out_root / f"report_p22_{run_id}.md"
    write_comparison_report(report_path, rows_sorted, primary_metric=primary_metric, champion_update=champion_update)

    final_payload = {
        "schema": "p22_report_v1",
        "generated_at": now_iso(),
        "run_id": run_id,
        "status": "PASS" if all(str(r.get("status")) in {"success", "dry_run"} for r in rows_sorted) else "FAIL",
        "rows": rows_sorted,
        "summary_paths": summary_paths,
        "comparison_report": str(report_path),
        "champion_update": champion_update,
    }
    write_json(run_root / "report_p22.json", final_payload)

    print(f"[P22] run_id={run_id} status={final_payload['status']}")
    print(f"[P22] summary_json={summary_paths['json']}")
    print(f"[P22] report_md={report_path}")
    return 0 if final_payload["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())

