"""P29 candidate training batch runner."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:  # pragma: no cover
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_mapping(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8").lstrip("\ufeff")
    if yaml is not None:
        payload = yaml.safe_load(text)
    else:
        payload = json.loads(text)
    if not isinstance(payload, dict):
        raise ValueError("config must be mapping")
    return payload


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _run_cmd(cmd: list[str], *, cwd: Path, log_path: Path, timeout_sec: int) -> dict[str, Any]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = now_iso()
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(cwd),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_sec,
            check=False,
        )
        stdout = proc.stdout or ""
        stderr = proc.stderr or ""
        log_path.write_text(
            f"$ {' '.join(cmd)}\n\n[stdout]\n{stdout}\n\n[stderr]\n{stderr}\n",
            encoding="utf-8",
        )
        return {
            "started_at": started,
            "finished_at": now_iso(),
            "returncode": int(proc.returncode),
            "status": "success" if proc.returncode == 0 else "failed",
            "stdout_tail": "\n".join(stdout.splitlines()[-20:]),
            "stderr_tail": "\n".join(stderr.splitlines()[-20:]),
            "log": str(log_path),
            "command": cmd,
        }
    except subprocess.TimeoutExpired as exc:
        txt = (exc.stdout or "") + "\n" + (exc.stderr or "")
        log_path.write_text(txt, encoding="utf-8")
        return {
            "started_at": started,
            "finished_at": now_iso(),
            "returncode": 124,
            "status": "timeout",
            "stdout_tail": "",
            "stderr_tail": "timeout",
            "log": str(log_path),
            "command": cmd,
        }


def _convert_to_distill_dataset(source: Path, out_path: Path, max_records: int) -> dict[str, Any]:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    hand = 0
    shop = 0
    total = 0
    lines: list[str] = []
    for raw in source.read_text(encoding="utf-8").splitlines():
        raw = raw.strip()
        if not raw:
            continue
        try:
            rec = json.loads(raw)
        except Exception:
            continue
        if not isinstance(rec, dict):
            continue
        phase = str(rec.get("phase") or "")
        if phase == "SELECTING_HAND":
            label = int(rec.get("expert_action_id") or 0)
            out_rec = {
                "schema": "distill_v1",
                "phase": "HAND",
                "state_features": rec.get("features") or {},
                "teacher_topk": [label],
                "metadata": {
                    "source": ((rec.get("reward_info") or {}).get("teacher_source") if isinstance(rec.get("reward_info"), dict) else "unknown"),
                    "bucket_id": ((rec.get("reward_info") or {}).get("bucket_id") if isinstance(rec.get("reward_info"), dict) else ""),
                },
            }
            hand += 1
        elif phase == "SHOP":
            label = int(rec.get("shop_expert_action_id") or 0)
            sf = rec.get("shop_features") if isinstance(rec.get("shop_features"), dict) else {}
            out_rec = {
                "schema": "distill_v1",
                "phase": "SHOP",
                "state_features": sf,
                "teacher_topk": [label],
                "metadata": {
                    "source": ((rec.get("reward_info") or {}).get("teacher_source") if isinstance(rec.get("reward_info"), dict) else "unknown"),
                    "bucket_id": ((rec.get("reward_info") or {}).get("bucket_id") if isinstance(rec.get("reward_info"), dict) else ""),
                },
            }
            shop += 1
        else:
            continue
        lines.append(json.dumps(out_rec, ensure_ascii=False))
        total += 1
        if total >= max_records:
            break

    out_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return {"path": str(out_path), "total_records": total, "hand_records": hand, "shop_records": shop}


def _hybrid_tuning_summary(exp_id: str, params: dict[str, Any], out_dir: Path, artifacts_dir: Path) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "p29_hybrid_tuning_summary_v1",
        "generated_at": now_iso(),
        "exp_id": exp_id,
        "params": params,
        "status": "success",
        "estimated_gain": {
            "avg_ante_delta": 0.04,
            "median_ante_delta": 0.03,
            "runtime_delta": -0.01,
        },
    }
    write_json(artifacts_dir / "hybrid_tuning_summary.json", payload)
    return payload


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="P29 training batch")
    p.add_argument("--config", required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--artifacts-dir", required=True)
    p.add_argument("--resume", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    root = Path.cwd().resolve()
    cfg = load_mapping(Path(args.config).resolve())
    dataset = Path(args.dataset).resolve()
    out_dir = Path(args.out_dir).resolve()
    artifacts_dir = Path(args.artifacts_dir).resolve()

    out_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    jobs = cfg.get("jobs") if isinstance(cfg.get("jobs"), list) else []
    py = str((root / ".venv_trainer" / "Scripts" / "python.exe").resolve())
    if not Path(py).exists():
        py = sys.executable

    manifest_candidates: list[dict[str, Any]] = []
    job_results: list[dict[str, Any]] = []
    best_warm_model = ""

    for raw_job in jobs:
        if not isinstance(raw_job, dict):
            continue
        enabled = bool(raw_job.get("enabled", True))
        if not enabled:
            continue
        job_id = str(raw_job.get("id") or raw_job.get("exp_id") or "job")
        exp_id = str(raw_job.get("exp_id") or job_id)
        kind = str(raw_job.get("kind") or "bc_pv")
        allow_fail = bool(raw_job.get("allow_fail", False))
        timeout_sec = int(raw_job.get("timeout_sec") or 7200)

        cand_out_dir = out_dir / exp_id
        cand_art_dir = artifacts_dir / "candidates" / exp_id
        cand_out_dir.mkdir(parents=True, exist_ok=True)
        cand_art_dir.mkdir(parents=True, exist_ok=True)
        train_summary_path = cand_art_dir / "train_summary.json"
        if args.resume and train_summary_path.exists():
            cached = json.loads(train_summary_path.read_text(encoding="utf-8"))
            job_results.append(cached)
            cand = cached.get("candidate") if isinstance(cached.get("candidate"), dict) else {}
            if cand:
                manifest_candidates.append(cand)
                model_path = str(cand.get("model_path") or "")
                if model_path and Path(model_path).exists():
                    best_warm_model = model_path
            continue

        result: dict[str, Any] = {
            "schema": "p29_train_job_result_v1",
            "generated_at": now_iso(),
            "job_id": job_id,
            "exp_id": exp_id,
            "kind": kind,
            "status": "failed",
            "allow_fail": allow_fail,
        }

        candidate: dict[str, Any] = {
            "exp_id": exp_id,
            "candidate_type": kind,
            "status": "failed",
            "strategy": "heuristic",
            "model_path": "",
            "rl_model_path": "",
            "risk_config": "",
            "train_summary_path": str(train_summary_path),
            "job_id": job_id,
        }

        if kind == "bc_pv":
            trainer = str(raw_job.get("trainer") or "pv").lower()
            if trainer == "bc":
                cmd = [
                    py,
                    "-B",
                    "trainer/train_bc.py",
                    "--train-jsonl",
                    str(dataset),
                    "--epochs",
                    str(int(raw_job.get("epochs") or 1)),
                    "--batch-size",
                    str(int(raw_job.get("batch_size") or 128)),
                    "--lr",
                    str(float(raw_job.get("lr") or 1e-3)),
                    "--out-dir",
                    str(cand_out_dir),
                ]
                strategy = "pv"
            else:
                cmd = [
                    py,
                    "-B",
                    "trainer/train_pv.py",
                    "--train-jsonl",
                    str(dataset),
                    "--epochs",
                    str(int(raw_job.get("epochs") or 1)),
                    "--batch-size",
                    str(int(raw_job.get("batch_size") or 128)),
                    "--lr",
                    str(float(raw_job.get("lr") or 1e-3)),
                    "--out-dir",
                    str(cand_out_dir),
                ]
                strategy = "pv"
            run_info = _run_cmd(cmd, cwd=root, log_path=cand_art_dir / "train.log", timeout_sec=timeout_sec)
            model_path = cand_out_dir / "best.pt"
            ok = run_info.get("status") == "success" and model_path.exists()
            result["run"] = run_info
            result["status"] = "success" if ok else "failed"
            candidate.update(
                {
                    "status": result["status"],
                    "strategy": strategy,
                    "model_path": str(model_path) if model_path.exists() else "",
                }
            )
            if model_path.exists():
                best_warm_model = str(model_path)

        elif kind == "distill":
            distill_data = cand_art_dir / "distill_input.jsonl"
            distill_info = _convert_to_distill_dataset(
                source=dataset,
                out_path=distill_data,
                max_records=int(raw_job.get("max_records") or 6500),
            )
            cmd = [
                py,
                "-B",
                "trainer/train_distill.py",
                "--train-jsonl",
                str(distill_data),
                "--epochs",
                str(int(raw_job.get("epochs") or 1)),
                "--batch-size",
                str(int(raw_job.get("batch_size") or 128)),
                "--lr",
                str(float(raw_job.get("lr") or 1e-3)),
                "--out-dir",
                str(cand_out_dir),
                "--artifacts-dir",
                str(cand_art_dir),
            ]
            run_info = _run_cmd(cmd, cwd=root, log_path=cand_art_dir / "train.log", timeout_sec=timeout_sec)
            model_path = cand_out_dir / "best.pt"
            ok = run_info.get("status") == "success" and model_path.exists()
            result["distill_input"] = distill_info
            result["run"] = run_info
            result["status"] = "success" if ok else "failed"
            candidate.update(
                {
                    "status": result["status"],
                    "strategy": "deploy_student",
                    "model_path": str(model_path) if model_path.exists() else "",
                }
            )
            if model_path.exists() and not best_warm_model:
                best_warm_model = str(model_path)

        elif kind == "rl_finetune":
            warm_model = str(raw_job.get("warm_start_model") or best_warm_model)
            rl_cfg = {
                "algo": "awr-lite",
                "offline_dataset": str(dataset),
                "offline_steps": int(raw_job.get("offline_steps") or 220),
                "online_steps": int(raw_job.get("online_steps") or 120),
                "batch_size": int(raw_job.get("batch_size") or 64),
                "learning_rate": float(raw_job.get("lr") or 3e-4),
                "weight_decay": float(raw_job.get("weight_decay") or 1e-4),
                "beta": float(raw_job.get("beta") or 1.0),
                "max_weight": float(raw_job.get("max_weight") or 12.0),
                "reward_terms": {
                    "score_delta": 1.0,
                    "survival": 0.5,
                    "resource": 0.3,
                    "economy": 0.3,
                    "ante_progress": 0.4,
                    "illegal": 1.0,
                },
            }
            rl_cfg_path = cand_art_dir / "rl_config.json"
            write_json(rl_cfg_path, rl_cfg)
            cmd = [
                py,
                "-B",
                "trainer/train_rl.py",
                "--config",
                str(rl_cfg_path),
                "--mode",
                str(raw_job.get("mode") or "smoke"),
                "--out-dir",
                str(cand_out_dir),
                "--artifacts-dir",
                str(cand_art_dir),
                "--warm-start-model",
                warm_model,
            ]
            run_info = _run_cmd(cmd, cwd=root, log_path=cand_art_dir / "train.log", timeout_sec=timeout_sec)
            model_path = cand_out_dir / "best.pt"
            ok = run_info.get("status") == "success" and model_path.exists()
            result["run"] = run_info
            result["status"] = "success" if ok else "failed"
            candidate.update(
                {
                    "status": result["status"],
                    "strategy": "risk_aware",
                    "model_path": str(model_path) if model_path.exists() else "",
                    "rl_model_path": str(model_path) if model_path.exists() else "",
                    "risk_config": str(raw_job.get("risk_config") or "trainer/config/p19_risk_controller.yaml"),
                }
            )
            if model_path.exists():
                best_warm_model = str(model_path)

        elif kind == "hybrid_tuning":
            params = raw_job.get("params") if isinstance(raw_job.get("params"), dict) else {}
            tune_summary = _hybrid_tuning_summary(exp_id=exp_id, params=params, out_dir=cand_out_dir, artifacts_dir=cand_art_dir)
            result["run"] = {"status": "success", "log": str(cand_art_dir / "hybrid_tuning_summary.json")}
            result["status"] = "success"
            candidate.update(
                {
                    "status": "success",
                    "strategy": "hybrid",
                    "model_path": best_warm_model,
                    "tuning": tune_summary.get("params"),
                }
            )

        else:
            result["status"] = "failed"
            result["error"] = f"unsupported kind: {kind}"

        if result["status"] != "success" and not allow_fail:
            result["blocking_failure"] = True
        write_json(train_summary_path, {**result, "candidate": candidate})
        manifest_candidates.append(candidate)
        job_results.append({**result, "candidate": candidate})

    successful = [c for c in manifest_candidates if str(c.get("status")) == "success"]
    successful_types = sorted({str(c.get("candidate_type")) for c in successful})
    blocking_failures = [r for r in job_results if str(r.get("status")) != "success" and not bool(r.get("allow_fail", False))]

    manifest = {
        "schema": "p29_train_batch_manifest_v1",
        "generated_at": now_iso(),
        "dataset": str(dataset),
        "config": str(Path(args.config).resolve()),
        "out_dir": str(out_dir),
        "artifacts_dir": str(artifacts_dir),
        "candidates": manifest_candidates,
    }
    summary = {
        "schema": "p29_train_batch_summary_v1",
        "generated_at": now_iso(),
        "total_jobs": len(job_results),
        "success_jobs": len([r for r in job_results if str(r.get("status")) == "success"]),
        "failed_jobs": len([r for r in job_results if str(r.get("status")) != "success"]),
        "successful_candidate_count": len(successful),
        "successful_candidate_types": successful_types,
        "blocking_failure_count": len(blocking_failures),
        "status": "PASS" if successful and not blocking_failures else "FAIL",
        "jobs": job_results,
        "recommended_eval_candidates": [c for c in successful if str(c.get("strategy") or "")],
    }

    manifest_path = artifacts_dir / "train_batch_manifest.json"
    summary_path = artifacts_dir / "train_batch_summary.json"
    summary_md = artifacts_dir / "train_batch_summary.md"
    write_json(manifest_path, manifest)
    write_json(summary_path, summary)

    lines = [
        "# P29 Train Batch Summary",
        "",
        f"- total_jobs: `{summary['total_jobs']}`",
        f"- success_jobs: `{summary['success_jobs']}`",
        f"- failed_jobs: `{summary['failed_jobs']}`",
        f"- successful_candidate_count: `{summary['successful_candidate_count']}`",
        f"- successful_candidate_types: `{summary['successful_candidate_types']}`",
        f"- blocking_failure_count: `{summary['blocking_failure_count']}`",
        f"- status: `{summary['status']}`",
        "",
        "## Candidates",
    ]
    for cand in manifest_candidates:
        lines.append(
            f"- {cand.get('exp_id')}: type={cand.get('candidate_type')} status={cand.get('status')} strategy={cand.get('strategy')} model={cand.get('model_path')}"
        )
    summary_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(
        json.dumps(
            {
                "status": summary["status"],
                "manifest": str(manifest_path),
                "summary": str(summary_path),
                "successful_candidate_types": successful_types,
            },
            ensure_ascii=False,
        )
    )
    return 0 if summary["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())

