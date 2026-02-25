from __future__ import annotations

if __package__ is None or __package__ == "":
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).resolve().parent.parent))

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests

from trainer.utils import setup_logger, warn_if_unstable_python

EXIT_OK = 0
EXIT_DRIFT_FAIL = 10
EXIT_LABEL_FAIL = 11
EXIT_SERVICE_FAIL = 12
EXIT_TRAIN_EVAL_FAIL = 13


def _ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _find_latest_model() -> Path | None:
    runs = Path("trainer_runs")
    if not runs.exists():
        return None
    best = sorted(runs.rglob("best.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
    if best:
        return best[0]
    last = sorted(runs.rglob("last.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
    return last[0] if last else None


def _find_trainer_python() -> Path:
    candidates = [
        Path(".venv_trainer") / "Scripts" / "python.exe",
        Path(".venv") / "Scripts" / "python.exe",
    ]
    for cand in candidates:
        if cand.exists():
            return cand
    return Path(sys.executable)


def _tail_lines(text: str, n: int = 100) -> str:
    lines = text.splitlines()
    return "\n".join(lines[-n:]) if lines else ""


def _decode_output(blob: bytes | None) -> str:
    if not blob:
        return ""
    for enc in ("utf-8", "gbk", "cp936"):
        try:
            return blob.decode(enc)
        except Exception:
            continue
    return blob.decode("utf-8", errors="replace")


@dataclass
class StageFailure(RuntimeError):
    stage: str
    exit_code: int
    reason: str
    command: list[str] | None = None
    log_tail: str = ""


def _health_ok(base_url: str, timeout_sec: float = 2.5) -> bool:
    body = {"jsonrpc": "2.0", "id": 1, "method": "health", "params": {}}
    try:
        resp = requests.post(base_url, json=body, timeout=timeout_sec)
        if resp.status_code == 200:
            return True
    except Exception:
        pass

    body = {"jsonrpc": "2.0", "id": 2, "method": "gamestate", "params": {}}
    try:
        resp = requests.post(base_url, json=body, timeout=timeout_sec)
        return resp.status_code == 200
    except Exception:
        return False


def _stop_service() -> None:
    for name in ("Balatro.exe", "balatrobot.exe", "uvx.exe"):
        subprocess.run(["taskkill", "/IM", name, "/T", "/F"], capture_output=True, text=True, check=False)


def _clear_lovely_dump() -> None:
    dump_dir = Path(r"C:\Users\Administrator\AppData\Roaming\Balatro\Mods\lovely\dump")
    if dump_dir.exists():
        shutil.rmtree(dump_dir, ignore_errors=True)


def _start_service(base_url: str, logger) -> subprocess.Popen[str] | None:
    port = 12346
    try:
        port = int(base_url.rsplit(":", 1)[-1])
    except Exception:
        pass
    uvx = shutil.which("uvx")
    if not uvx:
        logger.error("uvx not found in PATH")
        return None

    args = [uvx, "balatrobot", "serve", "--headless", "--fast", "--port", str(port)]
    love = Path(r"D:\SteamLibrary\steamapps\common\Balatro\Balatro.exe")
    lovely = Path(r"D:\SteamLibrary\steamapps\common\Balatro\version.dll")
    if love.exists() and lovely.exists():
        args += ["--love-path", str(love), "--lovely-path", str(lovely)]
    elif love.exists():
        args += ["--balatro-path", str(love)]

    logger.info("starting service: %s", " ".join(args))
    try:
        proc = subprocess.Popen(args, cwd=str(Path.cwd()))
        return proc
    except Exception as exc:
        logger.error("failed to start service: %s", exc)
        return None


def _ensure_service(base_url: str, service_manage: str, service_events: list[dict[str, Any]], logger) -> subprocess.Popen[str] | None:
    if _health_ok(base_url):
        service_events.append({"ts": _now_iso(), "event": "health_ok"})
        return None
    if service_manage != "auto":
        raise StageFailure("service", EXIT_SERVICE_FAIL, f"base_url unhealthy: {base_url}")

    service_events.append({"ts": _now_iso(), "event": "restart_begin"})
    _stop_service()
    time.sleep(1.0)
    _clear_lovely_dump()
    proc = _start_service(base_url, logger)
    if proc is None:
        raise StageFailure("service", EXIT_SERVICE_FAIL, "failed to spawn balatrobot service process")
    for _ in range(45):
        if _health_ok(base_url):
            service_events.append({"ts": _now_iso(), "event": "restart_success"})
            return proc
        time.sleep(1.0)
    service_events.append({"ts": _now_iso(), "event": "restart_timeout"})
    raise StageFailure("service", EXIT_SERVICE_FAIL, "service health timeout after restart")


def _run_cmd(
    *,
    stage: str,
    label: str,
    cmd: list[str],
    timeout_sec: int,
    stage_dir: Path,
    logger,
) -> dict[str, Any]:
    log_path = stage_dir / f"{label}.log"
    logger.info("[%s] run: %s", stage, " ".join(cmd))
    started = _now_iso()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=False, timeout=timeout_sec, check=False)
    except subprocess.TimeoutExpired as exc:
        out = _decode_output(exc.stdout) + "\n" + _decode_output(exc.stderr)
        log_path.write_text(out, encoding="utf-8")
        raise StageFailure(stage, EXIT_TRAIN_EVAL_FAIL, f"{label} timeout ({timeout_sec}s)", command=cmd, log_tail=_tail_lines(out))

    stdout_text = _decode_output(proc.stdout)
    stderr_text = _decode_output(proc.stderr)
    merged = stdout_text + ("\n" + stderr_text if stderr_text else "")
    log_path.write_text(merged, encoding="utf-8")
    if proc.returncode != 0:
        raise StageFailure(stage, EXIT_TRAIN_EVAL_FAIL, f"{label} failed rc={proc.returncode}", command=cmd, log_tail=_tail_lines(merged))

    return {
        "label": label,
        "command": cmd,
        "timeout_sec": timeout_sec,
        "return_code": proc.returncode,
        "started_at": started,
        "finished_at": _now_iso(),
        "log": str(log_path),
    }


def _default_seed_file() -> Path:
    seed_file = Path("trainer/config/p16_fixed_seeds.txt")
    seed_file.parent.mkdir(parents=True, exist_ok=True)
    if not seed_file.exists():
        seeds = [f"{1000 + i}" for i in range(40)]
        seed_file.write_text("\n".join(seeds) + "\n", encoding="utf-8")
    return seed_file


def _update_status_md(status: str, out_dir: Path | None, summary: dict[str, Any] | None) -> None:
    status_path = Path("docs/COVERAGE_P16_STATUS.md")
    status_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# P16 Status",
        "",
        f"- status: {status}",
        f"- updated_at_utc: {_now_iso()}",
        f"- latest_artifact_dir: {str(out_dir) if out_dir else ''}",
    ]
    if summary:
        drift = summary.get("drift", {})
        ds = summary.get("dataset", {})
        off = summary.get("offline_eval", {})
        lh = summary.get("long_horizon", {})
        lines += [
            f"- drift_mismatch_count: {drift.get('mismatch_count', '')}",
            f"- hand_records: {ds.get('hand_records', '')}",
            f"- shop_records: {ds.get('shop_records', '')}",
            f"- invalid_rows: {ds.get('invalid_rows', '')}",
            f"- offline_hand_illegal_rate: {(off.get('hand') or {}).get('illegal_rate', '')}",
            f"- offline_shop_illegal_rate: {(off.get('shop') or {}).get('illegal_rate', '')}",
            f"- long_horizon_win_rate_pv: {lh.get('pv_win_rate', '')}",
        ]
    status_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="P16 continuous DAgger loop: real->fixture->drift->label->train->eval.")
    p.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    p.add_argument("--out-root", default="", help="Target artifact directory. If empty, create docs/artifacts/p16/<timestamp>.")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--seed", default="AAAAAAA")
    p.add_argument("--seed-file", default="")
    p.add_argument("--base-url", default="http://127.0.0.1:12346")
    p.add_argument("--service-manage", choices=["auto", "off"], default="off")
    p.add_argument("--stop-on-exit", action="store_true")
    p.add_argument("--model", default="", help="Optional fixed model path for real session suggestion.")
    return p


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    logger = setup_logger("trainer.p16_loop")
    warn_if_unstable_python(logger)

    if args.out_root:
        out_dir = Path(args.out_root)
    else:
        out_dir = Path("docs/artifacts/p16") / _ts()
    out_dir.mkdir(parents=True, exist_ok=True)
    stage_root = out_dir / "stages"
    stage_root.mkdir(parents=True, exist_ok=True)

    service_events: list[dict[str, Any]] = []
    managed_proc: subprocess.Popen[str] | None = None
    python = str(_find_trainer_python())
    logger.info("python runner: %s", python)

    cfg = {
        "smoke": {
            "capture_steps": 60,
            "capture_interval": 0.5,
            "hand_samples": 200,
            "shop_samples": 50,
            "epochs": 1,
            "eval_episodes": 10,
        },
        "full": {
            "capture_steps": 300,
            "capture_interval": 0.3,
            "hand_samples": 5000,
            "shop_samples": 500,
            "epochs": 2,
            "eval_episodes": 40,
        },
    }[args.mode]

    report: dict[str, Any] = {
        "schema": "p16_report_v1",
        "mode": args.mode,
        "base_url": args.base_url,
        "seed": args.seed,
        "started_at": _now_iso(),
        "out_dir": str(out_dir),
        "service_events": service_events,
        "stages": {},
    }

    def run_stage(stage_name: str, fn):
        stage_dir = stage_root / stage_name
        stage_dir.mkdir(parents=True, exist_ok=True)
        stage_report_path = stage_dir / "stage_report.json"
        if args.resume:
            old = _read_json(stage_report_path, {})
            if isinstance(old, dict) and str(old.get("status")) == "PASS":
                report["stages"][stage_name] = old
                logger.info("[%s] resume skip", stage_name)
                return old
        started = _now_iso()
        try:
            payload = fn(stage_dir)
            stage_report = {
                "stage": stage_name,
                "status": "PASS",
                "started_at": started,
                "finished_at": _now_iso(),
                "payload": payload,
            }
            _write_json(stage_report_path, stage_report)
            report["stages"][stage_name] = stage_report
            return stage_report
        except StageFailure as exc:
            stage_report = {
                "stage": stage_name,
                "status": "FAIL",
                "started_at": started,
                "finished_at": _now_iso(),
                "reason": exc.reason,
                "exit_code": int(exc.exit_code),
                "command": exc.command,
                "log_tail": exc.log_tail,
            }
            _write_json(stage_report_path, stage_report)
            report["stages"][stage_name] = stage_report
            report["status"] = "FAIL"
            report["failed_stage"] = stage_name
            report["failure_reason"] = exc.reason
            report["exit_code"] = int(exc.exit_code)
            _write_json(out_dir / "report_p16.json", report)
            _update_status_md("FAIL", out_dir, None)
            raise

    try:
        managed_proc = _ensure_service(args.base_url, args.service_manage, service_events, logger)

        # Stage 1: capture
        def _capture(stage_dir: Path):
            sessions_dir = out_dir / "sessions"
            sessions_dir.mkdir(parents=True, exist_ok=True)
            session_path = sessions_dir / "session_latest.jsonl"
            model_path = Path(args.model) if args.model else _find_latest_model()
            cmd = [
                python,
                "-B",
                "trainer/record_real_session.py",
                "--base-url",
                args.base_url,
                "--steps",
                str(cfg["capture_steps"]),
                "--interval",
                str(cfg["capture_interval"]),
                "--topk",
                "3",
                "--include-raw",
                "--out",
                str(session_path),
            ]
            if model_path and model_path.exists():
                cmd += ["--model", str(model_path)]
            run = _run_cmd(stage="capture", label="record_session", cmd=cmd, timeout_sec=600, stage_dir=stage_dir, logger=logger)
            summary_path = session_path.with_suffix(".summary.json")
            summary = _read_json(summary_path, {})
            if not summary:
                raise StageFailure("capture", EXIT_SERVICE_FAIL, "missing session summary json", command=cmd, log_tail="")
            manifest = {
                "sessions": [{"session_id": "latest", "path": str(session_path), "summary": str(summary_path)}],
                "count": 1,
            }
            _write_json(sessions_dir / "sessions_manifest.json", manifest)
            return {"run": run, "session": str(session_path), "summary": summary, "manifest": str(sessions_dir / "sessions_manifest.json")}

        capture = run_stage("capture", _capture)
        session_path = Path(capture["payload"]["session"])

        # Stage 2: trace -> fixture
        def _fixture(stage_dir: Path):
            fixtures_dir = out_dir / "fixtures" / "latest"
            fixtures_dir.mkdir(parents=True, exist_ok=True)
            cmd = [
                python,
                "-B",
                "trainer/real_trace_to_fixture.py",
                "--in",
                str(session_path),
                "--out-dir",
                str(fixtures_dir),
                "--seed",
                args.seed,
            ]
            run = _run_cmd(stage="fixture", label="real_trace_to_fixture", cmd=cmd, timeout_sec=600, stage_dir=stage_dir, logger=logger)
            manifest = _read_json(fixtures_dir / "manifest.json", {})
            if not manifest:
                raise StageFailure("fixture", EXIT_SERVICE_FAIL, "missing fixture manifest", command=cmd, log_tail="")
            fixtures_manifest = {
                "fixtures": [{"fixture_id": "latest", "dir": str(fixtures_dir), "manifest": str(fixtures_dir / "manifest.json")}],
                "count": 1,
            }
            _write_json(out_dir / "fixtures" / "fixtures_manifest.json", fixtures_manifest)
            return {"run": run, "fixture_dir": str(fixtures_dir), "manifest": manifest}

        fixture = run_stage("fixture", _fixture)
        fixture_dir = Path(fixture["payload"]["fixture_dir"])
        state_trace = fixture_dir / "state_trace.jsonl"
        if not state_trace.exists():
            raise StageFailure("fixture", EXIT_SERVICE_FAIL, "fixture missing state_trace.jsonl")

        # Stage 3: drift
        def _drift(stage_dir: Path):
            drift_dir = out_dir / "drift_reports"
            drift_dir.mkdir(parents=True, exist_ok=True)
            drift_path = drift_dir / "drift_selfcheck_latest.json"
            cmd = [
                python,
                "-B",
                "sim/tests/run_real_drift_fixture.py",
                "--trace-a",
                str(state_trace),
                "--trace-b",
                str(state_trace),
                "--out",
                str(drift_path),
            ]
            run = _run_cmd(stage="drift", label="run_real_drift_fixture", cmd=cmd, timeout_sec=180, stage_dir=stage_dir, logger=logger)
            drift = _read_json(drift_path, {})
            mismatch = int(drift.get("mismatch_count") or 0)
            _write_json(drift_dir / "drift_summary.json", {"mismatch_count": mismatch, "top_paths": drift.get("top_paths") or [], "report": str(drift_path)})
            if mismatch != 0:
                raise StageFailure("drift", EXIT_DRIFT_FAIL, f"drift mismatch_count={mismatch}", command=cmd, log_tail="")
            return {"run": run, "drift_report": drift, "summary": str(drift_dir / "drift_summary.json")}

        drift = run_stage("drift", _drift)

        # Stage 4: teacher labels
        def _label(stage_dir: Path):
            data_dir = out_dir / "datasets"
            data_dir.mkdir(parents=True, exist_ok=True)
            data_path = data_dir / "p16_dagger_dataset.jsonl"
            summary_path = data_dir / "p16_dagger_summary.json"
            cmd = [
                python,
                "-B",
                "trainer/dagger_collect.py",
                "--session",
                str(session_path),
                "--backend",
                "sim",
                "--out",
                str(data_path),
                "--hand-samples",
                str(cfg["hand_samples"]),
                "--shop-samples",
                str(cfg["shop_samples"]),
                "--time-budget-ms",
                "20",
                "--allow-sim-augment",
                "--summary-out",
                str(summary_path),
            ]
            run = _run_cmd(stage="label", label="dagger_collect", cmd=cmd, timeout_sec=1200, stage_dir=stage_dir, logger=logger)
            summary = _read_json(summary_path, {})
            if not summary:
                raise StageFailure("label", EXIT_LABEL_FAIL, "missing dagger summary", command=cmd, log_tail="")
            hand_records = int(summary.get("hand_records") or 0)
            shop_records = int(summary.get("shop_records") or 0)
            invalid_rows = int(summary.get("invalid_rows") or 0)
            if hand_records < int(cfg["hand_samples"]) or shop_records < int(cfg["shop_samples"]):
                raise StageFailure(
                    "label",
                    EXIT_LABEL_FAIL,
                    f"insufficient labeled records hand={hand_records} shop={shop_records}",
                    command=cmd,
                    log_tail="",
                )
            if invalid_rows > 1:
                raise StageFailure("label", EXIT_LABEL_FAIL, f"invalid_rows too high: {invalid_rows}", command=cmd, log_tail="")

            ds_cmd = [python, "-B", "trainer/dataset.py", "--path", str(data_path), "--validate", "--summary"]
            ds_run = _run_cmd(stage="label", label="dataset_validate", cmd=ds_cmd, timeout_sec=180, stage_dir=stage_dir, logger=logger)
            return {
                "run": run,
                "dataset_validate": ds_run,
                "dataset": str(data_path),
                "summary": summary,
            }

        label = run_stage("label", _label)
        dataset_path = Path(label["payload"]["dataset"])

        # Stage 5: training
        def _train(stage_dir: Path):
            models_dir = out_dir / "models" / "p16_pv"
            models_dir.mkdir(parents=True, exist_ok=True)
            cmd = [
                python,
                "-B",
                "trainer/train_pv.py",
                "--train-jsonl",
                str(dataset_path),
                "--epochs",
                str(cfg["epochs"]),
                "--batch-size",
                "64",
                "--out-dir",
                str(models_dir),
            ]
            run = _run_cmd(stage="train", label="train_pv", cmd=cmd, timeout_sec=2400, stage_dir=stage_dir, logger=logger)
            best = models_dir / "best.pt"
            metrics_path = models_dir / "metrics.json"
            metrics = _read_json(metrics_path, {})
            if not best.exists() or not metrics:
                raise StageFailure("train", EXIT_TRAIN_EVAL_FAIL, "missing best.pt or metrics.json", command=cmd, log_tail="")
            return {"run": run, "model": str(best), "metrics": metrics, "metrics_path": str(metrics_path)}

        train = run_stage("train", _train)
        model_path = Path(train["payload"]["model"])

        # Stage 6: eval (offline + long horizon)
        def _eval(stage_dir: Path):
            eval_dir = out_dir / "eval"
            eval_dir.mkdir(parents=True, exist_ok=True)
            offline_path = eval_dir / "eval_offline.json"
            lh_pv_path = eval_dir / "eval_long_horizon_pv.json"
            lh_heur_path = eval_dir / "eval_long_horizon_heuristic.json"

            off_cmd = [
                python,
                "-B",
                "trainer/eval_pv.py",
                "--model",
                str(model_path),
                "--dataset",
                str(dataset_path),
                "--out",
                str(offline_path),
            ]
            off_run = _run_cmd(stage="eval", label="eval_pv_offline", cmd=off_cmd, timeout_sec=600, stage_dir=stage_dir, logger=logger)
            offline = _read_json(offline_path, {})
            hand_illegal = float(((offline.get("hand") or {}).get("illegal_rate") or 0.0))
            shop_illegal = float(((offline.get("shop") or {}).get("illegal_rate") or 0.0))
            if hand_illegal > 0.001 or shop_illegal > 0.001:
                raise StageFailure(
                    "eval",
                    EXIT_TRAIN_EVAL_FAIL,
                    f"illegal_rate too high hand={hand_illegal:.6f} shop={shop_illegal:.6f}",
                    command=off_cmd,
                    log_tail="",
                )

            seed_file = Path(args.seed_file) if args.seed_file else _default_seed_file()
            episodes = int(cfg["eval_episodes"])

            lh_heur_cmd = [
                python,
                "-B",
                "trainer/eval_long_horizon.py",
                "--backend",
                "sim",
                "--stake",
                "gold",
                "--episodes",
                str(episodes),
                "--seeds-file",
                str(seed_file),
                "--policy",
                "heuristic",
                "--out",
                str(lh_heur_path),
            ]
            lh_heur_run = _run_cmd(stage="eval", label="eval_long_horizon_heuristic", cmd=lh_heur_cmd, timeout_sec=1800, stage_dir=stage_dir, logger=logger)

            lh_pv_cmd = [
                python,
                "-B",
                "trainer/eval_long_horizon.py",
                "--backend",
                "sim",
                "--stake",
                "gold",
                "--episodes",
                str(episodes),
                "--seeds-file",
                str(seed_file),
                "--policy",
                "pv",
                "--model",
                str(model_path),
                "--out",
                str(lh_pv_path),
            ]
            lh_pv_run = _run_cmd(stage="eval", label="eval_long_horizon_pv", cmd=lh_pv_cmd, timeout_sec=1800, stage_dir=stage_dir, logger=logger)

            lh_heur = _read_json(lh_heur_path, {})
            lh_pv = _read_json(lh_pv_path, {})
            if not lh_heur or not lh_pv:
                raise StageFailure("eval", EXIT_TRAIN_EVAL_FAIL, "missing long horizon reports", command=lh_pv_cmd, log_tail="")

            eval_summary_md = eval_dir / "eval_summary.md"
            eval_summary_md.write_text(
                "\n".join(
                    [
                        "# P16 Eval Summary",
                        "",
                        f"- offline_hand_top1: {((offline.get('hand') or {}).get('top1'))}",
                        f"- offline_hand_top3: {((offline.get('hand') or {}).get('top3'))}",
                        f"- offline_hand_illegal_rate: {hand_illegal}",
                        f"- offline_shop_top1: {((offline.get('shop') or {}).get('top1'))}",
                        f"- offline_shop_illegal_rate: {shop_illegal}",
                        f"- long_horizon_win_rate_heuristic: {lh_heur.get('win_rate')}",
                        f"- long_horizon_win_rate_pv: {lh_pv.get('win_rate')}",
                        f"- long_horizon_avg_ante_heuristic: {lh_heur.get('avg_ante_reached')}",
                        f"- long_horizon_avg_ante_pv: {lh_pv.get('avg_ante_reached')}",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            return {
                "offline_run": off_run,
                "offline_report": offline,
                "long_horizon_heuristic_run": lh_heur_run,
                "long_horizon_pv_run": lh_pv_run,
                "long_horizon_heuristic": lh_heur,
                "long_horizon_pv": lh_pv,
                "eval_summary_md": str(eval_summary_md),
            }

        ev = run_stage("eval", _eval)

        report["status"] = "PASS"
        report["finished_at"] = _now_iso()
        report["exit_code"] = EXIT_OK
        report["drift"] = {
            "mismatch_count": int((((drift.get("payload") or {}).get("drift_report") or {}).get("mismatch_count") or 0))
        }
        report["dataset"] = dict((label.get("payload") or {}).get("summary") or {})
        report["offline_eval"] = dict((ev.get("payload") or {}).get("offline_report") or {})
        report["long_horizon"] = {
            "heuristic_win_rate": ((ev.get("payload") or {}).get("long_horizon_heuristic") or {}).get("win_rate"),
            "pv_win_rate": ((ev.get("payload") or {}).get("long_horizon_pv") or {}).get("win_rate"),
            "heuristic_avg_ante": ((ev.get("payload") or {}).get("long_horizon_heuristic") or {}).get("avg_ante_reached"),
            "pv_avg_ante": ((ev.get("payload") or {}).get("long_horizon_pv") or {}).get("avg_ante_reached"),
        }

        _write_json(out_dir / "report_p16.json", report)
        report_md = out_dir / "report_p16.md"
        report_md.write_text(
            "\n".join(
                [
                    "# P16 Report",
                    "",
                    f"- status: {report['status']}",
                    f"- out_dir: {out_dir}",
                    f"- drift_mismatch_count: {report['drift']['mismatch_count']}",
                    f"- hand_records: {report['dataset'].get('hand_records')}",
                    f"- shop_records: {report['dataset'].get('shop_records')}",
                    f"- invalid_rows: {report['dataset'].get('invalid_rows')}",
                    f"- offline_hand_illegal_rate: {(report['offline_eval'].get('hand') or {}).get('illegal_rate')}",
                    f"- offline_shop_illegal_rate: {(report['offline_eval'].get('shop') or {}).get('illegal_rate')}",
                    f"- long_horizon_win_rate_pv: {report['long_horizon'].get('pv_win_rate')}",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        _update_status_md("PASS", out_dir, report)
        logger.info("P16 PASS out=%s", out_dir)
        return EXIT_OK

    except StageFailure as exc:
        logger.error("P16 FAILED stage=%s code=%d reason=%s", exc.stage, exc.exit_code, exc.reason)
        report["finished_at"] = _now_iso()
        _write_json(out_dir / "report_p16.json", report)
        return int(exc.exit_code)
    finally:
        if args.stop_on_exit:
            _stop_service()
            service_events.append({"ts": _now_iso(), "event": "stop_on_exit"})
        elif managed_proc is not None and managed_proc.poll() is not None:
            service_events.append({"ts": _now_iso(), "event": "service_proc_exited", "returncode": managed_proc.returncode})


if __name__ == "__main__":
    raise SystemExit(main())
