from __future__ import annotations

import json
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any

from demo.training_status import infer_profile, latest_status_path, read_status, write_status


def _read_progress(path: Path, limit: int = 80) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if not text:
            continue
        try:
            rows.append(json.loads(text))
        except Exception:
            continue
    return rows[-limit:]


class TrainingManager:
    """以后台子进程方式运行 MVP-S2 训练流水线，并暴露最新状态。"""

    def __init__(self) -> None:
        self.project_root = Path(__file__).resolve().parent.parent
        self.status_path = latest_status_path()
        self._lock = threading.RLock()
        self._process: subprocess.Popen[str] | None = None

    def _profile(self, profile: str) -> dict[str, Any]:
        normalized = str(profile).lower()
        if normalized == "smoke":
            return {
                "profile": "smoke",
                "budget_minutes": 8,
                "episodes": 180,
                "max_steps": 28,
                "scenario_copies": 48,
                "final_epochs": 4,
                "sweep_epochs": 2,
                "batch_size": 256,
                "device": "auto",
            }
        if normalized == "fast":
            return {
                "profile": "fast",
                "budget_minutes": 25,
                "episodes": 480,
                "max_steps": 36,
                "scenario_copies": 128,
                "final_epochs": 10,
                "sweep_epochs": 5,
                "batch_size": 512,
                "device": "auto",
            }
        return {
            "profile": "standard",
            "budget_minutes": 120,
            "episodes": 1800,
            "max_steps": 44,
            "scenario_copies": 256,
            "final_epochs": 18,
            "sweep_epochs": 8,
            "batch_size": 768,
            "device": "auto",
        }

    def _augment_with_progress(self, payload: dict[str, Any]) -> dict[str, Any]:
        status = dict(payload)
        status["profile"] = infer_profile(status)
        progress_candidates = []
        training_run_dir = str((status.get("training") or {}).get("run_dir") or "")
        dataset_run_dir = str((status.get("dataset") or {}).get("run_dir") or "")
        final_run_dir = str(status.get("final_run_dir") or "")
        for raw in [training_run_dir, dataset_run_dir, final_run_dir]:
            if raw:
                progress_candidates.append(Path(raw) / "progress.jsonl")
        for candidate in progress_candidates:
            progress = _read_progress(candidate)
            if progress:
                status["progress"] = progress
                status.setdefault("artifacts", {})
                status["artifacts"]["progress_path"] = str(candidate)
                break
        return status

    def status(self) -> dict[str, Any]:
        with self._lock:
            status = read_status(self.status_path)
            if self._process is not None and self._process.poll() is not None:
                if status.get("status") in {"queued", "building_dataset", "training", "evaluating"}:
                    final_state = "finished" if self._process.returncode == 0 else "failed"
                    final_label = "训练完成" if self._process.returncode == 0 else "训练失败"
                    final_message = status.get("message") or ("训练任务已完成。" if self._process.returncode == 0 else f"训练任务异常退出，exit_code={self._process.returncode}")
                    status = write_status(
                        {
                            **status,
                            "status": final_state,
                            "status_label": final_label,
                            "message": final_message,
                        },
                        self.status_path,
                    )
                self._process = None
            return self._augment_with_progress(status)

    def running(self) -> bool:
        with self._lock:
            return self._process is not None and self._process.poll() is None

    def start(self, profile: str = "standard") -> dict[str, Any]:
        with self._lock:
            current = self.status()
            if self.running():
                return current

            cfg = self._profile(profile)
            write_status(
                {
                    "status": "queued",
                    "status_label": "排队中",
                    "message": "训练任务已创建，正在准备启动。",
                    "profile": cfg["profile"],
                    "budget_minutes": cfg["budget_minutes"],
                    "device": cfg["device"],
                    "dataset": {
                        "episodes_target": cfg["episodes"],
                        "max_steps": cfg["max_steps"],
                        "scenario_copies": cfg["scenario_copies"],
                    },
                    "training": {
                        "epochs": cfg["final_epochs"],
                        "batch_size": cfg["batch_size"],
                    },
                },
                self.status_path,
            )
            cmd = [
                sys.executable,
                "-B",
                "-m",
                "demo.train_mvp_pipeline",
                "--status-path",
                str(self.status_path),
                "--profile",
                str(cfg["profile"]),
                "--budget-minutes",
                str(cfg["budget_minutes"]),
                "--episodes",
                str(cfg["episodes"]),
                "--max-steps",
                str(cfg["max_steps"]),
                "--scenario-copies",
                str(cfg["scenario_copies"]),
                "--device",
                str(cfg["device"]),
                "--batch-size",
                str(cfg["batch_size"]),
                "--final-epochs",
                str(cfg["final_epochs"]),
                "--sweep-epochs",
                str(cfg["sweep_epochs"]),
            ]
            creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
            self._process = subprocess.Popen(cmd, cwd=str(self.project_root), creationflags=creationflags)
            return self.status()
