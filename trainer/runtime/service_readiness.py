from __future__ import annotations

if __package__ is None or __package__ == "":
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from trainer.env_client import _call_method


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception:
        return None


def _latest_window_state() -> dict[str, Any]:
    path = _repo_root() / "docs" / "artifacts" / "p53" / "window_supervisor" / "latest" / "window_state.json"
    payload = _read_json(path)
    return {
        "path": str(path.resolve()),
        "payload": payload if isinstance(payload, dict) else {},
    }


def _latest_background_validation() -> dict[str, Any]:
    path = _repo_root() / "docs" / "artifacts" / "p53" / "background_mode_validation" / "latest" / "background_mode_validation.json"
    payload = _read_json(path)
    return {
        "path": str(path.resolve()),
        "payload": payload if isinstance(payload, dict) else {},
    }


def _probe_once(base_url: str, *, timeout_sec: float, probe_method: str) -> dict[str, Any]:
    started = time.time()
    details: dict[str, Any] = {
        "health_ok": False,
        "gamestate_ok": False,
        "state_keys": [],
        "probe_method": probe_method,
    }
    try:
        _call_method(base_url, "health", timeout=timeout_sec)
        details["health_ok"] = True
    except Exception as exc:
        details["error"] = f"health:{exc}"
        details["elapsed_sec"] = time.time() - started
        return details

    if probe_method in {"health", "health_only"}:
        details["gamestate_ok"] = True
        details["elapsed_sec"] = time.time() - started
        return details

    try:
        state = _call_method(base_url, "gamestate", timeout=timeout_sec)
        if isinstance(state, dict):
            details["gamestate_ok"] = True
            details["state_keys"] = sorted(list(state.keys()))[:12]
            details["state_phase"] = str(state.get("state") or "")
            details["round"] = state.get("round") if isinstance(state.get("round"), dict) else {}
        else:
            details["error"] = "gamestate:not_dict"
    except Exception as exc:
        details["error"] = f"gamestate:{exc}"
    details["elapsed_sec"] = time.time() - started
    return details


def wait_for_service_ready(
    *,
    base_url: str,
    out_dir: str | Path | None = None,
    run_id: str = "",
    max_retries: int = 20,
    retry_interval_sec: float = 2.0,
    warmup_grace_sec: float = 8.0,
    consecutive_successes: int = 3,
    timeout_sec: float = 3.0,
    probe_method: str = "health_gamestate",
) -> dict[str, Any]:
    run_token = str(run_id or _now_stamp())
    output_root = (
        (_repo_root() / "docs" / "artifacts" / "p49" / "readiness").resolve()
        if out_dir is None
        else (Path(out_dir) if Path(out_dir).is_absolute() else (_repo_root() / out_dir).resolve())
    )
    run_dir = output_root / run_token
    run_dir.mkdir(parents=True, exist_ok=True)

    attempts: list[dict[str, Any]] = []
    first_success_ts: float | None = None
    consecutive = 0
    ready = False
    reason = "max_retries_exhausted"

    for attempt_idx in range(1, max(1, int(max_retries)) + 1):
        probe = _probe_once(
            base_url,
            timeout_sec=max(0.5, float(timeout_sec)),
            probe_method=str(probe_method or "health_gamestate"),
        )
        probe["attempt"] = attempt_idx
        probe["ts"] = _now_iso()
        attempts.append(probe)

        success = bool(probe.get("health_ok")) and bool(probe.get("gamestate_ok"))
        if success:
            if first_success_ts is None:
                first_success_ts = time.time()
            grace_elapsed = time.time() - first_success_ts
            if grace_elapsed >= max(0.0, float(warmup_grace_sec)):
                consecutive += 1
            else:
                consecutive = 0
                probe["warmup_wait_remaining_sec"] = max(0.0, float(warmup_grace_sec) - grace_elapsed)
            if consecutive >= max(1, int(consecutive_successes)):
                ready = True
                reason = "ready"
                break
        else:
            consecutive = 0
            first_success_ts = None

        if attempt_idx < max(1, int(max_retries)):
            time.sleep(max(0.1, float(retry_interval_sec)))

    payload = {
        "schema": "p49_service_readiness_report_v1",
        "generated_at": _now_iso(),
        "run_id": run_token,
        "base_url": str(base_url),
        "status": "ready" if ready else "failed",
        "reason": reason,
        "probe_method": str(probe_method or "health_gamestate"),
        "max_retries": int(max_retries),
        "retry_interval_sec": float(retry_interval_sec),
        "warmup_grace_sec": float(warmup_grace_sec),
        "consecutive_successes_required": int(consecutive_successes),
        "attempt_count": len(attempts),
        "attempts": attempts,
        "window_state": _latest_window_state(),
        "background_validation": _latest_background_validation(),
        "report_json": str((run_dir / "service_readiness_report.json").resolve()),
    }
    _write_json(run_dir / "service_readiness_report.json", payload)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Wait for the oracle service to become truly ready.")
    parser.add_argument("--base-url", default="http://127.0.0.1:12346")
    parser.add_argument("--out-dir", default="docs/artifacts/p49/readiness")
    parser.add_argument("--run-id", default="")
    parser.add_argument("--max-retries", type=int, default=20)
    parser.add_argument("--retry-interval-sec", type=float, default=2.0)
    parser.add_argument("--warmup-grace-sec", type=float, default=8.0)
    parser.add_argument("--consecutive-successes", type=int, default=3)
    parser.add_argument("--timeout-sec", type=float, default=3.0)
    parser.add_argument("--probe-method", default="health_gamestate")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = wait_for_service_ready(
        base_url=str(args.base_url),
        out_dir=(str(args.out_dir).strip() or None),
        run_id=str(args.run_id or ""),
        max_retries=max(1, int(args.max_retries)),
        retry_interval_sec=max(0.1, float(args.retry_interval_sec)),
        warmup_grace_sec=max(0.0, float(args.warmup_grace_sec)),
        consecutive_successes=max(1, int(args.consecutive_successes)),
        timeout_sec=max(0.5, float(args.timeout_sec)),
        probe_method=str(args.probe_method or "health_gamestate"),
    )
    print(json.dumps(summary, ensure_ascii=False))
    return 0 if str(summary.get("status")) == "ready" else 1


if __name__ == "__main__":
    raise SystemExit(main())
