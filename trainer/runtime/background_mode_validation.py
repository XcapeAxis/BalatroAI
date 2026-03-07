from __future__ import annotations

if __package__ is None or __package__ == "":
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from trainer.env_client import _call_method
from trainer.runtime.window_supervisor import (
    WINDOW_MODES,
    WindowSelector,
    default_window_settings,
    latest_window_state_path,
    set_window_mode,
)

VALIDATION_MODES = ("visible", "offscreen", "minimized", "hidden")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_markdown(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception:
        return None


def _artifacts_root() -> Path:
    return _repo_root() / "docs" / "artifacts" / "p53" / "background_mode_validation"


def _latest_root() -> Path:
    return _artifacts_root() / "latest"


def latest_validation_path() -> Path:
    return _latest_root() / "background_mode_validation.json"


def load_latest_validation() -> dict[str, Any]:
    payload = _read_json(latest_validation_path())
    return payload if isinstance(payload, dict) else {}


def _mode_result_index(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    rows = payload.get("mode_results") if isinstance(payload.get("mode_results"), list) else []
    out: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        key = str(row.get("requested_mode") or "")
        if key:
            out[key] = row
    return out


def resolve_effective_window_mode(
    requested_mode: str = "",
    *,
    fallback_mode: str = "",
    validation_payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    defaults = default_window_settings()
    requested = str(requested_mode or defaults.get("window_mode") or "visible").strip().lower() or "visible"
    fallback = str(fallback_mode or defaults.get("window_mode_fallback") or "offscreen").strip().lower() or "offscreen"
    validation = validation_payload if isinstance(validation_payload, dict) else load_latest_validation()
    recommended = str(validation.get("recommended_default_mode") or defaults.get("window_mode") or "visible").strip().lower() or "visible"
    validation_path = str(validation.get("report_json") or latest_validation_path().resolve())
    reason = "requested_mode_accepted"
    effective = requested if requested in WINDOW_MODES else recommended

    indexed = _mode_result_index(validation)
    row = indexed.get(requested) or {}
    if requested not in WINDOW_MODES:
        effective = recommended
        reason = "invalid_requested_mode"
    elif requested == "restore":
        effective = "restore"
    elif requested in {"visible", "offscreen"}:
        if row and str(row.get("status") or "") not in {"pass", "unsupported"}:
            effective = recommended if recommended in {"visible", "offscreen"} else "visible"
            reason = "requested_mode_failed_validation"
    elif row:
        status = str(row.get("status") or "")
        if status == "pass":
            effective = requested
        elif str(row.get("effective_mode") or "").strip().lower() in WINDOW_MODES:
            effective = str(row.get("effective_mode") or "").strip().lower()
            reason = f"{requested}_degraded_to_{effective}"
        else:
            effective = fallback if fallback in WINDOW_MODES else recommended
            reason = f"{requested}_downgraded_after_validation"
    else:
        effective = fallback if fallback in WINDOW_MODES else recommended
        reason = f"{requested}_downgraded_without_validation"

    if effective not in WINDOW_MODES:
        effective = "visible"
    return {
        "requested_mode": requested,
        "effective_mode": effective,
        "fallback_mode": fallback,
        "recommended_default_mode": recommended,
        "resolution_reason": reason,
        "validation_path": validation_path,
    }


def _selector() -> WindowSelector:
    return WindowSelector(process_names=("Balatro",))


def _sleep_for_window_transition() -> None:
    time.sleep(1.25)


def _probe_service(base_url: str) -> dict[str, Any]:
    out = {
        "health_ok": False,
        "gamestate_ok": False,
        "state_phase": "",
        "error": "",
    }
    try:
        _call_method(base_url, "health", timeout=3.0)
        out["health_ok"] = True
    except Exception as exc:
        out["error"] = f"health:{exc}"
        return out
    try:
        state = _call_method(base_url, "gamestate", timeout=3.0)
        if isinstance(state, dict):
            out["gamestate_ok"] = True
            out["state_phase"] = str(state.get("state") or "")
        else:
            out["error"] = "gamestate:not_dict"
    except Exception as exc:
        out["error"] = f"gamestate:{exc}"
    return out


def _run_p1_smoke(
    *,
    base_url: str,
    out_dir: Path,
    seed: str,
    scope: str,
    max_steps: int,
    timeout_sec: int,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "-B",
        str((_repo_root() / "sim" / "oracle" / "batch_build_p1_smoke.py").resolve()),
        "--base-url",
        str(base_url),
        "--out-dir",
        str(out_dir),
        "--max-steps",
        str(max_steps),
        "--scope",
        str(scope),
        "--seed",
        str(seed),
        "--dump-on-diff",
        str((out_dir / "dumps").resolve()),
    ]
    started = time.time()
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(_repo_root()),
            text=True,
            capture_output=True,
            timeout=max(60, int(timeout_sec)),
            check=False,
        )
        timed_out = False
        returncode = int(proc.returncode)
        stdout = str(proc.stdout or "")
        stderr = str(proc.stderr or "")
    except subprocess.TimeoutExpired as exc:
        timed_out = True
        returncode = 124
        stdout = str(exc.stdout or "")
        stderr = str(exc.stderr or "")
    report_path = out_dir / "report_p1.json"
    report = _read_json(report_path)
    if not isinstance(report, dict):
        report = {}
    return {
        "command": cmd,
        "returncode": returncode,
        "timed_out": timed_out,
        "elapsed_sec": time.time() - started,
        "stdout_tail": stdout[-4000:],
        "stderr_tail": stderr[-4000:],
        "report_path": str(report_path.resolve()),
        "report": report,
    }


def _report_signature(report: dict[str, Any]) -> dict[str, Any]:
    rows = report.get("results") if isinstance(report.get("results"), list) else []
    targets = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        targets.append(
            {
                "target": str(row.get("target") or ""),
                "status": str(row.get("status") or ""),
                "steps_used": int(row.get("steps_used") or 0),
                "first_diff_step": row.get("first_diff_step"),
                "first_diff_path": str(row.get("first_diff_path") or ""),
                "final_phase": str(row.get("final_phase") or ""),
            }
        )
    targets.sort(key=lambda item: item["target"])
    return {
        "total": int(report.get("total") or 0),
        "passed": int(report.get("passed") or 0),
        "diff_fail": int(report.get("diff_fail") or 0),
        "oracle_fail": int(report.get("oracle_fail") or 0),
        "gen_fail": int(report.get("gen_fail") or 0),
        "targets": targets,
    }


def _dominant_mode_from_state(payload: dict[str, Any]) -> str:
    rows = payload.get("window_mode_after") if isinstance(payload.get("window_mode_after"), list) else payload.get("windows")
    if not isinstance(rows, list):
        return ""
    for role in ("game_main", "other_balatro", "diagnostic_console"):
        for row in rows:
            if not isinstance(row, dict):
                continue
            if str(row.get("role") or "") == role:
                return str(row.get("mode") or "")
    for row in rows:
        if isinstance(row, dict) and str(row.get("mode") or "").strip():
            return str(row.get("mode") or "")
    return ""


def _compare_with_baseline(
    *,
    baseline_signature: dict[str, Any],
    current_signature: dict[str, Any],
) -> dict[str, Any]:
    return {
        "matches_baseline": current_signature == baseline_signature,
        "baseline_signature": baseline_signature,
        "current_signature": current_signature,
    }


def _mode_summary(
    requested_mode: str,
    effective_mode: str,
    *,
    switch_payload: dict[str, Any],
    probe_before: dict[str, Any],
    probe_after: dict[str, Any],
    smoke: dict[str, Any],
    baseline_signature: dict[str, Any] | None,
) -> dict[str, Any]:
    report = smoke.get("report") if isinstance(smoke.get("report"), dict) else {}
    signature = _report_signature(report)
    comparison = (
        _compare_with_baseline(baseline_signature=baseline_signature, current_signature=signature)
        if isinstance(baseline_signature, dict)
        else {"matches_baseline": True, "baseline_signature": signature, "current_signature": signature}
    )
    completed = int(smoke.get("returncode") or 0) == 0 and bool(report)
    passed = bool(report) and int(report.get("passed") or 0) == int(report.get("total") or 0) and int(report.get("total") or 0) > 0
    stable = completed and passed and bool(comparison.get("matches_baseline"))
    status = "pass" if stable else ("fail" if completed else "unsupported")
    if requested_mode != effective_mode and completed and passed:
        status = "degraded"
    return {
        "requested_mode": requested_mode,
        "effective_mode": effective_mode,
        "status": status,
        "operation_success": bool(switch_payload.get("operation_success")),
        "probe_before": probe_before,
        "probe_after": probe_after,
        "smoke_completed": completed,
        "smoke_passed": passed,
        "matches_visible_baseline": bool(comparison.get("matches_baseline")),
        "returncode": int(smoke.get("returncode") or 0),
        "timed_out": bool(smoke.get("timed_out")),
        "elapsed_sec": float(smoke.get("elapsed_sec") or 0.0),
        "report_path": str(smoke.get("report_path") or ""),
        "signature": signature,
        "switch_payload": {
            "artifact_dir": str(switch_payload.get("artifact_dir") or ""),
            "state_path": str(switch_payload.get("state_path") or ""),
            "log_path": str(switch_payload.get("log_path") or ""),
            "error_reason": str(switch_payload.get("error_reason") or ""),
        },
        "stdout_tail": str(smoke.get("stdout_tail") or ""),
        "stderr_tail": str(smoke.get("stderr_tail") or ""),
    }


def _build_recommendation(mode_results: list[dict[str, Any]]) -> dict[str, Any]:
    result_index = {str(row.get("requested_mode") or ""): row for row in mode_results if isinstance(row, dict)}
    preferred_order = ("offscreen", "minimized", "hidden")
    recommended = "visible"
    prohibited: list[str] = []
    debug_only: list[str] = ["visible"]
    for mode in preferred_order:
        row = result_index.get(mode) or {}
        if str(row.get("status") or "") == "pass" and bool(row.get("matches_visible_baseline")):
            recommended = mode
            break
    for mode in preferred_order:
        row = result_index.get(mode) or {}
        if str(row.get("status") or "") not in {"pass"}:
            prohibited.append(mode)
    if recommended != "visible":
        debug_only = ["visible"]
    fallback = "offscreen" if recommended != "visible" else "visible"
    return {
        "recommended_default_mode": recommended,
        "prohibited_modes": prohibited,
        "debug_only_modes": debug_only,
        "fallback_mode": fallback,
        "hidden_requests_downgrade_to": (fallback if "hidden" in prohibited else "hidden"),
        "summary": (
            f"default to {recommended}; downgrade unsupported or unstable hidden/minimized requests to {fallback}"
            if recommended != "visible"
            else "keep visible as default until a stable background mode exists"
        ),
    }


def validate_background_modes(
    *,
    base_url: str = "http://127.0.0.1:12346",
    out_dir: str | Path | None = None,
    run_id: str = "",
    modes: list[str] | None = None,
    seed: str = "AAAAAAA",
    scope: str = "p1_hand_score_observed_core",
    max_steps: int = 120,
    timeout_sec: int = 900,
) -> dict[str, Any]:
    chosen_run_id = str(run_id or _now_stamp())
    output_root = (
        _artifacts_root().resolve()
        if out_dir is None
        else (Path(out_dir) if Path(out_dir).is_absolute() else (_repo_root() / out_dir).resolve())
    )
    run_dir = output_root / chosen_run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    selector = _selector()

    requested_modes = []
    for mode in (modes or list(VALIDATION_MODES)):
        token = str(mode).strip().lower()
        if token in VALIDATION_MODES and token not in requested_modes:
            requested_modes.append(token)
    if "visible" not in requested_modes:
        requested_modes.insert(0, "visible")

    results: list[dict[str, Any]] = []
    baseline_signature: dict[str, Any] | None = None

    for mode in requested_modes:
        set_window_mode("visible", selector=selector, out_root=run_dir / "window_ops")
        _sleep_for_window_transition()
        switch_payload = set_window_mode(mode, selector=selector, out_root=run_dir / "window_ops")
        _sleep_for_window_transition()
        effective_mode = _dominant_mode_from_state(switch_payload) or mode
        probe_before = _probe_service(base_url)
        smoke = _run_p1_smoke(
            base_url=base_url,
            out_dir=run_dir / f"mode_{mode}",
            seed=seed,
            scope=scope,
            max_steps=max_steps,
            timeout_sec=timeout_sec,
        )
        probe_after = _probe_service(base_url)
        summary = _mode_summary(
            mode,
            effective_mode,
            switch_payload=switch_payload,
            probe_before=probe_before,
            probe_after=probe_after,
            smoke=smoke,
            baseline_signature=baseline_signature,
        )
        if mode == "visible" and summary.get("smoke_completed"):
            baseline_signature = dict(summary.get("signature") or {})
            summary["matches_visible_baseline"] = True
        elif baseline_signature is not None:
            summary["matches_visible_baseline"] = bool(
                _compare_with_baseline(
                    baseline_signature=baseline_signature,
                    current_signature=dict(summary.get("signature") or {}),
                ).get("matches_baseline")
            )
            if str(summary.get("status") or "") == "degraded":
                pass
            elif bool(summary.get("smoke_completed")) and bool(summary.get("smoke_passed")) and bool(summary.get("matches_visible_baseline")):
                summary["status"] = "pass"
            elif bool(summary.get("smoke_completed")):
                summary["status"] = "fail"
        results.append(summary)
        set_window_mode("restore", selector=selector, out_root=run_dir / "window_ops")
        _sleep_for_window_transition()

    recommendation = _build_recommendation(results)
    payload = {
        "schema": "p53_background_mode_validation_v1",
        "generated_at": _now_iso(),
        "run_id": chosen_run_id,
        "base_url": str(base_url),
        "seed": str(seed),
        "scope": str(scope),
        "max_steps": int(max_steps),
        "mode_results": results,
        "recommended_default_mode": str(recommendation.get("recommended_default_mode") or "visible"),
        "prohibited_modes": list(recommendation.get("prohibited_modes") or []),
        "debug_only_modes": list(recommendation.get("debug_only_modes") or []),
        "window_mode_fallback": str(recommendation.get("fallback_mode") or "visible"),
        "hidden_requests_downgrade_to": str(recommendation.get("hidden_requests_downgrade_to") or "visible"),
        "decision_summary": str(recommendation.get("summary") or ""),
        "latest_window_state_path": str(latest_window_state_path().resolve()),
        "report_json": str((run_dir / "background_mode_validation.json").resolve()),
        "report_md": str((run_dir / "background_mode_validation.md").resolve()),
    }
    _write_json(run_dir / "background_mode_validation.json", payload)
    _write_json(_latest_root() / "background_mode_validation.json", payload)

    lines = [
        "# P53 Background Mode Validation",
        "",
        f"- run_id: `{chosen_run_id}`",
        f"- base_url: `{base_url}`",
        f"- seed: `{seed}`",
        f"- scope: `{scope}`",
        f"- max_steps: `{max_steps}`",
        f"- recommended_default_mode: `{payload['recommended_default_mode']}`",
        f"- window_mode_fallback: `{payload['window_mode_fallback']}`",
        f"- prohibited_modes: `{', '.join(payload['prohibited_modes']) or 'none'}`",
        f"- hidden_requests_downgrade_to: `{payload['hidden_requests_downgrade_to']}`",
        "",
        "## Decision",
        f"- {payload['decision_summary']}",
        "",
        "## Mode Results",
    ]
    for row in results:
        lines.extend(
            [
                "- requested=`{requested}` effective=`{effective}` status=`{status}` smoke_passed={smoke_passed} matches_visible_baseline={matches} returncode={code} report=`{report}`".format(
                    requested=row.get("requested_mode"),
                    effective=row.get("effective_mode"),
                    status=row.get("status"),
                    smoke_passed=str(bool(row.get("smoke_passed"))).lower(),
                    matches=str(bool(row.get("matches_visible_baseline"))).lower(),
                    code=int(row.get("returncode") or 0),
                    report=row.get("report_path") or "",
                ),
            ]
        )
    _write_markdown(run_dir / "background_mode_validation.md", lines)
    _write_markdown(_latest_root() / "background_mode_validation.md", lines)
    return payload


def _parse_modes(text: str) -> list[str]:
    out: list[str] = []
    for token in str(text or "").split(","):
        item = token.strip().lower()
        if item and item in VALIDATION_MODES and item not in out:
            out.append(item)
    return out


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate background window modes against deterministic P1 smoke behavior.")
    parser.add_argument("--base-url", default="http://127.0.0.1:12346")
    parser.add_argument("--out-dir", default="docs/artifacts/p53/background_mode_validation")
    parser.add_argument("--run-id", default="")
    parser.add_argument("--modes", default="visible,offscreen,minimized,hidden")
    parser.add_argument("--seed", default="AAAAAAA")
    parser.add_argument("--scope", default="p1_hand_score_observed_core")
    parser.add_argument("--max-steps", type=int, default=120)
    parser.add_argument("--timeout-sec", type=int, default=900)
    parser.add_argument("--resolve-mode", action="store_true")
    parser.add_argument("--requested-mode", default="")
    parser.add_argument("--fallback-mode", default="")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if bool(args.resolve_mode):
        payload = resolve_effective_window_mode(
            str(args.requested_mode or ""),
            fallback_mode=str(args.fallback_mode or ""),
        )
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0
    payload = validate_background_modes(
        base_url=str(args.base_url),
        out_dir=(str(args.out_dir).strip() or None),
        run_id=str(args.run_id or ""),
        modes=_parse_modes(str(args.modes or "")),
        seed=str(args.seed or "AAAAAAA"),
        scope=str(args.scope or "p1_hand_score_observed_core"),
        max_steps=max(1, int(args.max_steps)),
        timeout_sec=max(60, int(args.timeout_sec)),
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
