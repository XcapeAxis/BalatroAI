from __future__ import annotations

if __package__ is None or __package__ == "":
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

import argparse
import json
import os
import subprocess
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, quote, unquote, urlparse

from trainer.monitoring.dashboard_build import build_dashboard
from trainer.ops_ui import routes
from trainer.ops_ui.state_loader import build_ops_state, path_in_repo, repo_root
from trainer.autonomy.attention_queue import resolve_attention_item
from trainer.registry.checkpoint_registry import list_entries, snapshot_registry
from trainer.registry.promotion_queue import build_promotion_queue_summary
from trainer.runtime.background_mode_validation import resolve_effective_window_mode
from trainer.runtime.window_supervisor import set_window_mode


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _artifacts_dir() -> Path:
    return repo_root() / "docs" / "artifacts" / "p53"


def _ops_ui_root() -> Path:
    return _artifacts_dir() / "ops_ui"


def _ops_ui_latest_state_path() -> Path:
    return _ops_ui_root() / "latest" / "ops_ui_state.json"


def _ops_jobs_root() -> Path:
    return _ops_ui_root() / "jobs"


def _ops_audit_path() -> Path:
    return _artifacts_dir() / "ops_audit" / "ops_audit.jsonl"


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as fp:
        fp.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _render_base(title: str, content: str, *, current_path: str, refresh_sec: int = 15) -> str:
    template_root = Path(__file__).resolve().parent / "templates"
    base_html = (template_root / "base.html").read_text(encoding="utf-8")
    style = (template_root / "style.css").read_text(encoding="utf-8")
    nav_items = [
        ("/", "Overview"),
        ("/autonomy", "Autonomy"),
        ("/environment", "Environment"),
        ("/campaigns", "Campaigns"),
        ("/registry", "Checkpoint Registry"),
        ("/promotion-queue", "Promotion Queue"),
        ("/attention-queue", "Attention Queue"),
        ("/morning-summary", "Morning Summary"),
        ("/blocked-campaigns", "Blocked Campaigns"),
        ("/router-calibration", "Router Calibration"),
        ("/router-canary", "Guard / Canary"),
        ("/runs", "Runs / Metrics"),
        ("/windows", "Windows"),
        ("/jobs", "Jobs / Audit"),
    ]
    nav_links: list[str] = []
    for href, label in nav_items:
        current_attr = ' aria-current="page"' if href == current_path else ""
        nav_links.append(f'<a href="{href}"{current_attr}>{label}</a>')
    nav_html = "".join(nav_links)
    return (
        base_html.replace("{{refresh_sec}}", str(int(refresh_sec)))
        .replace("{{title}}", title)
        .replace("{{style}}", style)
        .replace("{{generated_at}}", _now_iso())
        .replace("{{repo_root}}", str(repo_root()))
        .replace("{{nav}}", nav_html)
        .replace("{{content}}", content)
    )


def _server_state(host: str, port: int) -> dict[str, Any]:
    payload = {
        "schema": "p53_ops_ui_server_state_v1",
        "generated_at": _now_iso(),
        "host": host,
        "port": int(port),
        "url": f"http://{host}:{int(port)}/",
        "pid": os.getpid(),
    }
    _write_json(_ops_ui_latest_state_path(), payload)
    return payload


def _audit(action: str, *, target: str, success: bool, output_ref: str = "", details: dict[str, Any] | None = None) -> None:
    _append_jsonl(
        _ops_audit_path(),
        {
            "schema": "p53_ops_audit_v1",
            "timestamp": _now_iso(),
            "operator": "local_ui",
            "action": action,
            "target": target,
            "success": bool(success),
            "output_ref": str(output_ref or ""),
            "details": dict(details or {}),
        },
    )


def _spawn_job(action: str, command: list[str]) -> dict[str, Any]:
    job_id = f"{action}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    log_path = _ops_jobs_root() / f"{job_id}.log"
    job_path = _ops_jobs_root() / f"{job_id}.json"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8", newline="\n") as log_fp:
        proc = subprocess.Popen(
            command,
            cwd=str(repo_root()),
            stdout=log_fp,
            stderr=subprocess.STDOUT,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0) | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0),
        )
    payload = {
        "schema": "p53_ops_ui_job_v1",
        "created_at": _now_iso(),
        "job_id": job_id,
        "action": action,
        "status": "running",
        "pid": int(proc.pid),
        "command": " ".join(command),
        "log_path": str(log_path.resolve()),
        "job_path": str(job_path.resolve()),
    }
    _write_json(job_path, payload)
    _audit(action, target=" ".join(command), success=True, output_ref=str(log_path.resolve()), details={"job_id": job_id, "pid": proc.pid})
    return payload


def _latest_resume_command() -> str:
    state = build_ops_state()
    target = state.get("latest_resume_target") if isinstance(state.get("latest_resume_target"), dict) else {}
    return str(target.get("resume_command") or "powershell -ExecutionPolicy Bypass -File scripts\\run_p22.ps1 -ResumeLatestCampaign")


def _artifact_response(target: Path) -> tuple[bytes, str]:
    suffix = target.suffix.lower()
    if target.is_dir():
        items = sorted(target.iterdir(), key=lambda p: p.name)
        lines = [f"<h1>{target.name}</h1><ul>"]
        for item in items[:200]:
            href = "/artifact?path=" + quote(str(item.resolve()))
            lines.append(f'<li><a href="{href}">{item.name}</a></li>')
        lines.append("</ul>")
        return ("\n".join(lines)).encode("utf-8"), "text/html; charset=utf-8"
    if suffix == ".html":
        return target.read_bytes(), "text/html; charset=utf-8"
    if suffix in {".json", ".md", ".txt", ".log", ".jsonl", ".csv", ".ps1", ".py", ".yaml", ".yml"}:
        text = target.read_text(encoding="utf-8", errors="replace")
        body = f"<h1>{target.name}</h1><pre>{routes.esc(text)}</pre>"
        return body.encode("utf-8"), "text/html; charset=utf-8"
    return target.read_bytes(), "application/octet-stream"


class OpsRequestHandler(BaseHTTPRequestHandler):
    server_version = "BalatroOpsUI/1.0"

    def _send_html(self, html_text: str, *, status: int = 200) -> None:
        data = html_text.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _send_json(self, payload: Any, *, status: int = 200) -> None:
        data = (json.dumps(payload, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _redirect(self, location: str) -> None:
        self.send_response(HTTPStatus.SEE_OTHER)
        self.send_header("Location", location)
        self.end_headers()

    def _read_form(self) -> dict[str, str]:
        length = int(self.headers.get("Content-Length") or 0)
        data = self.rfile.read(length).decode("utf-8", errors="replace")
        parsed = parse_qs(data, keep_blank_values=True)
        return {key: values[-1] for key, values in parsed.items()}

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        route = parsed.path
        query = parse_qs(parsed.query)
        if route == "/health":
            self._send_json({"status": "ok", "ts": _now_iso(), "pid": os.getpid()})
            return
        if route == "/api/state":
            self._send_json(build_ops_state())
            return
        if route == "/artifact":
            target = path_in_repo(unquote(str((query.get("path") or [""])[-1])))
            if not isinstance(target, Path) or not target.exists():
                self._send_html(_render_base("Artifact Missing", "<section class='panel'><h2>Artifact not found.</h2></section>", current_path="/artifact"), status=404)
                return
            data, content_type = _artifact_response(target)
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return

        state = build_ops_state(
            registry_family=str((query.get("family") or [""])[-1]),
            registry_status=str((query.get("status") or [""])[-1]),
            registry_latest=str((query.get("latest") or ["0"])[-1]).lower() in {"1", "true", "yes"},
            registry_promoted=str((query.get("promoted") or ["0"])[-1]).lower() in {"1", "true", "yes"},
        )
        if route == "/":
            self._send_html(_render_base("Overview", routes.render_overview(state, current_path=route), current_path=route))
            return
        if route == "/autonomy":
            self._send_html(_render_base("Autonomy", routes.render_autonomy(state), current_path=route))
            return
        if route == "/environment":
            self._send_html(_render_base("Environment", routes.render_environment(state), current_path=route))
            return
        if route == "/campaigns":
            self._send_html(_render_base("Campaigns", routes.render_campaigns(state), current_path=route))
            return
        if route == "/registry":
            self._send_html(_render_base("Checkpoint Registry", routes.render_registry(state), current_path=route))
            return
        if route == "/promotion-queue":
            self._send_html(_render_base("Promotion Queue", routes.render_promotion_queue(state), current_path=route))
            return
        if route == "/attention-queue":
            self._send_html(_render_base("Attention Queue", routes.render_attention_queue(state), current_path=route))
            return
        if route == "/morning-summary":
            self._send_html(_render_base("Morning Summary", routes.render_morning_summary(state), current_path=route))
            return
        if route == "/blocked-campaigns":
            self._send_html(_render_base("Blocked Campaigns", routes.render_blocked_campaigns(state), current_path=route))
            return
        if route == "/router-calibration":
            self._send_html(_render_base("Router Calibration", routes.render_router_calibration(state), current_path=route))
            return
        if route == "/router-canary":
            self._send_html(_render_base("Guard / Canary", routes.render_router_guard_canary(state), current_path=route))
            return
        if route == "/runs":
            self._send_html(_render_base("Runs", routes.render_runs(state), current_path=route))
            return
        if route == "/windows":
            self._send_html(_render_base("Windows", routes.render_windows(state, current_path=route), current_path=route))
            return
        if route == "/jobs":
            self._send_html(_render_base("Jobs", routes.render_jobs(state), current_path=route))
            return
        self._send_html(_render_base("Not Found", "<section class='panel'><h2>Unknown route.</h2></section>", current_path=route), status=404)

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        form = self._read_form()
        return_to = str(form.get("return_to") or "/")
        try:
            if parsed.path == "/actions/run_p22_quick":
                _spawn_job("run_p22_quick", ["powershell", "-ExecutionPolicy", "Bypass", "-File", "scripts\\run_p22.ps1", "-Quick"])
            elif parsed.path == "/actions/run_p22_p53":
                _spawn_job("run_p22_p53", ["powershell", "-ExecutionPolicy", "Bypass", "-File", "scripts\\run_p22.ps1", "-RunP53"])
            elif parsed.path == "/actions/run_p22_p57":
                _spawn_job("run_p22_p57", ["powershell", "-ExecutionPolicy", "Bypass", "-File", "scripts\\run_p22.ps1", "-RunP57"])
            elif parsed.path == "/actions/run_p22_overnight":
                _spawn_job("run_p22_overnight", ["powershell", "-ExecutionPolicy", "Bypass", "-File", "scripts\\run_p22.ps1", "-Overnight"])
            elif parsed.path == "/actions/run_autonomy_quick":
                _spawn_job("run_autonomy_quick", ["powershell", "-ExecutionPolicy", "Bypass", "-File", "scripts\\run_autonomy.ps1", "-Quick"])
            elif parsed.path == "/actions/run_autonomy_overnight":
                _spawn_job("run_autonomy_overnight", ["powershell", "-ExecutionPolicy", "Bypass", "-File", "scripts\\run_autonomy.ps1", "-Overnight"])
            elif parsed.path == "/actions/run_autonomy_resume":
                _spawn_job("run_autonomy_resume", ["powershell", "-ExecutionPolicy", "Bypass", "-File", "scripts\\run_autonomy.ps1", "-ResumeLatest"])
            elif parsed.path == "/actions/run_doctor":
                _spawn_job("run_doctor", ["powershell", "-ExecutionPolicy", "Bypass", "-File", "scripts\\doctor.ps1"])
            elif parsed.path == "/actions/resume_latest_campaign":
                _spawn_job("resume_latest_campaign", _latest_resume_command().split(" "))
            elif parsed.path == "/actions/rebuild_dashboard":
                summary = build_dashboard(repo_root() / "docs" / "artifacts", repo_root() / "docs" / "artifacts" / "dashboard" / "latest")
                _audit("rebuild_dashboard", target="dashboard", success=True, output_ref=str(summary.get("index_html") or ""))
            elif parsed.path == "/actions/refresh_registry":
                out_path = _artifacts_dir() / "ops_ui" / "latest" / ("registry_snapshot_" + datetime.now().strftime("%Y%m%d-%H%M%S") + ".json")
                snapshot_registry(out_path=out_path)
                queue_path = _artifacts_dir() / "ops_ui" / "latest" / "promotion_queue_snapshot.json"
                _write_json(queue_path, build_promotion_queue_summary(list_entries()))
                _audit("refresh_registry", target="registry", success=True, output_ref=str(out_path.resolve()))
            elif parsed.path == "/actions/window_mode":
                requested = str(form.get("mode") or "visible")
                resolved = resolve_effective_window_mode(requested)
                payload = set_window_mode(str(resolved.get("effective_mode") or requested))
                _audit("window_mode", target=requested, success=bool(payload.get("operation_success")), output_ref=str(payload.get("state_path") or ""), details=resolved)
            elif parsed.path == "/actions/resolve_attention":
                attention_id = str(form.get("attention_id") or "")
                note = str(form.get("resolution_note") or "resolved in ops ui")
                payload = resolve_attention_item(attention_id, resolution_note=note)
                _audit("resolve_attention", target=attention_id, success=True, output_ref=str(payload.get("item_md_path") or ""), details={"status": payload.get("status")})
            else:
                raise ValueError(f"unsupported action: {parsed.path}")
        except Exception as exc:
            _audit(parsed.path.rsplit("/", 1)[-1], target=str(form), success=False, output_ref="", details={"error": str(exc)})
        self._redirect(return_to)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="P53 local Ops UI server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    host = str(args.host or "127.0.0.1")
    port = max(1, int(args.port))
    _server_state(host, port)
    server = ThreadingHTTPServer((host, port), OpsRequestHandler)
    print(json.dumps({"status": "ok", "url": f"http://{host}:{port}/", "pid": os.getpid()}, ensure_ascii=False))
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
