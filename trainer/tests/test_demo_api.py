from __future__ import annotations

import http.client
import json
import threading
from pathlib import Path
from types import SimpleNamespace

from demo.api import DemoRequestError, create_handler
from http.server import ThreadingHTTPServer


class _SessionStub:
    def load_scenario(self, scenario_id: str) -> dict[str, str]:
        raise DemoRequestError(404, f"unknown scenario '{scenario_id}'")

    def recommendations(self, *, policy: str, topk: int) -> dict[str, object]:
        return {"policy": policy, "topk": topk, "recommendations": []}

    def step(self, *, action: dict[str, object] | None = None, policy: str | None = None) -> dict[str, object]:
        return {"ok": True, "action": action or {}, "policy": policy or "model"}

    def autoplay(self, *, steps: int = 4, policy: str | None = None) -> dict[str, object]:
        return {"ok": True, "steps_executed": steps, "policy": policy or "model"}


def _make_app(static_root: Path) -> SimpleNamespace:
    return SimpleNamespace(
        static_root=static_root,
        session=_SessionStub(),
        health=lambda: {"status": "ok"},
        scenarios_payload=lambda: {"scenarios": []},
        model_payload=lambda: {"loaded": False},
        training_status_payload=lambda: {"status": "idle"},
        start_training=lambda profile="standard": {"status": "queued", "profile": profile},
    )


def _run_server(app: SimpleNamespace) -> tuple[ThreadingHTTPServer, threading.Thread]:
    server = ThreadingHTTPServer(("127.0.0.1", 0), create_handler(app))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread


def _request(
    server: ThreadingHTTPServer,
    method: str,
    path: str,
    *,
    body: bytes = b"",
    headers: dict[str, str] | None = None,
) -> tuple[int, dict[str, str], bytes]:
    conn = http.client.HTTPConnection(server.server_address[0], server.server_address[1], timeout=5)
    try:
        conn.request(method, path, body=body, headers=headers or {})
        response = conn.getresponse()
        payload = response.read()
        return response.status, dict(response.getheaders()), payload
    finally:
        conn.close()


def test_invalid_json_body_returns_400(tmp_path: Path) -> None:
    static_root = tmp_path / "static"
    static_root.mkdir()
    (static_root / "index.html").write_text("<html></html>", encoding="utf-8")
    server, thread = _run_server(_make_app(static_root))
    try:
        status, headers, payload = _request(
            server,
            "POST",
            "/api/recommend",
            body=b"{bad json",
            headers={"Content-Type": "application/json", "Content-Length": "9"},
        )
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()

    assert status == 400
    assert headers["Content-Type"].startswith("application/json")
    body = json.loads(payload.decode("utf-8"))
    assert "invalid JSON body" in body["error"]


def test_unknown_scenario_returns_404(tmp_path: Path) -> None:
    static_root = tmp_path / "static"
    static_root.mkdir()
    (static_root / "index.html").write_text("<html></html>", encoding="utf-8")
    server, thread = _run_server(_make_app(static_root))
    try:
        raw = json.dumps({"scenario_id": "missing"}).encode("utf-8")
        status, _, payload = _request(
            server,
            "POST",
            "/api/scenario/load",
            body=raw,
            headers={"Content-Type": "application/json", "Content-Length": str(len(raw))},
        )
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()

    assert status == 404
    body = json.loads(payload.decode("utf-8"))
    assert "unknown scenario 'missing'" in body["error"]


def test_static_path_traversal_returns_403(tmp_path: Path) -> None:
    static_root = tmp_path / "static"
    static_root.mkdir()
    (static_root / "index.html").write_text("<html></html>", encoding="utf-8")
    (tmp_path / "secret.txt").write_text("top-secret", encoding="utf-8")
    server, thread = _run_server(_make_app(static_root))
    try:
        status, _, payload = _request(server, "GET", "/static/../secret.txt")
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()

    assert status == 403
    body = json.loads(payload.decode("utf-8"))
    assert "forbidden static path" in body["error"]


def test_invalid_integer_query_returns_400(tmp_path: Path) -> None:
    static_root = tmp_path / "static"
    static_root.mkdir()
    (static_root / "index.html").write_text("<html></html>", encoding="utf-8")
    server, thread = _run_server(_make_app(static_root))
    try:
        raw = json.dumps({"topk": "oops"}).encode("utf-8")
        status, _, payload = _request(
            server,
            "POST",
            "/api/recommend",
            body=raw,
            headers={"Content-Type": "application/json", "Content-Length": str(len(raw))},
        )
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()

    assert status == 400
    body = json.loads(payload.decode("utf-8"))
    assert "invalid integer for 'topk'" in body["error"]
