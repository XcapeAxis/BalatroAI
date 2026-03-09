from __future__ import annotations

if __package__ is None or __package__ == "":
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).resolve().parent.parent))

import json
import mimetypes
import threading
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from sim.core.engine import SimEnv

from demo.model_inference import load_latest_bundle, model_info, recommend_actions
from demo.scenario_loader import DemoScenario, load_scenarios
from demo.state_adapter import action_label, build_state_payload, compute_resource_delta, now_iso
from demo.training_manager import TrainingManager


class DemoSession:
    def __init__(self, scenarios: dict[str, DemoScenario]):
        self._lock = threading.RLock()
        self.scenarios = scenarios
        self.bundle = load_latest_bundle()
        self.env = SimEnv(seed="MVP")
        self.timeline: list[dict[str, Any]] = []
        self.policy = "model"
        self.mode = "manual"
        self.current_scenario_id = next(iter(sorted(self.scenarios)))
        self.load_scenario(self.current_scenario_id)

    def refresh_bundle(self) -> None:
        self.bundle = load_latest_bundle()

    def _scenario_summary(self) -> dict[str, Any]:
        return self.scenarios[self.current_scenario_id].to_summary()

    def load_scenario(self, scenario_id: str) -> dict[str, Any]:
        with self._lock:
            scenario = self.scenarios[str(scenario_id)]
            self.current_scenario_id = scenario.scenario_id
            seed = str(((scenario.snapshot.get("rng") or {}).get("seed")) or scenario.scenario_id.upper())
            self.env = SimEnv(seed=seed)
            self.env.reset(from_snapshot=scenario.snapshot)
            self.timeline = [
                {
                    "timestamp": now_iso(),
                    "kind": "scenario_loaded",
                    "kind_label": "场景载入",
                    "scenario_id": scenario.scenario_id,
                    "label": f"已载入：{scenario.name}",
                    "summary": scenario.summary,
                    "focus": scenario.focus,
                }
            ]
            self.mode = "manual"
            return self.state()

    def state(self) -> dict[str, Any]:
        with self._lock:
            return build_state_payload(
                self.env.get_state(),
                scenario=self._scenario_summary(),
                timeline=self.timeline,
                mode=self.mode,
                policy=self.policy,
                model_name=self.bundle.model_name,
            )

    def recommendations(self, *, policy: str | None = None, topk: int = 3) -> dict[str, Any]:
        with self._lock:
            self.refresh_bundle()
            self.policy = str(policy or self.policy or "model").lower()
            return recommend_actions(
                self.env.get_state(),
                env=self.env,
                policy=self.policy,
                topk=max(1, int(topk)),
                bundle=self.bundle,
            )

    def _record_transition(
        self,
        *,
        before: dict[str, Any],
        after: dict[str, Any],
        action: dict[str, Any],
        reward: float,
        done: bool,
        info: dict[str, Any],
        recommendation: dict[str, Any] | None,
    ) -> dict[str, Any]:
        transition = {
            "timestamp": now_iso(),
            "kind": "step",
            "kind_label": "执行一步",
            "action": action,
            "label": action_label(action, before),
            "reward": float(reward),
            "done": bool(done),
            "delta": compute_resource_delta(before, after),
            "phase_before": str(before.get("state") or "UNKNOWN"),
            "phase_after": str(after.get("state") or "UNKNOWN"),
            "info": dict(info or {}),
            "recommendation": recommendation,
        }
        self.timeline.append(transition)
        return transition

    def step(self, *, action: dict[str, Any] | None = None, policy: str | None = None) -> dict[str, Any]:
        with self._lock:
            self.refresh_bundle()
            self.policy = str(policy or self.policy or "model").lower()
            recommendation = None
            if action is None:
                payload = recommend_actions(
                    self.env.get_state(),
                    env=self.env,
                    policy=self.policy,
                    topk=3,
                    bundle=self.bundle,
                )
                recommendations = list(payload.get("recommendations") or [])
                if not recommendations:
                    raise RuntimeError("当前状态没有可执行推荐。")
                recommendation = recommendations[0]
                action = dict(recommendation["action"])

            before = self.env.get_state()
            after, reward, done, info = self.env.step(action)
            transition = self._record_transition(
                before=before,
                after=after,
                action=action,
                reward=reward,
                done=done,
                info=info,
                recommendation=recommendation,
            )
            self.mode = "manual"
            return {"ok": True, "action": action, "transition": transition, "state": self.state()}

    def autoplay(self, *, steps: int = 4, policy: str | None = None) -> dict[str, Any]:
        with self._lock:
            self.refresh_bundle()
            self.policy = str(policy or self.policy or "model").lower()
            self.mode = "autoplay"
            transitions: list[dict[str, Any]] = []
            for _ in range(max(1, int(steps))):
                payload = recommend_actions(
                    self.env.get_state(),
                    env=self.env,
                    policy=self.policy,
                    topk=3,
                    bundle=self.bundle,
                )
                recommendations = list(payload.get("recommendations") or [])
                if not recommendations:
                    break
                recommendation = recommendations[0]
                action = dict(recommendation["action"])
                before = self.env.get_state()
                after, reward, done, info = self.env.step(action)
                transitions.append(
                    self._record_transition(
                        before=before,
                        after=after,
                        action=action,
                        reward=reward,
                        done=done,
                        info=info,
                        recommendation=recommendation,
                    )
                )
                if done:
                    break
            self.mode = "manual"
            return {"ok": True, "steps_executed": len(transitions), "transitions": transitions, "state": self.state()}


class DemoApplication:
    def __init__(self, static_root: Path | None = None):
        self.static_root = static_root or (Path(__file__).resolve().parent / "static")
        self.scenarios = load_scenarios()
        self.session = DemoSession(self.scenarios)
        self.training = TrainingManager()

    def health(self) -> dict[str, Any]:
        training = self.training.status()
        return {
            "status": "ok",
            "scenario_count": len(self.scenarios),
            "current_scenario": self.session.current_scenario_id,
            "model_loaded": bool(self.session.bundle.loaded),
            "model_name": self.session.bundle.model_name,
            "training_status": training.get("status"),
        }

    def scenarios_payload(self) -> dict[str, Any]:
        return {"scenarios": [scenario.to_summary() for scenario in self.scenarios.values()]}

    def model_payload(self) -> dict[str, Any]:
        self.session.refresh_bundle()
        return model_info(self.session.bundle)

    def training_status_payload(self) -> dict[str, Any]:
        return self.training.status()

    def start_training(self, profile: str = "standard") -> dict[str, Any]:
        return self.training.start(profile=profile)


def _json_bytes(payload: Any) -> bytes:
    return (json.dumps(payload, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def _read_body(handler: BaseHTTPRequestHandler) -> dict[str, Any]:
    length = int(handler.headers.get("Content-Length") or 0)
    if length <= 0:
        return {}
    raw = handler.rfile.read(length).decode("utf-8", errors="replace")
    if not raw.strip():
        return {}
    return json.loads(raw)


def _serve_static(app: DemoApplication, request_path: str) -> tuple[int, bytes, str]:
    if request_path == "/":
        file_path = app.static_root / "index.html"
    else:
        file_path = app.static_root / request_path.removeprefix("/static/")
    if not file_path.exists() or not file_path.is_file():
        return 404, _json_bytes({"error": f"静态资源不存在：{request_path}"}), "application/json; charset=utf-8"
    content_type = mimetypes.guess_type(str(file_path))[0] or "application/octet-stream"
    return 200, file_path.read_bytes(), content_type


def create_handler(app: DemoApplication):
    class DemoRequestHandler(BaseHTTPRequestHandler):
        server_version = "BalatroMVP/2.0"

        def _send(self, status: int, payload: bytes, content_type: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def _send_json(self, payload: Any, status: int = 200) -> None:
            self._send(status, _json_bytes(payload), "application/json; charset=utf-8")

        def do_GET(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            try:
                if parsed.path == "/api/health":
                    self._send_json(app.health())
                    return
                if parsed.path == "/api/scenarios":
                    self._send_json(app.scenarios_payload())
                    return
                if parsed.path == "/api/state":
                    self._send_json(app.session.state())
                    return
                if parsed.path == "/api/model_info":
                    self._send_json(app.model_payload())
                    return
                if parsed.path in {"/api/training/status", "/api/training_status"}:
                    self._send_json(app.training_status_payload())
                    return
                if parsed.path == "/" or parsed.path.startswith("/static/"):
                    status, payload, content_type = _serve_static(app, parsed.path)
                    self._send(status, payload, content_type)
                    return
                self._send_json({"error": f"未知路由：{parsed.path}"}, status=404)
            except Exception as exc:
                self._send_json({"error": str(exc)}, status=500)

        def do_POST(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            try:
                body = _read_body(self)
                if parsed.path == "/api/scenario/load":
                    self._send_json(app.session.load_scenario(str(body.get("scenario_id") or body.get("id") or "")))
                    return
                if parsed.path == "/api/recommend":
                    self._send_json(app.session.recommendations(policy=str(body.get("policy") or "model"), topk=int(body.get("topk") or 3)))
                    return
                if parsed.path == "/api/step":
                    self._send_json(
                        app.session.step(
                            action=body.get("action") if isinstance(body.get("action"), dict) else None,
                            policy=str(body.get("policy") or "model"),
                        )
                    )
                    return
                if parsed.path == "/api/autoplay":
                    self._send_json(app.session.autoplay(steps=int(body.get("steps") or 4), policy=str(body.get("policy") or "model")))
                    return
                if parsed.path == "/api/training/start":
                    raw_profile = str(body.get("profile") or "").lower()
                    profile = raw_profile if raw_profile in {"smoke", "fast", "standard"} else "standard"
                    self._send_json(app.start_training(profile=profile))
                    return
                self._send_json({"error": f"未知路由：{parsed.path}"}, status=404)
            except Exception as exc:
                self._send_json({"error": str(exc)}, status=500)

        def log_message(self, format: str, *args) -> None:  # noqa: A003
            return None

    return DemoRequestHandler
