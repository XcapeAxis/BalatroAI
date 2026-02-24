if __package__ is None or __package__ == "":
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).resolve().parent.parent))


import json
import logging
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import urlparse

import requests
from requests.adapters import HTTPAdapter

from trainer.action_space import DISCARD, PLAY


class RPCError(RuntimeError):
    pass


class ConnectionError(RuntimeError):
    pass


class StateError(RuntimeError):
    pass


class EnvBackend(Protocol):
    def reset(self, seed: str | None = None) -> dict[str, Any]:
        ...

    def get_state(self) -> dict[str, Any]:
        ...

    def step(self, action: dict[str, Any]) -> tuple[dict[str, Any], float, bool, dict[str, Any]]:
        ...

    def health(self) -> bool:
        ...

    def close(self) -> None:
        ...


_SERVE_SUPPORT_CACHE: dict[tuple[str, ...], bool] = {}
_FALLBACK_WARNED = False
_CONFIG_CACHE: dict | None = None

CONFIG_PATH = Path(__file__).resolve().parent / "config.json"


@dataclass
class RPCConfig:
    timeout: float = 5.0
    retries: int = 3
    backoff_sec: float = 0.15


def _build_session() -> requests.Session:
    s = requests.Session()
    adapter = HTTPAdapter(pool_connections=8, pool_maxsize=32, max_retries=0)
    s.mount("http://", adapter)
    s.mount("https://", adapter)
    s.headers.update({"Connection": "keep-alive"})
    return s


class _ClientPool:
    def __init__(self):
        self._sessions: dict[str, requests.Session] = {}

    def get(self, base_url: str) -> requests.Session:
        if base_url not in self._sessions:
            self._sessions[base_url] = _build_session()
        return self._sessions[base_url]

    def close_all(self) -> None:
        for s in self._sessions.values():
            try:
                s.close()
            except Exception:
                pass
        self._sessions.clear()


_POOL = _ClientPool()


def load_config(force_reload: bool = False) -> dict:
    global _CONFIG_CACHE
    if _CONFIG_CACHE is not None and not force_reload:
        return dict(_CONFIG_CACHE)

    cfg = {"index_base": 0}
    if CONFIG_PATH.exists():
        try:
            raw = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                cfg.update(raw)
        except Exception:
            pass

    idx = int(cfg.get("index_base", 0))
    cfg["index_base"] = 0 if idx not in (0, 1) else idx
    _CONFIG_CACHE = dict(cfg)
    return dict(cfg)


def save_config(cfg: dict) -> None:
    global _CONFIG_CACHE
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    CONFIG_PATH.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
    _CONFIG_CACHE = dict(cfg)


def get_index_base(force_reload: bool = False) -> int:
    cfg = load_config(force_reload=force_reload)
    value = int(cfg.get("index_base", 0))
    return 0 if value not in (0, 1) else value


def set_index_base(index_base: int) -> None:
    if index_base not in (0, 1):
        raise ValueError("index_base must be 0 or 1")
    cfg = load_config(force_reload=False)
    cfg["index_base"] = index_base
    save_config(cfg)


def _to_api_indices(local_indices: list[int], index_base: int) -> list[int]:
    out = []
    for idx in local_indices:
        idx_i = int(idx)
        if idx_i < 0:
            raise ValueError(f"card index must be >= 0, got {idx_i}")
        out.append(idx_i + index_base)
    return out


def _call_method(base_url: str, method: str, params: dict | None = None, timeout: float = 5.0):
    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": method,
        "params": params or {},
    }
    session = _POOL.get(base_url)
    try:
        response = session.post(base_url, json=payload, timeout=timeout)
        response.raise_for_status()
    except requests.exceptions.RequestException as exc:
        raise ConnectionError(f"RPC transport failed for {method}: {exc}") from exc

    try:
        body = response.json()
    except Exception as exc:
        raise RPCError(f"RPC invalid JSON response for {method}: {exc}") from exc

    if "error" in body:
        msg = body["error"].get("message", "unknown error")
        raise RPCError(f"RPC {method} error: {msg}")

    return body.get("result")


def health(base_url: str) -> bool:
    try:
        _call_method(base_url, "health", timeout=2.0)
        return True
    except Exception:
        try:
            _call_method(base_url, "gamestate", timeout=2.0)
            return True
        except Exception:
            return False


def get_state(base_url: str, timeout: float = 5.0) -> dict:
    result = _call_method(base_url, "gamestate", timeout=timeout)
    if not isinstance(result, dict):
        raise StateError(f"gamestate is not dict: {type(result)}")
    return result


def _fallback_act_via_select(
    base_url: str,
    action_type: str,
    local_indices: list[int],
    timeout: float,
    logger: logging.Logger | None,
    index_base: int,
):
    global _FALLBACK_WARNED
    if not _FALLBACK_WARNED and logger is not None:
        logger.warning("act_batch fallback enabled (select+play/discard). This path is slower; add batch RPC for speed.")
        _FALLBACK_WARNED = True

    api_indices = _to_api_indices(local_indices, index_base)
    for idx in api_indices:
        _call_method(base_url, "select", {"index": int(idx)}, timeout=timeout)

    method = "play" if action_type == PLAY else "discard"
    try:
        _call_method(base_url, method, {}, timeout=timeout)
    except RPCError:
        _call_method(base_url, method, {"cards": api_indices}, timeout=timeout)


def act_batch(
    base_url: str,
    action_type: str,
    indices: list[int],
    timeout: float = 8.0,
    logger: logging.Logger | None = None,
) -> dict:
    action_type = action_type.upper()
    if action_type not in {PLAY, DISCARD}:
        raise ValueError(f"Unsupported action_type: {action_type}")

    index_base = get_index_base(force_reload=False)
    api_indices = _to_api_indices(indices, index_base)

    method = "play" if action_type == PLAY else "discard"
    try:
        _call_method(base_url, method, {"cards": api_indices}, timeout=timeout)
    except RPCError as exc:
        msg = str(exc).lower()
        needs_fallback = any(
            marker in msg
            for marker in (
                "invalid params",
                "method not found",
                "requires one of these states",
                "button not found",
            )
        )
        if not needs_fallback:
            raise
        _fallback_act_via_select(base_url, action_type, indices, timeout=timeout, logger=logger, index_base=index_base)
    return get_state(base_url, timeout=timeout)


def _round_chips(state: dict[str, Any]) -> float:
    return float((state.get("round") or {}).get("chips") or 0.0)


def _phase_default_action(state: dict[str, Any], seed: str) -> dict[str, Any]:
    phase = str(state.get("state") or "UNKNOWN")
    if phase == "BLIND_SELECT":
        return {"action_type": "SELECT", "index": 0}
    if phase == "SELECTING_HAND":
        hand = (state.get("hand") or {}).get("cards") or []
        if hand:
            return {"action_type": PLAY, "indices": [0]}
        return {"action_type": "WAIT"}
    if phase == "ROUND_EVAL":
        return {"action_type": "CASH_OUT"}
    if phase == "SHOP":
        return {"action_type": "NEXT_ROUND"}
    if phase in {"MENU", "GAME_OVER"}:
        return {"action_type": "START", "seed": seed}
    return {"action_type": "WAIT"}


@dataclass
class RealBackend:
    base_url: str
    timeout_sec: float = 8.0
    seed: str = "AAAAAAA"
    logger: logging.Logger | None = None

    def health(self) -> bool:
        return health(self.base_url)

    def get_state(self) -> dict[str, Any]:
        return get_state(self.base_url, timeout=self.timeout_sec)

    def reset(self, seed: str | None = None) -> dict[str, Any]:
        if seed is not None:
            self.seed = seed
        state = self.get_state()
        phase = str(state.get("state") or "UNKNOWN")
        if phase in {"MENU", "GAME_OVER"}:
            _call_method(
                self.base_url,
                "start",
                {"deck": "RED", "stake": "WHITE", "seed": self.seed},
                timeout=self.timeout_sec,
            )
            state = self.get_state()
        return state

    def step(self, action: dict[str, Any]) -> tuple[dict[str, Any], float, bool, dict[str, Any]]:
        if not isinstance(action, dict):
            raise ValueError("action must be dict")

        before = self.get_state()
        action_type = str(action.get("action_type") or "WAIT").upper()

        if action_type == "AUTO":
            action = _phase_default_action(before, self.seed)
            action_type = str(action.get("action_type") or "WAIT").upper()

        if action_type in {PLAY, DISCARD}:
            indices = [int(x) for x in (action.get("indices") or [])]
            after = act_batch(self.base_url, action_type, indices, timeout=self.timeout_sec, logger=self.logger)

        elif action_type == "SELECT":
            _call_method(self.base_url, "select", {"index": int(action.get("index", 0))}, timeout=self.timeout_sec)
            after = self.get_state()

        elif action_type == "CASH_OUT":
            _call_method(self.base_url, "cash_out", {}, timeout=self.timeout_sec)
            after = self.get_state()

        elif action_type == "NEXT_ROUND":
            _call_method(self.base_url, "next_round", {}, timeout=self.timeout_sec)
            after = self.get_state()

        elif action_type == "START":
            seed = str(action.get("seed") or self.seed)
            self.seed = seed
            params = {"deck": "RED", "stake": "WHITE", "seed": seed}
            _call_method(self.base_url, "start", params, timeout=self.timeout_sec)
            after = self.get_state()

        elif action_type == "MENU":
            _call_method(self.base_url, "menu", {}, timeout=self.timeout_sec)
            after = self.get_state()

        elif action_type == "SKIP":
            _call_method(self.base_url, "skip", {}, timeout=self.timeout_sec)
            after = self.get_state()

        elif action_type == "REROLL":
            _call_method(self.base_url, "reroll", {}, timeout=self.timeout_sec)
            after = self.get_state()

        elif action_type == "BUY":
            params = action.get("params") if isinstance(action.get("params"), dict) else {}
            if not params:
                for key in ("card", "pack", "voucher"):
                    if key in action:
                        params[key] = action[key]
            _call_method(self.base_url, "buy", params, timeout=self.timeout_sec)
            after = self.get_state()

        elif action_type == "SELL":
            params = action.get("params") if isinstance(action.get("params"), dict) else {}
            if not params and "joker" in action:
                params["joker"] = action["joker"]
            _call_method(self.base_url, "sell", params, timeout=self.timeout_sec)
            after = self.get_state()

        elif action_type == "PACK":
            params = action.get("params") if isinstance(action.get("params"), dict) else {}
            if not params:
                for key in ("card", "skip"):
                    if key in action:
                        params[key] = action[key]
            _call_method(self.base_url, "pack", params, timeout=self.timeout_sec)
            after = self.get_state()

        elif action_type == "USE":
            params = action.get("params") if isinstance(action.get("params"), dict) else {}
            if not params and "consumable" in action:
                params["consumable"] = action["consumable"]
            _call_method(self.base_url, "use", params, timeout=self.timeout_sec)
            after = self.get_state()

        elif action_type == "WAIT":
            time.sleep(max(0.0, float(action.get("sleep") or 0.05)))
            after = self.get_state()

        else:
            fallback = _phase_default_action(before, self.seed)
            if self.logger is not None:
                self.logger.warning("Unknown action_type=%s, fallback to %s", action_type, fallback)
            return self.step(fallback)

        reward = _round_chips(after) - _round_chips(before)
        done = str(after.get("state") or "") == "GAME_OVER"
        info = {"backend": "real", "action_type": action_type}
        return after, float(reward), done, info

    def close(self) -> None:
        return None


class SimBackend:
    def __init__(self, seed: str = "AAAAAAA"):
        from sim.pybind.sim_env import SimEnvBackend as _SimEnvBackend

        self.seed = seed
        self._backend = _SimEnvBackend(seed=seed)

    def health(self) -> bool:
        return True

    def get_state(self) -> dict[str, Any]:
        return self._backend.get_state()

    def reset(self, seed: str | None = None) -> dict[str, Any]:
        if seed is not None:
            self.seed = seed
        return self._backend.reset(seed=self.seed)

    def step(self, action: dict[str, Any]) -> tuple[dict[str, Any], float, bool, dict[str, Any]]:
        return self._backend.step(action)

    def close(self) -> None:
        self._backend.close()


def create_backend(
    backend: str,
    *,
    base_url: str | None = None,
    timeout_sec: float = 8.0,
    seed: str = "AAAAAAA",
    logger: logging.Logger | None = None,
) -> EnvBackend:
    kind = str(backend or "real").lower()
    if kind == "real":
        if not base_url:
            raise ValueError("base_url is required for RealBackend")
        return RealBackend(base_url=base_url, timeout_sec=timeout_sec, seed=seed, logger=logger)
    if kind == "sim":
        return SimBackend(seed=seed)
    raise ValueError(f"Unknown backend: {backend}")


def _supports_serve_subcommand(cmd_prefix: list[str]) -> bool:
    key = tuple(cmd_prefix)
    if key in _SERVE_SUPPORT_CACHE:
        return _SERVE_SUPPORT_CACHE[key]
    try:
        result = subprocess.run(
            cmd_prefix + ["serve", "--help"],
            capture_output=True,
            text=True,
            timeout=3,
            check=False,
        )
        ok = result.returncode == 0
    except Exception:
        ok = False
    _SERVE_SUPPORT_CACHE[key] = ok
    return ok


def _build_launch_cmd(
    launcher: str,
    uvx_path: str,
    balatrobot_cmd: str,
    port: int,
    love_path: str,
    lovely_path: str,
) -> list[str]:
    if launcher == "uvx":
        prefix = [uvx_path, "balatrobot"]
    else:
        prefix = [balatrobot_cmd]

    cmd = list(prefix)
    if _supports_serve_subcommand(prefix):
        cmd.append("serve")

    cmd.extend(
        [
            "--headless",
            "--fast",
            "--port",
            str(port),
            "--love-path",
            love_path,
            "--lovely-path",
            lovely_path,
        ]
    )
    return cmd


def _parse_port(base_url: str) -> int:
    parsed = urlparse(base_url)
    if parsed.port is None:
        raise ValueError(f"base_url must include port: {base_url}")
    return parsed.port


@dataclass
class EnvHandle:
    base_url: str
    launcher: str = "uvx"
    uvx_path: str = "uvx"
    balatrobot_cmd: str = "balatrobot"
    love_path: str | None = None
    lovely_path: str | None = None
    logs_path: str | None = None
    cwd: str | None = None
    logger: logging.Logger | None = None

    proc: subprocess.Popen | None = None

    def _log(self, level: int, message: str, *args) -> None:
        if self.logger is not None:
            self.logger.log(level, message, *args)

    def start(self) -> None:
        if not self.love_path or not self.lovely_path:
            raise ValueError("love_path and lovely_path are required for managed launch")
        if self.proc is not None and self.proc.poll() is None:
            return

        port = _parse_port(self.base_url)
        cmd = _build_launch_cmd(
            launcher=self.launcher,
            uvx_path=self.uvx_path,
            balatrobot_cmd=self.balatrobot_cmd,
            port=port,
            love_path=self.love_path,
            lovely_path=self.lovely_path,
        )

        env = os.environ.copy()
        if self.logs_path:
            env["BALATROBOT_LOGS_PATH"] = self.logs_path

        kwargs = {
            "stdout": subprocess.DEVNULL,
            "stderr": subprocess.DEVNULL,
            "env": env,
        }
        if self.cwd:
            kwargs["cwd"] = self.cwd
        if os.name == "nt":
            kwargs["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0)

        try:
            self.proc = subprocess.Popen(cmd, **kwargs)
        except Exception as exc:
            cmd_text = subprocess.list2cmdline(cmd)
            raise RuntimeError(f"Failed to launch env process: {cmd_text} | {exc}") from exc

        time.sleep(0.2)
        if self.proc.poll() is not None:
            raise RuntimeError(
                f"Env process exited early on {self.base_url}, returncode={self.proc.returncode}."
            )

        self._log(logging.INFO, "Launched env %s", self.base_url)

    def wait_healthy(self, timeout_sec: float = 60.0, poll_sec: float = 0.3) -> bool:
        deadline = time.perf_counter() + timeout_sec
        while time.perf_counter() < deadline:
            if health(self.base_url):
                return True
            time.sleep(poll_sec)
        return False

    def stop(self) -> None:
        if self.proc is None:
            return
        if os.name == "nt" and self.proc.pid:
            subprocess.run(
                ["taskkill", "/PID", str(self.proc.pid), "/T", "/F"],
                capture_output=True,
                text=True,
                check=False,
            )
        try:
            if self.proc.poll() is None:
                self.proc.terminate()
                self.proc.wait(timeout=8)
        except Exception:
            try:
                self.proc.kill()
            except Exception:
                pass
        self._log(logging.INFO, "Stopped env %s", self.base_url)
        self.proc = None

    def restart(self, timeout_sec: float = 60.0) -> bool:
        self.stop()
        self.start()
        ok = self.wait_healthy(timeout_sec=timeout_sec)
        if not ok:
            self._log(logging.ERROR, "Env failed health check after restart: %s", self.base_url)
        return ok

    def is_alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None


def close_pool() -> None:
    _POOL.close_all()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Minimal CLI for env_client adapter.")
    parser.add_argument("--base-url", default="http://127.0.0.1:12346")
    parser.add_argument("--backend", choices=["real", "sim"], default="real")
    parser.add_argument("--health", action="store_true", help="Check env health")
    parser.add_argument("--get-state", action="store_true", help="Fetch and print gamestate")
    parser.add_argument("--act", choices=[PLAY, DISCARD], help="Execute hand action")
    parser.add_argument("--indices", default="", help="Comma-separated card indices for --act")
    parser.add_argument("--timeout", type=float, default=8.0)
    parser.add_argument("--show-config", action="store_true", help="Print trainer/config.json")
    parser.add_argument("--set-index-base", type=int, choices=[0, 1], help="Set index base and save config")
    args = parser.parse_args()

    if args.set_index_base is not None:
        set_index_base(args.set_index_base)

    if args.show_config:
        print(json.dumps(load_config(force_reload=True), ensure_ascii=False, indent=2))

    if args.backend == "real":
        backend = create_backend("real", base_url=args.base_url, timeout_sec=args.timeout)
    else:
        backend = create_backend("sim", seed="AAAAAAA")

    if args.health:
        print(json.dumps({"healthy": backend.health()}, ensure_ascii=False))

    if args.get_state:
        state = backend.get_state()
        print(json.dumps(state, ensure_ascii=False, indent=2))

    if args.act:
        idx = [int(x) for x in args.indices.split(",") if x.strip()]
        state, reward, done, info = backend.step({"action_type": args.act, "indices": idx})
        print(json.dumps({"state": state, "reward": reward, "done": done, "info": info}, ensure_ascii=False, indent=2))

    if not args.health and not args.get_state and not args.act and not args.show_config and args.set_index_base is None:
        print("No action selected. Use --help for options.")

    backend.close()

