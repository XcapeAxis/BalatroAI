import argparse
import multiprocessing as mp
import os
import queue
import shutil
import subprocess
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path
from urllib.parse import urlparse

import requests
from requests.adapters import HTTPAdapter

DEFAULT_BASE_URL = "http://127.0.0.1"
DEFAULT_START_PORT = 12346
DEFAULT_URL = f"{DEFAULT_BASE_URL}:{DEFAULT_START_PORT}"
PROJECT_ROOT = Path(__file__).resolve().parent

ACTIONABLE_STATES = {"BLIND_SELECT", "SELECTING_HAND", "ROUND_EVAL", "SHOP"}
TRANSITION_STATES = {
    "HAND_PLAYED",
    "DRAW_TO_HAND",
    "NEW_ROUND",
    "PLAY_TAROT",
    "TAROT_PACK",
    "PLANET_PACK",
    "SPECTRAL_PACK",
    "STANDARD_PACK",
    "BUFFOON_PACK",
}

JSON_BACKEND = "stdlib"
ORJSON = None
DEFAULT_CLIENT = None
SERVE_SUBCOMMAND_SUPPORT_CACHE = {}
IDEMPOTENT_RPC_METHODS = {"gamestate", "health"}


def ensure_project_root_cwd():
    try:
        if Path.cwd().resolve() != PROJECT_ROOT:
            os.chdir(PROJECT_ROOT)
    except Exception:
        # CWD enforcement is best-effort; benchmark continues even if chdir fails.
        pass


def build_session():
    session = requests.Session()
    adapter = HTTPAdapter(pool_connections=16, pool_maxsize=64, max_retries=0)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    session.headers.update({"Connection": "keep-alive"})
    return session


def percentile(values, q):
    if not values:
        return 0.0
    sorted_values = sorted(values)
    idx = int((len(sorted_values) - 1) * q)
    return sorted_values[idx]


def summarize(values):
    if not values:
        return {
            "count": 0,
            "avg": 0.0,
            "p50": 0.0,
            "p95": 0.0,
            "max": 0.0,
        }
    return {
        "count": len(values),
        "avg": sum(values) / len(values),
        "p50": percentile(values, 0.50),
        "p95": percentile(values, 0.95),
        "max": max(values),
    }


def is_transient_state_error(exc):
    message = str(exc)
    transient_markers = [
        "requires one of these states",
        "INVALID_STATE",
        "button not found",
        "cash_out_button",
        "no blind on deck",
    ]
    return any(marker in message for marker in transient_markers)


def _try_import_orjson():
    try:
        import orjson as _orjson

        return _orjson
    except Exception:
        return None


def _attempt_install_orjson():
    cmd = [sys.executable, "-m", "pip", "install", "orjson"]
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="ignore",
            check=False,
            timeout=180,
        )
        return result.returncode == 0, (result.stderr or "").strip()
    except Exception as exc:
        return False, str(exc)


def configure_json_backend(requested_backend):
    global JSON_BACKEND, ORJSON
    if requested_backend == "stdlib":
        JSON_BACKEND = "stdlib"
        ORJSON = None
        return JSON_BACKEND

    module = _try_import_orjson()
    if module is not None:
        ORJSON = module
        JSON_BACKEND = "orjson"
        return JSON_BACKEND

    print("orjson not found; attempting automatic install...")
    ok, err = _attempt_install_orjson()
    if not ok:
        print(f"WARNING: Failed to install orjson ({err}). Falling back to stdlib backend.")
        JSON_BACKEND = "stdlib"
        ORJSON = None
        return JSON_BACKEND

    module = _try_import_orjson()
    if module is None:
        print("WARNING: orjson still unavailable after install attempt. Falling back to stdlib backend.")
        JSON_BACKEND = "stdlib"
        ORJSON = None
        return JSON_BACKEND

    ORJSON = module
    JSON_BACKEND = "orjson"
    return JSON_BACKEND


class RPCClient:
    def __init__(self, url, json_backend="stdlib", orjson_module=None):
        self.url = url
        self.json_backend = json_backend
        self.orjson_module = orjson_module
        self.session = build_session()
        self.lat_by_method = defaultdict(list)

    def close(self):
        try:
            self.session.close()
        except Exception:
            pass

    def _reset_session(self):
        try:
            self.session.close()
        except Exception:
            pass
        self.session = build_session()

    def call(self, method, params=None, timeout=10):
        payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": method,
            "params": params or {},
        }

        started = time.perf_counter()
        last_exc = None
        is_idempotent = method in IDEMPOTENT_RPC_METHODS
        max_attempts = 40 if is_idempotent else 1
        for attempt in range(max_attempts):
            try:
                if self.json_backend == "orjson" and self.orjson_module is not None:
                    response = self.session.post(
                        self.url,
                        data=self.orjson_module.dumps(payload),
                        timeout=timeout,
                        headers={"Content-Type": "application/json"},
                    )
                    response.raise_for_status()
                    body = self.orjson_module.loads(response.content)
                else:
                    response = self.session.post(self.url, json=payload, timeout=timeout)
                    response.raise_for_status()
                    body = response.json()
                break
            except requests.exceptions.RequestException as exc:
                last_exc = exc
                self._reset_session()
                if attempt < max_attempts - 1:
                    time.sleep(min(0.05 * (attempt + 1), 0.25))
                    continue
                retry_note = "" if is_idempotent else " (non-idempotent method not retried)"
                raise RuntimeError(f"RPC {method} transport error{retry_note}: {exc}") from exc
        else:
            raise RuntimeError(f"RPC {method} transport error: {last_exc}")

        self.lat_by_method[method].append(time.perf_counter() - started)

        if "error" in body:
            message = body["error"].get("message", "Unknown error")
            raise RuntimeError(f"RPC {method} failed: {message}")

        return body["result"]


def call(method, params=None, timeout=10):
    if DEFAULT_CLIENT is None:
        raise RuntimeError("Default RPC client is not initialized.")
    return DEFAULT_CLIENT.call(method, params=params, timeout=timeout)


class BenchmarkRunner:
    def __init__(self, client, idle_sleep=0.05, prefer_fast_restart=True, mode="action_only", seed="AAAAAAA"):
        self.client = client
        self.idle_sleep = idle_sleep
        self.prefer_fast_restart = prefer_fast_restart
        self.mode = mode
        self.seed = seed

        self.wait_sleep_durations = []
        self.wait_sleep_count = 0
        self.wait_sleep_total_sec = 0.0
        self.action_methods_used = set()
        self.blind_select_retry_at = 0.0

    def _is_actionable_or_transition(self, phase):
        return phase in ACTIONABLE_STATES or phase in TRANSITION_STATES

    def _blind_select_ready(self, state):
        if state.get("state") != "BLIND_SELECT":
            return True
        round_blind = state.get("round", {}).get("blind")
        blinds = state.get("blinds", {})
        has_selectable = any(
            isinstance(info, dict) and info.get("status") in {"SELECT", "CURRENT"} for info in blinds.values()
        )
        return bool(round_blind) or has_selectable

    def _wait_for_blind_select_ready(self, state, timeout=3.0):
        if self._blind_select_ready(state):
            return state

        deadline = time.perf_counter() + timeout
        current = state
        while time.perf_counter() < deadline:
            sleep_for = min(self.idle_sleep, 0.05)
            self.record_wait_sleep(sleep_for)
            time.sleep(sleep_for)
            current = self.client.call("gamestate")
            if current.get("state") != "BLIND_SELECT":
                return current
            if self._blind_select_ready(current):
                return current

        return current

    def _state_fingerprint(self, state):
        round_state = state.get("round", {})
        hand_cards = state.get("hand", {}).get("cards", [])
        return (
            state.get("state"),
            round_state.get("hands_left"),
            round_state.get("discards_left"),
            round_state.get("ante"),
            round_state.get("blind"),
            len(hand_cards),
            state.get("money"),
        )

    def _is_transport_error(self, exc):
        return "transport error" in str(exc).lower()

    def _transient_backoff(self, exc):
        message = str(exc).lower()
        if "no blind on deck" in message:
            backoff = max(self.idle_sleep * 4, 0.2)
            self.blind_select_retry_at = time.perf_counter() + backoff
            return backoff
        return min(self.idle_sleep, 0.05)

    def _recover_transport_action(self, method, prev_state, exc):
        if not self._is_transport_error(exc):
            return False
        if method not in {"play", "discard", "select", "cash_out", "next_round"}:
            return False

        try:
            current = self.client.call("gamestate")
        except Exception:
            return False

        current_phase = current.get("state")
        prev_phase = prev_state.get("state")
        if current_phase in TRANSITION_STATES:
            return True
        if current_phase != prev_phase:
            return True
        return self._state_fingerprint(current) != self._state_fingerprint(prev_state)

    def _start_and_recover_if_already_started(self, params, mode_label):
        exc = None
        try:
            return self.client.call("start", params), mode_label
        except RuntimeError as err:
            exc = err
            recovered = self.client.call("gamestate")
            recovered_phase = recovered.get("state")
            if self._is_actionable_or_transition(recovered_phase):
                return recovered, mode_label
            raise exc

    def reset_runtime_stats(self):
        self.client.lat_by_method.clear()
        self.wait_sleep_durations.clear()
        self.action_methods_used.clear()
        self.wait_sleep_count = 0
        self.wait_sleep_total_sec = 0.0

    def record_wait_sleep(self, duration):
        self.wait_sleep_count += 1
        self.wait_sleep_total_sec += duration
        self.wait_sleep_durations.append(duration)

    def random_action(self, state):
        phase = state.get("state")

        if phase == "BLIND_SELECT":
            if time.perf_counter() < self.blind_select_retry_at:
                return ("__wait__", None)
            if not self._blind_select_ready(state):
                return ("__wait__", None)
            return ("select", {"index": 0})

        if phase == "SELECTING_HAND":
            hand_cards = state.get("hand", {}).get("cards", [])
            discards_left = state.get("round", {}).get("discards_left", 0)
            hands_left = state.get("round", {}).get("hands_left", 1)
            if hand_cards and hands_left <= 0 and discards_left > 0:
                return ("discard", {"cards": [0]})
            if hand_cards:
                return ("play", {"cards": [0]})
            return None

        if phase == "ROUND_EVAL":
            return ("cash_out", {})

        if phase == "SHOP":
            return ("next_round", {})

        if phase == "GAME_OVER":
            return ("__reset__", None)

        if phase in TRANSITION_STATES:
            return ("__wait__", None)

        return None

    def wait_until_actionable(self, timeout=15.0, poll_interval=0.05, stop_on_game_over=False):
        deadline = time.perf_counter() + timeout
        last_state = None

        while time.perf_counter() < deadline:
            state = self.client.call("gamestate")
            last_state = state.get("state")
            if last_state in ACTIONABLE_STATES:
                return state
            if stop_on_game_over and last_state == "GAME_OVER":
                return state
            self.record_wait_sleep(poll_interval)
            time.sleep(poll_interval)

        raise TimeoutError(f"Timed out waiting for actionable state, last_state={last_state}")

    def start_run(self, seed="AAAAAAA", prefer_fast_restart=True):
        params = {"deck": "RED", "stake": "WHITE", "seed": seed}
        for _ in range(6):
            current = self.client.call("gamestate")
            phase = current.get("state")

            if phase in TRANSITION_STATES:
                current = self.wait_until_actionable(poll_interval=0.05, stop_on_game_over=True)
                phase = current.get("state")

            if phase in ACTIONABLE_STATES:
                current = self._wait_for_blind_select_ready(current)
                if current.get("state") == "BLIND_SELECT" and not self._blind_select_ready(current):
                    time.sleep(0.05)
                    continue
                return current, "fast_start" if prefer_fast_restart else "menu_start"

            if phase == "GAME_OVER":
                if prefer_fast_restart:
                    try:
                        state, mode = self._start_and_recover_if_already_started(params, "fast_start")
                        state = self._wait_for_blind_select_ready(state)
                        return state, mode
                    except RuntimeError:
                        time.sleep(0.05)
                        continue

                # Legacy fallback path when fast restart is disabled.
                try:
                    self.client.call("menu")
                except RuntimeError:
                    pass
                time.sleep(0.05)
                continue

            if phase == "MENU":
                try:
                    state, mode = self._start_and_recover_if_already_started(params, "menu_start")
                    state = self._wait_for_blind_select_ready(state)
                    return state, mode
                except RuntimeError:
                    time.sleep(0.05)
                    continue

            time.sleep(0.05)

        raise RuntimeError("Failed to start run after retries")

    def measure_reset_latency(self, seed="AAAAAAA", prefer_fast_restart=True, poll_interval=0.05):
        t0 = time.perf_counter()
        state, mode = self.start_run(seed=seed, prefer_fast_restart=prefer_fast_restart)
        if state.get("state") not in ACTIONABLE_STATES:
            state = self.wait_until_actionable(poll_interval=poll_interval)
        latency = time.perf_counter() - t0
        return latency, state, mode

    def _run_action_only_loop(self, steps, reset_latencies, reset_mode_counts):
        step_count = 0
        idle_count = 0
        max_idle = 400
        reset_time_total = 0.0
        action_time_total = 0.0
        rl_step_time_total = 0.0

        while step_count < steps:
            state = self.client.call("gamestate")
            action = self.random_action(state)

            if action is None:
                idle_count += 1
                if idle_count >= max_idle:
                    print(f"Stopped after {idle_count} idle polls at state={state.get('state')}")
                    break
                time.sleep(self.idle_sleep)
                continue

            method, params = action

            if method == "__wait__":
                idle_count += 1
                if idle_count >= max_idle:
                    print(f"Stopped after {idle_count} idle polls at state={state.get('state')}")
                    break
                self.record_wait_sleep(self.idle_sleep)
                time.sleep(self.idle_sleep)
                continue

            if method == "__reset__":
                idle_count = 0
                reset_latency, _, reset_mode = self.measure_reset_latency(
                    seed=self.seed,
                    prefer_fast_restart=self.prefer_fast_restart,
                    poll_interval=self.idle_sleep,
                )
                reset_latencies.append(reset_latency)
                reset_time_total += reset_latency
                reset_mode_counts[reset_mode] = reset_mode_counts.get(reset_mode, 0) + 1
                continue

            self.action_methods_used.add(method)
            t0 = time.perf_counter()
            try:
                self.client.call(method, params)
            except RuntimeError as exc:
                if is_transient_state_error(exc):
                    transient_sleep = self._transient_backoff(exc)
                    self.record_wait_sleep(transient_sleep)
                    time.sleep(transient_sleep)
                    continue
                if self._recover_transport_action(method, state, exc):
                    action_time_total += time.perf_counter() - t0
                    idle_count = 0
                    step_count += 1
                    continue
                if self._is_transport_error(exc):
                    time.sleep(min(self.idle_sleep, 0.05))
                    continue
                raise
            action_time_total += time.perf_counter() - t0
            idle_count = 0
            step_count += 1

        return step_count, reset_time_total, action_time_total, rl_step_time_total

    def _run_rl_step_loop(self, steps, reset_latencies, reset_mode_counts):
        step_count = 0
        idle_count = 0
        max_idle = 400
        reset_time_total = 0.0
        action_time_total = 0.0
        rl_step_time_total = 0.0
        obs_cache = None

        while step_count < steps:
            step_started = time.perf_counter()
            if obs_cache is not None:
                obs = obs_cache
                obs_cache = None
            else:
                obs = self.client.call("gamestate")

            phase = obs.get("state")
            if phase not in ACTIONABLE_STATES:
                if phase == "GAME_OVER":
                    reset_latency, obs, reset_mode = self.measure_reset_latency(
                        seed=self.seed,
                        prefer_fast_restart=self.prefer_fast_restart,
                        poll_interval=self.idle_sleep,
                    )
                    reset_latencies.append(reset_latency)
                    reset_time_total += reset_latency
                    reset_mode_counts[reset_mode] = reset_mode_counts.get(reset_mode, 0) + 1
                    obs_cache = obs
                    continue

                obs = self.wait_until_actionable(poll_interval=self.idle_sleep, stop_on_game_over=True)
                phase = obs.get("state")
                if phase == "GAME_OVER":
                    reset_latency, obs, reset_mode = self.measure_reset_latency(
                        seed=self.seed,
                        prefer_fast_restart=self.prefer_fast_restart,
                        poll_interval=self.idle_sleep,
                    )
                    reset_latencies.append(reset_latency)
                    reset_time_total += reset_latency
                    reset_mode_counts[reset_mode] = reset_mode_counts.get(reset_mode, 0) + 1
                    obs_cache = obs
                    continue

            action = self.random_action(obs)
            if action is None:
                idle_count += 1
                if idle_count >= max_idle:
                    print(f"Stopped after {idle_count} idle polls at state={obs.get('state')}")
                    break
                time.sleep(self.idle_sleep)
                continue

            method, params = action
            if method == "__wait__":
                idle_count += 1
                if idle_count >= max_idle:
                    print(f"Stopped after {idle_count} idle polls at state={obs.get('state')}")
                    break
                self.record_wait_sleep(self.idle_sleep)
                time.sleep(self.idle_sleep)
                continue

            if method == "__reset__":
                reset_latency, obs, reset_mode = self.measure_reset_latency(
                    seed=self.seed,
                    prefer_fast_restart=self.prefer_fast_restart,
                    poll_interval=self.idle_sleep,
                )
                reset_latencies.append(reset_latency)
                reset_time_total += reset_latency
                reset_mode_counts[reset_mode] = reset_mode_counts.get(reset_mode, 0) + 1
                obs_cache = obs
                continue

            self.action_methods_used.add(method)
            t0 = time.perf_counter()
            try:
                self.client.call(method, params)
            except RuntimeError as exc:
                if is_transient_state_error(exc):
                    transient_sleep = self._transient_backoff(exc)
                    self.record_wait_sleep(transient_sleep)
                    time.sleep(transient_sleep)
                    continue
                if self._recover_transport_action(method, obs, exc):
                    action_time_total += time.perf_counter() - t0
                    idle_count = 0
                    step_count += 1
                    next_obs = self.wait_until_actionable(poll_interval=self.idle_sleep, stop_on_game_over=True)
                    if next_obs.get("state") == "GAME_OVER":
                        reset_latency, next_obs, reset_mode = self.measure_reset_latency(
                            seed=self.seed,
                            prefer_fast_restart=self.prefer_fast_restart,
                            poll_interval=self.idle_sleep,
                        )
                        reset_latencies.append(reset_latency)
                        reset_time_total += reset_latency
                        reset_mode_counts[reset_mode] = reset_mode_counts.get(reset_mode, 0) + 1
                    obs_cache = next_obs
                    rl_step_time_total += time.perf_counter() - step_started
                    continue
                if self._is_transport_error(exc):
                    time.sleep(min(self.idle_sleep, 0.05))
                    continue
                raise
            action_time_total += time.perf_counter() - t0
            idle_count = 0
            step_count += 1

            next_obs = self.wait_until_actionable(poll_interval=self.idle_sleep, stop_on_game_over=True)
            if next_obs.get("state") == "GAME_OVER":
                reset_latency, next_obs, reset_mode = self.measure_reset_latency(
                    seed=self.seed,
                    prefer_fast_restart=self.prefer_fast_restart,
                    poll_interval=self.idle_sleep,
                )
                reset_latencies.append(reset_latency)
                reset_time_total += reset_latency
                reset_mode_counts[reset_mode] = reset_mode_counts.get(reset_mode, 0) + 1

            obs_cache = next_obs
            rl_step_time_total += time.perf_counter() - step_started

        return step_count, reset_time_total, action_time_total, rl_step_time_total

    def run(self, steps):
        self.start_run(seed=self.seed, prefer_fast_restart=self.prefer_fast_restart)
        self.reset_runtime_stats()

        benchmark_start = time.perf_counter()
        reset_latencies = []
        reset_mode_counts = {"fast_start": 0, "menu_start": 0}

        if self.mode == "rl_step":
            step_count, reset_time_total, action_time_total, rl_step_time_total = self._run_rl_step_loop(
                steps,
                reset_latencies,
                reset_mode_counts,
            )
        else:
            step_count, reset_time_total, action_time_total, rl_step_time_total = self._run_action_only_loop(
                steps,
                reset_latencies,
                reset_mode_counts,
            )

        total_wall = time.perf_counter() - benchmark_start
        step_wall = max(total_wall - reset_time_total, 1e-9)

        return {
            "success": True,
            "steps": step_count,
            "total_wall": total_wall,
            "step_wall": step_wall,
            "action_time_total": action_time_total,
            "rl_step_time_total": rl_step_time_total,
            "reset_time_total": reset_time_total,
            "reset_latencies": reset_latencies,
            "reset_mode_counts": reset_mode_counts,
            "lat_by_method": {k: list(v) for k, v in self.client.lat_by_method.items()},
            "action_methods": sorted(self.action_methods_used),
            "wait_sleep_durations": list(self.wait_sleep_durations),
            "wait_sleep_count": self.wait_sleep_count,
            "wait_sleep_total_sec": self.wait_sleep_total_sec,
            "mode": self.mode,
            "url": self.client.url,
            "seed": self.seed,
        }


def collect_action_latencies(lat_by_method, action_methods):
    values = []
    for method in action_methods:
        values.extend(lat_by_method.get(method, []))
    return values


def print_main_output(stats, prefer_fast_restart=True, mode="action_only"):
    step_count = stats.get("steps", 0)
    total_wall = stats.get("total_wall", 0.0)
    step_wall = max(stats.get("step_wall", 0.0), 1e-9)
    action_time = max(stats.get("action_time_total", 0.0), 1e-9)
    action_only_tps = step_count / step_wall

    print(f"Steps: {step_count}")
    print(f"Benchmark wall time: {total_wall:.4f} sec")
    print(f"Step wall time (reset excluded): {step_wall:.4f} sec")
    print(f"Step throughput (wall, reset excluded): {action_only_tps:.2f} steps/sec")
    print(f"Step throughput (action RPC only): {step_count / action_time:.2f} steps/sec")

    reset_latencies = stats.get("reset_latencies", [])
    reset_mode_counts = stats.get("reset_mode_counts", {})
    if reset_latencies:
        mean_reset = sum(reset_latencies) / len(reset_latencies)
        print(f"Resets: {len(reset_latencies)}")
        print(
            "Reset mode counts: "
            f"fast_start={reset_mode_counts.get('fast_start', 0)}, "
            f"menu_start={reset_mode_counts.get('menu_start', 0)}"
        )
        if prefer_fast_restart and reset_mode_counts.get("fast_start", 0) == 0:
            print("Fast restart unsupported by current balatrobot process; fallback menu+start was used.")
        print(f"Reset latency avg: {mean_reset:.4f} sec")
        print(f"Reset latency p50: {percentile(reset_latencies, 0.50):.4f} sec")
        print(f"Reset latency p95: {percentile(reset_latencies, 0.95):.4f} sec")
        print(f"Reset latency max: {max(reset_latencies):.4f} sec")
    else:
        print("Resets: 0 (no reset observed)")

    if mode == "rl_step":
        rl_step_denom = max(stats.get("rl_step_time_total", 0.0), 1e-9)
        print(f"Steps/sec (rl_step): {step_count / rl_step_denom:.2f}")
        print(f"Steps/sec (action_only): {action_only_tps:.2f}")


def print_extra_stats(lat_by_method, action_methods, wait_sleep_durations, wait_sleep_count, wait_sleep_total_sec):
    print("Extra stats:")

    gamestate_stats = summarize(lat_by_method.get("gamestate", []))
    print(
        "gamestate latency: "
        f"count={gamestate_stats['count']}, "
        f"avg={gamestate_stats['avg'] * 1000:.3f}ms, "
        f"p50={gamestate_stats['p50'] * 1000:.3f}ms, "
        f"p95={gamestate_stats['p95'] * 1000:.3f}ms, "
        f"max={gamestate_stats['max'] * 1000:.3f}ms"
    )

    action_latencies = collect_action_latencies(lat_by_method, action_methods)
    action_stats = summarize(action_latencies)
    print(
        "action RPC latency: "
        f"count={action_stats['count']}, "
        f"avg={action_stats['avg'] * 1000:.3f}ms, "
        f"p50={action_stats['p50'] * 1000:.3f}ms, "
        f"p95={action_stats['p95'] * 1000:.3f}ms, "
        f"max={action_stats['max'] * 1000:.3f}ms"
    )

    wait_stats = summarize(wait_sleep_durations)
    print(
        "__wait__ sleep: "
        f"count={wait_sleep_count}, "
        f"total={wait_sleep_total_sec:.4f}s, "
        f"avg={wait_stats['avg']:.4f}s, "
        f"p50={wait_stats['p50']:.4f}s, "
        f"p95={wait_stats['p95']:.4f}s"
    )

    print("RPC latency by method:")
    for method in sorted(lat_by_method):
        stats = summarize(lat_by_method[method])
        print(
            f"  {method}: "
            f"count={stats['count']}, "
            f"avg={stats['avg'] * 1000:.3f}ms, "
            f"p50={stats['p50'] * 1000:.3f}ms, "
            f"p95={stats['p95'] * 1000:.3f}ms, "
            f"max={stats['max'] * 1000:.3f}ms"
        )


def benchmark(steps=100, idle_sleep=0.05, prefer_fast_restart=True, mode="action_only", url=DEFAULT_URL, seed="AAAAAAA"):
    global DEFAULT_CLIENT
    client = RPCClient(url=url, json_backend=JSON_BACKEND, orjson_module=ORJSON)
    DEFAULT_CLIENT = client
    try:
        runner = BenchmarkRunner(
            client=client,
            idle_sleep=idle_sleep,
            prefer_fast_restart=prefer_fast_restart,
            mode=mode,
            seed=seed,
        )
        stats = runner.run(steps)
        print_main_output(stats, prefer_fast_restart=prefer_fast_restart, mode=mode)
        print_extra_stats(
            lat_by_method=stats["lat_by_method"],
            action_methods=stats["action_methods"],
            wait_sleep_durations=stats["wait_sleep_durations"],
            wait_sleep_count=stats["wait_sleep_count"],
            wait_sleep_total_sec=stats["wait_sleep_total_sec"],
        )
        return stats
    finally:
        DEFAULT_CLIENT = None
        client.close()


def normalize_base_url(base_url):
    candidate = (base_url or "").strip().rstrip("/")
    parsed = urlparse(candidate)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError("--base-url must start with http:// or https://")
    if not parsed.hostname:
        raise ValueError("--base-url must include a host")
    if parsed.path not in {"", "/"} or parsed.query or parsed.fragment or parsed.params:
        raise ValueError("--base-url must be host-only (no path/query/fragment)")
    if parsed.port is not None:
        raise ValueError("--base-url must not include a port; use --ports to specify ports")

    host = parsed.hostname
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"
    return f"{parsed.scheme}://{host}"


def parse_ports(ports_arg, instances):
    if not ports_arg:
        return [DEFAULT_START_PORT + i for i in range(instances)]

    parts = [piece.strip() for piece in ports_arg.split(",") if piece.strip()]
    if len(parts) != instances:
        raise ValueError(f"--ports count ({len(parts)}) must equal --instances ({instances})")

    ports = []
    for piece in parts:
        try:
            port = int(piece)
        except ValueError as exc:
            raise ValueError(f"Invalid port value: {piece}") from exc
        if port <= 0 or port > 65535:
            raise ValueError(f"Invalid port value: {port}")
        ports.append(port)
    return ports


def build_urls(base_url, ports):
    return [f"{base_url}:{port}" for port in ports]


def parse_cpu_affinity(cpu_affinity_arg, instances):
    if cpu_affinity_arg is None:
        return None
    parts = [piece.strip() for piece in cpu_affinity_arg.split(",") if piece.strip()]
    if len(parts) != instances:
        raise ValueError(f"--cpu-affinity count ({len(parts)}) must equal --instances ({instances})")
    cpus = []
    for piece in parts:
        try:
            cpu_id = int(piece)
        except ValueError as exc:
            raise ValueError(f"Invalid CPU id: {piece}") from exc
        if cpu_id < 0:
            raise ValueError(f"Invalid CPU id: {cpu_id}")
        cpus.append(cpu_id)
    return cpus


def apply_cpu_affinity(cpu_id):
    if cpu_id is None:
        return None
    try:
        import psutil
    except Exception:
        return "psutil not installed; ignoring --cpu-affinity for this worker."
    try:
        psutil.Process(os.getpid()).cpu_affinity([cpu_id])
    except Exception as exc:
        return f"Failed to set CPU affinity to core {cpu_id}: {exc}"
    return None


def build_worker_seed(seed_base, worker_id, instances):
    return f"{seed_base}-{worker_id}"


def worker_main(config, result_queue):
    ensure_project_root_cwd()
    warnings = []
    client = None
    try:
        affinity_warning = apply_cpu_affinity(config.get("cpu_affinity"))
        if affinity_warning:
            warnings.append(affinity_warning)

        backend = config.get("json_backend", "stdlib")
        orjson_module = None
        if backend == "orjson":
            orjson_module = _try_import_orjson()
            if orjson_module is None:
                warnings.append("orjson unavailable in worker; falling back to stdlib backend.")
                backend = "stdlib"

        client = RPCClient(config["url"], json_backend=backend, orjson_module=orjson_module)
        runner = BenchmarkRunner(
            client=client,
            idle_sleep=config["idle_sleep"],
            prefer_fast_restart=config["prefer_fast_restart"],
            mode=config["mode"],
            seed=config["seed"],
        )
        stats = runner.run(config["steps_per_instance"])
        stats.update(
            {
                "success": True,
                "error": "",
                "traceback": "",
                "worker_id": config["worker_id"],
                "url": config["url"],
                "warnings": warnings,
                "json_backend_used": backend,
            }
        )
        result_queue.put(stats)
    except Exception as exc:
        result_queue.put(
            {
                "success": False,
                "error": str(exc),
                "traceback": traceback.format_exc(),
                "worker_id": config.get("worker_id", -1),
                "url": config.get("url", ""),
                "warnings": warnings,
                "steps": 0,
                "total_wall": 0.0,
                "step_wall": 0.0,
                "action_time_total": 0.0,
                "rl_step_time_total": 0.0,
                "reset_time_total": 0.0,
                "reset_latencies": [],
                "reset_mode_counts": {"fast_start": 0, "menu_start": 0},
                "lat_by_method": {},
                "action_methods": [],
                "wait_sleep_durations": [],
                "wait_sleep_count": 0,
                "wait_sleep_total_sec": 0.0,
                "mode": config.get("mode", "action_only"),
                "seed": config.get("seed", ""),
            }
        )
    finally:
        if client is not None:
            client.close()


def supports_serve_subcommand(cmd_prefix):
    cache_key = tuple(cmd_prefix)
    if cache_key in SERVE_SUBCOMMAND_SUPPORT_CACHE:
        return SERVE_SUBCOMMAND_SUPPORT_CACHE[cache_key]

    supports = False
    try:
        result = subprocess.run(
            cmd_prefix + ["serve", "--help"],
            capture_output=True,
            text=True,
            timeout=3,
            check=False,
        )
        supports = result.returncode == 0
    except Exception:
        supports = False

    SERVE_SUBCOMMAND_SUPPORT_CACHE[cache_key] = supports
    return supports


def build_launcher_cmd(args, port):
    if args.launcher == "uvx":
        cmd_prefix = [args.uvx_path, "balatrobot"]
    else:
        cmd_prefix = [args.balatrobot_cmd]

    cmd = cmd_prefix + (["serve"] if supports_serve_subcommand(cmd_prefix) else [])

    cmd.extend(
        [
            "--headless",
            "--fast",
            "--port",
            str(port),
            "--love-path",
            args.love_path,
            "--lovely-path",
            args.lovely_path,
        ]
    )
    return cmd


def _resolve_executable_path(candidate):
    resolved = shutil.which(candidate)
    if resolved:
        return resolved
    candidate_path = Path(candidate)
    if candidate_path.exists():
        return str(candidate_path)
    return None


def resolve_launcher_executable(args):
    if args.launcher == "uvx":
        key = "--uvx-path"
        value = args.uvx_path
    else:
        key = "--balatrobot-cmd"
        value = args.balatrobot_cmd

    resolved = _resolve_executable_path(value)
    if resolved:
        return resolved

    raise RuntimeError(
        f"Launcher '{args.launcher}' executable not found: {key}={value!r}. "
        "Check PATH or pass full executable path."
    )


def list_process_pids_by_image(image_name):
    cmd = ["tasklist", "/FO", "CSV", "/NH", "/FI", f"IMAGENAME eq {image_name}"]
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="ignore",
            check=False,
        )
    except Exception:
        return set()
    if result.returncode != 0 or result.stdout is None:
        return set()

    pids = set()
    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    for line in lines:
        if line.upper().startswith("INFO:"):
            continue
        # CSV row format: "Image Name","PID","Session Name","Session#","Mem Usage"
        parts = [part.strip().strip('"') for part in line.split('","')]
        if len(parts) < 2:
            continue
        try:
            pids.add(int(parts[1].strip('"')))
        except ValueError:
            continue
    return pids


def get_pid_listening_on_port(port):
    if os.name != "nt":
        return None
    cmd = ["netstat", "-ano", "-p", "TCP"]
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="ignore",
            check=False,
            timeout=10,
        )
    except Exception:
        return None
    if result.returncode != 0 or result.stdout is None:
        return None

    target = f":{port} "
    for line in result.stdout.splitlines():
        if "LISTENING" in line and target in line:
            parts = line.split()
            if len(parts) >= 5:
                try:
                    return int(parts[-1])
                except ValueError:
                    continue
    return None


def kill_process_by_pid(pid):
    if pid is None or pid <= 0:
        return False
    if os.name == "nt":
        try:
            result = subprocess.run(
                ["taskkill", "/PID", str(pid), "/T", "/F"],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="ignore",
                check=False,
                timeout=5,
            )
            return result.returncode == 0
        except Exception:
            return False
    return False


def ensure_ports_available(ports, game_image_name="Balatro.exe"):
    unavailable = []
    for port in ports:
        pid = get_pid_listening_on_port(port)
        if pid is not None:
            unavailable.append((port, pid))

    if not unavailable:
        return True

    print(f"WARNING: Found {len(unavailable)} port(s) already in use, attempting cleanup...")
    for port, pid in unavailable:
        print(f"  Port {port} is used by PID {pid}, terminating...")
        if kill_process_by_pid(pid):
            print(f"  Terminated PID {pid}")
        else:
            print(f"  Failed to terminate PID {pid}")

    time.sleep(0.5)

    still_unavailable = []
    for port, _ in unavailable:
        pid = get_pid_listening_on_port(port)
        if pid is not None:
            still_unavailable.append((port, pid))

    if still_unavailable:
        port_list = ", ".join(f"{p}:{pid}" for p, pid in still_unavailable)
        raise RuntimeError(
            f"Ports still in use after cleanup attempt: {port_list}. "
            "Please manually close the conflicting processes and try again."
        )

    return True


def _default_mod_source_dir():
    env_mod_dir = os.environ.get("LOVELY_MOD_DIR")
    if env_mod_dir:
        return Path(env_mod_dir)

    appdata = os.environ.get("APPDATA")
    if appdata:
        return Path(appdata) / "Balatro" / "Mods"

    return Path.home() / "AppData" / "Roaming" / "Balatro" / "Mods"


def _ignore_mod_runtime_artifacts(dir_path, names):
    dir_path = Path(dir_path)
    parent_name = dir_path.parent.name.lower() if dir_path.parent else ""
    if parent_name != "mods" or dir_path.name.lower() != "lovely":
        return []
    ignored = []
    for name in ("dump", "game-dump", "log"):
        if name in names:
            ignored.append(name)
    return ignored


def prepare_isolated_mod_dir(port):
    source_mod_dir = _default_mod_source_dir()
    if not source_mod_dir.exists():
        raise RuntimeError(
            f"Cannot locate source mod directory: {source_mod_dir}. "
            "Install/check mods first or set LOVELY_MOD_DIR."
        )

    runtime_mod_root = PROJECT_ROOT / "runtime" / "lovely_mods"
    runtime_mod_root.mkdir(parents=True, exist_ok=True)
    target_mod_dir = runtime_mod_root / f"port_{port}"
    try:
        if source_mod_dir.resolve() == target_mod_dir.resolve():
            return target_mod_dir
    except Exception:
        pass

    if target_mod_dir.exists():
        shutil.rmtree(target_mod_dir)

    try:
        shutil.copytree(
            source_mod_dir,
            target_mod_dir,
            ignore=_ignore_mod_runtime_artifacts,
        )
    except Exception as exc:
        raise RuntimeError(
            f"Failed to prepare isolated mod dir for port {port}: "
            f"source={source_mod_dir}, target={target_mod_dir}, error={exc}"
        ) from exc

    for runtime_dir in ("dump", "game-dump", "log"):
        shutil.rmtree(target_mod_dir / "lovely" / runtime_dir, ignore_errors=True)

    lovely_dir = target_mod_dir / "lovely"
    if lovely_dir.exists():
        for json_file in lovely_dir.rglob("*.lua.json"):
            try:
                json_file.unlink()
            except Exception:
                pass

    return target_mod_dir


def launch_serve_instances(args, ports):
    game_image_name = Path(args.love_path).name if args.love_path else "Balatro.exe"
    ensure_ports_available(ports, game_image_name)
    preexisting_game_pids = list_process_pids_by_image(game_image_name)
    logs_root = PROJECT_ROOT / "logs"
    logs_root.mkdir(parents=True, exist_ok=True)
    processes = []
    for index, port in enumerate(ports):
        cmd = build_launcher_cmd(args, port)
        isolated_mod_dir = prepare_isolated_mod_dir(port)
        launch_env = os.environ.copy()
        launch_env["BALATROBOT_LOGS_PATH"] = str(logs_root)
        launch_env["LOVELY_MOD_DIR"] = str(isolated_mod_dir)
        try:
            popen_kwargs = dict(
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                cwd=str(PROJECT_ROOT),
                env=launch_env,
            )
            if os.name == "nt":
                popen_kwargs["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0)

            proc = subprocess.Popen(
                cmd,
                **popen_kwargs,
            )
            proc._benchmark_port = port
            time.sleep(0.2)
            returncode = proc.poll()
            if returncode is not None:
                cmd_text = subprocess.list2cmdline(cmd)
                raise RuntimeError(
                    f"process exited early on port {port}; cmd={cmd_text}; returncode={returncode}. "
                    "Please copy this command and run it manually in terminal to inspect startup errors."
                )
        except Exception as exc:
            cmd_text = subprocess.list2cmdline(cmd)
            raise RuntimeError(
                f"failed to start instance on port {port}; cmd={cmd_text}; error={exc}. "
                "Check PATH or pass full executable path via --uvx-path/--balatrobot-cmd."
            ) from exc
        processes.append(proc)
        if args.stagger_start > 0 and index < len(ports) - 1:
            time.sleep(args.stagger_start)
    cleanup_ctx = {
        "game_image_name": game_image_name,
        "preexisting_game_pids": preexisting_game_pids,
    }
    return processes, cleanup_ctx


def terminate_processes(processes, cleanup_ctx=None):
    if not processes:
        return

    ports_to_clean = []
    for proc in processes:
        port = getattr(proc, "_benchmark_port", None)
        if port:
            ports_to_clean.append(port)

    if os.name == "nt":
        for proc in processes:
            pid = getattr(proc, "pid", None)
            if not pid:
                continue
            subprocess.run(
                ["taskkill", "/PID", str(pid), "/T", "/F"],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="ignore",
                check=False,
            )

    for proc in processes:
        try:
            if proc.poll() is None:
                proc.terminate()
        except Exception:
            pass
    deadline = time.time() + 5.0
    for proc in processes:
        while proc.poll() is None and time.time() < deadline:
            time.sleep(0.05)
        if proc.poll() is None:
            try:
                proc.kill()
            except Exception:
                pass
    for proc in processes:
        try:
            proc.wait(timeout=1.0)
        except Exception:
            pass

    if cleanup_ctx:
        game_image_name = cleanup_ctx.get("game_image_name")
        preexisting_game_pids = cleanup_ctx.get("preexisting_game_pids", set())
        if game_image_name:
            current_pids = list_process_pids_by_image(game_image_name)
            leaked_pids = sorted(pid for pid in current_pids if pid not in preexisting_game_pids)
            for pid in leaked_pids:
                subprocess.run(
                    ["taskkill", "/PID", str(pid), "/T", "/F"],
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="ignore",
                    check=False,
                )

    for port in ports_to_clean:
        pid = get_pid_listening_on_port(port)
        if pid is not None:
            kill_process_by_pid(pid)


def _is_instance_healthy(client):
    try:
        client.call("health", timeout=2.0)
        return True
    except RuntimeError as exc:
        msg = str(exc).lower()
        if "method not found" in msg or "unknown" in msg:
            try:
                client.call("gamestate", timeout=2.0)
                return True
            except Exception:
                return False
        return False
    except Exception:
        return False


def wait_for_instances_health(urls, timeout_sec=60.0, poll_sec=0.3):
    clients = [RPCClient(url, json_backend="stdlib", orjson_module=None) for url in urls]
    healthy = {url: False for url in urls}
    deadline = time.perf_counter() + timeout_sec
    try:
        while time.perf_counter() < deadline:
            for client in clients:
                if healthy[client.url]:
                    continue
                healthy[client.url] = _is_instance_healthy(client)
            if all(healthy.values()):
                return True, []
            time.sleep(poll_sec)
        unhealthy = [url for url, ok in healthy.items() if not ok]
        return False, unhealthy
    finally:
        for client in clients:
            client.close()


def extract_port_from_url(url):
    try:
        parsed = urlparse(url)
        return parsed.port
    except Exception:
        return None


def find_latest_port_log(port):
    logs_root = PROJECT_ROOT / "logs"
    if not logs_root.exists():
        return None

    candidates = []
    target = f"{port}.log"
    try:
        for run_dir in logs_root.iterdir():
            if not run_dir.is_dir():
                continue
            candidate = run_dir / target
            if candidate.exists():
                candidates.append(candidate)
    except Exception:
        return None

    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def read_log_tail(path, max_lines=80):
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()
        return [line.rstrip() for line in lines[-max_lines:]]
    except Exception:
        return []


def extract_crash_summary(lines):
    if not lines:
        return ""
    markers = ("Oops! The game crashed", "screenwipe", "attempt to index field", "StackTrace")
    matched = [line.strip() for line in lines if any(marker.lower() in line.lower() for marker in markers)]
    if matched:
        return " | ".join(matched[:4])
    return ""


def build_failure_diagnostic(result, serve_returncodes_by_port=None):
    url = result.get("url", "")
    port = extract_port_from_url(url)
    if port is None:
        return None
    returncode = None
    if serve_returncodes_by_port is not None:
        returncode = serve_returncodes_by_port.get(port)
    log_path = find_latest_port_log(port)
    if log_path is None:
        return {"port": port, "log_path": None, "summary": "", "returncode": returncode}
    tail_lines = read_log_tail(log_path)
    return {
        "port": port,
        "log_path": str(log_path),
        "summary": extract_crash_summary(tail_lines),
        "returncode": returncode,
    }


def merge_latency_maps(results):
    lat_by_method = defaultdict(list)
    for result in results:
        for method, values in result.get("lat_by_method", {}).items():
            lat_by_method[method].extend(values)
    return lat_by_method


def aggregate_success_results(success_results, wall_total, mode):
    reset_mode_counts = {"fast_start": 0, "menu_start": 0}
    reset_latencies = []
    action_methods = set()
    wait_sleep_durations = []
    wait_sleep_count = 0
    wait_sleep_total_sec = 0.0

    total_steps = 0
    total_step_wall = 0.0
    total_action_time = 0.0
    total_rl_step_time = 0.0

    for result in success_results:
        total_steps += result.get("steps", 0)
        total_step_wall += result.get("step_wall", 0.0)
        total_action_time += result.get("action_time_total", 0.0)
        total_rl_step_time += result.get("rl_step_time_total", 0.0)
        reset_latencies.extend(result.get("reset_latencies", []))
        reset_mode_counts["fast_start"] += result.get("reset_mode_counts", {}).get("fast_start", 0)
        reset_mode_counts["menu_start"] += result.get("reset_mode_counts", {}).get("menu_start", 0)
        action_methods.update(result.get("action_methods", []))
        wait_sleep_durations.extend(result.get("wait_sleep_durations", []))
        wait_sleep_count += result.get("wait_sleep_count", 0)
        wait_sleep_total_sec += result.get("wait_sleep_total_sec", 0.0)

    lat_by_method = merge_latency_maps(success_results)
    if total_step_wall <= 0:
        total_step_wall = 1e-9

    return {
        "success": True,
        "steps": total_steps,
        "total_wall": wall_total,
        "step_wall": total_step_wall,
        "action_time_total": total_action_time,
        "rl_step_time_total": total_rl_step_time,
        "reset_time_total": 0.0,
        "reset_latencies": reset_latencies,
        "reset_mode_counts": reset_mode_counts,
        "lat_by_method": {k: list(v) for k, v in lat_by_method.items()},
        "action_methods": sorted(action_methods),
        "wait_sleep_durations": wait_sleep_durations,
        "wait_sleep_count": wait_sleep_count,
        "wait_sleep_total_sec": wait_sleep_total_sec,
        "mode": mode,
    }


def make_worker_failure(worker_id, url, mode, error):
    return {
        "success": False,
        "error": error,
        "traceback": "",
        "worker_id": worker_id,
        "url": url,
        "warnings": [],
        "steps": 0,
        "total_wall": 0.0,
        "step_wall": 0.0,
        "action_time_total": 0.0,
        "rl_step_time_total": 0.0,
        "reset_time_total": 0.0,
        "reset_latencies": [],
        "reset_mode_counts": {"fast_start": 0, "menu_start": 0},
        "lat_by_method": {},
        "action_methods": [],
        "wait_sleep_durations": [],
        "wait_sleep_count": 0,
        "wait_sleep_total_sec": 0.0,
        "mode": mode,
    }


def run_multi_instance(configs, stagger_start, fail_fast_on_error=True):
    ctx = mp.get_context("spawn")
    result_queue = ctx.Queue()
    process_by_worker = {}
    config_by_worker = {cfg["worker_id"]: cfg for cfg in configs}

    wall_start = time.perf_counter()
    for idx, config in enumerate(configs):
        proc = ctx.Process(target=worker_main, args=(config, result_queue))
        proc.start()
        process_by_worker[config["worker_id"]] = proc
        if stagger_start > 0 and idx < len(configs) - 1:
            time.sleep(stagger_start)

    results_by_worker = {}
    abort_run = False
    while len(results_by_worker) < len(configs):
        try:
            result = result_queue.get(timeout=1.0)
            results_by_worker[result["worker_id"]] = result
            if fail_fast_on_error and not result.get("success", False):
                abort_run = True
                break
        except queue.Empty:
            all_exited = all(proc.exitcode is not None for proc in process_by_worker.values())
            if all_exited:
                break
            if fail_fast_on_error:
                for worker_id, proc in process_by_worker.items():
                    if worker_id in results_by_worker:
                        continue
                    if proc.exitcode not in (None, 0):
                        cfg = config_by_worker[worker_id]
                        results_by_worker[worker_id] = make_worker_failure(
                            worker_id=worker_id,
                            url=cfg["url"],
                            mode=configs[0]["mode"],
                            error=f"Worker exited before reporting result (exitcode={proc.exitcode})",
                        )
                        abort_run = True
                        break
                if abort_run:
                    break
            continue

    if abort_run:
        for worker_id, proc in process_by_worker.items():
            if worker_id in results_by_worker:
                continue
            if proc.is_alive():
                proc.terminate()

    for worker_id, proc in process_by_worker.items():
        proc.join(timeout=1.0)
        if worker_id not in results_by_worker and proc.exitcode is not None:
            cfg = config_by_worker[worker_id]
            results_by_worker[worker_id] = make_worker_failure(
                worker_id=worker_id,
                url=cfg["url"],
                mode=configs[0]["mode"],
                error=f"Worker exited before reporting result (exitcode={proc.exitcode})",
            )

    for worker_id, proc in process_by_worker.items():
        if proc.is_alive():
            proc.terminate()
            proc.join(timeout=2.0)
        if worker_id not in results_by_worker:
            cfg = config_by_worker[worker_id]
            error = "Worker timed out or crashed without result"
            if abort_run:
                error = "Aborted after another worker failed"
            results_by_worker[worker_id] = make_worker_failure(
                worker_id=worker_id,
                url=cfg["url"],
                mode=configs[0]["mode"],
                error=error,
            )

    wall_total = time.perf_counter() - wall_start
    ordered_results = [results_by_worker[cfg["worker_id"]] for cfg in configs]
    return ordered_results, wall_total


def print_scaling_summary(instances, worker_results, aggregate_stats):
    steps_total = aggregate_stats.get("steps", 0)
    wall_total = max(aggregate_stats.get("total_wall", 0.0), 1e-9)
    print(
        f"Scaling summary: instances={instances}, "
        f"steps_total={steps_total}, steps/s_total={steps_total / wall_total:.2f}"
    )
    print("Per instance:")
    for result in worker_results:
        worker_id = result.get("worker_id", -1)
        url = result.get("url", "")
        if not result.get("success", False):
            print(
                f"  i={worker_id}, url={url}, steps=0, steps/s=0.00, "
                f"action_avg_ms=0.000, gamestate_avg_ms=0.000, error={result.get('error', 'unknown')}"
            )
            continue

        steps = result.get("steps", 0)
        wall = max(result.get("total_wall", 0.0), 1e-9)
        lat_by_method = result.get("lat_by_method", {})
        action_methods = result.get("action_methods", [])
        action_avg_ms = summarize(collect_action_latencies(lat_by_method, action_methods))["avg"] * 1000.0
        gamestate_avg_ms = summarize(lat_by_method.get("gamestate", []))["avg"] * 1000.0
        print(
            f"  i={worker_id}, url={url}, steps={steps}, steps/s={steps / wall:.2f}, "
            f"action_avg_ms={action_avg_ms:.3f}, gamestate_avg_ms={gamestate_avg_ms:.3f}"
        )

    print("Aggregate RPC latency by method:")
    lat_by_method = aggregate_stats.get("lat_by_method", {})
    for method in sorted(lat_by_method):
        stats = summarize(lat_by_method[method])
        print(
            f"  {method}: "
            f"count={stats['count']}, "
            f"avg={stats['avg'] * 1000:.3f}ms, "
            f"p50={stats['p50'] * 1000:.3f}ms, "
            f"p95={stats['p95'] * 1000:.3f}ms, "
            f"max={stats['max'] * 1000:.3f}ms"
        )


def resolve_steps_per_instance(args):
    argv = sys.argv[1:]
    steps_set = any(token == "--steps" or token.startswith("--steps=") for token in argv)
    spi_set = any(token == "--steps-per-instance" or token.startswith("--steps-per-instance=") for token in argv)
    if spi_set and steps_set:
        print("Both --steps and --steps-per-instance were provided; using --steps-per-instance.")
        return args.steps_per_instance
    if spi_set:
        return args.steps_per_instance
    if steps_set:
        return args.steps
    return args.steps_per_instance


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark BalatroBot step throughput and reset latency.")
    parser.add_argument("--steps", type=int, default=100, help="Number of in-run steps to execute.")
    parser.add_argument(
        "--steps-per-instance",
        type=int,
        default=100,
        help="Per-instance step target (overrides --steps if both are set).",
    )
    parser.add_argument("--idle-sleep", type=float, default=0.05, help="Sleep time between idle polls.")
    parser.add_argument(
        "--no-fast-restart",
        action="store_true",
        help="Disable GAME_OVER direct start() fast path and always use menu+start.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["action_only", "rl_step"],
        default="action_only",
        help="Benchmark mode. action_only keeps legacy semantics; rl_step uses RL step semantics.",
    )
    parser.add_argument(
        "--json-backend",
        type=str,
        choices=["stdlib", "orjson"],
        default="stdlib",
        help="JSON backend for requests payload/response handling.",
    )
    parser.add_argument("--instances", type=int, default=1, help="Number of instances to benchmark in parallel.")
    parser.add_argument("--base-url", type=str, default=DEFAULT_BASE_URL, help="Base URL without port.")
    parser.add_argument("--ports", type=str, default=None, help="Comma-separated list of ports.")
    parser.add_argument("--seed-base", type=str, default="AAAAAAA", help="Base seed for runs.")
    parser.add_argument(
        "--stagger-start",
        type=float,
        default=0.0,
        help="Stagger seconds between starting serve processes and worker processes.",
    )
    parser.add_argument(
        "--cpu-affinity",
        type=str,
        default=None,
        help="Comma-separated CPU IDs, one per worker (requires psutil).",
    )
    parser.add_argument(
        "--launch-instances",
        action="store_true",
        help="Auto-launch balatrobot instances before running benchmark.",
    )
    parser.add_argument(
        "--launcher",
        type=str,
        choices=["uvx", "direct"],
        default="direct",
        help="Launcher backend for auto-launch mode.",
    )
    parser.add_argument(
        "--balatrobot-cmd",
        type=str,
        default="balatrobot",
        help="balatrobot executable/command (used when --launcher direct).",
    )
    parser.add_argument(
        "--uvx-path",
        type=str,
        default="uvx",
        help="uvx executable path (used when --launcher uvx).",
    )
    parser.add_argument("--love-path", type=str, default=None, help="Balatro executable path (required when launching).")
    parser.add_argument(
        "--lovely-path",
        type=str,
        default=None,
        help="Lovely DLL path (required when launching).",
    )
    return parser.parse_args()


def main():
    ensure_project_root_cwd()
    args = parse_args()

    if args.instances <= 0:
        print("ERROR: --instances must be >= 1")
        return 2
    if args.idle_sleep <= 0:
        print("ERROR: --idle-sleep must be > 0")
        return 2
    if args.stagger_start < 0:
        print("ERROR: --stagger-start must be >= 0")
        return 2

    steps_per_instance = resolve_steps_per_instance(args)
    if steps_per_instance <= 0:
        print("ERROR: step count must be > 0")
        return 2

    try:
        base_url = normalize_base_url(args.base_url)
        ports = parse_ports(args.ports, args.instances)
        urls = build_urls(base_url, ports)
        cpu_affinity = parse_cpu_affinity(args.cpu_affinity, args.instances)
    except ValueError as exc:
        print(f"ERROR: {exc}")
        return 2

    configure_json_backend(args.json_backend)

    serve_processes = []
    launch_cleanup_ctx = None
    serve_returncodes_by_port = {}
    try:
        if args.launch_instances:
            if not args.love_path or not args.lovely_path:
                print("ERROR: --love-path and --lovely-path are required when --launch-instances is enabled.")
                return 2
            try:
                resolve_launcher_executable(args)
            except RuntimeError as exc:
                print(f"ERROR: {exc}")
                return 1
            try:
                serve_processes, launch_cleanup_ctx = launch_serve_instances(args, ports)
            except Exception as exc:
                print(f"ERROR: failed to launch balatrobot instances: {exc}")
                return 1

            healthy, unhealthy = wait_for_instances_health(urls)
            if not healthy:
                print(f"ERROR: instances failed health check: {', '.join(unhealthy)}")
                return 1

        if args.instances == 1:
            if cpu_affinity is not None:
                warning = apply_cpu_affinity(cpu_affinity[0])
                if warning:
                    print(f"WARNING: {warning}")
            seed = build_worker_seed(args.seed_base, 0, 1)
            benchmark(
                steps=steps_per_instance,
                idle_sleep=args.idle_sleep,
                prefer_fast_restart=not args.no_fast_restart,
                mode=args.mode,
                url=urls[0],
                seed=seed,
            )
            return 0

        worker_configs = []
        for index, url in enumerate(urls):
            worker_configs.append(
                {
                    "worker_id": index,
                    "url": url,
                    "mode": args.mode,
                    "steps_per_instance": steps_per_instance,
                    "seed": build_worker_seed(args.seed_base, index, args.instances),
                    "prefer_fast_restart": not args.no_fast_restart,
                    "idle_sleep": args.idle_sleep,
                    "json_backend": JSON_BACKEND,
                    "cpu_affinity": None if cpu_affinity is None else cpu_affinity[index],
                }
            )

        worker_results, wall_total = run_multi_instance(worker_configs, args.stagger_start)
        for proc in serve_processes:
            port = getattr(proc, "_benchmark_port", None)
            if port is not None:
                serve_returncodes_by_port[port] = proc.poll()

        for result in worker_results:
            for warning in result.get("warnings", []):
                print(f"WARNING(worker {result.get('worker_id', -1)}): {warning}")

        success_results = [result for result in worker_results if result.get("success")]
        failed_results = [result for result in worker_results if not result.get("success")]

        if not success_results:
            print("ERROR: all workers failed; no benchmark stats available.")
            for result in failed_results:
                print(f"Worker {result.get('worker_id', -1)} ({result.get('url', '')}) failed: {result.get('error')}")
                diag = build_failure_diagnostic(result, serve_returncodes_by_port=serve_returncodes_by_port)
                if diag:
                    print(
                        f"  port={diag.get('port')}, returncode={diag.get('returncode')}, "
                        f"log={diag.get('log_path') or 'not found under PROJECT_ROOT/logs'}"
                    )
                    if diag.get("summary"):
                        print(f"  crash_summary={diag['summary']}")
                if result.get("traceback"):
                    print(result["traceback"].rstrip())
            return 1

        aggregate = aggregate_success_results(success_results, wall_total=wall_total, mode=args.mode)
        print_main_output(aggregate, prefer_fast_restart=not args.no_fast_restart, mode=args.mode)
        print("NOTE: multi-instance total throughput is reported in Scaling summary steps/s_total.")
        print_extra_stats(
            lat_by_method=aggregate["lat_by_method"],
            action_methods=aggregate["action_methods"],
            wait_sleep_durations=aggregate["wait_sleep_durations"],
            wait_sleep_count=aggregate["wait_sleep_count"],
            wait_sleep_total_sec=aggregate["wait_sleep_total_sec"],
        )
        print_scaling_summary(args.instances, worker_results, aggregate)

        if failed_results:
            print("Worker failures:")
            for result in failed_results:
                print(f"  worker={result.get('worker_id', -1)}, url={result.get('url', '')}, error={result.get('error')}")
                diag = build_failure_diagnostic(result, serve_returncodes_by_port=serve_returncodes_by_port)
                if diag:
                    print(
                        f"    port={diag.get('port')}, returncode={diag.get('returncode')}, "
                        f"log={diag.get('log_path') or 'not found under PROJECT_ROOT/logs'}"
                    )
                    if diag.get("summary"):
                        print(f"    crash_summary={diag['summary']}")
            return 1
        return 0
    finally:
        terminate_processes(serve_processes, cleanup_ctx=launch_cleanup_ctx)


if __name__ == "__main__":
    sys.exit(main())
