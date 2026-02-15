if __package__ is None or __package__ == "":
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).resolve().parent.parent))
import argparse
import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from trainer import action_space
from trainer.dataset import JsonlWriter
from trainer.env_client import (
    ConnectionError,
    EnvHandle,
    RPCError,
    StateError,
    _call_method,
    act_batch,
    close_pool,
    get_state,
    health,
)
from trainer.expert_policy import choose_action
from trainer.features import extract_features
from trainer.utils import (
    RunStats,
    add_common_launch_args,
    build_urls_from_ports,
    parse_base_urls,
    set_global_seed,
    setup_logger,
    timestamp,
    warn_if_unstable_python,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rollout data generator for Balatro BC training.")
    parser.add_argument(
        "--base-urls",
        type=str,
        default="http://127.0.0.1:12346",
        help="Comma-separated base urls with ports.",
    )
    parser.add_argument("--episodes", type=int, default=10, help="Total episodes to collect.")
    parser.add_argument("--out", type=str, default="trainer_data/dataset.jsonl", help="Output jsonl path.")
    parser.add_argument("--include-obs-raw", action="store_true", help="Include raw gamestate in each record.")
    parser.add_argument("--restart-on-fail", action="store_true", help="Restart or skip failed instances and continue.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility.")
    parser.add_argument("--seed-prefix", type=str, default="AAAAAAA", help="Seed prefix used for start() macro action.")
    parser.add_argument("--max-steps-per-episode", type=int, default=300, help="Step cap per episode.")
    parser.add_argument("--timeout-sec", type=float, default=8.0, help="RPC timeout seconds.")
    parser.add_argument("--workers", type=int, default=0, help="Worker threads. 0 means len(base_urls).")
    parser.add_argument("--idle-sleep", type=float, default=0.05, help="Sleep seconds for wait macro actions.")
    parser.add_argument(
        "--max-consecutive-failures",
        type=int,
        default=3,
        help="Stop an instance worker after this many consecutive failures.",
    )
    add_common_launch_args(parser)
    return parser.parse_args()


def _resolve_urls(args: argparse.Namespace) -> list[str]:
    urls = parse_base_urls(args.base_urls)
    if args.launch_instances:
        if not urls:
            count = args.workers if args.workers > 0 else 1
            urls = build_urls_from_ports(args.base_host, args.base_port, count)
    if not urls:
        raise ValueError("No base URLs resolved. Provide --base-urls or use --launch-instances.")
    return urls


def _make_handles(args: argparse.Namespace, urls: list[str], logger):
    handles: dict[str, EnvHandle] = {}
    if not args.launch_instances:
        return handles
    if not args.love_path or not args.lovely_path:
        raise ValueError("--love-path and --lovely-path are required when --launch-instances is enabled.")

    logs_path = str((Path.cwd() / "logs").resolve())
    for url in urls:
        handles[url] = EnvHandle(
            base_url=url,
            launcher=args.launcher,
            uvx_path=args.uvx_path,
            balatrobot_cmd=args.balatrobot_cmd,
            love_path=args.love_path,
            lovely_path=args.lovely_path,
            logs_path=logs_path,
            cwd=str(Path.cwd()),
            logger=logger,
        )
    return handles


def _start_handles(handles: dict[str, EnvHandle], stagger_start: float, logger) -> None:
    if not handles:
        return
    urls = list(handles.keys())
    for i, url in enumerate(urls):
        handle = handles[url]
        handle.start()
        if not handle.wait_healthy(timeout_sec=60.0):
            raise RuntimeError(f"Handle failed health check: {url}")
        logger.info("Healthy instance: %s", url)
        if stagger_start > 0 and i < len(urls) - 1:
            time.sleep(stagger_start)


def _stop_handles(handles: dict[str, EnvHandle]) -> None:
    for handle in handles.values():
        try:
            handle.stop()
        except Exception:
            pass


def _calc_reward(prev_state: dict | None, cur_state: dict) -> tuple[float, str]:
    cur = int((cur_state.get("round") or {}).get("chips") or 0)
    if prev_state is None:
        return 0.0, "chips_delta_proxy"
    prev = int((prev_state.get("round") or {}).get("chips") or 0)
    return float(cur - prev), "chips_delta_proxy"


def _apply_macro(base_url: str, macro_action: str, macro_params: dict, timeout_sec: float, idle_sleep: float) -> dict:
    macro_action = macro_action or "wait"
    macro_params = macro_params or {}

    if macro_action == "wait":
        time.sleep(max(0.0, idle_sleep))
        return get_state(base_url, timeout=timeout_sec)

    if macro_action in {
        "select",
        "cash_out",
        "next_round",
        "menu",
        "skip",
        "reroll",
        "start",
    }:
        _call_method(base_url, macro_action, macro_params, timeout=timeout_sec)
        return get_state(base_url, timeout=timeout_sec)

    # Unknown macro action: fail open with wait.
    time.sleep(max(0.0, idle_sleep))
    return get_state(base_url, timeout=timeout_sec)


def _run_episode(
    base_url: str,
    instance_id: int,
    episode_id: int,
    timeout_sec: float,
    include_obs_raw: bool,
    max_steps: int,
    idle_sleep: float,
    seed_prefix: str,
):
    records: list[dict] = []

    state = get_state(base_url, timeout=timeout_sec)
    if state.get("state") in {"MENU", "GAME_OVER"}:
        start_seed = f"{seed_prefix}-{episode_id}"
        _call_method(base_url, "start", {"deck": "RED", "stake": "WHITE", "seed": start_seed}, timeout=timeout_sec)
        state = get_state(base_url, timeout=timeout_sec)

    prev_state = None

    for step_id in range(max_steps):
        phase = str(state.get("state") or "UNKNOWN")
        feat = extract_features(state)
        done = phase == "GAME_OVER"
        reward, reward_info = _calc_reward(prev_state, state)

        decision = choose_action(state, start_seed=f"{seed_prefix}-{episode_id}")

        hand_cards = (state.get("hand") or {}).get("cards") or []
        hand_size = min(len(hand_cards), action_space.MAX_HAND)

        record = {
            "schema_version": "record_v1",
            "timestamp": timestamp(),
            "episode_id": episode_id,
            "step_id": step_id,
            "instance_id": instance_id,
            "base_url": base_url,
            "phase": phase,
            "done": bool(done),
            "hand_size": hand_size,
            "legal_action_ids": [],
            "expert_action_id": None,
            "macro_action": decision.macro_action,
            "reward": reward,
            "reward_info": reward_info,
            "features": feat,
        }

        if include_obs_raw:
            record["obs_raw"] = state

        if phase == "SELECTING_HAND" and hand_size > 0 and decision.action_type is not None and decision.mask_int is not None:
            legal_ids = action_space.legal_action_ids(hand_size)
            expert_action_id = action_space.encode(hand_size, decision.action_type, decision.mask_int)
            indices = action_space.mask_to_indices(decision.mask_int, hand_size)

            record["legal_action_ids"] = legal_ids
            record["expert_action_id"] = expert_action_id
            record["macro_action"] = None

            next_state = act_batch(
                base_url,
                decision.action_type,
                indices,
                timeout=timeout_sec,
            )
        else:
            macro_action = decision.macro_action or "wait"
            macro_params = decision.macro_params or {}
            next_state = _apply_macro(
                base_url,
                macro_action=macro_action,
                macro_params=macro_params,
                timeout_sec=timeout_sec,
                idle_sleep=idle_sleep,
            )

        records.append(record)
        prev_state = state
        state = next_state

        if done:
            break

    return records


def main() -> int:
    args = parse_args()
    logger = setup_logger("trainer.rollout")
    warn_if_unstable_python(logger)
    set_global_seed(args.seed)

    if args.episodes <= 0:
        logger.error("--episodes must be > 0")
        return 2
    if args.max_steps_per_episode <= 0:
        logger.error("--max-steps-per-episode must be > 0")
        return 2

    try:
        urls = _resolve_urls(args)
    except ValueError as exc:
        logger.error("%s", exc)
        return 2

    workers = args.workers if args.workers > 0 else len(urls)
    workers = max(1, min(workers, len(urls)))

    handles = _make_handles(args, urls, logger)

    try:
        _start_handles(handles, args.stagger_start, logger)
    except Exception as exc:
        logger.error("Failed to start managed instances: %s", exc)
        _stop_handles(handles)
        close_pool()
        return 1

    for url in urls:
        if not health(url):
            logger.warning("Initial health check failed for %s", url)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    episode_queue: queue.Queue[int] = queue.Queue()
    for episode_id in range(args.episodes):
        episode_queue.put(episode_id)

    stats = RunStats()
    stats_lock = threading.Lock()
    write_lock = threading.Lock()

    def worker_fn(worker_idx: int, base_url: str):
        handle = handles.get(base_url)
        consecutive_failures = 0

        while True:
            try:
                episode_id = episode_queue.get_nowait()
            except queue.Empty:
                return

            with stats_lock:
                stats.episodes_started += 1

            try:
                records = _run_episode(
                    base_url=base_url,
                    instance_id=worker_idx,
                    episode_id=episode_id,
                    timeout_sec=args.timeout_sec,
                    include_obs_raw=args.include_obs_raw,
                    max_steps=args.max_steps_per_episode,
                    idle_sleep=args.idle_sleep,
                    seed_prefix=args.seed_prefix,
                )
                hand_steps = sum(1 for r in records if r["phase"] == "SELECTING_HAND")

                with write_lock:
                    for record in records:
                        writer.write_record(record)

                with stats_lock:
                    stats.episodes_succeeded += 1
                    stats.steps_total += len(records)
                    stats.hand_steps += hand_steps

                consecutive_failures = 0
                logger.info(
                    "worker=%d episode=%d done records=%d hand_records=%d",
                    worker_idx,
                    episode_id,
                    len(records),
                    hand_steps,
                )
            except (ConnectionError, RPCError, StateError, TimeoutError, RuntimeError) as exc:
                with stats_lock:
                    stats.episodes_failed += 1
                logger.warning(
                    "worker=%d episode=%d failed on %s: %s",
                    worker_idx,
                    episode_id,
                    base_url,
                    exc,
                )

                consecutive_failures += 1
                if args.restart_on_fail:
                    if handle is not None:
                        ok = handle.restart(timeout_sec=60.0)
                        if not ok:
                            logger.error("worker=%d restart failed for %s", worker_idx, base_url)
                    else:
                        healthy = health(base_url)
                        if not healthy:
                            logger.warning("worker=%d base_url still unhealthy (non-managed): %s", worker_idx, base_url)
                if consecutive_failures >= args.max_consecutive_failures:
                    logger.error(
                        "worker=%d reached max consecutive failures=%d, stopping this worker",
                        worker_idx,
                        args.max_consecutive_failures,
                    )
                    return
            finally:
                episode_queue.task_done()

    rc = 0
    with JsonlWriter(out_path) as writer:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for idx in range(workers):
                base_url = urls[idx % len(urls)]
                pool.submit(worker_fn, idx, base_url)

            episode_queue.join()

    logger.info(
        "Rollout finished: episodes_started=%d succeeded=%d failed=%d steps_total=%d hand_steps=%d out=%s",
        stats.episodes_started,
        stats.episodes_succeeded,
        stats.episodes_failed,
        stats.steps_total,
        stats.hand_steps,
        out_path,
    )

    if stats.episodes_succeeded == 0:
        rc = 1

    _stop_handles(handles)
    close_pool()
    return rc


if __name__ == "__main__":
    raise SystemExit(main())



