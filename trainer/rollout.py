if __package__ is None or __package__ == "":
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).resolve().parent.parent))
import argparse
import queue
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from trainer import action_space
from trainer import action_space_shop
from trainer.dataset import JsonlWriter
from trainer.env_client import (
    ConnectionError,
    EnvHandle,
    RPCError,
    StateError,
    close_pool,
    create_backend,
)
from trainer.expert_policy import choose_action
from trainer.features import extract_features
from trainer.expert_policy_shop import choose_shop_action
from trainer.features_shop import extract_shop_features
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
    parser.add_argument("--backend", choices=["real", "sim"], default="real", help="Environment backend.")
    parser.add_argument(
        "--base-urls",
        type=str,
        default="http://127.0.0.1:12346",
        help="Comma-separated base urls with ports (real backend) or worker ids (sim backend).",
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


def _resolve_targets(args: argparse.Namespace) -> list[str]:
    if args.backend == "real":
        urls = parse_base_urls(args.base_urls)
        if args.launch_instances and not urls:
            count = args.workers if args.workers > 0 else 1
            urls = build_urls_from_ports(args.base_host, args.base_port, count)
        if not urls:
            raise ValueError("No base URLs resolved. Provide --base-urls or use --launch-instances.")
        return urls

    # simulator backend
    tokens = [t.strip() for t in str(args.base_urls or "").split(",") if t.strip()]
    if not tokens:
        count = args.workers if args.workers > 0 else 1
        tokens = [f"sim://{i}" for i in range(count)]
    return tokens


def _make_handles(args: argparse.Namespace, targets: list[str], logger):
    handles: dict[str, EnvHandle] = {}
    if args.backend != "real" or not args.launch_instances:
        return handles
    if not args.love_path or not args.lovely_path:
        raise ValueError("--love-path and --lovely-path are required when --launch-instances is enabled.")

    logs_path = str((Path.cwd() / "logs").resolve())
    for url in targets:
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
            import time

            time.sleep(stagger_start)


def _stop_handles(handles: dict[str, EnvHandle]) -> None:
    for handle in handles.values():
        try:
            handle.stop()
        except Exception:
            pass


def _calc_reward(prev_state: dict | None, cur_state: dict) -> tuple[float, str]:
    cur = float((cur_state.get("round") or {}).get("chips") or 0)
    if prev_state is None:
        return 0.0, "chips_delta_proxy"
    prev = float((prev_state.get("round") or {}).get("chips") or 0)
    return float(cur - prev), "chips_delta_proxy"


def _build_macro_action(decision, idle_sleep: float) -> dict:
    macro_action = str(decision.macro_action or "wait").upper()
    macro_params = dict(decision.macro_params or {})
    action = {"action_type": macro_action}
    if macro_action == "WAIT":
        action["sleep"] = max(0.0, float(idle_sleep))
    action.update(macro_params)
    return action


def _run_episode(
    backend,
    source_id: str,
    instance_id: int,
    episode_id: int,
    include_obs_raw: bool,
    max_steps: int,
    idle_sleep: float,
    seed_prefix: str,
):
    records: list[dict] = []

    state = backend.reset(seed=f"{seed_prefix}-{episode_id}")
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
            "base_url": source_id,
            "phase": phase,
            "done": bool(done),
            "hand_size": hand_size,
            "legal_action_ids": [],
            "expert_action_id": None,
            "macro_action": decision.macro_action,
            "reward": reward,
            "reward_info": reward_info,
            "features": feat,
            "shop_legal_action_ids": [],
            "shop_expert_action_id": None,
            "shop_features": None,
        }

        if include_obs_raw:
            record["obs_raw"] = state

        records.append(record)
        if done:
            break

        if phase == "SELECTING_HAND" and hand_size > 0 and decision.action_type is not None and decision.mask_int is not None:
            legal_ids = action_space.legal_action_ids(hand_size)
            expert_action_id = action_space.encode(hand_size, decision.action_type, decision.mask_int)
            indices = action_space.mask_to_indices(decision.mask_int, hand_size)

            record["legal_action_ids"] = legal_ids
            record["expert_action_id"] = expert_action_id
            record["macro_action"] = None

            action = {
                "action_type": decision.action_type,
                "indices": indices,
            }
        elif phase in {"SHOP", "SMODS_BOOSTER_OPENED"}:
            shop_feat = extract_shop_features(state)
            shop_legal_ids = action_space_shop.legal_action_ids(state)
            shop_decision = choose_shop_action(state)

            record["shop_legal_action_ids"] = shop_legal_ids
            record["shop_expert_action_id"] = int(shop_decision.action_id)
            record["shop_features"] = shop_feat
            record["macro_action"] = str((shop_decision.action or {}).get("action_type") or "WAIT")

            action = dict(shop_decision.action)
        else:
            action = _build_macro_action(decision, idle_sleep=idle_sleep)

        next_state, _, _, _ = backend.step(action)
        prev_state = state
        state = next_state

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
        targets = _resolve_targets(args)
    except ValueError as exc:
        logger.error("%s", exc)
        return 2

    workers = args.workers if args.workers > 0 else len(targets)
    workers = max(1, min(workers, len(targets)))

    handles = _make_handles(args, targets, logger)

    try:
        _start_handles(handles, args.stagger_start, logger)
    except Exception as exc:
        logger.error("Failed to start managed instances: %s", exc)
        _stop_handles(handles)
        close_pool()
        return 1

    if args.backend == "real":
        for target in targets:
            backend = create_backend("real", base_url=target, timeout_sec=args.timeout_sec, seed=args.seed_prefix, logger=logger)
            if not backend.health():
                logger.warning("Initial health check failed for %s", target)
            backend.close()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    episode_queue: queue.Queue[int] = queue.Queue()
    for episode_id in range(args.episodes):
        episode_queue.put(episode_id)

    stats = RunStats()
    stats_lock = threading.Lock()
    write_lock = threading.Lock()

    def worker_fn(worker_idx: int, target: str):
        backend = create_backend(
            args.backend,
            base_url=target if args.backend == "real" else None,
            timeout_sec=args.timeout_sec,
            seed=f"{args.seed_prefix}-w{worker_idx}",
            logger=logger,
        )
        handle = handles.get(target)
        consecutive_failures = 0

        try:
            while True:
                try:
                    episode_id = episode_queue.get_nowait()
                except queue.Empty:
                    return

                with stats_lock:
                    stats.episodes_started += 1

                try:
                    records = _run_episode(
                        backend=backend,
                        source_id=target,
                        instance_id=worker_idx,
                        episode_id=episode_id,
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
                        "worker=%d episode=%d done records=%d hand_records=%d backend=%s target=%s",
                        worker_idx,
                        episode_id,
                        len(records),
                        hand_steps,
                        args.backend,
                        target,
                    )
                except (ConnectionError, RPCError, StateError, TimeoutError, RuntimeError, ValueError) as exc:
                    with stats_lock:
                        stats.episodes_failed += 1
                    logger.warning(
                        "worker=%d episode=%d failed on %s: %s",
                        worker_idx,
                        episode_id,
                        target,
                        exc,
                    )

                    consecutive_failures += 1
                    if args.restart_on_fail:
                        if args.backend == "real" and handle is not None:
                            ok = handle.restart(timeout_sec=60.0)
                            if not ok:
                                logger.error("worker=%d restart failed for %s", worker_idx, target)
                        elif args.backend == "real":
                            if not backend.health():
                                logger.warning("worker=%d target still unhealthy (non-managed): %s", worker_idx, target)
                        else:
                            backend.reset(seed=f"{args.seed_prefix}-recover-{worker_idx}")

                    if consecutive_failures >= args.max_consecutive_failures:
                        logger.error(
                            "worker=%d reached max consecutive failures=%d, stopping this worker",
                            worker_idx,
                            args.max_consecutive_failures,
                        )
                        return
                finally:
                    episode_queue.task_done()
        finally:
            backend.close()

    rc = 0
    with JsonlWriter(out_path) as writer:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for idx in range(workers):
                target = targets[idx % len(targets)]
                pool.submit(worker_fn, idx, target)

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
