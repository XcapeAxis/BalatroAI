import argparse
import logging
import os
import random
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Iterable, Sequence
from urllib.parse import urlparse


def setup_logger(name: str = "trainer", level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        logger.setLevel(level)
        return logger

    logger.setLevel(level)
    handler = logging.StreamHandler(sys.stdout)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s", "%H:%M:%S")
    handler.setFormatter(fmt)
    logger.addHandler(handler)
    logger.propagate = False
    return logger


def timestamp() -> str:
    return datetime.now().isoformat(timespec="seconds")


def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def set_global_seed(seed: int | None) -> None:
    if seed is None:
        return
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def warn_if_unstable_python(logger: logging.Logger) -> None:
    version = sys.version_info
    release = version.releaselevel
    if release not in {"final"}:
        logger.warning(
            "Detected non-final Python runtime: %s. Recommended for training: Python 3.12/3.13 stable.",
            sys.version.split()[0],
        )
    elif version.major == 3 and version.minor >= 15:
        logger.warning(
            "Python %s may be newer than tested stack. Recommended for training: Python 3.12/3.13 stable.",
            sys.version.split()[0],
        )


def parse_base_urls(raw: str | None) -> list[str]:
    if not raw:
        return []
    urls: list[str] = []
    for token in raw.split(","):
        candidate = token.strip().rstrip("/")
        if not candidate:
            continue
        parsed = urlparse(candidate)
        if parsed.scheme not in {"http", "https"}:
            raise ValueError(f"Invalid URL (missing http/https): {candidate}")
        if not parsed.hostname:
            raise ValueError(f"Invalid URL host: {candidate}")
        if parsed.port is None:
            raise ValueError(f"URL must include port: {candidate}")
        urls.append(candidate)
    return urls


def build_urls_from_ports(base_host: str, base_port: int, count: int) -> list[str]:
    if count <= 0:
        raise ValueError("count must be > 0")
    host = base_host.rstrip("/")
    if not host.startswith("http://") and not host.startswith("https://"):
        raise ValueError("base_host must include http:// or https://")
    return [f"{host}:{base_port + i}" for i in range(count)]


def parse_int_csv(raw: str | None) -> list[int]:
    if not raw:
        return []
    values = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        values.append(int(token))
    return values


def retry_with_backoff(
    fn: Callable[[], object],
    retries: int,
    base_delay: float,
    max_delay: float,
    exceptions: tuple[type[BaseException], ...],
    logger: logging.Logger | None = None,
    context: str = "",
):
    last_exc: BaseException | None = None
    for attempt in range(retries):
        try:
            return fn()
        except exceptions as exc:  # type: ignore[arg-type]
            last_exc = exc
            if attempt >= retries - 1:
                break
            delay = min(max_delay, base_delay * (2**attempt))
            if logger is not None:
                logger.warning(
                    "Retry %d/%d for %s after error: %s (sleep %.2fs)",
                    attempt + 1,
                    retries,
                    context or "operation",
                    exc,
                    delay,
                )
            time.sleep(delay)
    if last_exc is not None:
        raise last_exc
    raise RuntimeError("retry_with_backoff reached unexpected path")


@dataclass
class RunStats:
    episodes_started: int = 0
    episodes_succeeded: int = 0
    episodes_failed: int = 0
    steps_total: int = 0
    hand_steps: int = 0


def add_common_launch_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--launch-instances", action="store_true", help="Auto-launch balatrobot instances.")
    parser.add_argument(
        "--launcher",
        choices=["uvx", "direct"],
        default="uvx",
        help="Launcher backend used when --launch-instances is enabled.",
    )
    parser.add_argument("--uvx-path", default="uvx", help="uvx executable path (launcher=uvx).")
    parser.add_argument(
        "--balatrobot-cmd",
        default="balatrobot",
        help="balatrobot executable/command (launcher=direct).",
    )
    parser.add_argument("--love-path", default=None, help="Path to Balatro executable.")
    parser.add_argument("--lovely-path", default=None, help="Path to lovely/version DLL.")
    parser.add_argument("--base-host", default="http://127.0.0.1", help="Base host for auto-generated urls.")
    parser.add_argument("--base-port", type=int, default=12346, help="Start port for auto-generated urls.")
    parser.add_argument("--stagger-start", type=float, default=0.2, help="Start interval between instances.")


def format_action(action_type: str, indices: Sequence[int]) -> str:
    return f"{action_type}[{','.join(str(i) for i in indices)}]"


def flatten(it: Iterable[Iterable[int]]) -> list[int]:
    out: list[int] = []
    for seq in it:
        out.extend(seq)
    return out
