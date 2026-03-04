from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def to_abs_path(repo_root: Path, raw_path: str | Path) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return (repo_root / path).resolve()


def read_json(path: Path) -> dict[str, Any] | list[Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception:
        return None


def read_jsonl(path: Path, *, max_rows: int = 0) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    out: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", errors="replace") as fp:
        for line in fp:
            text = line.strip()
            if not text:
                continue
            try:
                obj = json.loads(text)
            except Exception:
                continue
            if isinstance(obj, dict):
                out.append(obj)
            if max_rows > 0 and len(out) >= max_rows:
                break
    return out


def count_jsonl_rows(path: Path, *, max_scan_rows: int = 0) -> tuple[int, bool]:
    if not path.exists():
        return 0, False
    count = 0
    truncated = False
    with path.open("r", encoding="utf-8", errors="replace") as fp:
        for _line in fp:
            count += 1
            if max_scan_rows > 0 and count >= max_scan_rows:
                truncated = True
                break
    return count, truncated


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_markdown(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as fp:
        fp.write(json.dumps(payload, ensure_ascii=False) + "\n")


def stable_seed_hash(seeds: list[str]) -> str:
    packed = json.dumps([str(s) for s in seeds], ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(packed.encode("utf-8")).hexdigest()


def stable_hash_text(text: str) -> str:
    return hashlib.sha256(str(text).encode("utf-8")).hexdigest()


_RUN_ID_PATTERN = re.compile(r"^\d{8}-\d{6}$")


def infer_source_run_id(path: str | Path) -> str:
    p = Path(path)
    for part in reversed(p.parts):
        token = str(part).strip()
        if _RUN_ID_PATTERN.match(token):
            return token
        if token.startswith("champion_eval_"):
            return token
    return ""


def infer_generation_method(source_type: str, path: str, metadata: dict[str, Any] | None = None) -> str:
    md = metadata if isinstance(metadata, dict) else {}
    explicit = str(md.get("generation_method") or "").strip()
    if explicit:
        return explicit
    st = str(source_type or "").strip().lower()
    low = str(path or "").lower()
    if st == "p10_long_episode":
        return "oracle_trace"
    if st == "p13_dagger_or_real":
        if "dagger" in low:
            return "dagger_collect"
        return "real_trace_to_fixture"
    if st == "selfsup_replay":
        return "selfsup_replay_build"
    if st == "arena_failures":
        return "arena_failure_mining"
    return "unknown"


def make_sample_id(parts: list[str]) -> str:
    packed = json.dumps([str(x) for x in parts], ensure_ascii=False, separators=(",", ":"))
    return stable_hash_text(packed)[:24]


def build_seeds_payload(seeds: list[str], *, seed_policy_version: str = "p40.explicit") -> dict[str, Any]:
    normalized = [str(s).strip() for s in seeds if str(s).strip()]
    return {
        "schema": "p40_seeds_used_v1",
        "generated_at": now_iso(),
        "seed_policy_version": seed_policy_version,
        "seed_count": len(normalized),
        "seed_hash": stable_seed_hash(normalized),
        "seeds": normalized,
    }
