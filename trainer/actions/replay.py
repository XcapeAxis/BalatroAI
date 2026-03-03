from __future__ import annotations

if __package__ is None or __package__ == "":
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

from trainer.env_client import create_backend
from trainer.utils import timestamp

ActionMode = Literal["sim", "real"]


class ActionReplayError(RuntimeError):
    pass


def _stable_hash(payload: Any) -> str:
    import hashlib

    text = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _ensure_int_list(values: Any) -> list[int]:
    if not isinstance(values, list):
        return []
    out: list[int] = []
    for item in values:
        out.append(int(item))
    return out


ACTION_TYPE_ALIASES: dict[str, str] = {
    "REROLL": "SHOP_REROLL",
    "BUY": "SHOP_BUY",
    "PACK": "PACK_OPEN",
    "USE": "CONSUMABLE_USE",
    "SWAP_HAND_CARDS": "SWAP_HAND_CARD",
    "SWAP_JOKERS": "SWAP_JOKER",
}


def _canonical_action_type(value: Any) -> str:
    raw = str(value or "WAIT").strip().upper()
    return ACTION_TYPE_ALIASES.get(raw, raw)


def _read_index_base(src: dict[str, Any], params: dict[str, Any]) -> int:
    candidate = src.get("index_base", params.get("index_base", 0))
    try:
        base = int(candidate)
    except Exception:
        base = 0
    return 1 if base == 1 else 0


def _normalize_index(value: Any, *, index_base: int, default: int = 0) -> int:
    try:
        idx = int(value)
    except Exception:
        idx = int(default)
    norm = idx - int(index_base)
    return max(0, norm)


def _normalize_params_indices(params: dict[str, Any], *, index_base: int) -> dict[str, Any]:
    out = dict(params)
    if index_base not in {0, 1}:
        index_base = 0

    scalar_fields = (
        "index",
        "card",
        "pack",
        "voucher",
        "consumable",
        "joker",
        "shop_index",
        "pack_index",
        "choice_index",
        "voucher_index",
        "consumable_index",
        "joker_index",
    )
    for field in scalar_fields:
        if field in out:
            out[field] = _normalize_index(out.get(field), index_base=index_base, default=0)

    list_fields = ("cards", "targets", "hand_indices")
    for field in list_fields:
        if field in out and isinstance(out.get(field), list):
            values = _ensure_int_list(out.get(field))
            out[field] = [_normalize_index(v, index_base=index_base, default=0) for v in values]

    out["index_base"] = 0
    return out


def normalize_high_level_action(
    action: dict[str, Any] | None,
    *,
    phase: str = "",
) -> dict[str, Any]:
    """Normalize action_v1-ish payload into a deterministic replay-friendly shape.

    This function is intentionally permissive for backward compatibility with older
    recorder/infer payloads. Unsupported experimental hooks are left intact and
    flagged through `meta.unimplemented_hook`.
    """
    src = dict(action or {})
    params = src.get("params") if isinstance(src.get("params"), dict) else {}
    action_type = _canonical_action_type(src.get("action_type") or params.get("action_type") or "WAIT")
    out: dict[str, Any] = {
        "schema_version": str(src.get("schema_version") or "action_v1"),
        "phase": str(src.get("phase") or phase or "UNKNOWN"),
        "action_type": action_type,
    }
    index_base = _read_index_base(src, params)

    if action_type in {"PLAY", "DISCARD"}:
        raw_indices = src.get("indices") if "indices" in src else params.get("indices")
        if not isinstance(raw_indices, list):
            raw_indices = src.get("cards") if isinstance(src.get("cards"), list) else params.get("cards")
        out["indices"] = [_normalize_index(v, index_base=index_base, default=0) for v in _ensure_int_list(raw_indices)]
    elif action_type == "SELECT":
        out["index"] = int(src.get("index", params.get("index", 0)))
    elif action_type in {"SHOP_BUY", "SELL", "PACK_OPEN", "CONSUMABLE_USE"}:
        merged = dict(params)
        for key in (
            "card",
            "pack",
            "voucher",
            "joker",
            "consumable",
            "shop_index",
            "pack_index",
            "choice_index",
            "voucher_index",
            "consumable_index",
            "joker_index",
            "cards",
            "hand_indices",
            "skip",
            "targets",
            "target_side",
            "kind",
            "key",
            "index_base",
        ):
            if key in src and key not in merged:
                merged[key] = src[key]
        out["params"] = _normalize_params_indices(merged, index_base=index_base)
    elif action_type in {"REORDER_HAND", "REORDER_JOKERS"}:
        permutation = src.get("permutation") if "permutation" in src else params.get("permutation")
        out["permutation"] = _ensure_int_list(permutation)
        out["index_base"] = 0
    elif action_type in {"MOVE_HAND_CARD", "MOVE_JOKER"}:
        src_index = src.get("src_index", src.get("from_index", params.get("src_index", params.get("from_index", 0))))
        dst_index = src.get("dst_index", src.get("to_index", params.get("dst_index", params.get("to_index", 0))))
        out["src_index"] = _normalize_index(src_index, index_base=index_base, default=0)
        out["dst_index"] = _normalize_index(dst_index, index_base=index_base, default=0)
        out["index_base"] = 0
    elif action_type in {"SWAP_HAND_CARD", "SWAP_JOKER"}:
        out["i"] = _normalize_index(src.get("i", params.get("i", 0)), index_base=index_base, default=0)
        out["j"] = _normalize_index(src.get("j", params.get("j", 0)), index_base=index_base, default=0)
        out["index_base"] = 0
    elif action_type in {"CARD_REORDER", "JOKER_REORDER", "APPLY_TAROT_SWAP", "USE_CONSUMABLE_ON_HAND_CARD"}:
        # Reserved P33 hook names for future contract expansion.
        out["params"] = dict(params)
        out["meta"] = {"unimplemented_hook": action_type}
    else:
        # Keep passthrough fields for compatibility with legacy action dictionaries.
        passthrough = dict(src)
        passthrough.pop("schema_version", None)
        passthrough.pop("phase", None)
        passthrough.pop("action_type", None)
        if passthrough:
            out["params"] = passthrough
    rng_replay = src.get("rng_replay")
    if isinstance(rng_replay, dict):
        out["rng_replay"] = rng_replay
    return out


@dataclass
class ReplayResult:
    ok: bool
    mode: ActionMode
    action_input: dict[str, Any]
    action_normalized: dict[str, Any]
    before_hash: str
    after_hash: str
    reward: float
    done: bool
    info: dict[str, Any]
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ActionReplayer:
    """Unified adapter for replaying high-level action dictionaries on sim/real backends."""

    def __init__(
        self,
        *,
        mode: ActionMode,
        backend: Any | None = None,
        base_url: str = "http://127.0.0.1:12346",
        timeout_sec: float = 8.0,
        seed: str = "AAAAAAA",
        logger=None,
        debug_dir: str | Path = "",
        strict: bool = True,
    ) -> None:
        self.mode = str(mode).lower().strip()
        if self.mode not in {"sim", "real"}:
            raise ValueError(f"unsupported replay mode: {mode}")
        self.logger = logger
        self.strict = bool(strict)
        self._owns_backend = backend is None
        self.backend = backend or create_backend(self.mode, base_url=base_url, timeout_sec=timeout_sec, seed=seed, logger=logger)
        self.debug_dir = Path(debug_dir) if str(debug_dir).strip() else None

    def _debug_write(self, event: dict[str, Any]) -> None:
        if self.debug_dir is None:
            return
        try:
            self.debug_dir.mkdir(parents=True, exist_ok=True)
            path = self.debug_dir / "action_replayer.jsonl"
            with path.open("a", encoding="utf-8", newline="\n") as fp:
                fp.write(json.dumps(event, ensure_ascii=False) + "\n")
        except Exception:
            if self.logger is not None:
                self.logger.warning("failed to write action replayer debug event")

    def replay_single_action(
        self,
        state_context: dict[str, Any] | None,
        high_level_action: dict[str, Any] | None,
    ) -> ReplayResult:
        before_state = state_context if isinstance(state_context, dict) else self.backend.get_state()
        phase = str(before_state.get("state") or before_state.get("phase") or "UNKNOWN")
        normalized = normalize_high_level_action(high_level_action, phase=phase)
        before_hash = _stable_hash(before_state)
        action_type = str(normalized.get("action_type") or "WAIT").upper()

        hook_meta = normalized.get("meta") if isinstance(normalized.get("meta"), dict) else {}
        if str(hook_meta.get("unimplemented_hook") or ""):
            message = f"unimplemented action hook: {action_type}"
            result = ReplayResult(
                ok=False,
                mode=self.mode,  # type: ignore[arg-type]
                action_input=dict(high_level_action or {}),
                action_normalized=normalized,
                before_hash=before_hash,
                after_hash=before_hash,
                reward=0.0,
                done=False,
                info={"backend": self.mode, "skipped": True, "reason": message},
                error=message,
            )
            self._debug_write({"ts": timestamp(), **result.to_dict()})
            if self.strict:
                raise ActionReplayError(message)
            return result

        try:
            after_state, reward, done, info = self.backend.step(normalized)
            after_hash = _stable_hash(after_state)
            merged_info = dict(info or {})
            merged_info.setdefault("backend", self.mode)
            merged_info["replayer_phase"] = phase
            result = ReplayResult(
                ok=True,
                mode=self.mode,  # type: ignore[arg-type]
                action_input=dict(high_level_action or {}),
                action_normalized=normalized,
                before_hash=before_hash,
                after_hash=after_hash,
                reward=float(reward),
                done=bool(done),
                info=merged_info,
            )
            self._debug_write({"ts": timestamp(), **result.to_dict()})
            return result
        except Exception as exc:
            message = f"{type(exc).__name__}: {exc}"
            if self.logger is not None:
                self.logger.error("ActionReplayer failure mode=%s action=%s err=%s", self.mode, action_type, message)
            result = ReplayResult(
                ok=False,
                mode=self.mode,  # type: ignore[arg-type]
                action_input=dict(high_level_action or {}),
                action_normalized=normalized,
                before_hash=before_hash,
                after_hash=before_hash,
                reward=0.0,
                done=False,
                info={"backend": self.mode, "failed": True},
                error=message,
            )
            self._debug_write({"ts": timestamp(), **result.to_dict()})
            if self.strict:
                raise ActionReplayError(message) from exc
            return result

    def close(self) -> None:
        if self._owns_backend and self.backend is not None:
            try:
                self.backend.close()
            except Exception:
                pass


__all__ = [
    "ActionReplayError",
    "ActionReplayer",
    "ReplayResult",
    "normalize_high_level_action",
]
