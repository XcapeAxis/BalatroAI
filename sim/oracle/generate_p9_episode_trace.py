from __future__ import annotations

if __package__ is None or __package__ == "":
    import sys
    from pathlib import Path as _Path

    sys.path.append(str(_Path(__file__).resolve().parent.parent.parent))

import argparse
import json
import time
from pathlib import Path
from typing import Any

from sim.oracle.generate_p0_trace import (
    canonical_snapshot,
    detect_index_base,
    hard_reset_fixture,
    prepare_selecting_hand,
    state_phase,
)
from trainer.env_client import ConnectionError, RPCError, _call_method, get_state, health

DEFAULT_TARGETS = [
    "p9_episode_01_blind_alpha_tag_alpha",
    "p9_episode_02_blind_beta_tag_beta",
    "p9_episode_03_blind_gamma_tag_gamma",
    "p9_episode_04_blind_delta_tag_delta",
    "p9_episode_05_blind_epsilon_tag_epsilon",
    "p9_episode_06_blind_zeta_tag_zeta",
]

CONSUMABLE_KEYS = [
    "c_mercury",
    "c_venus",
    "c_earth",
    "c_mars",
    "c_jupiter",
    "c_saturn",
    "c_uranus",
    "c_neptune",
    "c_pluto",
    "c_fool",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate P9 episode trace (multi-round hand+shop+pack+use loop).")
    parser.add_argument("--base-url", default="http://127.0.0.1:12346")
    parser.add_argument("--target", required=True)
    parser.add_argument("--max-steps", type=int, default=800)
    parser.add_argument("--seed", default="AAAAAAA")
    parser.add_argument("--timeout-sec", type=float, default=8.0)
    parser.add_argument("--wait-sleep", type=float, default=0.05)
    parser.add_argument("--out-dir", default="")
    return parser.parse_args()


def _classify_connection_failure_text(text: str) -> str:
    low = str(text or "").lower()
    if any(token in low for token in ("10061", "connection refused", "actively refused", "refused")):
        return "connection refused"
    if any(token in low for token in ("timed out", "timeout", "read timed out", "connect timeout")):
        return "timeout"
    return "health check failed"


def _probe_connection_failure(base_url: str, timeout_sec: float) -> str | None:
    if health(base_url):
        return None
    timeout = max(1.0, min(float(timeout_sec), 3.0))
    reasons: list[str] = []
    for method in ("health", "gamestate"):
        try:
            _call_method(base_url, method, timeout=timeout)
            return None
        except Exception as exc:
            reasons.append(str(exc))
    return _classify_connection_failure_text(" | ".join(reasons))


def _extract_hand_brief(state: dict[str, Any]) -> str:
    hand = ((state.get("hand") or {}).get("cards") or []) if isinstance(state.get("hand"), dict) else []
    out: list[str] = []
    for card in hand[:8]:
        if not isinstance(card, dict):
            continue
        value = card.get("value") if isinstance(card.get("value"), dict) else {}
        rank = str(card.get("rank") or value.get("rank") or "?")
        suit = str(card.get("suit") or value.get("suit") or "?")
        out.append(f"{rank}-{suit}")
    return "[" + ",".join(out) + "]"


def _state_context_summary(state: dict[str, Any]) -> str:
    phase = state_phase(state)
    round_info = state.get("round") or {}
    hands_left = int(round_info.get("hands_left") or 0)
    discards_left = int(round_info.get("discards_left") or 0)
    return f"final_phase={phase}; hands_left={hands_left}; discards_left={discards_left}; hand={_extract_hand_brief(state)}"


def _save_start_state(base_url: str, out_dir: Path, target: str, timeout_sec: float) -> str | None:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"oracle_start_state_{target}.jkr"
    try:
        _call_method(base_url, "save", {"path": str(path)}, timeout=timeout_sec)
        return str(path)
    except Exception:
        return None


def _save_state_if_possible(base_url: str, path: str | None, timeout_sec: float) -> None:
    if not path:
        return
    try:
        _call_method(base_url, "save", {"path": path}, timeout=timeout_sec)
    except Exception:
        pass


def _to_api_indices(local_indices: list[int], index_base: int) -> list[int]:
    return [int(i) + (1 if int(index_base) == 1 else 0) for i in local_indices]


def _market_index(state: dict[str, Any], field: str, key_hint: str | None = None) -> int:
    market = state.get(field) if isinstance(state.get(field), dict) else {}
    cards = market.get("cards") if isinstance(market.get("cards"), list) else []
    if key_hint:
        low = str(key_hint).strip().lower()
        for idx, card in enumerate(cards):
            if isinstance(card, dict) and str(card.get("key") or "").strip().lower() == low:
                return idx
    return 0


def _prepare_shop_start(
    base_url: str,
    seed: str,
    timeout_sec: float,
    wait_sleep: float,
    index_base: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    hard_reset_fixture(base_url, seed, timeout_sec, wait_sleep)
    state = prepare_selecting_hand(base_url, seed, timeout_sec, wait_sleep)

    # Light preparation only; do not force phase transitions during generation.
    try:
        _call_method(base_url, "set", {"money": 300}, timeout=timeout_sec)
    except Exception:
        pass

    prep_meta: dict[str, Any] = {"voucher_key": None, "pack_key": None, "consumable_key": None}

    try:
        _call_method(base_url, "add", {"key": "v_paint_brush"}, timeout=timeout_sec)
        prep_meta["voucher_key"] = "v_paint_brush"
    except Exception:
        pass

    try:
        _call_method(base_url, "add", {"key": "p_arcana_normal_1"}, timeout=timeout_sec)
        prep_meta["pack_key"] = "p_arcana_normal_1"
    except Exception:
        pass

    try:
        _call_method(base_url, "add", {"key": "j_joker"}, timeout=timeout_sec)
    except Exception:
        pass

    for key in CONSUMABLE_KEYS:
        try:
            _call_method(base_url, "add", {"key": key}, timeout=timeout_sec)
            prep_meta["consumable_key"] = key
            break
        except Exception:
            continue

    state = get_state(base_url, timeout=timeout_sec)
    prep_meta["voucher_idx"] = _market_index(state, "vouchers", prep_meta.get("voucher_key"))
    prep_meta["pack_idx"] = _market_index(state, "packs", prep_meta.get("pack_key"))
    prep_meta["card_idx"] = _market_index(state, "shop", None)
    prep_meta["joker_idx"] = 0
    prep_meta["consumable_idx"] = 0

    return state, prep_meta


def _episode_actions(prep_meta: dict[str, Any], variant: int) -> list[dict[str, Any]]:
    voucher_idx = int(prep_meta.get("voucher_idx") or 0)
    pack_idx = int(prep_meta.get("pack_idx") or 0)
    card_idx = int(prep_meta.get("card_idx") or 0)
    consumable_idx = int(prep_meta.get("consumable_idx") or 0)
    joker_idx = int(prep_meta.get("joker_idx") or 0)

    base_actions: list[dict[str, Any]] = [
        {"action_type": "DISCARD", "indices": [0], "params": {}},
        {"action_type": "PLAY", "indices": [0], "params": {}},
        {"action_type": "PLAY", "indices": [0], "params": {}},
        {"action_type": "PLAY", "indices": [0], "params": {}},
        {"action_type": "CASH_OUT", "params": {}},
        {"action_type": "BUY", "params": {"voucher": voucher_idx}},
        {"action_type": "BUY", "params": {"pack": pack_idx}},
        {"action_type": "PACK", "params": {"card": 0}},
        {"action_type": "USE", "params": {"consumable": consumable_idx}},
        {"action_type": "REROLL", "params": {}},
        {"action_type": "NEXT_ROUND", "params": {}},
        {"action_type": "SELECT", "index": 0, "params": {}},
        {"action_type": "DISCARD", "indices": [0], "params": {}},
        {"action_type": "PLAY", "indices": [0], "params": {}},
        {"action_type": "PLAY", "indices": [0], "params": {}},
        {"action_type": "CASH_OUT", "params": {}},
        {"action_type": "BUY", "params": {"card": card_idx}},
        {"action_type": "REROLL", "params": {}},
        {"action_type": "BUY", "params": {"pack": 0}},
        {"action_type": "PACK", "params": {"skip": True}},
        {"action_type": "NEXT_ROUND", "params": {}},
        {"action_type": "SELECT", "index": 0, "params": {}},
        {"action_type": "DISCARD", "indices": [0], "params": {}},
        {"action_type": "PLAY", "indices": [0], "params": {}},
        {"action_type": "PLAY", "indices": [0], "params": {}},
        {"action_type": "CASH_OUT", "params": {}},
        {"action_type": "BUY", "params": {"card": 0}},
        {"action_type": "REROLL", "params": {}},
        {"action_type": "NEXT_ROUND", "params": {}},
        {"action_type": "SELECT", "index": 0, "params": {}},
        {"action_type": "PLAY", "indices": [0], "params": {}},
        {"action_type": "WAIT", "params": {}},
        {"action_type": "WAIT", "params": {}},
    ]

    # Minor deterministic variants to diversify traces while preserving requirements.
    if variant % 3 == 1:
        base_actions[16] = {"action_type": "SELL", "params": {"joker": joker_idx}}
        base_actions[26] = {"action_type": "BUY", "params": {"pack": 0}}
        base_actions[27] = {"action_type": "PACK", "params": {"card": 0}}
    elif variant % 3 == 2:
        base_actions[10] = {"action_type": "BUY", "params": {"card": card_idx}}
        base_actions[17] = {"action_type": "USE", "params": {"consumable": consumable_idx}}
        base_actions[30] = {"action_type": "REROLL", "params": {}}

    return base_actions


def _execute_action(base_url: str, action: dict[str, Any], timeout_sec: float, wait_sleep: float, index_base: int) -> dict[str, Any]:
    at = str(action.get("action_type") or "WAIT").upper()

    if at == "PLAY":
        rpc_indices = _to_api_indices([int(x) for x in (action.get("indices") or [])], index_base)
        _call_method(base_url, "play", {"cards": rpc_indices}, timeout=timeout_sec)
    elif at == "DISCARD":
        rpc_indices = _to_api_indices([int(x) for x in (action.get("indices") or [])], index_base)
        _call_method(base_url, "discard", {"cards": rpc_indices}, timeout=timeout_sec)
    elif at == "SELECT":
        _call_method(base_url, "select", {"index": int(action.get("index", 0))}, timeout=timeout_sec)
    elif at == "CASH_OUT":
        _call_method(base_url, "cash_out", {}, timeout=timeout_sec)
    elif at == "NEXT_ROUND":
        _call_method(base_url, "next_round", {}, timeout=timeout_sec)
    elif at == "REROLL":
        _call_method(base_url, "reroll", {}, timeout=timeout_sec)
    elif at == "BUY":
        _call_method(base_url, "buy", dict(action.get("params") or {}), timeout=timeout_sec)
    elif at == "PACK":
        _call_method(base_url, "pack", dict(action.get("params") or {}), timeout=timeout_sec)
    elif at == "SELL":
        _call_method(base_url, "sell", dict(action.get("params") or {}), timeout=timeout_sec)
    elif at == "USE":
        params = dict(action.get("params") or {})
        attempts = [params]
        if "consumable" not in params:
            attempts.append({"consumable": 0})
        if "card" not in params:
            attempts.append({"consumable": int(params.get("consumable", 0)), "card": 0})
        last_exc: Exception | None = None
        for p in attempts:
            try:
                _call_method(base_url, "use", p, timeout=timeout_sec)
                break
            except Exception as exc:
                last_exc = exc
        else:
            assert last_exc is not None
            raise last_exc
    elif at == "SKIP":
        _call_method(base_url, "skip", {}, timeout=timeout_sec)
    elif at == "WAIT":
        time.sleep(max(0.0, float(action.get("sleep") or wait_sleep)))
    else:
        raise ValueError(f"unsupported action_type:{at}")

    return get_state(base_url, timeout=timeout_sec)


def _parse_target_parts(target: str) -> tuple[str, str]:
    parts = str(target).split("_")
    if len(parts) >= 6:
        # p9_episode_01_<boss...>_<tag...>
        boss = parts[4]
        tag = parts[5]
        return boss, tag
    return "", ""


def load_supported_entries(project_root: Path | None = None) -> list[dict[str, Any]]:
    root = project_root or Path(__file__).resolve().parent.parent.parent
    targets_file = root / "balatro_mechanics" / "derived" / "p9_supported_targets.txt"
    targets: list[str] = []
    if targets_file.exists():
        for raw in targets_file.read_text(encoding="utf-8-sig").splitlines():
            t = raw.strip()
            if t and not t.startswith("#"):
                targets.append(t)
    if not targets:
        targets = list(DEFAULT_TARGETS)

    entries: list[dict[str, Any]] = []
    for i, target in enumerate(targets):
        boss_key, tag_key = _parse_target_parts(target)
        entries.append(
            {
                "target": target,
                "category": "episode_loop",
                "template": "p9_episode_loop_v1",
                "boss_key": boss_key,
                "tag_key": tag_key,
                "variant": i,
                "description": "multi-round hand+shop+pack+use loop",
            }
        )
    return entries


def select_entries(targets_csv: str | None, limit: int | None, project_root: Path | None = None) -> list[dict[str, Any]]:
    entries = load_supported_entries(project_root)
    by_target = {str(e.get("target") or ""): e for e in entries}

    if targets_csv:
        selected: list[dict[str, Any]] = []
        seen: set[str] = set()
        for raw in str(targets_csv).split(","):
            t = raw.strip()
            if not t or t in seen:
                continue
            if t in by_target:
                selected.append(by_target[t])
                seen.add(t)
        entries = selected

    if limit is not None and int(limit) > 0:
        entries = entries[: int(limit)]
    return entries


def _validate_requirements(action_trace: list[dict[str, Any]]) -> tuple[bool, str]:
    if len(action_trace) < 25:
        return False, "requirement_failed:actions_lt_25"

    counts: dict[str, int] = {}
    for action in action_trace:
        at = str(action.get("action_type") or "").upper()
        counts[at] = counts.get(at, 0) + 1

    checks = [
        (counts.get("NEXT_ROUND", 0) >= 2, "next_round_lt_2"),
        ((counts.get("PLAY", 0) + counts.get("DISCARD", 0)) >= 2, "hand_decisions_lt_2"),
        ((counts.get("REROLL", 0) + counts.get("BUY", 0) + counts.get("SELL", 0) + counts.get("SKIP", 0)) >= 2, "shop_actions_lt_2"),
        (counts.get("PACK", 0) >= 1, "pack_lt_1"),
        (counts.get("USE", 0) >= 1, "use_lt_1"),
        (counts.get("SELECT", 0) >= 1, "blind_select_lt_1"),
    ]
    for ok, reason in checks:
        if not ok:
            return False, f"requirement_failed:{reason}"
    return True, ""


def generate_one_trace(
    *,
    base_url: str,
    entry: dict[str, Any],
    max_steps: int,
    seed: str,
    timeout_sec: float,
    wait_sleep: float,
    out_dir: Path | None,
) -> dict[str, Any]:
    target = str(entry.get("target") or "")

    connection_reason = _probe_connection_failure(base_url, timeout_sec)
    if connection_reason is not None:
        return {
            "success": False,
            "target": target,
            "template": "p9_episode_loop_v1",
            "steps_used": 0,
            "final_phase": "UNKNOWN",
            "hit_info": {},
            "index_base": 0,
            "failure_reason": connection_reason,
            "start_snapshot": None,
            "action_trace": [],
            "oracle_start_state_path": None,
        }

    try:
        index_base, index_probe = detect_index_base(base_url, seed, timeout_sec, wait_sleep)
        state, prep_meta = _prepare_shop_start(base_url, seed, timeout_sec, wait_sleep, index_base)
    except Exception as exc:
        return {
            "success": False,
            "target": target,
            "template": "p9_episode_loop_v1",
            "steps_used": 0,
            "final_phase": "UNKNOWN",
            "hit_info": {},
            "index_base": 0,
            "failure_reason": _classify_connection_failure_text(str(exc)),
            "start_snapshot": None,
            "action_trace": [],
            "oracle_start_state_path": None,
        }

    hit_info: dict[str, Any] = {
        "category": entry.get("category"),
        "description": entry.get("description"),
        "boss_key": entry.get("boss_key"),
        "tag_key": entry.get("tag_key"),
        "prep_meta": prep_meta,
    }

    try:
        action_trace: list[dict[str, Any]] = []
        steps = 0

        oracle_start_state_path: str | None = None
        if out_dir is not None:
            oracle_start_state_path = _save_start_state(base_url, out_dir, target, timeout_sec)

        start_snapshot = canonical_snapshot(state, seed=seed)

        actions_plan = _episode_actions(prep_meta, int(entry.get("variant") or 0))
        for spec in actions_plan:
            if steps >= int(max_steps):
                break
            action_type = str(spec.get("action_type") or "WAIT").upper()
            params = dict(spec.get("params") or {})

            action_line: dict[str, Any] = {
                "schema_version": "action_v1",
                "phase": state_phase(state),
                "action_type": action_type,
                "params": {"index_base": int(index_base), **params},
            }
            if "indices" in spec:
                action_line["indices"] = [int(i) for i in (spec.get("indices") or [])]
            if "index" in spec:
                action_line["index"] = int(spec.get("index") or 0)

            action_trace.append(action_line)
            steps += 1

        ok, reason = _validate_requirements(action_trace)
        if not ok:
            return {
                "success": False,
                "target": target,
                "template": "p9_episode_loop_v1",
                "steps_used": steps,
                "final_phase": state_phase(state),
                "hit_info": hit_info,
                "index_base": int(index_base),
                "failure_reason": f"{reason}; {_state_context_summary(state)}",
                "start_snapshot": start_snapshot,
                "action_trace": action_trace,
                "oracle_start_state_path": oracle_start_state_path,
            }

        _save_state_if_possible(base_url, oracle_start_state_path, timeout_sec)

        meta = start_snapshot.setdefault("_meta", {})
        if isinstance(meta, dict):
            meta["index_base_detected"] = int(index_base)
            meta["index_probe"] = dict(index_probe)
            meta["p9_target"] = target
            meta["p9_template"] = "p9_episode_loop_v1"
            meta["p9_boss_key"] = entry.get("boss_key")
            meta["p9_tag_key"] = entry.get("tag_key")
            meta["p9_prep_meta"] = prep_meta

        return {
            "success": True,
            "target": target,
            "template": "p9_episode_loop_v1",
            "steps_used": steps,
            "final_phase": state_phase(state),
            "hit_info": hit_info,
            "index_base": int(index_base),
            "failure_reason": None,
            "start_snapshot": start_snapshot,
            "action_trace": action_trace,
            "oracle_start_state_path": oracle_start_state_path,
        }

    except (RPCError, ConnectionError, ValueError) as exc:
        st = get_state(base_url, timeout=timeout_sec)
        return {
            "success": False,
            "target": target,
            "template": "p9_episode_loop_v1",
            "steps_used": 0,
            "final_phase": state_phase(st),
            "hit_info": hit_info,
            "index_base": int(index_base),
            "failure_reason": f"{exc}; {_state_context_summary(st)}",
            "start_snapshot": canonical_snapshot(st, seed=seed),
            "action_trace": [],
            "oracle_start_state_path": None,
        }


def main() -> int:
    args = parse_args()
    if not health(args.base_url):
        print(f"base_url unhealthy: {args.base_url}. Start balatrobot serve first.")
        return 2

    project_root = Path(__file__).resolve().parent.parent.parent
    entry = next((e for e in load_supported_entries(project_root) if str(e.get("target") or "") == args.target), None)
    if entry is None:
        print(f"ERROR: unknown target: {args.target}")
        print("available targets:")
        for e in load_supported_entries(project_root):
            print(" -", e.get("target"))
        return 2

    out_dir: Path | None = None
    if args.out_dir:
        out_dir = Path(args.out_dir)
        if not out_dir.is_absolute():
            out_dir = project_root / out_dir
        out_dir.mkdir(parents=True, exist_ok=True)

    result = generate_one_trace(
        base_url=args.base_url,
        entry=entry,
        max_steps=int(args.max_steps),
        seed=args.seed,
        timeout_sec=float(args.timeout_sec),
        wait_sleep=float(args.wait_sleep),
        out_dir=out_dir,
    )

    print(
        f"target={result.get('target')} success={result.get('success')} "
        f"steps_used={result.get('steps_used')} final_phase={result.get('final_phase')}"
    )
    print("hit_info=", json.dumps(result.get("hit_info") or {}, ensure_ascii=False))

    if out_dir is not None and result.get("start_snapshot") is not None:
        snapshot_path = out_dir / f"oracle_start_snapshot_{result['target']}.json"
        action_path = out_dir / f"action_trace_{result['target']}.jsonl"
        snapshot_path.write_text(json.dumps(result["start_snapshot"], ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        with action_path.open("w", encoding="utf-8", newline="\n") as fp:
            for action in result.get("action_trace") or []:
                fp.write(json.dumps(action, ensure_ascii=False) + "\n")
        print(f"wrote: {snapshot_path}")
        print(f"wrote: {action_path}")

    return 0 if bool(result.get("success")) else 1


if __name__ == "__main__":
    raise SystemExit(main())
