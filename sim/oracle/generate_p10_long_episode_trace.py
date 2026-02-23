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
    {"target": "p10_episode_01_normal_a", "stake": "white", "category": "long_episode", "variant": 0},
    {"target": "p10_episode_02_normal_b", "stake": "white", "category": "long_episode", "variant": 1},
    {"target": "p10_episode_03_gold", "stake": "gold", "category": "long_episode_gold", "variant": 2},
    {"target": "p10_episode_04_stress", "stake": "gold", "category": "stress_shop_pack_use", "variant": 3},
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
    parser = argparse.ArgumentParser(description="Generate P10 long-episode trace (stake+boss/tag+shop/pack/use).")
    parser.add_argument("--base-url", default="http://127.0.0.1:12346")
    parser.add_argument("--target", required=True)
    parser.add_argument("--max-steps", type=int, default=1200)
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


def _market_index(state: dict[str, Any], field: str, key_hint: str | None = None) -> int:
    market = state.get(field) if isinstance(state.get(field), dict) else {}
    cards = market.get("cards") if isinstance(market.get("cards"), list) else []
    if key_hint:
        low = str(key_hint).strip().lower()
        for idx, card in enumerate(cards):
            if isinstance(card, dict) and str(card.get("key") or "").strip().lower() == low:
                return idx
    return 0


def _prepare_episode_start(base_url: str, seed: str, timeout_sec: float, wait_sleep: float) -> tuple[dict[str, Any], int, dict[str, Any]]:
    hard_reset_fixture(base_url, seed, timeout_sec, wait_sleep)
    state = prepare_selecting_hand(base_url, seed, timeout_sec, wait_sleep)
    index_base, _ = detect_index_base(base_url, seed, timeout_sec, wait_sleep)

    prep_meta: dict[str, Any] = {"voucher_key": None, "pack_key": None, "consumable_key": None}

    try:
        _call_method(base_url, "set", {"money": 300}, timeout=timeout_sec)
    except Exception:
        pass

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
    return state, int(index_base), prep_meta


def _episode_actions_long(prep_meta: dict[str, Any], variant: int, stake_name: str, seed: str) -> list[dict[str, Any]]:
    voucher_idx = int(prep_meta.get("voucher_idx") or 0)
    pack_idx = int(prep_meta.get("pack_idx") or 0)
    card_idx = int(prep_meta.get("card_idx") or 0)
    consumable_idx = int(prep_meta.get("consumable_idx") or 0)
    joker_idx = int(prep_meta.get("joker_idx") or 0)

    actions: list[dict[str, Any]] = [{"action_type": "START", "seed": seed, "stake": str(stake_name).upper()}]
    cycles = 7 if variant != 3 else 8
    for i in range(cycles):
        blind_index = i % 3
        actions.extend(
            [
                {"action_type": "SELECT", "index": blind_index, "params": {}},
                {"action_type": "DISCARD", "indices": [0], "params": {}},
                {"action_type": "PLAY", "indices": [0], "params": {}},
                {"action_type": "PLAY", "indices": [0], "params": {}},
                {"action_type": "CASH_OUT", "params": {}},
                {"action_type": "BUY", "params": {"voucher": voucher_idx}},
                {"action_type": "BUY", "params": {"pack": pack_idx}},
                {"action_type": "PACK", "params": {"card": 0}},
                {"action_type": "USE", "params": {"consumable": consumable_idx}},
                {"action_type": "REROLL", "params": {}},
                {"action_type": "BUY", "params": {"card": card_idx}},
                {"action_type": "NEXT_ROUND", "params": {}},
            ]
        )
        if i % 2 == 1:
            actions.append({"action_type": "SELL", "params": {"joker": joker_idx}})
    actions.extend([{"action_type": "WAIT", "params": {}}, {"action_type": "WAIT", "params": {}}])
    return actions


def _validate_requirements(action_trace: list[dict[str, Any]], require_stake: str) -> tuple[bool, str]:
    if len(action_trace) < 80:
        return False, "requirement_failed:actions_lt_80"
    counts: dict[str, int] = {}
    for action in action_trace:
        at = str(action.get("action_type") or "").upper()
        counts[at] = counts.get(at, 0) + 1
    checks = [
        (counts.get("NEXT_ROUND", 0) >= 3, "next_round_lt_3"),
        ((counts.get("PLAY", 0) + counts.get("DISCARD", 0)) >= 2, "hand_decisions_lt_2"),
        ((counts.get("REROLL", 0) + counts.get("BUY", 0) + counts.get("SELL", 0) + counts.get("SKIP", 0)) >= 2, "shop_actions_lt_2"),
        (counts.get("PACK", 0) >= 1, "pack_lt_1"),
        (counts.get("USE", 0) >= 1, "use_lt_1"),
        (counts.get("SELECT", 0) >= 1, "blind_select_lt_1"),
        (any(str(a.get("stake") or "").strip().lower() == require_stake.lower() for a in action_trace), "stake_not_applied"),
    ]
    for ok, reason in checks:
        if not ok:
            return False, f"requirement_failed:{reason}"
    return True, ""


def load_supported_entries(project_root: Path | None = None) -> list[dict[str, Any]]:
    root = project_root or Path(__file__).resolve().parent.parent.parent
    derived = root / "balatro_mechanics" / "derived"
    targets_file = derived / "p10_supported_targets.txt"
    if targets_file.exists():
        out: list[dict[str, Any]] = []
        for i, raw in enumerate(targets_file.read_text(encoding="utf-8-sig").splitlines()):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            target = line
            stake = "gold" if "gold" in target.lower() else "white"
            out.append(
                {
                    "target": target,
                    "stake": stake,
                    "category": "long_episode_gold" if stake == "gold" else "long_episode",
                    "variant": i,
                }
            )
        if out:
            return out
    return [dict(x) for x in DEFAULT_TARGETS]


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
    stake_name = str(entry.get("stake") or "white").strip().lower() or "white"

    connection_reason = _probe_connection_failure(base_url, timeout_sec)
    if connection_reason is not None:
        return {
            "success": False,
            "target": target,
            "template": "p10_long_episode_v1",
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
        state, index_base, prep_meta = _prepare_episode_start(base_url, seed, timeout_sec, wait_sleep)
    except Exception as exc:
        return {
            "success": False,
            "target": target,
            "template": "p10_long_episode_v1",
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
        "description": "p10 long episode loop",
        "stake": stake_name,
        "prep_meta": prep_meta,
    }

    try:
        action_trace: list[dict[str, Any]] = []
        steps = 0

        oracle_start_state_path: str | None = None
        if out_dir is not None:
            oracle_start_state_path = _save_start_state(base_url, out_dir, target, timeout_sec)

        start_snapshot = canonical_snapshot(state, seed=seed)
        actions_plan = _episode_actions_long(prep_meta, int(entry.get("variant") or 0), stake_name, f"{seed}-{target}")

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
            if "seed" in spec:
                action_line["seed"] = str(spec.get("seed") or "")
            if "stake" in spec:
                action_line["stake"] = str(spec.get("stake") or "").upper()
            action_trace.append(action_line)
            steps += 1

        ok, reason = _validate_requirements(action_trace, require_stake=stake_name)
        if not ok:
            return {
                "success": False,
                "target": target,
                "template": "p10_long_episode_v1",
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
            meta["p10_target"] = target
            meta["p10_template"] = "p10_long_episode_v1"
            meta["p10_stake"] = stake_name
            meta["p10_prep_meta"] = prep_meta

        return {
            "success": True,
            "target": target,
            "template": "p10_long_episode_v1",
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
            "template": "p10_long_episode_v1",
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

