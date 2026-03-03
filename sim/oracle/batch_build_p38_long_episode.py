from __future__ import annotations

if __package__ is None or __package__ == "":
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

import argparse
import json
import re
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sim.oracle.canonicalize_real import canonicalize_real_state
from sim.oracle.run_oracle_trace import apply_action, phase_default_action
from trainer.env_client import _call_method, get_state, health


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build P38 long-horizon oracle/sim stress episodes.")
    parser.add_argument("--base-url", default="http://127.0.0.1:12346")
    parser.add_argument("--out-dir", default="docs/artifacts/p38/long_episode")
    parser.add_argument("--run-id", default="")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=260)
    parser.add_argument("--seeds", default="AAAAAAA,BBBBBBB,CCCCCCC,DDDDDDD,EEEEEEE")
    parser.add_argument("--scope", default="p37_action_fidelity_core")
    parser.add_argument("--timeout-sec", type=float, default=8.0)
    parser.add_argument("--wait-sleep", type=float, default=0.05)
    parser.add_argument("--python", default=sys.executable)
    return parser.parse_args()


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def _parse_seed_list(raw: str) -> list[str]:
    seeds: list[str] = []
    seen: set[str] = set()
    for part in str(raw or "").split(","):
        seed = part.strip()
        if not seed or seed in seen:
            continue
        seen.add(seed)
        seeds.append(seed)
    return seeds


def _phase(state: dict[str, Any]) -> str:
    return str(state.get("state") or "UNKNOWN").upper()


def _market_cards(state: dict[str, Any], field: str) -> list[dict[str, Any]]:
    raw = state.get(field)
    if isinstance(raw, dict):
        cards = raw.get("cards")
    else:
        cards = raw
    if not isinstance(cards, list):
        return []
    out: list[dict[str, Any]] = []
    for item in cards:
        if isinstance(item, dict):
            out.append(item)
    return out


def _policy_action(state: dict[str, Any], *, step_idx: int, seed: str) -> dict[str, Any]:
    phase = _phase(state)
    if phase in {"MENU", "GAME_OVER"}:
        return {
            "schema_version": "action_v1",
            "phase": phase,
            "action_type": "START",
            "seed": seed,
            "stake": "WHITE",
        }

    if phase == "BLIND_SELECT":
        return {
            "schema_version": "action_v1",
            "phase": phase,
            "action_type": "SELECT",
            "index": 0,
            "params": {"index_base": 0},
        }

    if phase == "SELECTING_HAND":
        hand_cards = _market_cards(state, "hand")
        round_info = state.get("round") if isinstance(state.get("round"), dict) else {}
        hands_left = int(round_info.get("hands_left") or 0)
        discards_left = int(round_info.get("discards_left") or 0)
        if hands_left > 0 and hand_cards:
            take = max(1, min(5, len(hand_cards)))
            return {
                "schema_version": "action_v1",
                "phase": phase,
                "action_type": "PLAY",
                "indices": list(range(take)),
                "params": {"index_base": 0},
            }
        if discards_left > 0 and hand_cards:
            take = max(1, min(3, len(hand_cards)))
            return {
                "schema_version": "action_v1",
                "phase": phase,
                "action_type": "DISCARD",
                "indices": list(range(take)),
                "params": {"index_base": 0},
            }
        return {"schema_version": "action_v1", "phase": phase, "action_type": "WAIT", "params": {"index_base": 0}}

    if phase == "ROUND_EVAL":
        return {"schema_version": "action_v1", "phase": phase, "action_type": "CASH_OUT", "params": {"index_base": 0}}

    if phase == "SHOP":
        shop_cards = _market_cards(state, "shop")
        vouchers = _market_cards(state, "vouchers")
        packs = _market_cards(state, "packs")
        consumables = _market_cards(state, "consumables")

        mod = step_idx % 11
        if packs and mod in {2, 7}:
            return {
                "schema_version": "action_v1",
                "phase": phase,
                "action_type": "SHOP_BUY",
                "params": {"pack_index": 0, "index_base": 0},
            }
        if mod == 3:
            return {"schema_version": "action_v1", "phase": phase, "action_type": "SHOP_REROLL", "params": {"index_base": 0}}
        if shop_cards and mod in {4, 8}:
            return {
                "schema_version": "action_v1",
                "phase": phase,
                "action_type": "SHOP_BUY",
                "params": {"shop_index": 0, "index_base": 0},
            }
        if vouchers and mod == 5:
            return {
                "schema_version": "action_v1",
                "phase": phase,
                "action_type": "SHOP_BUY",
                "params": {"voucher_index": 0, "index_base": 0},
            }
        if consumables and mod in {6, 9}:
            return {
                "schema_version": "action_v1",
                "phase": phase,
                "action_type": "CONSUMABLE_USE",
                "params": {"consumable_index": 0, "hand_indices": [0], "cards": [0], "index_base": 0},
            }
        return {"schema_version": "action_v1", "phase": phase, "action_type": "SHOP_REROLL", "params": {"index_base": 0}}

    if "BOOSTER" in phase or "PACK" in phase:
        return {
            "schema_version": "action_v1",
            "phase": phase,
            "action_type": "PACK_OPEN",
            "params": {"choice_index": 0, "index_base": 0},
        }

    return {"schema_version": "action_v1", "phase": phase, "action_type": "WAIT", "params": {"index_base": 0}}


def _normalize_action(action: dict[str, Any], *, fallback_phase: str) -> dict[str, Any]:
    out = dict(action)
    out["schema_version"] = "action_v1"
    out["phase"] = str(out.get("phase") or fallback_phase or "UNKNOWN").upper()
    out["action_type"] = str(out.get("action_type") or "WAIT").upper()
    params = dict(out.get("params") or {})
    if "index_base" not in params:
        params["index_base"] = 0
    out["params"] = params
    return out


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as fp:
        for row in rows:
            fp.write(json.dumps(row, ensure_ascii=False) + "\n")


def _run_cmd(cmd: list[str], *, cwd: Path) -> tuple[int, str, str]:
    proc = subprocess.run(cmd, cwd=str(cwd), text=True, capture_output=True, check=False)
    return int(proc.returncode), str(proc.stdout or ""), str(proc.stderr or "")


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if not path.exists():
        return out
    with path.open("r", encoding="utf-8-sig") as fp:
        for line in fp:
            text = line.strip()
            if not text:
                continue
            try:
                obj = json.loads(text)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                out.append(obj)
    return out


def _extract_numeric(value: Any, *, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _extract_int(value: Any, *, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _trace_metrics(trace_lines: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, dict[str, int]]]:
    snapshots: list[dict[str, Any]] = []
    actions: list[str] = []
    shop_offer_keys: Counter[str] = Counter()
    pack_types: Counter[str] = Counter()
    joker_keys: Counter[str] = Counter()
    shop_state_count = 0

    for line in trace_lines:
        action = line.get("action") if isinstance(line.get("action"), dict) else {}
        actions.append(str(action.get("action_type") or "").upper())
        snap = line.get("canonical_state_snapshot")
        if not isinstance(snap, dict):
            continue
        snapshots.append(snap)
        phase = str(snap.get("phase") or line.get("phase") or "").upper()
        if phase == "SHOP":
            shop_state_count += 1

        shop_cards = _market_cards(snap, "shop")
        for card in shop_cards:
            key = str(card.get("key") or "").strip().lower()
            if key:
                shop_offer_keys[key] += 1

        pack_cards = _market_cards(snap, "packs")
        for card in pack_cards:
            pack_type = str(card.get("set") or card.get("kind") or card.get("key") or "").strip().upper()
            if pack_type:
                pack_types[pack_type] += 1

        jokers = snap.get("jokers") if isinstance(snap.get("jokers"), list) else []
        for joker in jokers:
            if not isinstance(joker, dict):
                continue
            jkey = str(joker.get("joker_id") or joker.get("key") or "").strip().lower()
            if jkey:
                joker_keys[jkey] += 1

    first_snap = snapshots[0] if snapshots else {}
    last_snap = snapshots[-1] if snapshots else {}
    first_money = _extract_numeric((first_snap.get("economy") or {}).get("money") if isinstance(first_snap.get("economy"), dict) else 0.0)
    last_money = _extract_numeric((last_snap.get("economy") or {}).get("money") if isinstance(last_snap.get("economy"), dict) else 0.0)
    score_block = last_snap.get("score") if isinstance(last_snap.get("score"), dict) else {}
    round_block = last_snap.get("round") if isinstance(last_snap.get("round"), dict) else {}

    total_score = _extract_numeric(score_block.get("chips"), default=_extract_numeric(round_block.get("chips"), default=0.0))
    rounds_survived = _extract_int(round_block.get("round_num"), default=0)

    metrics = {
        "trace_steps": int(len(trace_lines)),
        "state_snapshots": int(len(snapshots)),
        "total_score": float(total_score),
        "rounds_survived": int(rounds_survived),
        "money_earned": float(last_money - first_money),
        "rerolls_count": int(sum(1 for x in actions if x in {"SHOP_REROLL", "REROLL"})),
        "packs_opened": int(sum(1 for x in actions if x in {"PACK_OPEN", "PACK"})),
        "consumables_used": int(sum(1 for x in actions if x in {"CONSUMABLE_USE", "USE"})),
        "shop_state_count": int(shop_state_count),
    }

    frequencies = {
        "shop_offer_keys": {k: int(v) for k, v in sorted(shop_offer_keys.items()) if v > 0},
        "pack_types": {k: int(v) for k, v in sorted(pack_types.items()) if v > 0},
        "joker_keys": {k: int(v) for k, v in sorted(joker_keys.items()) if v > 0},
    }
    return metrics, frequencies


def _parse_mismatch_count(text: str, returncode: int) -> int:
    if not text:
        return 1 if returncode != 0 else 0
    m = re.search(r"mismatches=(\d+)", text)
    if m:
        return int(m.group(1))
    mismatches = len(re.findall(r"^MISMATCH step=", text, flags=re.MULTILINE))
    if mismatches > 0:
        return int(mismatches)
    if "START_MISMATCH" in text:
        return 1
    return 1 if returncode != 0 else 0


def _collect_episode_actions(
    *,
    base_url: str,
    seed: str,
    max_steps: int,
    timeout_sec: float,
    wait_sleep: float,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any], str | None]:
    state = _bootstrap_shop_state(base_url=base_url, seed=seed, timeout_sec=timeout_sec)

    start_snapshot = canonicalize_real_state(state, seed=seed, rng_events=[], rng_cursor=0)
    actions: list[dict[str, Any]] = []
    error_text: str | None = None
    last_state = state
    for step_idx in range(int(max_steps)):
        phase = _phase(last_state)

        action = _policy_action(last_state, step_idx=step_idx, seed=seed)
        action = _normalize_action(action, fallback_phase=phase)
        executed = dict(action)
        try:
            next_state = apply_action(base_url, action, timeout_sec=timeout_sec, wait_sleep=wait_sleep, seed=seed)
        except Exception as exc:
            fallback = _normalize_action(phase_default_action(last_state, seed), fallback_phase=phase)
            fallback["_fallback_reason"] = str(exc)
            executed = fallback
            try:
                next_state = apply_action(base_url, fallback, timeout_sec=timeout_sec, wait_sleep=wait_sleep, seed=seed)
            except Exception as exc_fallback:
                error_text = f"policy_and_fallback_failed@step={step_idx}:{exc_fallback}"
                break

        actions.append(executed)
        last_state = next_state

    return actions, start_snapshot, last_state, error_text


def _bootstrap_shop_state(*, base_url: str, seed: str, timeout_sec: float) -> dict[str, Any]:
    _call_method(base_url, "menu", {}, timeout=timeout_sec)
    _call_method(
        base_url,
        "start",
        {"deck": "RED", "stake": "WHITE", "seed": seed},
        timeout=timeout_sec,
    )
    _call_method(base_url, "select", {"index": 0}, timeout=timeout_sec)
    _call_method(base_url, "set", {"chips": 999999}, timeout=timeout_sec)
    _call_method(base_url, "play", {"cards": [0]}, timeout=timeout_sec)
    _call_method(base_url, "cash_out", {}, timeout=timeout_sec)
    _call_method(base_url, "set", {"money": 999999}, timeout=timeout_sec)
    state = get_state(base_url, timeout=timeout_sec)
    if _phase(state) != "SHOP":
        raise RuntimeError(f"failed to bootstrap SHOP state, got phase={_phase(state)}")
    return state


def main() -> int:
    args = _parse_args()
    if not health(args.base_url):
        print(f"base_url unhealthy: {args.base_url}")
        return 2

    repo_root = Path(__file__).resolve().parents[2]
    seeds = _parse_seed_list(args.seeds)
    if not seeds:
        raise SystemExit("empty seeds list")

    run_id = str(args.run_id or _now_stamp())
    out_root = Path(args.out_dir)
    if not out_root.is_absolute():
        out_root = (repo_root / out_root).resolve()
    run_dir = out_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    hard_fail_count = 0
    for episode_idx in range(int(args.episodes)):
        seed = seeds[episode_idx % len(seeds)]
        episode_name = f"episode_{episode_idx + 1:03d}"
        episode_dir = run_dir / episode_name
        episode_dir.mkdir(parents=True, exist_ok=True)

        start_snapshot_path = episode_dir / "oracle_start_snapshot.json"
        start_state_path = episode_dir / "oracle_start_state.jkr"
        action_trace_path = episode_dir / "action_trace.jsonl"
        oracle_trace_path = episode_dir / "oracle_trace.jsonl"
        sim_trace_path = episode_dir / "sim_trace.jsonl"
        diff_dump_dir = episode_dir / "diff_dump"
        episode_json_path = run_dir / f"{episode_name}.json"

        row: dict[str, Any] = {
            "schema": "p38_long_episode_row_v1",
            "generated_at": _now_iso(),
            "run_id": run_id,
            "episode_index": int(episode_idx + 1),
            "seed": seed,
            "scope": str(args.scope),
            "max_steps": int(args.max_steps),
            "status": "fail",
            "mismatch_count": 0,
            "paths": {
                "episode_dir": str(episode_dir),
                "start_snapshot": str(start_snapshot_path),
                "start_state": str(start_state_path),
                "action_trace": str(action_trace_path),
                "oracle_trace": str(oracle_trace_path),
                "sim_trace": str(sim_trace_path),
                "diff_dump": str(diff_dump_dir),
            },
            "oracle": {"metrics": {}, "frequencies": {}},
            "sim": {"metrics": {}, "frequencies": {}},
            "commands": {},
            "errors": [],
        }

        collect_error = None
        try:
            actions, start_snapshot, _last_state, collect_error = _collect_episode_actions(
                base_url=args.base_url,
                seed=seed,
                max_steps=int(args.max_steps),
                timeout_sec=float(args.timeout_sec),
                wait_sleep=float(args.wait_sleep),
            )
            _write_json(start_snapshot_path, start_snapshot)
            _write_jsonl(action_trace_path, actions)
            _bootstrap_shop_state(base_url=args.base_url, seed=seed, timeout_sec=float(args.timeout_sec))
            _call_method(args.base_url, "save", {"path": str(start_state_path)}, timeout=float(args.timeout_sec))
        except Exception as exc:
            collect_error = f"collect_failed:{exc}"

        if collect_error:
            row["status"] = "collect_fail"
            row["errors"].append(str(collect_error))
            row["mismatch_count"] = 1
            _write_json(episode_json_path, row)
            rows.append(row)
            hard_fail_count += 1
            print(f"[P38] {episode_name} status=collect_fail seed={seed} error={collect_error}")
            continue

        try:
            _call_method(args.base_url, "load", {"path": str(start_state_path)}, timeout=float(args.timeout_sec))
        except Exception as exc:
            row["status"] = "load_fail"
            row["errors"].append(f"load_start_state_failed:{exc}")
            row["mismatch_count"] = 1
            _write_json(episode_json_path, row)
            rows.append(row)
            hard_fail_count += 1
            print(f"[P38] {episode_name} status=load_fail seed={seed}")
            continue

        oracle_cmd = [
            str(args.python),
            "-B",
            str(repo_root / "sim" / "oracle" / "run_oracle_trace.py"),
            "--base-url",
            str(args.base_url),
            "--seed",
            str(seed),
            "--action-trace",
            str(action_trace_path),
            "--out",
            str(oracle_trace_path),
            "--snapshot-every",
            "1",
            "--timeout-sec",
            str(float(args.timeout_sec)),
            "--wait-sleep",
            str(float(args.wait_sleep)),
        ]
        row["commands"]["oracle_trace"] = oracle_cmd
        oracle_code, oracle_out, oracle_err = _run_cmd(oracle_cmd, cwd=repo_root)
        row["commands"]["oracle_trace_returncode"] = int(oracle_code)
        row["commands"]["oracle_trace_stdout_tail"] = oracle_out[-2000:]
        row["commands"]["oracle_trace_stderr_tail"] = oracle_err[-2000:]
        if oracle_code != 0:
            row["status"] = "oracle_fail"
            row["errors"].append("oracle_trace_failed")
            row["mismatch_count"] = 1
            _write_json(episode_json_path, row)
            rows.append(row)
            hard_fail_count += 1
            print(f"[P38] {episode_name} status=oracle_fail seed={seed}")
            continue

        directed_cmd = [
            str(args.python),
            "-B",
            str(repo_root / "sim" / "tests" / "run_directed_fixture.py"),
            "--oracle-snapshot",
            str(start_snapshot_path),
            "--action-trace",
            str(action_trace_path),
            "--oracle-trace",
            str(oracle_trace_path),
            "--scope",
            str(args.scope),
            "--check-start",
            "--no-fail-fast",
            "--snapshot-every",
            "1",
            "--out-trace",
            str(sim_trace_path),
            "--dump-on-diff",
            str(diff_dump_dir),
        ]
        row["commands"]["directed_trace"] = directed_cmd
        directed_code, directed_out, directed_err = _run_cmd(directed_cmd, cwd=repo_root)
        merged_directed = (directed_out + "\n" + directed_err).strip()
        mismatch_count = _parse_mismatch_count(merged_directed, directed_code)
        row["mismatch_count"] = int(mismatch_count)
        row["commands"]["directed_trace_returncode"] = int(directed_code)
        row["commands"]["directed_trace_stdout_tail"] = directed_out[-3000:]
        row["commands"]["directed_trace_stderr_tail"] = directed_err[-3000:]

        oracle_lines = _load_jsonl(oracle_trace_path)
        sim_lines = _load_jsonl(sim_trace_path)
        oracle_metrics, oracle_freq = _trace_metrics(oracle_lines)
        sim_metrics, sim_freq = _trace_metrics(sim_lines)
        row["oracle"] = {"metrics": oracle_metrics, "frequencies": oracle_freq}
        row["sim"] = {"metrics": sim_metrics, "frequencies": sim_freq}

        if int(mismatch_count) > 0:
            row["status"] = "diff_fail"
            row["errors"].append("mismatch_count_gt_0")
            hard_fail_count += 1
        elif directed_code != 0:
            row["status"] = "directed_fail"
            row["errors"].append("directed_trace_failed_without_mismatch_token")
            hard_fail_count += 1
        else:
            row["status"] = "pass"

        _write_json(episode_json_path, row)
        rows.append(row)
        print(
            f"[P38] {episode_name} status={row['status']} seed={seed} "
            f"steps={oracle_metrics.get('trace_steps', 0)} mismatches={row['mismatch_count']}"
        )

    summary_counter: Counter[str] = Counter(str(x.get("status") or "unknown") for x in rows)
    report = {
        "schema": "p38_long_episode_batch_v1",
        "generated_at": _now_iso(),
        "run_id": run_id,
        "base_url": str(args.base_url),
        "scope": str(args.scope),
        "episodes_requested": int(args.episodes),
        "episodes_total": int(len(rows)),
        "episodes_pass": int(summary_counter.get("pass", 0)),
        "hard_fail_count": int(hard_fail_count),
        "diff_fail_count": int(summary_counter.get("diff_fail", 0)),
        "oracle_fail_count": int(summary_counter.get("oracle_fail", 0)),
        "collect_fail_count": int(summary_counter.get("collect_fail", 0)),
        "seeds": list(seeds),
        "max_steps": int(args.max_steps),
        "run_dir": str(run_dir),
        "results": rows,
        "status": "PASS" if hard_fail_count == 0 else "FAIL",
    }
    report_path = run_dir / "report_p38_long_episode.json"
    _write_json(report_path, report)

    print(
        json.dumps(
            {
                "status": report["status"],
                "run_id": run_id,
                "episodes_total": report["episodes_total"],
                "episodes_pass": report["episodes_pass"],
                "hard_fail_count": report["hard_fail_count"],
                "report": str(report_path),
            },
            ensure_ascii=False,
        )
    )
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
