from __future__ import annotations

if __package__ is None or __package__ == "":
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

import argparse
import json
import math
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sim.core.engine import SimEnv
from sim.oracle.canonicalize_real import canonicalize_real_state
from sim.oracle.extract_rng_outcomes import extract_rng_outcomes
from trainer.env_client import RPCError, _call_method, get_state, health


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="P37 shop/pack probability parity audit (oracle vs sim replay).")
    parser.add_argument("--base-url", default="http://127.0.0.1:12346")
    parser.add_argument("--seed", default="P37PROB01")
    parser.add_argument("--samples", type=int, default=240, help="Target reroll sample count (recommended 200-500).")
    parser.add_argument("--pack-interval", type=int, default=5, help="Every N rerolls attempt one pack buy/open path.")
    parser.add_argument("--timeout-sec", type=float, default=8.0)
    parser.add_argument("--sleep-sec", type=float, default=0.01)
    parser.add_argument("--warn-kl", type=float, default=0.08)
    parser.add_argument("--warn-l1", type=float, default=0.15)
    parser.add_argument("--out-dir", default="docs/artifacts/p37")
    return parser.parse_args()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def _market_cards(state: dict[str, Any], field: str) -> list[dict[str, Any]]:
    block = state.get(field) if isinstance(state.get(field), dict) else {}
    cards = block.get("cards") if isinstance(block.get("cards"), list) else []
    out: list[dict[str, Any]] = []
    for idx, card in enumerate(cards):
        if not isinstance(card, dict):
            continue
        key = str(card.get("key") or "").strip().lower()
        set_name = str(card.get("set") or card.get("kind") or "").strip().upper()
        out.append({"key": key, "set": set_name, "slot": int(card.get("slot_index") if isinstance(card.get("slot_index"), int) else idx)})
    return out


def _pack_choice_keys(state: dict[str, Any]) -> list[str]:
    keys: list[str] = []
    candidates = [state.get("pack_choices"), state.get("pack"), state.get("booster"), state.get("booster_pack")]
    for cand in candidates:
        cards: list[Any] = []
        if isinstance(cand, list):
            cards = cand
        elif isinstance(cand, dict):
            raw_cards = cand.get("cards")
            if isinstance(raw_cards, list):
                cards = raw_cards
        for item in cards:
            if isinstance(item, dict):
                key = str(item.get("key") or "").strip().lower()
            else:
                key = str(item or "").strip().lower()
            if key:
                keys.append(key)
    return sorted(set(keys))


def _init_metric_counters() -> dict[str, Counter[str]]:
    return {
        "shop_set": Counter(),
        "shop_key": Counter(),
        "voucher_key": Counter(),
        "pack_offer_set": Counter(),
        "pack_offer_key": Counter(),
        "pack_choice_key": Counter(),
    }


def _ingest_state_metrics(counters: dict[str, Counter[str]], state: dict[str, Any]) -> None:
    for item in _market_cards(state, "shop"):
        if item["set"]:
            counters["shop_set"][item["set"]] += 1
        if item["key"]:
            counters["shop_key"][item["key"]] += 1
    for item in _market_cards(state, "vouchers"):
        if item["key"]:
            counters["voucher_key"][item["key"]] += 1
    for item in _market_cards(state, "packs"):
        if item["set"]:
            counters["pack_offer_set"][item["set"]] += 1
        if item["key"]:
            counters["pack_offer_key"][item["key"]] += 1
    for key in _pack_choice_keys(state):
        counters["pack_choice_key"][key] += 1


def _goto_shop(base_url: str, seed: str, timeout_sec: float) -> dict[str, Any]:
    try:
        _call_method(base_url, "menu", {}, timeout=timeout_sec)
    except Exception:
        pass
    _call_method(base_url, "start", {"deck": "RED", "stake": "WHITE", "seed": seed}, timeout=timeout_sec)
    _call_method(base_url, "select", {"index": 0}, timeout=timeout_sec)
    _call_method(base_url, "set", {"chips": 999999}, timeout=timeout_sec)
    _call_method(base_url, "play", {"cards": [0]}, timeout=timeout_sec)
    _call_method(base_url, "cash_out", {}, timeout=timeout_sec)
    _call_method(base_url, "set", {"money": 999999}, timeout=timeout_sec)
    state = get_state(base_url, timeout=timeout_sec)
    if str(state.get("state") or "") != "SHOP":
        raise RuntimeError(f"expected SHOP state after bootstrap, got {state.get('state')!r}")
    return state


@dataclass
class OracleCollection:
    start_state: dict[str, Any] | None
    steps: list[dict[str, Any]]
    metrics: dict[str, Counter[str]]
    reroll_count: int
    pack_buy_count: int
    pack_open_count: int
    errors: list[str]


def _record_oracle_step(
    *,
    base_url: str,
    timeout_sec: float,
    before: dict[str, Any],
    action: dict[str, Any],
    method: str,
    params: dict[str, Any],
    step_idx: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    _call_method(base_url, method, params, timeout=timeout_sec)
    after = get_state(base_url, timeout=timeout_sec)
    outcomes = extract_rng_outcomes(before, after, action=action, step_id=step_idx)
    return after, {"action": action, "outcomes": [dict(x) for x in outcomes if isinstance(x, dict)]}


def _collect_oracle(args: argparse.Namespace) -> OracleCollection:
    errors: list[str] = []
    steps: list[dict[str, Any]] = []
    metrics = _init_metric_counters()
    reroll_count = 0
    pack_buy_count = 0
    pack_open_count = 0

    state = _goto_shop(args.base_url, args.seed, float(args.timeout_sec))
    start_state = dict(state)
    _ingest_state_metrics(metrics, state)

    for sample_id in range(max(1, int(args.samples))):
        if str(state.get("state") or "") != "SHOP":
            try:
                state = _goto_shop(args.base_url, f"{args.seed}_{sample_id}", float(args.timeout_sec))
            except Exception as exc:
                errors.append(f"goto_shop_failed@{sample_id}:{exc}")
                break
        try:
            _call_method(args.base_url, "set", {"money": 999999}, timeout=float(args.timeout_sec))
            state = get_state(args.base_url, timeout=float(args.timeout_sec))
        except Exception as exc:
            errors.append(f"set_money_failed@{sample_id}:{exc}")

        # Periodically force a pack buy/open path so pack-choice distributions are sampled too.
        if int(args.pack_interval) > 0 and (sample_id % int(args.pack_interval) == 0):
            packs = _market_cards(state, "packs")
            if packs:
                before_buy = state
                buy_action = {
                    "schema_version": "action_v1",
                    "phase": "SHOP",
                    "action_type": "SHOP_BUY",
                    "params": {"pack_index": 0, "index_base": 0},
                }
                try:
                    state, step = _record_oracle_step(
                        base_url=args.base_url,
                        timeout_sec=float(args.timeout_sec),
                        before=before_buy,
                        action=buy_action,
                        method="buy",
                        params={"pack": 0},
                        step_idx=len(steps),
                    )
                    steps.append(step)
                    pack_buy_count += 1
                    _ingest_state_metrics(metrics, state)
                except (RPCError, RuntimeError, ValueError, ConnectionError) as exc:
                    errors.append(f"pack_buy_failed@{sample_id}:{exc}")

                if str(state.get("state") or "") == "SMODS_BOOSTER_OPENED":
                    before_open = state
                    pack_action = {
                        "schema_version": "action_v1",
                        "phase": "SMODS_BOOSTER_OPENED",
                        "action_type": "PACK_OPEN",
                        "params": {"choice_index": 0, "index_base": 0},
                    }
                    try:
                        state, step = _record_oracle_step(
                            base_url=args.base_url,
                            timeout_sec=float(args.timeout_sec),
                            before=before_open,
                            action=pack_action,
                            method="pack",
                            params={"card": 0},
                            step_idx=len(steps),
                        )
                        steps.append(step)
                        pack_open_count += 1
                        _ingest_state_metrics(metrics, state)
                    except (RPCError, RuntimeError, ValueError, ConnectionError) as exc:
                        errors.append(f"pack_open_failed@{sample_id}:{exc}")

        before_reroll = state
        reroll_action = {"schema_version": "action_v1", "phase": "SHOP", "action_type": "SHOP_REROLL", "params": {"index_base": 0}}
        try:
            state, step = _record_oracle_step(
                base_url=args.base_url,
                timeout_sec=float(args.timeout_sec),
                before=before_reroll,
                action=reroll_action,
                method="reroll",
                params={},
                step_idx=len(steps),
            )
            steps.append(step)
            reroll_count += 1
            _ingest_state_metrics(metrics, state)
        except (RPCError, RuntimeError, ValueError, ConnectionError) as exc:
            errors.append(f"reroll_failed@{sample_id}:{exc}")
            break

        time.sleep(max(0.0, float(args.sleep_sec)))

    return OracleCollection(
        start_state=start_state,
        steps=steps,
        metrics=metrics,
        reroll_count=reroll_count,
        pack_buy_count=pack_buy_count,
        pack_open_count=pack_open_count,
        errors=errors,
    )


@dataclass
class SimReplay:
    metrics: dict[str, Counter[str]]
    replay_errors: list[str]
    replayed_steps: int


def _replay_sim(*, start_state: dict[str, Any], steps: list[dict[str, Any]], seed: str) -> SimReplay:
    replay_errors: list[str] = []
    counters = _init_metric_counters()
    replayed_steps = 0

    canonical_start = canonicalize_real_state(start_state, seed=seed, rng_events=[], rng_cursor=0)
    env = SimEnv(seed=seed)
    env.reset(from_snapshot=canonical_start)
    state = env.get_state()
    _ingest_state_metrics(counters, state)

    for idx, step in enumerate(steps):
        action = step.get("action") if isinstance(step.get("action"), dict) else {}
        outcomes = step.get("outcomes") if isinstance(step.get("outcomes"), list) else []
        action_use = dict(action)
        action_use["phase"] = str(state.get("state") or action_use.get("phase") or "UNKNOWN")
        action_use["rng_replay"] = {
            "enabled": len(outcomes) > 0,
            "source": "p37_probability_audit_oracle_outcomes",
            "outcomes": [dict(x) for x in outcomes if isinstance(x, dict)],
        }
        try:
            state, _reward, _done, _info = env.step(action_use)
            replayed_steps += 1
            _ingest_state_metrics(counters, state)
        except Exception as exc:
            replay_errors.append(f"replay_failed@{idx}:{exc}")
            break

    return SimReplay(metrics=counters, replay_errors=replay_errors, replayed_steps=replayed_steps)


def _counter_probs(counter: Counter[str]) -> dict[str, float]:
    total = float(sum(counter.values()))
    if total <= 0:
        return {}
    return {k: float(v) / total for k, v in sorted(counter.items()) if v > 0}


def _kl_divergence(p: dict[str, float], q: dict[str, float], eps: float = 1e-12) -> float:
    keys = sorted(set(p.keys()) | set(q.keys()))
    if not keys:
        return 0.0
    acc = 0.0
    for key in keys:
        pv = float(p.get(key, 0.0))
        qv = float(q.get(key, 0.0))
        if pv <= 0.0:
            continue
        acc += pv * math.log((pv + eps) / (qv + eps))
    return float(acc)


def _chi_square(obs: Counter[str], exp: Counter[str]) -> float:
    obs_total = float(sum(obs.values()))
    exp_total = float(sum(exp.values()))
    if obs_total <= 0.0 or exp_total <= 0.0:
        return 0.0
    chi2 = 0.0
    keys = sorted(set(obs.keys()) | set(exp.keys()))
    for key in keys:
        expected = (float(exp.get(key, 0)) / exp_total) * obs_total
        if expected <= 1e-12:
            continue
        observed = float(obs.get(key, 0))
        chi2 += ((observed - expected) ** 2) / expected
    return float(chi2)


def _l1_distance(p: dict[str, float], q: dict[str, float]) -> float:
    keys = sorted(set(p.keys()) | set(q.keys()))
    if not keys:
        return 0.0
    return float(sum(abs(float(p.get(k, 0.0)) - float(q.get(k, 0.0))) for k in keys))


def _compare_metric(name: str, oracle_counter: Counter[str], sim_counter: Counter[str]) -> dict[str, Any]:
    oracle_probs = _counter_probs(oracle_counter)
    sim_probs = _counter_probs(sim_counter)
    return {
        "metric": name,
        "oracle_total": int(sum(oracle_counter.values())),
        "sim_total": int(sum(sim_counter.values())),
        "oracle_unique": int(len(oracle_counter)),
        "sim_unique": int(len(sim_counter)),
        "overlap_unique": int(len(set(oracle_counter.keys()) & set(sim_counter.keys()))),
        "kl_oracle_to_sim": _kl_divergence(oracle_probs, sim_probs),
        "kl_sim_to_oracle": _kl_divergence(sim_probs, oracle_probs),
        "chi_square_oracle_vs_sim": _chi_square(oracle_counter, sim_counter),
        "l1_distance": _l1_distance(oracle_probs, sim_probs),
        "top_oracle": [{"key": k, "count": int(v)} for k, v in oracle_counter.most_common(10)],
        "top_sim": [{"key": k, "count": int(v)} for k, v in sim_counter.most_common(10)],
    }


def main() -> int:
    args = _parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    json_path = out_dir / f"probability_audit_{stamp}.json"
    md_path = out_dir / f"probability_audit_{stamp}.md"

    known_gaps = [
        "Mechanics CSV exports in balatro_mechanics do not provide official runtime weight constants for all shop/pack draws.",
        "Sim shop/pack stochasticity is replay-driven in parity paths; native standalone weighted sampler is not a complete source-of-truth yet.",
    ]

    if not health(args.base_url):
        payload = {
            "schema": "p37_probability_parity_v1",
            "status": "ORACLE_UNAVAILABLE",
            "base_url": args.base_url,
            "seed": args.seed,
            "samples_requested": int(args.samples),
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "known_gaps": known_gaps
            + [
                f"Oracle endpoint {args.base_url} is unhealthy in this environment; live sampling skipped.",
                "Start balatrobot with valid --love-path/--lovely-path and rerun this script.",
            ],
            "metrics": [],
        }
        _write_json(json_path, payload)
        md = "\n".join(
            [
                "# P37 Probability Parity Audit",
                "",
                "- status: ORACLE_UNAVAILABLE",
                f"- base_url: {args.base_url}",
                f"- samples_requested: {int(args.samples)}",
                "",
                "## Known Gaps",
                *[f"- {x}" for x in payload["known_gaps"]],
                "",
            ]
        )
        md_path.write_text(md, encoding="utf-8")
        print(json.dumps({"status": payload["status"], "json": str(json_path), "md": str(md_path)}, ensure_ascii=False))
        return 0

    oracle = _collect_oracle(args)
    if oracle.start_state is None:
        raise RuntimeError("oracle collection failed to produce a start_state")
    sim = _replay_sim(start_state=oracle.start_state, steps=oracle.steps, seed=str(args.seed))

    metric_names = ["shop_set", "shop_key", "voucher_key", "pack_offer_set", "pack_offer_key", "pack_choice_key"]
    metric_rows: list[dict[str, Any]] = []
    warnings: list[str] = []
    for metric in metric_names:
        row = _compare_metric(metric, oracle.metrics[metric], sim.metrics[metric])
        metric_rows.append(row)
        if float(row["kl_oracle_to_sim"]) > float(args.warn_kl):
            warnings.append(f"{metric}: kl_oracle_to_sim={row['kl_oracle_to_sim']:.6f} > warn_kl={float(args.warn_kl):.6f}")
        if float(row["l1_distance"]) > float(args.warn_l1):
            warnings.append(f"{metric}: l1_distance={row['l1_distance']:.6f} > warn_l1={float(args.warn_l1):.6f}")

    status = "PASS"
    if oracle.errors or sim.replay_errors or warnings:
        status = "WARN"

    payload = {
        "schema": "p37_probability_parity_v1",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "status": status,
        "base_url": args.base_url,
        "seed": args.seed,
        "samples_requested": int(args.samples),
        "samples_collected_reroll": int(oracle.reroll_count),
        "pack_buy_count": int(oracle.pack_buy_count),
        "pack_open_count": int(oracle.pack_open_count),
        "oracle_steps": int(len(oracle.steps)),
        "sim_replayed_steps": int(sim.replayed_steps),
        "oracle_errors": oracle.errors,
        "sim_replay_errors": sim.replay_errors,
        "soft_thresholds": {"warn_kl": float(args.warn_kl), "warn_l1": float(args.warn_l1)},
        "warnings": warnings,
        "metrics": metric_rows,
        "known_gaps": known_gaps,
    }
    _write_json(json_path, payload)

    lines = [
        "# P37 Probability Parity Audit",
        "",
        f"- status: {status}",
        f"- base_url: {args.base_url}",
        f"- seed: {args.seed}",
        f"- samples_requested: {int(args.samples)}",
        f"- samples_collected_reroll: {int(oracle.reroll_count)}",
        f"- pack_buy_count: {int(oracle.pack_buy_count)}",
        f"- pack_open_count: {int(oracle.pack_open_count)}",
        f"- oracle_steps: {int(len(oracle.steps))}",
        f"- sim_replayed_steps: {int(sim.replayed_steps)}",
        "",
        "## Divergence Metrics (soft warnings only)",
        "",
        "| metric | oracle_total | sim_total | overlap_unique | kl_oracle_to_sim | kl_sim_to_oracle | l1_distance | chi_square_oracle_vs_sim |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in metric_rows:
        lines.append(
            "| {metric} | {oracle_total} | {sim_total} | {overlap_unique} | {kl_oracle_to_sim:.6f} | {kl_sim_to_oracle:.6f} | {l1_distance:.6f} | {chi_square_oracle_vs_sim:.6f} |".format(
                **row
            )
        )

    lines.extend(["", "## Warnings"])
    if warnings:
        lines.extend([f"- {x}" for x in warnings])
    else:
        lines.append("- none")

    lines.extend(["", "## Known Gaps"])
    lines.extend([f"- {x}" for x in known_gaps])

    if oracle.errors:
        lines.extend(["", "## Oracle Errors"])
        lines.extend([f"- {x}" for x in oracle.errors])
    if sim.replay_errors:
        lines.extend(["", "## Sim Replay Errors"])
        lines.extend([f"- {x}" for x in sim.replay_errors])

    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": status, "json": str(json_path), "md": str(md_path)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

