from __future__ import annotations

if __package__ is None or __package__ == "":
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

import argparse
import csv
import json
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any

from trainer.env_client import RPCError, _call_method, get_state, health


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample shop refreshes and summarize empirical offer distributions.")
    parser.add_argument("--base-url", default="http://127.0.0.1:12346")
    parser.add_argument("--seed", default="P32SHOP01")
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--timeout-sec", type=float, default=8.0)
    parser.add_argument("--sleep-sec", type=float, default=0.02)
    return parser.parse_args()


def _norm_key(text: str) -> str:
    key = str(text or "").strip().lower()
    key = re.sub(r"[^a-z0-9]+", "_", key).strip("_")
    return key


def _joker_key_from_name(name: str) -> str:
    base = _norm_key(name)
    return f"j_{base}" if base else ""


def _load_joker_rarity_map(mech_root: Path) -> tuple[dict[str, str], dict[str, float]]:
    rarity_map: dict[str, str] = {}
    rarity_counts: Counter[str] = Counter()
    path = mech_root / "jokers.csv"
    if not path.exists():
        return rarity_map, {}
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not isinstance(row, dict):
                continue
            name = str(row.get("joker_name") or "").strip()
            rarity = str(row.get("rarity") or "").strip().lower()
            if not name or not rarity:
                continue
            key = _joker_key_from_name(name)
            if key:
                rarity_map[key] = rarity
            rarity_counts[rarity] += 1
    total = sum(rarity_counts.values())
    expected_probs = {k: float(v / total) for k, v in rarity_counts.items()} if total > 0 else {}
    return rarity_map, expected_probs


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
    _call_method(base_url, "set", {"money": 9999}, timeout=timeout_sec)
    state = get_state(base_url, timeout=timeout_sec)
    if str(state.get("state") or "") != "SHOP":
        raise RuntimeError(f"failed to enter SHOP phase, got {state.get('state')!r}")
    return state


def _card_keys(block: Any) -> list[str]:
    if not isinstance(block, dict):
        return []
    cards = block.get("cards") if isinstance(block.get("cards"), list) else []
    out: list[str] = []
    for item in cards:
        if not isinstance(item, dict):
            continue
        key = str(item.get("key") or "").strip().lower()
        if key:
            out.append(key)
    return out


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _chi_square(obs_counts: dict[str, int], expected_probs: dict[str, float]) -> float | None:
    total = sum(obs_counts.values())
    if total <= 0 or not expected_probs:
        return None
    chi2 = 0.0
    used = 0
    for key, prob in expected_probs.items():
        exp = float(prob) * float(total)
        if exp <= 1e-9:
            continue
        obs = float(obs_counts.get(key, 0))
        chi2 += (obs - exp) ** 2 / exp
        used += 1
    return chi2 if used > 0 else None


def main() -> int:
    args = _parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not health(args.base_url):
        raise RuntimeError(f"base_url unhealthy: {args.base_url}")

    project_root = Path(__file__).resolve().parent.parent.parent
    rarity_map, expected_rarity_probs = _load_joker_rarity_map(project_root / "balatro_mechanics")

    state = _goto_shop(args.base_url, args.seed, float(args.timeout_sec))

    joker_counter: Counter[str] = Counter()
    rarity_counter: Counter[str] = Counter()
    pack_counter: Counter[str] = Counter()
    voucher_counter: Counter[str] = Counter()
    sample_rows: list[dict[str, Any]] = []

    for sample_id in range(max(1, int(args.samples))):
        if str(state.get("state") or "") != "SHOP":
            state = _goto_shop(args.base_url, f"{args.seed}_{sample_id}", float(args.timeout_sec))
        try:
            _call_method(args.base_url, "set", {"money": 999999}, timeout=float(args.timeout_sec))
            state = get_state(args.base_url, timeout=float(args.timeout_sec))
        except Exception:
            pass

        shop_keys = _card_keys(state.get("shop"))
        pack_keys = _card_keys(state.get("packs"))
        voucher_keys = _card_keys(state.get("vouchers"))
        for key in shop_keys:
            joker_counter[key] += 1
            rarity_counter[rarity_map.get(key, "unknown")] += 1
        for key in pack_keys:
            pack_counter[key] += 1
        for key in voucher_keys:
            voucher_counter[key] += 1

        sample_rows.append(
            {
                "sample_id": sample_id,
                "phase": str(state.get("state") or ""),
                "shop_count": len(shop_keys),
                "pack_count": len(pack_keys),
                "voucher_count": len(voucher_keys),
                "shop_keys": ";".join(shop_keys),
                "pack_keys": ";".join(pack_keys),
                "voucher_keys": ";".join(voucher_keys),
            }
        )

        if sample_id + 1 < int(args.samples):
            try:
                _call_method(args.base_url, "reroll", {}, timeout=float(args.timeout_sec))
                state = get_state(args.base_url, timeout=float(args.timeout_sec))
            except RPCError as exc:
                raise RuntimeError(f"reroll failed at sample {sample_id}: {exc}") from exc
            time.sleep(max(0.0, float(args.sleep_sec)))

    rarity_total = sum(rarity_counter.values())
    rarity_rows: list[dict[str, Any]] = []
    for rarity, count in sorted(rarity_counter.items(), key=lambda x: x[0]):
        observed = float(count / rarity_total) if rarity_total > 0 else 0.0
        expected = float(expected_rarity_probs.get(rarity, 0.0))
        rarity_rows.append(
            {
                "rarity": rarity,
                "count": int(count),
                "observed_prob": observed,
                "expected_prob_proxy": expected,
                "delta": observed - expected,
            }
        )
    chi2 = _chi_square(dict(rarity_counter), expected_rarity_probs)

    top_jokers = [{"key": k, "count": int(v)} for k, v in joker_counter.most_common(20)]
    top_packs = [{"key": k, "count": int(v)} for k, v in pack_counter.most_common(20)]
    top_vouchers = [{"key": k, "count": int(v)} for k, v in voucher_counter.most_common(20)]

    summary = {
        "schema": "p32_shop_probability_summary_v1",
        "base_url": args.base_url,
        "seed": args.seed,
        "samples": int(args.samples),
        "rarity_total_offers": int(rarity_total),
        "rarity_distribution": rarity_rows,
        "chi_square_vs_mechanics_proxy": chi2,
        "top_shop_keys": top_jokers,
        "top_pack_keys": top_packs,
        "top_voucher_keys": top_vouchers,
        "notes": [
            "Expected rarity is a mechanics CSV proxy (count-based), not official weight guarantees.",
            "Observed frequencies are sampled from runtime rerolls in SHOP phase.",
        ],
    }

    _write_json(out_dir / "shop_probability_summary.json", summary)
    _write_csv(out_dir / "shop_probability_samples.csv", list(sample_rows[0].keys()) if sample_rows else ["sample_id"], sample_rows)
    _write_csv(out_dir / "shop_rarity_distribution.csv", ["rarity", "count", "observed_prob", "expected_prob_proxy", "delta"], rarity_rows)

    md_lines = [
        "# P32 Shop/RNG Probability Alignment",
        "",
        f"- base_url: {args.base_url}",
        f"- seed: {args.seed}",
        f"- samples: {int(args.samples)}",
        f"- rarity_total_offers: {int(rarity_total)}",
        f"- chi_square_vs_mechanics_proxy: {('%.6f' % chi2) if isinstance(chi2, float) else 'n/a'}",
        "",
        "## Rarity Distribution",
        "",
        "| rarity | count | observed_prob | expected_prob_proxy | delta |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in rarity_rows:
        md_lines.append(
            f"| {row['rarity']} | {row['count']} | {row['observed_prob']:.6f} | {row['expected_prob_proxy']:.6f} | {row['delta']:.6f} |"
        )
    md_lines.extend(
        [
            "",
            "## Notes",
            "- Expected distribution is a proxy based on mechanics CSV rarity counts.",
            "- This report validates internal alignment consistency, not an official drop-rate contract.",
        ]
    )
    (out_dir / "shop_probability_summary.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    print(json.dumps({"out_dir": str(out_dir), "samples": int(args.samples), "rarity_total_offers": int(rarity_total)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
