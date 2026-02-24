from __future__ import annotations

if __package__ is None or __package__ == "":
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).resolve().parent.parent))

import argparse
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any

from trainer.env_client import create_backend
from trainer.utils import setup_logger, warn_if_unstable_python


def _phase(state: dict[str, Any]) -> str:
    return str(state.get("state") or "UNKNOWN")


def _keys_from_cards(cards: Any) -> list[str]:
    if not isinstance(cards, list):
        return []
    out: list[str] = []
    for item in cards:
        if isinstance(item, dict):
            out.append(str(item.get("key") or ""))
    return [k for k in out if k]


def _market_keys(state: dict[str, Any], field: str) -> list[str]:
    block = state.get(field)
    if not isinstance(block, dict):
        return []
    return _keys_from_cards(block.get("cards"))


def _projection(state: dict[str, Any]) -> dict[str, Any]:
    round_info = state.get("round") if isinstance(state.get("round"), dict) else {}
    score = state.get("score") if isinstance(state.get("score"), dict) else {}
    economy = state.get("economy") if isinstance(state.get("economy"), dict) else {}
    hand = state.get("hand") if isinstance(state.get("hand"), dict) else {}
    jokers = state.get("jokers") if isinstance(state.get("jokers"), list) else []
    tags = state.get("tags") if isinstance(state.get("tags"), list) else []
    consumables = state.get("consumables") if isinstance(state.get("consumables"), dict) else {}

    joker_keys = [str(j.get("key") or j.get("joker_id") or "") for j in jokers if isinstance(j, dict)]
    tag_keys = [str(t.get("key") or "") for t in tags if isinstance(t, dict)]

    return {
        "phase": _phase(state),
        "resources": {
            "hands_left": int(round_info.get("hands_left") or 0),
            "discards_left": int(round_info.get("discards_left") or 0),
            "ante": int(round_info.get("ante") or 0),
            "round_num": int(round_info.get("round_num") or 0),
        },
        "score_observed": {
            "chips": float(round_info.get("chips") or score.get("chips") or 0.0),
            "mult": float(score.get("mult") or 0.0),
            "target_chips": float(score.get("target_chips") or 0.0),
        },
        "economy": {"money": float(economy.get("money") or 0.0)},
        "hand_keys": _keys_from_cards(hand.get("cards")),
        "shop_keys": _market_keys(state, "shop"),
        "voucher_keys": _market_keys(state, "vouchers"),
        "pack_keys": _market_keys(state, "packs"),
        "consumable_keys": _keys_from_cards(consumables.get("cards")),
        "joker_keys": [k for k in joker_keys if k],
        "tag_keys": [k for k in tag_keys if k],
    }


def _diff(a: Any, b: Any, prefix: str = "$") -> list[tuple[str, Any, Any]]:
    diffs: list[tuple[str, Any, Any]] = []
    if type(a) is not type(b):
        diffs.append((prefix, a, b))
        return diffs
    if isinstance(a, dict):
        keys = sorted(set(a.keys()) | set(b.keys()))
        for k in keys:
            pa = a.get(k)
            pb = b.get(k)
            diffs.extend(_diff(pa, pb, f"{prefix}.{k}"))
        return diffs
    if isinstance(a, list):
        if a != b:
            diffs.append((prefix, a, b))
        return diffs
    if a != b:
        diffs.append((prefix, a, b))
    return diffs


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare observable projection between real and sim backends.")
    parser.add_argument("--base-url", default="http://127.0.0.1:12346")
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--interval", type=float, default=1.0)
    parser.add_argument("--seed", default="AAAAAAA")
    parser.add_argument("--timeout-sec", type=float, default=8.0)
    parser.add_argument("--with-sim", action="store_true", help="If enabled, compare real projection vs sim projection.")
    parser.add_argument("--out", default="docs/artifacts/p12/drift_latest.json")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    logger = setup_logger("trainer.sim_real_drift")
    warn_if_unstable_python(logger)

    real_backend = create_backend("real", base_url=args.base_url, timeout_sec=args.timeout_sec, seed=args.seed, logger=logger)
    sim_backend = create_backend("sim", seed=args.seed, logger=logger) if args.with_sim else None
    if sim_backend is not None:
        sim_backend.reset(seed=args.seed)

    rows: list[dict[str, Any]] = []
    path_counter: Counter[str] = Counter()

    try:
        for i in range(max(1, int(args.samples))):
            real_state = real_backend.get_state()
            real_proj = _projection(real_state)

            row: dict[str, Any] = {"sample": i, "real": real_proj}
            if sim_backend is not None:
                sim_state = sim_backend.get_state()
                sim_proj = _projection(sim_state)
                diffs = _diff(real_proj, sim_proj, "$")
                for p, _, _ in diffs:
                    path_counter[p] += 1
                row["sim"] = sim_proj
                row["mismatch_count"] = len(diffs)
                row["mismatches"] = [{"path": p, "real": rv, "sim": sv} for p, rv, sv in diffs[:20]]
                try:
                    sim_backend.step({"action_type": "AUTO"})
                except Exception:
                    pass
                logger.info("sample=%d mismatch=%d", i, len(diffs))
            else:
                logger.info("sample=%d phase=%s", i, real_proj.get("phase"))
            rows.append(row)
            if i + 1 < int(args.samples):
                time.sleep(max(0.05, float(args.interval)))
    finally:
        real_backend.close()
        if sim_backend is not None:
            sim_backend.close()

    summary: dict[str, Any] = {
        "samples": len(rows),
        "with_sim": bool(args.with_sim),
        "total_mismatches": int(sum(int(r.get("mismatch_count") or 0) for r in rows)),
        "top_mismatch_paths": path_counter.most_common(15),
        "rows": rows,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("wrote drift report: %s", out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
