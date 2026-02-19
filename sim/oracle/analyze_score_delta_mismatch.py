from __future__ import annotations

if __package__ is None or __package__ == "":
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

from sim.core.score_basic import evaluate_selected
from sim.score.expected_basic import compute_expected_for_action

RANK_CHIPS = {
    "2": 2,
    "3": 3,
    "4": 4,
    "5": 5,
    "6": 6,
    "7": 7,
    "8": 8,
    "9": 9,
    "10": 10,
    "T": 10,
    "J": 10,
    "Q": 10,
    "K": 10,
    "A": 11,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze step-0 score_observed delta mismatch between oracle and sim traces.")
    parser.add_argument("--fixtures-dir", required=True, help="Fixture runtime directory, e.g. sim/tests/fixtures_runtime/oracle_p0_v5")
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _load_first_jsonl(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as fp:
        for line in fp:
            text = line.strip()
            if not text:
                continue
            obj = json.loads(text)
            if isinstance(obj, dict):
                return obj
    raise ValueError(f"empty jsonl: {path}")


def _normalize_rank(rank: Any) -> str:
    text = str(rank or "").strip().upper()
    if text == "T":
        return "10"
    return text


def _rank_from_key(key: Any) -> str:
    text = str(key or "").strip().upper()
    if "_" in text:
        _, r = text.split("_", 1)
        if r == "T":
            return "10"
        return r
    return ""


def _index_base_from_action(action: dict[str, Any]) -> int:
    params = action.get("params") if isinstance(action.get("params"), dict) else {}
    try:
        value = int(params.get("index_base", 0))
    except Exception:
        value = 0
    return 1 if value == 1 else 0


def _indices_zero_based(action: dict[str, Any]) -> list[int]:
    indices = [int(i) for i in (action.get("indices") or [])]
    if _index_base_from_action(action) == 1:
        return [i - 1 for i in indices]
    return indices


def _extract_hand_cards(snapshot: dict[str, Any]) -> list[dict[str, Any]]:
    zones = snapshot.get("zones") if isinstance(snapshot.get("zones"), dict) else {}
    hand = zones.get("hand") if isinstance(zones.get("hand"), list) else []
    cards: list[dict[str, Any]] = []
    for card in hand:
        if not isinstance(card, dict):
            continue
        cards.append(card)
    return cards


def _selected_cards_by_indices(snapshot: dict[str, Any], indices_zero: list[int]) -> list[dict[str, Any]]:
    hand = _extract_hand_cards(snapshot)
    out: list[dict[str, Any]] = []
    for i in indices_zero:
        if i < 0 or i >= len(hand):
            continue
        out.append(hand[i])
    return out


def _card_key(card: dict[str, Any]) -> str:
    key = str(card.get("key") or "").strip().upper()
    if key:
        return key
    rank = _normalize_rank(card.get("rank"))
    suit = str(card.get("suit") or "").strip().upper()
    if suit:
        suit = suit[:1]
    if rank and suit:
        k_rank = "T" if rank == "10" else rank
        return f"{suit}_{k_rank}"
    return ""


def _sum_rank_chips(cards: list[dict[str, Any]]) -> float:
    total = 0.0
    for card in cards:
        rank = _normalize_rank(card.get("rank"))
        if not rank:
            rank = _rank_from_key(card.get("key"))
        total += float(RANK_CHIPS.get(rank, 0.0))
    return total


def _sim_hand_type_from_trace_line(trace_line: dict[str, Any]) -> str:
    snap = trace_line.get("canonical_state_snapshot") if isinstance(trace_line.get("canonical_state_snapshot"), dict) else {}
    score = snap.get("score") if isinstance(snap.get("score"), dict) else {}
    return str(score.get("last_hand_type") or "").strip().upper()


def _format_float(value: Any) -> str:
    if value is None:
        return ""
    try:
        v = float(value)
    except Exception:
        return str(value)
    return f"{v:.6f}"


def _collect_targets(fixtures_dir: Path) -> list[str]:
    out: list[str] = []
    for p in fixtures_dir.glob("oracle_start_snapshot_*.json"):
        name = p.name
        prefix = "oracle_start_snapshot_"
        if not name.startswith(prefix):
            continue
        target = name[len(prefix) : -len(".json")]
        out.append(target)
    return sorted(set(out))


def main() -> int:
    args = parse_args()
    fixtures_dir = Path(args.fixtures_dir)
    if not fixtures_dir.exists():
        print(f"ERROR: fixtures dir not found: {fixtures_dir}")
        return 2

    rows: list[dict[str, Any]] = []
    conclusion_counter: Counter[str] = Counter()

    for target in _collect_targets(fixtures_dir):
        snapshot_path = fixtures_dir / f"oracle_start_snapshot_{target}.json"
        action_path = fixtures_dir / f"action_trace_{target}.jsonl"
        oracle_trace_path = fixtures_dir / f"oracle_trace_{target}.jsonl"
        sim_trace_path = fixtures_dir / f"sim_trace_{target}.jsonl"

        if not (snapshot_path.exists() and action_path.exists() and oracle_trace_path.exists() and sim_trace_path.exists()):
            continue

        snapshot = _load_json(snapshot_path)
        action0 = _load_first_jsonl(action_path)
        oracle0 = _load_first_jsonl(oracle_trace_path)
        sim0 = _load_first_jsonl(sim_trace_path)

        action_type = str(action0.get("action_type") or "").upper()
        indices = _indices_zero_based(action0)
        selected_cards = _selected_cards_by_indices(snapshot, indices)
        played_keys = [_card_key(c) for c in selected_cards]

        expected_from_csv = compute_expected_for_action(
            {
                "hand": {"cards": selected_cards},
            },
            {"action_type": action_type, "indices": list(range(len(selected_cards))), "params": {"index_base": 0}},
        )

        detected_hand_type_expected = ""
        base_chips = None
        base_mult = None
        if bool(expected_from_csv.get("available")):
            detected_hand_type_expected = str(expected_from_csv.get("hand_type") or "")
            base_chips = float(expected_from_csv.get("base_chips") or 0.0)
            base_mult = float(expected_from_csv.get("base_mult") or 1.0)
        else:
            cards_for_eval = [{"rank": _normalize_rank(c.get("rank") or _rank_from_key(c.get("key"))), "suit": str(c.get("suit") or "")} for c in selected_cards]
            h_type, b_chips, b_mult = evaluate_selected(cards_for_eval)
            detected_hand_type_expected = str(h_type or "")
            base_chips = float(b_chips)
            base_mult = float(b_mult)

        detected_hand_type_sim = _sim_hand_type_from_trace_line(sim0)
        sum_rank = _sum_rank_chips(selected_cards)
        predicted_core = (float(base_chips or 0.0) + sum_rank) * float(base_mult or 1.0)

        oracle_score_obs = oracle0.get("score_observed") if isinstance(oracle0.get("score_observed"), dict) else {}
        sim_score_obs = sim0.get("score_observed") if isinstance(sim0.get("score_observed"), dict) else {}
        oracle_delta = float(oracle_score_obs.get("delta") or 0.0)
        sim_delta = float(sim_score_obs.get("delta") or 0.0)

        oracle_minus_pred = oracle_delta - predicted_core
        sim_minus_pred = sim_delta - predicted_core

        played_keys_sim_zone: list[str] = []
        sim_snap = sim0.get("canonical_state_snapshot") if isinstance(sim0.get("canonical_state_snapshot"), dict) else {}
        sim_zones = sim_snap.get("zones") if isinstance(sim_snap.get("zones"), dict) else {}
        sim_played = sim_zones.get("played") if isinstance(sim_zones.get("played"), list) else []
        for c in sim_played:
            if isinstance(c, dict):
                played_keys_sim_zone.append(_card_key(c))

        # classification
        category = "ok"
        if action_type != "PLAY":
            category = "non_play_action"
        elif detected_hand_type_sim and detected_hand_type_expected and detected_hand_type_sim != detected_hand_type_expected:
            category = "hand_type_mismatch"
        elif set(played_keys_sim_zone) != set(played_keys):
            category = "played_set_mismatch"
        else:
            oracle_close = abs(oracle_minus_pred) < 1e-6
            sim_close = abs(sim_minus_pred) < 1e-6
            if oracle_close and not sim_close:
                category = "sim_scoring_core_mismatch"
            elif not oracle_close and sim_close:
                category = "oracle_extra_component"
            elif not oracle_close and not sim_close:
                category = "both_off_vs_predicted_core"
            else:
                category = "aligned"

        conclusion_counter[category] += 1

        row = {
            "target": target,
            "action_type": action_type,
            "action_play_cards_indices": json.dumps(indices, ensure_ascii=False),
            "played_cards_keys": json.dumps(played_keys, ensure_ascii=False),
            "played_cards_keys_sim_zone": json.dumps(played_keys_sim_zone, ensure_ascii=False),
            "detected_hand_type_sim": detected_hand_type_sim,
            "detected_hand_type_expected": detected_hand_type_expected,
            "base_chips": base_chips,
            "base_mult": base_mult,
            "sum_rank_chips": sum_rank,
            "predicted_core": predicted_core,
            "oracle_delta": oracle_delta,
            "sim_delta": sim_delta,
            "oracle_delta_minus_predicted_core": oracle_minus_pred,
            "sim_delta_minus_predicted_core": sim_minus_pred,
            "classification": category,
        }
        rows.append(row)

    rows.sort(key=lambda x: str(x.get("target") or ""))

    csv_path = fixtures_dir / "score_mismatch_table.csv"
    md_path = fixtures_dir / "score_mismatch_table.md"

    fieldnames = [
        "target",
        "action_type",
        "action_play_cards_indices",
        "played_cards_keys",
        "played_cards_keys_sim_zone",
        "detected_hand_type_sim",
        "detected_hand_type_expected",
        "base_chips",
        "base_mult",
        "sum_rank_chips",
        "predicted_core",
        "oracle_delta",
        "sim_delta",
        "oracle_delta_minus_predicted_core",
        "sim_delta_minus_predicted_core",
        "classification",
    ]

    with csv_path.open("w", encoding="utf-8-sig", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    lines: list[str] = []
    lines.append("# Score Delta Mismatch Table")
    lines.append("")
    lines.append("| " + " | ".join(fieldnames) + " |")
    lines.append("| " + " | ".join(["---"] * len(fieldnames)) + " |")
    for row in rows:
        vals = []
        for k in fieldnames:
            v = row.get(k)
            if isinstance(v, float):
                vals.append(_format_float(v))
            else:
                vals.append(str(v).replace("|", "\\|"))
        lines.append("| " + " | ".join(vals) + " |")

    lines.append("")
    lines.append("## Classification Summary")
    for key, value in sorted(conclusion_counter.items(), key=lambda x: x[0]):
        lines.append(f"- {key}: {value}")

    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"wrote: {csv_path}")
    print(f"wrote: {md_path}")
    print("classification_summary:")
    for key, value in sorted(conclusion_counter.items(), key=lambda x: x[0]):
        print(f"  {key}: {value}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
