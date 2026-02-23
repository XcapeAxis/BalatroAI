from __future__ import annotations

if __package__ is None or __package__ == "":
    import sys
    from pathlib import Path as _Path

    sys.path.append(str(_Path(__file__).resolve().parent.parent.parent))

import argparse
import csv
import json
import re
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sim.oracle.p3_joker_classifier import build_and_write


STOPWORDS = {
    "a",
    "an",
    "and",
    "any",
    "at",
    "be",
    "by",
    "card",
    "cards",
    "each",
    "for",
    "from",
    "give",
    "gives",
    "has",
    "if",
    "in",
    "is",
    "it",
    "joker",
    "of",
    "on",
    "or",
    "per",
    "played",
    "scored",
    "score",
    "scoring",
    "the",
    "to",
    "when",
    "with",
    "your",
}


@dataclass
class UnsupportedRow:
    joker_name: str
    joker_key: str
    reason: str
    trigger: str
    effect_summary: str
    key_mechanics: str
    constraints_or_conditions: str
    stacking_rules_or_notes: str


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent


def _normalize_key(name: str) -> str:
    low = str(name or "").strip().lower()
    low = re.sub(r"[^a-z0-9]+", "_", low)
    low = re.sub(r"_+", "_", low).strip("_")
    return f"j_{low}" if low else "j_unknown"


def _tokenize(text: str) -> list[str]:
    toks = re.findall(r"[a-z0-9]+", str(text or "").lower())
    return [t for t in toks if t and t not in STOPWORDS and len(t) >= 2]


def _timing_bucket(trigger: str) -> str:
    low = str(trigger or "").strip().lower()
    if "score" in low or "played" in low:
        return "on_score"
    if "discard" in low:
        return "on_discard"
    if "shop" in low or "buy" in low or "reroll" in low:
        return "on_shop"
    if "round" in low and "end" in low:
        return "on_round_end"
    if "hand" in low and "end" in low:
        return "on_hand_end"
    if low in {"passive", "always"}:
        return "passive"
    return "unknown"


def _extract_statefulness(text: str) -> list[str]:
    low = str(text or "").lower()
    out: list[str] = []
    if "this joker gains" in low or "currently" in low or "resets" in low:
        out.append("cross_round_state")
    if "shop" in low or "reroll" in low or "buy" in low:
        out.append("requires_shop")
    if "pack" in low or "booster" in low:
        out.append("requires_pack")
    if "$" in low or "money" in low or "sell value" in low or "debt" in low:
        out.append("depends_on_money")
    if "for each joker" in low or "joker to the right" in low or "uncommon jokers" in low:
        out.append("depends_on_joker_count")
    if "planet card" in low or "tarot card" in low or "consumable" in low:
        out.append("depends_on_consumable")
    if "chance" in low or "1 in " in low or "random" in low:
        out.append("probabilistic")
    return sorted(set(out))


def _extract_numeric_patterns(text: str) -> list[str]:
    low = str(text or "").lower()
    out: list[str] = []
    if re.search(r"\+\s*\d+(\.\d+)?\s*chips", low):
        out.append("plus_chips")
    if re.search(r"\+\s*\d+(\.\d+)?\s*mult", low):
        out.append("plus_mult")
    if re.search(r"x\s*\d+(\.\d+)?\s*mult", low):
        out.append("xmult")
    if "per card" in low or "for each" in low:
        out.append("per_card")
    if "suit" in low:
        out.append("per_suit")
    if "rank" in low or any(k in low for k in ("ace", "king", "queen", "jack")):
        out.append("per_rank")
    if "first played" in low or "first scoring" in low:
        out.append("first_scoring_card")
    if "last" in low and "card" in low:
        out.append("last_card")
    if "held in hand" in low or "held card" in low:
        out.append("held_cards")
    return sorted(set(out))


def _required_signals(timing: str, stateful: list[str], patterns: list[str]) -> list[str]:
    sig: list[str] = ["score_observed.delta", "round.hands_left", "round.discards_left", f"timing:{timing}"]
    if any(x in patterns for x in ("per_suit", "per_rank", "per_card", "first_scoring_card", "last_card")):
        sig.append("zones.played[min_card_fields]")
    if "held_cards" in patterns:
        sig.append("zones.hand[min_card_fields]")
    if "depends_on_money" in stateful:
        sig.append("economy.money")
    if "depends_on_joker_count" in stateful:
        sig.append("jokers.count")
    if "depends_on_consumable" in stateful:
        sig.append("consumables.cards")
    if "cross_round_state" in stateful:
        sig.append("persistent_counters(meta)")
    if "requires_shop" in stateful:
        sig.append("shop_state")
    if "requires_pack" in stateful:
        sig.append("pack_state")
    return sig


def _template_idea(timing: str, stateful: list[str], patterns: list[str], reasons: Counter[str]) -> tuple[str, str]:
    if "probabilistic" in stateful:
        return f"probabilistic_{timing}", "high"
    if "requires_shop" in stateful or "depends_on_money" in stateful:
        return f"shop_or_economy_{timing}", "high"
    if "requires_pack" in stateful:
        return f"pack_event_{timing}", "high"
    if "cross_round_state" in stateful:
        return f"cross_round_counter_{timing}", "high"
    if "first_scoring_card" in patterns and "xmult" in patterns:
        return "first_scoring_card_xmult", "med"
    if "held_cards" in patterns and "plus_mult" in patterns:
        return "held_cards_mult_add", "med"
    if "per_suit" in patterns and "plus_mult" in patterns:
        return "suit_scoring_mult_add", "low"
    if "per_suit" in patterns and "plus_chips" in patterns:
        return "suit_scoring_chips_add", "low"
    if "per_rank" in patterns and "plus_mult" in patterns:
        return "rank_scoring_mult_add", "low"
    if "per_rank" in patterns and "plus_chips" in patterns:
        return "rank_scoring_chips_add", "low"
    if "xmult" in patterns:
        return "conditional_xmult", "med"
    if "plus_mult" in patterns:
        return "flat_or_conditional_mult_add", "med"
    if "plus_chips" in patterns:
        return "flat_or_conditional_chips_add", "med"
    if reasons.get("insufficient_structured_fields", 0) > 0:
        return "metadata_required_template", "high"
    return "complex_or_unknown_template", "high"


def _load_jokers_csv(path: Path) -> dict[str, dict[str, str]]:
    rows: dict[str, dict[str, str]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as fp:
        reader = csv.DictReader(fp)
        for row in reader:
            name = str(row.get("joker_name") or "").strip()
            if not name:
                continue
            key = _normalize_key(name)
            rows[key] = {k: str(v or "") for k, v in row.items()}
    return rows


def _record_text(rec: UnsupportedRow) -> str:
    parts = [
        rec.effect_summary,
        rec.key_mechanics,
        rec.constraints_or_conditions,
        rec.stacking_rules_or_notes,
    ]
    return " | ".join([str(x or "") for x in parts])


def _signature(rec: UnsupportedRow) -> set[str]:
    text = _record_text(rec)
    tok = _tokenize(text)
    cnt = Counter(tok)
    top = [k for k, _ in cnt.most_common(20)]
    timing = _timing_bucket(rec.trigger)
    stateful = _extract_statefulness(text)
    patterns = _extract_numeric_patterns(text)
    sig = set(top)
    sig.add(f"timing:{timing}")
    for s in stateful:
        sig.add(f"state:{s}")
    for p in patterns:
        sig.add(f"pattern:{p}")
    return sig


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    uni = len(a | b)
    return float(inter) / float(uni) if uni else 0.0


def build_clusters(records: list[UnsupportedRow], threshold: float = 0.34) -> list[dict[str, Any]]:
    clusters: list[dict[str, Any]] = []

    for rec in sorted(records, key=lambda x: x.joker_name.lower()):
        sig = _signature(rec)
        timing = _timing_bucket(rec.trigger)

        best_idx = -1
        best_score = -1.0
        for i, c in enumerate(clusters):
            c_timing = c["dominant_timing"]
            score = _jaccard(sig, set(c["rep_tokens"]))
            if c_timing == timing:
                score += 0.08
            if score > best_score:
                best_score = score
                best_idx = i

        if best_idx >= 0 and best_score >= threshold:
            c = clusters[best_idx]
            c["members"].append(rec)
            c["token_counter"].update(_signature(rec))
            c["timing_counter"][timing] += 1
            c["reason_counter"][str(rec.reason or "unknown")] += 1
            c["rep_tokens"] = [k for k, _ in c["token_counter"].most_common(36)]
            c["dominant_timing"] = c["timing_counter"].most_common(1)[0][0]
            continue

        counter = Counter(_signature(rec))
        clusters.append(
            {
                "members": [rec],
                "token_counter": counter,
                "rep_tokens": [k for k, _ in counter.most_common(36)],
                "timing_counter": Counter([timing]),
                "reason_counter": Counter([str(rec.reason or "unknown")]),
                "dominant_timing": timing,
            }
        )

    payload: list[dict[str, Any]] = []
    ordered = sorted(clusters, key=lambda c: len(c["members"]), reverse=True)
    for idx, c in enumerate(ordered, start=1):
        members: list[UnsupportedRow] = c["members"]
        text = " ".join(_record_text(m) for m in members)
        stateful = Counter()
        patterns = Counter()
        keywords = Counter(_tokenize(text))
        for m in members:
            stateful.update(_extract_statefulness(_record_text(m)))
            patterns.update(_extract_numeric_patterns(_record_text(m)))
        idea, difficulty = _template_idea(c["dominant_timing"], list(stateful.keys()), list(patterns.keys()), c["reason_counter"])
        payload.append(
            {
                "cluster_id": f"p6_cluster_{idx:03d}",
                "size": len(members),
                "dominant_timing": c["dominant_timing"],
                "timing_distribution": dict(c["timing_counter"].most_common()),
                "reason_distribution": dict(c["reason_counter"].most_common()),
                "statefulness_distribution": dict(stateful.most_common()),
                "numeric_pattern_distribution": dict(patterns.most_common()),
                "top_keywords": [k for k, _ in keywords.most_common(16)],
                "representatives": [m.joker_name for m in members[:10]],
                "member_keys": [m.joker_key for m in members],
                "member_names": [m.joker_name for m in members],
                "template_candidate": idea,
                "difficulty": difficulty,
                "required_signals": _required_signals(c["dominant_timing"], list(stateful.keys()), list(patterns.keys())),
            }
        )
    return payload


def write_clusters_md(path: Path, clusters: list[dict[str, Any]], total_unsupported: int) -> None:
    lines: list[str] = []
    lines.append("# P6 P3 Unsupported Joker Clusters")
    lines.append("")
    lines.append(f"Generated at: `{datetime.now(timezone.utc).isoformat()}`")
    lines.append("")
    lines.append(f"- Unsupported total: **{total_unsupported}**")
    lines.append(f"- Cluster count: **{len(clusters)}**")
    lines.append("- Method: rule features (timing/statefulness/numeric patterns) + token-similarity greedy clustering.")
    lines.append("")
    lines.append("## Top Clusters")
    for c in clusters[:20]:
        lines.append("")
        lines.append(f"### {c['cluster_id']} | size={c['size']} | difficulty={c['difficulty']}")
        lines.append(f"- template_candidate: `{c['template_candidate']}`")
        lines.append(f"- dominant_timing: `{c['dominant_timing']}`")
        lines.append(f"- top_keywords: `{', '.join(c['top_keywords'][:12])}`")
        lines.append(f"- reason_distribution: `{json.dumps(c['reason_distribution'], ensure_ascii=False)}`")
        lines.append(f"- statefulness_distribution: `{json.dumps(c['statefulness_distribution'], ensure_ascii=False)}`")
        lines.append(f"- numeric_pattern_distribution: `{json.dumps(c['numeric_pattern_distribution'], ensure_ascii=False)}`")
        lines.append(f"- required_signals: `{', '.join(c['required_signals'])}`")
        lines.append("- representatives:")
        for name in c["representatives"][:10]:
            lines.append(f"  - {name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_backlog_md(path: Path, clusters: list[dict[str, Any]]) -> None:
    candidates = [c for c in clusters if c.get("difficulty") in {"low", "med"}]
    candidates.sort(key=lambda x: int(x.get("size") or 0), reverse=True)
    top = candidates[:10]

    lines: list[str] = []
    lines.append("# P6 Template Backlog (Next Batch)")
    lines.append("")
    lines.append(f"Generated at: `{datetime.now(timezone.utc).isoformat()}`")
    lines.append("")
    lines.append("- Selection rule: low/med difficulty first, then larger cluster size.")
    lines.append("")
    for i, c in enumerate(top, start=1):
        lines.append(f"## {i}. {c['template_candidate']} (from {c['cluster_id']}, size={c['size']})")
        lines.append(f"- trigger_timing: `{c['dominant_timing']}`")
        lines.append(f"- expected_scope: `{', '.join(c['required_signals'])}`")
        lines.append(f"- representative_jokers: `{', '.join(c['representatives'][:8])}`")
        lines.append("- fixture_strategy:")
        lines.append("  - use `add joker` + controlled hand construction")
        lines.append("  - isolate single-step action (prefer one PLAY; fallback DISCARD for non-scoring)")
        lines.append("  - compare on observed scope and keep identity/economy/rng noise excluded")
        risk: list[str] = []
        reason_dist = str(c.get("reason_distribution") or "")
        if "probabilistic" in reason_dist:
            risk.append("probabilistic trigger: require deterministic harness or keep unsupported")
        if "economy" in reason_dist or "shop" in reason_dist:
            risk.append("shop/economy coupling: may need dedicated scope and fixture setup")
        if "cross_round" in reason_dist:
            risk.append("cross-round counters: likely needs persistent state fields/meta")
        if not risk:
            risk.append("low direct risk under current observed scoring scope")
        lines.append(f"- risk_notes: `{'; '.join(risk)}`")
        lines.append("")

    if not top:
        lines.append("No low/med candidates discovered; keep unsupported until richer observable scope is available.")
        lines.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cluster unsupported P3 joker items and build next-template backlog.")
    parser.add_argument("--out-derived", default="balatro_mechanics/derived")
    parser.add_argument("--out-docs", default="docs")
    parser.add_argument("--threshold", type=float, default=0.34)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    project_root = _project_root()

    summary = build_and_write(project_root)
    unsupported_path = Path(summary["unsupported_path"])
    if not unsupported_path.exists():
        fallback = project_root / "balatro_mechanics" / "derived" / "joker_template_unsupported.json"
        unsupported_path = fallback
    if not unsupported_path.exists():
        print(f"ERROR: unsupported source not found: {unsupported_path}")
        return 2

    jokers_csv = project_root / "balatro_mechanics" / "jokers.csv"
    if not jokers_csv.exists():
        print(f"ERROR: jokers.csv not found at {jokers_csv}")
        return 2

    unsupported = json.loads(unsupported_path.read_text(encoding="utf-8"))
    if not isinstance(unsupported, list):
        print(f"ERROR: unsupported payload is not a list: {unsupported_path}")
        return 2

    rows = _load_jokers_csv(jokers_csv)
    records: list[UnsupportedRow] = []
    for item in unsupported:
        if not isinstance(item, dict):
            continue
        key = str(item.get("joker_key") or "").strip().lower()
        name = str(item.get("joker_name") or "").strip()
        row = rows.get(key, {})
        records.append(
            UnsupportedRow(
                joker_name=name,
                joker_key=key,
                reason=str(item.get("reason") or "unknown").strip(),
                trigger=str((row.get("trigger_timing") or item.get("trigger_timing") or "unknown")).strip(),
                effect_summary=str((row.get("effect_summary") or item.get("effect_summary") or "")).strip(),
                key_mechanics=str((row.get("key_mechanics") or item.get("key_mechanics") or "")).strip(),
                constraints_or_conditions=str((row.get("constraints_or_conditions") or "")).strip(),
                stacking_rules_or_notes=str((row.get("stacking_rules_or_notes") or "")).strip(),
            )
        )

    clusters = build_clusters(records, threshold=float(args.threshold))

    out_derived = project_root / args.out_derived if not Path(args.out_derived).is_absolute() else Path(args.out_derived)
    out_docs = project_root / args.out_docs if not Path(args.out_docs).is_absolute() else Path(args.out_docs)
    out_derived.mkdir(parents=True, exist_ok=True)
    out_docs.mkdir(parents=True, exist_ok=True)

    clusters_json = out_derived / "p6_p3_unsupported_clusters.json"
    clusters_md = out_docs / "P6_P3_UNSUPPORTED_CLUSTERS.md"
    backlog_md = out_docs / "P6_TEMPLATE_BACKLOG_NEXT.md"

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_unsupported": str(unsupported_path),
        "total_unsupported": len(records),
        "cluster_count": len(clusters),
        "clusters": clusters,
    }
    clusters_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    write_clusters_md(clusters_md, clusters, total_unsupported=len(records))
    write_backlog_md(backlog_md, clusters)

    print(f"unsupported={len(records)} clusters={len(clusters)}")
    print(f"clusters_json={clusters_json}")
    print(f"clusters_md={clusters_md}")
    print(f"backlog_md={backlog_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
