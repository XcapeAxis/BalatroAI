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
    "are",
    "as",
    "at",
    "be",
    "by",
    "can",
    "card",
    "cards",
    "each",
    "for",
    "from",
    "gain",
    "gains",
    "get",
    "gives",
    "hand",
    "has",
    "have",
    "if",
    "in",
    "is",
    "it",
    "its",
    "joker",
    "of",
    "on",
    "or",
    "per",
    "played",
    "scored",
    "score",
    "scoring",
    "that",
    "the",
    "this",
    "to",
    "when",
    "with",
    "your",
}


@dataclass
class JokerRecord:
    joker_name: str
    joker_key: str
    trigger_timing: str
    reason: str
    effect_summary: str
    key_mechanics: str
    constraints_or_conditions: str
    stacking_rules_or_notes: str



def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent



def _tokenize(text: str) -> list[str]:
    toks = re.findall(r"[a-z0-9]+", str(text or "").lower())
    return [t for t in toks if t and t not in STOPWORDS and len(t) >= 2]



def _signature(rec: JokerRecord) -> set[str]:
    parts = [
        rec.effect_summary,
        rec.key_mechanics,
        rec.constraints_or_conditions,
        rec.stacking_rules_or_notes,
    ]
    tokens: list[str] = []
    for p in parts:
        tokens.extend(_tokenize(p))
    # Keep deterministic high-signal tokens (frequency then lexical)
    cnt = Counter(tokens)
    ordered = sorted(cnt.items(), key=lambda kv: (-kv[1], kv[0]))
    top = [k for k, _ in ordered[:28]]
    sig = set(top)
    if rec.trigger_timing:
        sig.add(f"trigger:{rec.trigger_timing.lower()}")
    return sig



def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    uni = len(a | b)
    if uni <= 0:
        return 0.0
    return inter / uni



def _cluster_similarity(sig: set[str], cluster: dict[str, Any]) -> float:
    rep = set(cluster.get("rep_tokens") or [])
    score = _jaccard(sig, rep)
    trig = str(cluster.get("dominant_trigger") or "")
    if trig and f"trigger:{trig}" in sig:
        score += 0.08
    return score



def _normalize_key(name: str) -> str:
    low = str(name or "").strip().lower()
    low = re.sub(r"[^a-z0-9]+", "_", low)
    low = re.sub(r"_+", "_", low).strip("_")
    return f"j_{low}" if low else "j_unknown"



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



def _suggest_template(tokens: list[str], trigger: str, reasons: Counter[str]) -> tuple[str, str]:
    tset = set(tokens)

    if "probabilistic_trigger" in reasons:
        return f"probabilistic_{trigger or 'mixed'}", "high"
    if "economy_or_shop_related" in reasons:
        return f"economy_{trigger or 'mixed'}", "high"
    if "cross_round_state" in reasons:
        return f"cross_round_{trigger or 'mixed'}", "high"

    if "held" in tset and "mult" in tset:
        return "held_cards_mult_rule", "med"
    if "discard" in tset and "chips" in tset:
        return "discard_to_chips_rule", "med"
    if "face" in tset and "mult" in tset:
        return "face_mult_rule", "low"
    if "suit" in tset and "mult" in tset:
        return "suit_mult_rule", "low"
    if "suit" in tset and "chips" in tset:
        return "suit_chips_rule", "low"
    if "rank" in tset and "mult" in tset:
        return "rank_mult_rule", "low"
    if "xmult" in tset or ("x" in tset and "mult" in tset):
        return "xmult_condition_rule", "med"

    if trigger in {"on_scored", "on_play"}:
        return "deterministic_scoring_rule", "med"

    return "complex_or_unknown_rule", "high"



def _reason_bucket(reason: str) -> str:
    r = str(reason or "").strip().lower()
    if not r:
        return "unknown"
    if "probabilistic" in r:
        return "probabilistic"
    if "economy" in r or "shop" in r:
        return "economy"
    if "cross_round" in r:
        return "cross_round"
    if "insufficient" in r:
        return "insufficient_fields"
    return r



def build_clusters(records: list[JokerRecord], threshold: float = 0.34) -> list[dict[str, Any]]:
    clusters: list[dict[str, Any]] = []

    for rec in sorted(records, key=lambda x: x.joker_name.lower()):
        sig = _signature(rec)

        best_idx = -1
        best_score = -1.0
        for idx, cluster in enumerate(clusters):
            score = _cluster_similarity(sig, cluster)
            if score > best_score:
                best_score = score
                best_idx = idx

        if best_idx >= 0 and best_score >= threshold:
            cluster = clusters[best_idx]
            cluster["members"].append(rec)
            cluster["all_tokens"].update(sig)
            cluster["trigger_counter"][rec.trigger_timing or "unknown"] += 1
            cluster["reason_counter"][_reason_bucket(rec.reason)] += 1
            # Recompute representative tokens by frequency over members.
            token_counter = Counter()
            for m in cluster["members"]:
                token_counter.update(_signature(m))
            rep_tokens = [tok for tok, _ in token_counter.most_common(32)]
            cluster["rep_tokens"] = rep_tokens
            dominant_trigger = cluster["trigger_counter"].most_common(1)[0][0]
            cluster["dominant_trigger"] = dominant_trigger
            continue

        clusters.append(
            {
                "members": [rec],
                "all_tokens": set(sig),
                "rep_tokens": list(sig),
                "trigger_counter": Counter([rec.trigger_timing or "unknown"]),
                "reason_counter": Counter([_reason_bucket(rec.reason)]),
                "dominant_trigger": rec.trigger_timing or "unknown",
            }
        )

    # Build export form.
    out: list[dict[str, Any]] = []
    for i, cluster in enumerate(sorted(clusters, key=lambda c: len(c["members"]), reverse=True), start=1):
        members: list[JokerRecord] = cluster["members"]
        top_tokens = [t for t in cluster["rep_tokens"] if not t.startswith("trigger:")][:16]
        trigger_dist = dict(cluster["trigger_counter"].most_common())
        reason_dist = dict(cluster["reason_counter"].most_common())
        suggested_template, difficulty = _suggest_template(top_tokens, cluster["dominant_trigger"], cluster["reason_counter"])

        out.append(
            {
                "cluster_id": f"cluster_{i:03d}",
                "size": len(members),
                "dominant_trigger": cluster["dominant_trigger"],
                "trigger_distribution": trigger_dist,
                "reason_distribution": reason_dist,
                "top_tokens": top_tokens,
                "suggested_template": suggested_template,
                "difficulty": difficulty,
                "representatives": [m.joker_name for m in members[:10]],
                "members": [m.joker_name for m in members],
                "member_keys": [m.joker_key for m in members],
            }
        )

    return out



def write_backlog_md(path: Path, clusters: list[dict[str, Any]], records: list[JokerRecord]) -> None:
    reason_counter = Counter(_reason_bucket(r.reason) for r in records)

    lines: list[str] = []
    lines.append("# P3 Joker Template Backlog")
    lines.append("")
    lines.append(f"Generated at: `{datetime.now(timezone.utc).isoformat()}`")
    lines.append("")
    lines.append(f"- Unsupported jokers: **{len(records)}**")
    lines.append(f"- Clusters discovered: **{len(clusters)}**")
    lines.append("- Clustering method: token signature + greedy Jaccard (conservative, deterministic)")
    lines.append("")
    lines.append("## Unsupported Top Reasons")
    for reason, cnt in reason_counter.most_common(12):
        lines.append(f"- `{reason}`: {cnt}")
    lines.append("")
    lines.append("## Top Clusters")

    top_clusters = clusters[:20]
    for c in top_clusters:
        lines.append("")
        lines.append(f"### {c['cluster_id']} | size={c['size']} | difficulty={c['difficulty']}")
        lines.append(f"- suggested_template: `{c['suggested_template']}`")
        lines.append(f"- dominant_trigger: `{c['dominant_trigger']}`")
        lines.append(f"- trigger_distribution: `{json.dumps(c['trigger_distribution'], ensure_ascii=False)}`")
        lines.append(f"- reason_distribution: `{json.dumps(c['reason_distribution'], ensure_ascii=False)}`")
        lines.append(f"- top_tokens: `{', '.join(c['top_tokens'][:12])}`")
        lines.append("- representatives:")
        for name in c["representatives"]:
            lines.append(f"  - {name}")

    lines.append("")
    lines.append("## De-prioritized Buckets")
    lines.append("- probabilistic: keep unsupported for now unless oracle evidence allows deterministic harness")
    lines.append("- economy/shop: isolate later with dedicated economy trace scope")
    lines.append("- cross_round_state: postpone until persistent counters/state machine are modeled")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Discover unsupported joker template clusters and generate P3 backlog report.")
    parser.add_argument("--out-json", default="balatro_mechanics/derived/p3_discovery_clusters.json")
    parser.add_argument("--out-md", default="docs/P3_TEMPLATE_BACKLOG.md")
    parser.add_argument("--min-clusters", type=int, default=20)
    return parser.parse_args()



def main() -> int:
    args = parse_args()
    project_root = _project_root()

    summary = build_and_write(project_root)
    unsupported_path = Path(summary["unsupported_path"])

    mechanics_root = project_root / "balatro_mechanics"
    jokers_csv = mechanics_root / "jokers.csv"
    if not jokers_csv.exists():
        print(f"ERROR: jokers.csv not found at {jokers_csv}")
        return 2

    unsupported = json.loads(unsupported_path.read_text(encoding="utf-8"))
    rows_by_key = _load_jokers_csv(jokers_csv)

    records: list[JokerRecord] = []
    for item in unsupported:
        if not isinstance(item, dict):
            continue
        key = str(item.get("joker_key") or "").strip().lower()
        name = str(item.get("joker_name") or "").strip()
        row = rows_by_key.get(key, {})

        records.append(
            JokerRecord(
                joker_name=name,
                joker_key=key,
                trigger_timing=str(row.get("trigger_timing") or item.get("trigger_timing") or "unknown").strip().lower(),
                reason=str(item.get("reason") or "unknown").strip(),
                effect_summary=str(row.get("effect_summary") or item.get("effect_summary") or "").strip(),
                key_mechanics=str(row.get("key_mechanics") or item.get("key_mechanics") or "").strip(),
                constraints_or_conditions=str(row.get("constraints_or_conditions") or "").strip(),
                stacking_rules_or_notes=str(row.get("stacking_rules_or_notes") or "").strip(),
            )
        )

    clusters = build_clusters(records)

    out_json = project_root / args.out_json if not Path(args.out_json).is_absolute() else Path(args.out_json)
    out_md = project_root / args.out_md if not Path(args.out_md).is_absolute() else Path(args.out_md)

    out_json.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "total_unsupported": len(records),
        "cluster_count": len(clusters),
        "clusters": clusters,
    }
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    write_backlog_md(out_md, clusters, records)

    print(f"unsupported={len(records)} clusters={len(clusters)}")
    print(f"clusters_json={out_json}")
    print(f"backlog_md={out_md}")
    if len(clusters) < int(args.min_clusters):
        print(f"WARNING: clusters below requested min ({len(clusters)} < {int(args.min_clusters)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
