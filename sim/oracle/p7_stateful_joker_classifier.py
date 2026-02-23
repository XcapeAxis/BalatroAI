from __future__ import annotations

if __package__ is None or __package__ == "":
    import sys
    from pathlib import Path as _Path

    sys.path.append(str(_Path(__file__).resolve().parent.parent.parent))

import argparse
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sim.oracle.p3_joker_classifier import ensure_balatro_mechanics_root

SUPPORTED_TEMPLATES = {
    "accumulate_per_hand_then_apply",
    "accumulate_per_discard_then_apply",
    "end_round_payout_or_multiplier",
    "streak_or_consecutive_hand_rule",
}

# Prefer deterministic cross-round candidates from current unsupported set.
PRIORITY_KEYS = [
    "j_ancient_joker",
    "j_burnt_joker",
    "j_ice_cream",
    "j_loyalty_card",
    "j_marble_joker",
    "j_merry_andy",
    "j_popcorn",
    "j_ramen",
    "j_seltzer",
    "j_supernova",
    "j_the_idol",
    "j_troubadour",
    "j_turtle_bean",
    "j_dna",
]


KEY_TEMPLATE_OVERRIDES = {
    "j_seltzer": "accumulate_per_hand_then_apply",
    "j_supernova": "accumulate_per_hand_then_apply",
    "j_the_idol": "end_round_payout_or_multiplier",
    "j_troubadour": "end_round_payout_or_multiplier",
    "j_turtle_bean": "end_round_payout_or_multiplier",
    "j_dna": "streak_or_consecutive_hand_rule",
}

UNSUPPORTED_STOPWORDS = {
    "no_safe_template_match",
    "insufficient_structured_fields",
}


def _slug(text: str) -> str:
    low = str(text or "").strip().lower()
    low = re.sub(r"[^a-z0-9]+", "_", low)
    low = re.sub(r"_+", "_", low).strip("_")
    return low


def _record_text(entry: dict[str, Any]) -> str:
    parts = [
        str(entry.get("effect_summary") or ""),
        str(entry.get("key_mechanics") or ""),
        str(entry.get("trigger_timing") or ""),
        str(entry.get("reason") or ""),
    ]
    return " | ".join(parts).strip().lower()


def _choose_template(entry: dict[str, Any]) -> tuple[str | None, list[str], str | None]:
    text = _record_text(entry)
    rules: list[str] = []

    if any(x in text for x in ("chance", "random", "1 in ")):
        return None, rules, "probabilistic_or_unstable"

    if any(x in text for x in ("hands played", "per hand played", "remaining", "every hand")):
        rules.append("keyword:hands_played_or_remaining")
        return "accumulate_per_hand_then_apply", rules, None

    if any(x in text for x in ("discard", "first discarded", "card discarded", "per card discarded")):
        rules.append("keyword:discard_counter")
        return "accumulate_per_discard_then_apply", rules, None

    if any(x in text for x in ("per round", "end of round", "blind is selected", "when blind", "changes at end of round")):
        rules.append("keyword:round_transition")
        return "end_round_payout_or_multiplier", rules, None

    if any(x in text for x in ("consecutive", "streak", "every 6 hands", "remaining")):
        rules.append("keyword:streak_or_periodic")
        return "streak_or_consecutive_hand_rule", rules, None

    return None, rules, "not_stateful_deterministic"


def _load_json(path: Path) -> Any:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def classify_stateful_jokers(project_root: Path, mechanics_root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    derived_dir = mechanics_root / "derived"
    unsupported_path = derived_dir / "joker_template_unsupported.json"
    clusters_path = derived_dir / "p6_p3_unsupported_clusters.json"

    unsupported = _load_json(unsupported_path)
    if not isinstance(unsupported, list):
        raise FileNotFoundError(f"missing or invalid unsupported file: {unsupported_path}")

    clusters = _load_json(clusters_path)
    cluster_by_key: dict[str, str] = {}
    if isinstance(clusters, dict):
        for cluster in clusters.get("clusters") or []:
            if not isinstance(cluster, dict):
                continue
            cid = str(cluster.get("cluster_id") or "")
            for k in cluster.get("member_keys") or []:
                ks = str(k or "").strip().lower()
                if ks:
                    cluster_by_key[ks] = cid

    entries_by_key: dict[str, dict[str, Any]] = {}
    for row in unsupported:
        if not isinstance(row, dict):
            continue
        key = str(row.get("joker_key") or "").strip().lower()
        if not key:
            continue
        entries_by_key[key] = row

    mapping: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []

    ordered_keys = [k for k in PRIORITY_KEYS if k in entries_by_key]
    for k in sorted(entries_by_key.keys()):
        if k not in ordered_keys:
            ordered_keys.append(k)

    for key in ordered_keys:
        row = entries_by_key[key]
        name = str(row.get("joker_name") or key)
        template, rules, reject_reason = _choose_template(row)
        reason = str(row.get("reason") or "")

        if template is None and key in KEY_TEMPLATE_OVERRIDES:
            template = KEY_TEMPLATE_OVERRIDES[key]
            rules = list(rules) + [f"override:{key}:{template}"]
            reject_reason = None

        if reason not in UNSUPPORTED_STOPWORDS and template is None:
            reject_reason = reject_reason or f"unsupported_reason:{reason}"

        if template is None:
            rejected.append(
                {
                    "joker_name": name,
                    "joker_key": key,
                    "target": f"p7_{key}",
                    "reason": reject_reason or "not_selected",
                    "source_reason": reason,
                    "effect_summary": row.get("effect_summary"),
                }
            )
            continue

        params = {
            "strategy": "stateful_two_step_cycle",
            "family": template,
            "requires_multi_action": True,
            "state_schema": {
                "counter": "int",
                "armed": "bool",
                "last_round": "int",
            },
            "initial_state": {
                "counter": 0,
                "armed": False,
                "last_round": 0,
            },
        }

        mapping.append(
            {
                "joker_name": name,
                "joker_key": key,
                "template": template,
                "confidence": 0.74,
                "matched_rules": rules,
                "raw_fields_used": {
                    "effect_summary": row.get("effect_summary"),
                    "trigger_timing": row.get("trigger_timing"),
                    "key_mechanics": row.get("key_mechanics"),
                    "reason": row.get("reason"),
                    "cluster_id": cluster_by_key.get(key),
                },
                "params": params,
                "target": f"p7_{key}",
            }
        )

    metadata = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_unsupported": str(unsupported_path),
        "source_clusters": str(clusters_path),
        "total_unsupported_input": len(unsupported),
    }
    return mapping, rejected, metadata


def write_outputs(
    project_root: Path,
    mechanics_root: Path,
    mapping: list[dict[str, Any]],
    unsupported: list[dict[str, Any]],
    metadata: dict[str, Any],
    out_derived: Path | None = None,
    export_supported_targets: Path | None = None,
) -> dict[str, Any]:
    derived_dir = out_derived if out_derived is not None else (mechanics_root / "derived")
    derived_dir.mkdir(parents=True, exist_ok=True)

    map_path = derived_dir / "p7_template_map.json"
    unsupported_path = derived_dir / "p7_template_unsupported.json"
    supported_targets_path = export_supported_targets if export_supported_targets is not None else (derived_dir / "p7_supported_targets.txt")
    if not supported_targets_path.is_absolute():
        supported_targets_path = project_root / supported_targets_path
    supported_targets_path.parent.mkdir(parents=True, exist_ok=True)

    map_payload = {
        "metadata": metadata,
        "items": mapping,
    }
    map_path.write_text(json.dumps(map_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    unsupported_path.write_text(json.dumps(unsupported, ensure_ascii=False, indent=2), encoding="utf-8")

    supported_targets = sorted(set(str(x.get("target") or "") for x in mapping if str(x.get("template") or "") in SUPPORTED_TEMPLATES))
    supported_targets_path.write_text("\n".join(supported_targets) + "\n", encoding="utf-8")

    template_counter: Counter[str] = Counter(str(x.get("template") or "unknown") for x in mapping)
    unsupported_counter: Counter[str] = Counter(str(x.get("reason") or "unknown") for x in unsupported)

    coverage_doc = project_root / "docs" / "COVERAGE_P7_STATEFUL_JOKERS.md"
    coverage_doc.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    lines.append("# P7 Stateful Joker Coverage")
    lines.append("")
    lines.append(f"- Generated at: `{metadata.get('generated_at')}`")
    lines.append(f"- Input unsupported: **{metadata.get('total_unsupported_input', 0)}**")
    lines.append(f"- Supported stateful jokers: **{len(mapping)}**")
    lines.append(f"- Unsupported after P7 classifier: **{len(unsupported)}**")
    lines.append("")
    lines.append("## Template Family Counts")
    for tpl in sorted(SUPPORTED_TEMPLATES):
        lines.append(f"- `{tpl}`: {template_counter.get(tpl, 0)}")
    lines.append("")
    lines.append("## Unsupported Top Reasons")
    if unsupported_counter:
        for reason, count in unsupported_counter.most_common(20):
            lines.append(f"- `{reason}`: {count}")
    else:
        lines.append("- none")
    coverage_doc.write_text("\n".join(lines) + "\n", encoding="utf-8")

    status_doc = project_root / "docs" / "COVERAGE_P7_STATUS.md"
    status_doc.write_text(
        "\n".join(
            [
                "# P7 Stateful Joker Status",
                "",
                f"- supported: **{len(mapping)}**",
                f"- unsupported: **{len(unsupported)}**",
                f"- supported_targets_file: `{supported_targets_path}`",
                f"- template_families: `{', '.join(sorted(set(str(x.get('template') or '') for x in mapping)))}`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    return {
        "map_path": map_path,
        "unsupported_path": unsupported_path,
        "supported_targets_path": supported_targets_path,
        "coverage_doc": coverage_doc,
        "status_doc": status_doc,
        "total": int(metadata.get("total_unsupported_input") or 0),
        "supported": len(mapping),
        "unsupported": len(unsupported),
        "template_counts": dict(template_counter),
        "unsupported_reasons": dict(unsupported_counter),
    }


def build_and_write(
    project_root: Path,
    out_derived: Path | None = None,
    export_supported_targets: Path | None = None,
) -> dict[str, Any]:
    mechanics_root = ensure_balatro_mechanics_root(project_root)
    mapping, unsupported, metadata = classify_stateful_jokers(project_root, mechanics_root)
    summary = write_outputs(
        project_root=project_root,
        mechanics_root=mechanics_root,
        mapping=mapping,
        unsupported=unsupported,
        metadata=metadata,
        out_derived=out_derived,
        export_supported_targets=export_supported_targets,
    )
    summary["mechanics_root"] = mechanics_root
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build conservative P7 stateful joker template map and supported targets.")
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--out-derived", default="balatro_mechanics/derived")
    parser.add_argument("--export-supported-targets", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    project_root = Path(args.project_root).resolve()

    out_derived = Path(args.out_derived)
    if not out_derived.is_absolute():
        out_derived = project_root / out_derived

    export_supported_targets: Path | None = None
    if args.export_supported_targets:
        export_supported_targets = Path(args.export_supported_targets)

    summary = build_and_write(
        project_root=project_root,
        out_derived=out_derived,
        export_supported_targets=export_supported_targets,
    )

    print(f"mechanics_root={summary['mechanics_root']}")
    print(f"total={summary['total']} supported={summary['supported']} unsupported={summary['unsupported']}")
    print(f"map={summary['map_path']}")
    print(f"unsupported={summary['unsupported_path']}")
    print(f"supported_targets={summary['supported_targets_path']}")
    print(f"coverage_doc={summary['coverage_doc']}")
    print(f"status_doc={summary['status_doc']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
