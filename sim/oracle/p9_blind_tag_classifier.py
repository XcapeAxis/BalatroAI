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
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sim.oracle.p3_joker_classifier import ensure_balatro_mechanics_root


BOSS_SUPPORTED_MIN = 8
TAG_SUPPORTED_MIN = 8


def _slug(text: str) -> str:
    low = str(text or "").strip().lower()
    low = re.sub(r"[^a-z0-9]+", "_", low)
    low = re.sub(r"_+", "_", low).strip("_")
    return low


def _load_csv_rows(path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as fp:
        reader = csv.DictReader(fp)
        for row in reader:
            if isinstance(row, dict):
                rows.append({str(k): str(v or "") for k, v in row.items()})
    return rows


def _classify_boss(row: dict[str, str]) -> tuple[str | None, float, list[str], str | None]:
    text = " | ".join(
        [
            row.get("boss_name", ""),
            row.get("restriction_or_rule_change", ""),
            row.get("what_it_blocks_or_forces", ""),
            row.get("strategy_notes_if_explicit_in_wiki", ""),
            row.get("notes", ""),
        ]
    ).lower()

    if any(x in text for x in ("chance", "random", "probability", "1 in ")):
        return None, 0.0, [], "probabilistic_or_unstable"

    rules: list[str] = []
    if "discard" in text:
        rules.append("kw:discard")
        return "boss_limit_discards", 0.78, rules, None
    if "hand" in text and any(x in text for x in ("reduce", "fewer", "disable", "limit")):
        rules.append("kw:hand_limit")
        return "boss_limit_hands", 0.78, rules, None
    if "select" in text or "forced" in text:
        rules.append("kw:forced_selection")
        return "boss_forced_selection", 0.76, rules, None
    if any(x in text for x in ("face", "rank", "suit", "debuff")):
        rules.append("kw:card_rule")
        return "boss_card_rule", 0.74, rules, None
    if any(x in text for x in ("shop", "pack", "voucher")):
        rules.append("kw:shop_pack")
        return "boss_shop_rule", 0.72, rules, None

    return "boss_rule_observed", 0.70, ["fallback:boss_rule_observed"], None


def _classify_tag(row: dict[str, str]) -> tuple[str | None, float, list[str], str | None]:
    text = " | ".join(
        [
            row.get("tag_name", ""),
            row.get("effect_summary", ""),
            row.get("trigger_timing", ""),
            row.get("notes", ""),
        ]
    ).lower()

    if any(x in text for x in ("chance", "random", "probability", "1 in ")):
        return None, 0.0, [], "probabilistic_or_unstable"

    rules: list[str] = []
    if "boss blind" in text and any(x in text for x in ("re-roll", "reroll", "re roll")):
        rules.append("kw:boss_reroll")
        return "tag_boss_reroll", 0.80, rules, None
    if "pack" in text and "free" in text:
        rules.append("kw:free_pack")
        return "tag_free_pack", 0.78, rules, None
    if "shop" in text:
        rules.append("kw:shop_event")
        return "tag_shop_event", 0.76, rules, None
    if any(x in text for x in ("gain", "reward", "double", "interest", "money", "$")):
        rules.append("kw:reward_event")
        return "tag_reward_event", 0.72, rules, None

    return "tag_event_observed", 0.70, ["fallback:tag_event_observed"], None


def _make_boss_item(row: dict[str, str]) -> dict[str, Any]:
    name = row.get("boss_name", "").strip()
    key = _slug(name)
    template, confidence, rules, reason = _classify_boss(row)
    item = {
        "boss_name": name,
        "boss_key": key,
        "template": template,
        "confidence": confidence,
        "matched_rules": rules,
        "raw_fields_used": {
            "ante_or_tier_info": row.get("ante_or_tier_info", ""),
            "restriction_or_rule_change": row.get("restriction_or_rule_change", ""),
            "what_it_blocks_or_forces": row.get("what_it_blocks_or_forces", ""),
            "notes": row.get("notes", ""),
        },
        "unsupported_reason": reason,
    }
    return item


def _make_tag_item(row: dict[str, str]) -> dict[str, Any]:
    name = row.get("tag_name", "").strip()
    key = _slug(name)
    template, confidence, rules, reason = _classify_tag(row)
    item = {
        "tag_name": name,
        "tag_key": key,
        "template": template,
        "confidence": confidence,
        "matched_rules": rules,
        "raw_fields_used": {
            "effect_summary": row.get("effect_summary", ""),
            "trigger_timing": row.get("trigger_timing", ""),
            "notes": row.get("notes", ""),
        },
        "unsupported_reason": reason,
    }
    return item


def classify_blinds_tags(mechanics_root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    boss_rows = _load_csv_rows(mechanics_root / "boss_blinds.csv")
    tag_rows = _load_csv_rows(mechanics_root / "tags.csv")

    boss_items = [_make_boss_item(row) for row in boss_rows]
    tag_items = [_make_tag_item(row) for row in tag_rows]

    boss_supported = [x for x in boss_items if isinstance(x.get("template"), str) and not x.get("unsupported_reason")]
    boss_unsupported = [x for x in boss_items if x not in boss_supported]
    tag_supported = [x for x in tag_items if isinstance(x.get("template"), str) and not x.get("unsupported_reason")]
    tag_unsupported = [x for x in tag_items if x not in tag_supported]

    # Keep deterministic first wave conservative but wide enough.
    boss_supported = sorted(boss_supported, key=lambda x: str(x.get("boss_key") or ""))
    tag_supported = sorted(tag_supported, key=lambda x: str(x.get("tag_key") or ""))

    boss_payload = {
        "metadata": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "total": len(boss_items),
            "supported": len(boss_supported),
            "unsupported": len(boss_unsupported),
            "supported_min_required": BOSS_SUPPORTED_MIN,
        },
        "items": boss_supported,
    }
    boss_unsupported_payload = {
        "metadata": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "total": len(boss_items),
            "supported": len(boss_supported),
            "unsupported": len(boss_unsupported),
        },
        "items": boss_unsupported,
    }

    tag_payload = {
        "metadata": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "total": len(tag_items),
            "supported": len(tag_supported),
            "unsupported": len(tag_unsupported),
            "supported_min_required": TAG_SUPPORTED_MIN,
        },
        "items": tag_supported,
    }
    tag_unsupported_payload = {
        "metadata": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "total": len(tag_items),
            "supported": len(tag_supported),
            "unsupported": len(tag_unsupported),
        },
        "items": tag_unsupported,
    }

    return (
        {
            "supported": boss_payload,
            "unsupported": boss_unsupported_payload,
        },
        {
            "supported": tag_payload,
            "unsupported": tag_unsupported_payload,
        },
    )


def _build_episode_targets(
    boss_supported: list[dict[str, Any]],
    tag_supported: list[dict[str, Any]],
) -> list[str]:
    count = min(6, len(boss_supported), len(tag_supported))
    targets: list[str] = []
    for idx in range(count):
        b = str(boss_supported[idx].get("boss_key") or f"boss_{idx+1}")
        t = str(tag_supported[idx].get("tag_key") or f"tag_{idx+1}")
        targets.append(f"p9_episode_{idx+1:02d}_{b}_{t}")
    return targets


def write_outputs(
    project_root: Path,
    out_derived: Path | None = None,
    export_supported_targets: Path | None = None,
) -> dict[str, Any]:
    mechanics_root = ensure_balatro_mechanics_root(project_root)
    derived_dir = out_derived if out_derived is not None else (mechanics_root / "derived")
    derived_dir.mkdir(parents=True, exist_ok=True)

    (boss_data, tag_data) = classify_blinds_tags(mechanics_root)

    boss_map_path = derived_dir / "p9_blinds_template_map.json"
    boss_unsupported_path = derived_dir / "p9_blinds_template_unsupported.json"
    tag_map_path = derived_dir / "p9_tags_template_map.json"
    tag_unsupported_path = derived_dir / "p9_tags_template_unsupported.json"

    boss_map_path.write_text(json.dumps(boss_data["supported"], ensure_ascii=False, indent=2), encoding="utf-8")
    boss_unsupported_path.write_text(json.dumps(boss_data["unsupported"], ensure_ascii=False, indent=2), encoding="utf-8")
    tag_map_path.write_text(json.dumps(tag_data["supported"], ensure_ascii=False, indent=2), encoding="utf-8")
    tag_unsupported_path.write_text(json.dumps(tag_data["unsupported"], ensure_ascii=False, indent=2), encoding="utf-8")

    boss_supported_items = list(boss_data["supported"].get("items") or [])
    tag_supported_items = list(tag_data["supported"].get("items") or [])
    supported_targets = _build_episode_targets(boss_supported_items, tag_supported_items)

    targets_path = export_supported_targets if export_supported_targets is not None else (derived_dir / "p9_supported_targets.txt")
    if not targets_path.is_absolute():
        targets_path = project_root / targets_path
    targets_path.parent.mkdir(parents=True, exist_ok=True)
    targets_path.write_text("\n".join(supported_targets) + "\n", encoding="utf-8")

    coverage_path = project_root / "docs" / "COVERAGE_P9_BLINDS_TAGS.md"
    status_path = project_root / "docs" / "COVERAGE_P9_STATUS.md"
    coverage_path.parent.mkdir(parents=True, exist_ok=True)

    boss_tpl_counter = Counter(str(x.get("template") or "unknown") for x in boss_supported_items)
    tag_tpl_counter = Counter(str(x.get("template") or "unknown") for x in tag_supported_items)
    boss_uns_counter = Counter(str(x.get("unsupported_reason") or "unknown") for x in (boss_data["unsupported"].get("items") or []))
    tag_uns_counter = Counter(str(x.get("unsupported_reason") or "unknown") for x in (tag_data["unsupported"].get("items") or []))

    coverage_lines: list[str] = []
    coverage_lines.append("# P9 Blinds/Tags Coverage")
    coverage_lines.append("")
    coverage_lines.append(f"- generated_at: `{datetime.now(timezone.utc).isoformat()}`")
    coverage_lines.append(f"- boss_supported: **{len(boss_supported_items)}** / {boss_data['supported']['metadata']['total']}")
    coverage_lines.append(f"- tag_supported: **{len(tag_supported_items)}** / {tag_data['supported']['metadata']['total']}")
    coverage_lines.append("")
    coverage_lines.append("## Boss Template Counts")
    for k, v in sorted(boss_tpl_counter.items()):
        coverage_lines.append(f"- `{k}`: {v}")
    coverage_lines.append("")
    coverage_lines.append("## Tag Template Counts")
    for k, v in sorted(tag_tpl_counter.items()):
        coverage_lines.append(f"- `{k}`: {v}")
    coverage_lines.append("")
    coverage_lines.append("## Boss Unsupported Top Reasons")
    if boss_uns_counter:
        for k, v in boss_uns_counter.most_common(10):
            coverage_lines.append(f"- `{k}`: {v}")
    else:
        coverage_lines.append("- none")
    coverage_lines.append("")
    coverage_lines.append("## Tag Unsupported Top Reasons")
    if tag_uns_counter:
        for k, v in tag_uns_counter.most_common(10):
            coverage_lines.append(f"- `{k}`: {v}")
    else:
        coverage_lines.append("- none")
    coverage_lines.append("")
    coverage_lines.append("## Episode Targets (for P9 batch)")
    for t in supported_targets:
        coverage_lines.append(f"- `{t}`")
    coverage_path.write_text("\n".join(coverage_lines) + "\n", encoding="utf-8")

    status_lines = [
        "# P9 Status",
        "",
        f"- boss_supported: **{len(boss_supported_items)}** (required >= {BOSS_SUPPORTED_MIN})",
        f"- tag_supported: **{len(tag_supported_items)}** (required >= {TAG_SUPPORTED_MIN})",
        f"- targets_file: `{targets_path}`",
        f"- boss_map: `{boss_map_path}`",
        f"- tag_map: `{tag_map_path}`",
    ]
    status_path.write_text("\n".join(status_lines) + "\n", encoding="utf-8")

    return {
        "mechanics_root": str(mechanics_root),
        "boss_supported": len(boss_supported_items),
        "boss_total": int(boss_data["supported"]["metadata"]["total"]),
        "tag_supported": len(tag_supported_items),
        "tag_total": int(tag_data["supported"]["metadata"]["total"]),
        "boss_map_path": str(boss_map_path),
        "boss_unsupported_path": str(boss_unsupported_path),
        "tag_map_path": str(tag_map_path),
        "tag_unsupported_path": str(tag_unsupported_path),
        "targets_path": str(targets_path),
        "coverage_path": str(coverage_path),
        "status_path": str(status_path),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Classify deterministic boss blinds and tags for P9 episode fixtures.")
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

    export_path: Path | None = None
    if args.export_supported_targets:
        export_path = Path(args.export_supported_targets)

    summary = write_outputs(project_root, out_derived=out_derived, export_supported_targets=export_path)
    print(f"mechanics_root={summary['mechanics_root']}")
    print(f"boss_supported={summary['boss_supported']}/{summary['boss_total']}")
    print(f"tag_supported={summary['tag_supported']}/{summary['tag_total']}")
    print(f"targets={summary['targets_path']}")
    print(f"coverage={summary['coverage_path']}")
    print(f"status={summary['status_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
