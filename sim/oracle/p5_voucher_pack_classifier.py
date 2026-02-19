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

SUPPORTED_TEMPLATES = {
    "voucher_shop_event_observed",
    "pack_open_pick_first_observed",
}


def _slug(text: str) -> str:
    t = re.sub(r"\([^)]*\)", "", str(text or ""))
    t = t.strip().lower()
    t = re.sub(r"[^a-z0-9]+", "_", t)
    t = re.sub(r"_+", "_", t).strip("_")
    return t


def _read_csv(path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as fp:
        reader = csv.DictReader(fp)
        for row in reader:
            rows.append({k: str(v or "") for k, v in row.items()})
    return rows


def _voucher_key_candidates(voucher_name: str) -> list[str]:
    s = _slug(voucher_name)
    cands = [f"v_{s}"] if s else []
    # Handle apostrophe collapse (director_s_cut -> directors_cut)
    if "_s_" in s:
        cands.append(f"v_{s.replace('_s_', 's_')}")
    cands.append(f"v_{s}_norm")
    cands.append(f"v_{s}_plus")

    out: list[str] = []
    seen: set[str] = set()
    for c in cands:
        c = c.strip().lower()
        if c and c not in seen:
            seen.add(c)
            out.append(c)
    return out


def _pack_key(pack_type: str, size_variant: str) -> str:
    p = _slug(pack_type)
    s = _slug(size_variant)
    return f"p_{p}_{s}_1"


def classify_vouchers(mechanics_root: Path, capability: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    csv_path = mechanics_root / "vouchers.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"missing CSV: {csv_path}")

    rows = _read_csv(csv_path)
    mapping: list[dict[str, Any]] = []
    unsupported: list[dict[str, Any]] = []

    voucher_strategy = str(((capability.get("voucher") or {}).get("recommended_strategy") or "unsupported")).strip().lower()

    for row in rows:
        name = row.get("voucher_name", "").strip()
        slug = _slug(name)
        target = f"p5_voucher_{slug}"
        key_cands = _voucher_key_candidates(name)

        if voucher_strategy == "unsupported":
            mapping.append(
                {
                    "set_type": "voucher",
                    "item_name": name,
                    "item_key": slug,
                    "template": None,
                    "confidence": 0.0,
                    "matched_rules": [],
                    "params": {},
                    "target": target,
                    "raw_fields_used": row,
                }
            )
            unsupported.append(
                {
                    "set_type": "voucher",
                    "item_name": name,
                    "item_key": slug,
                    "target": target,
                    "reason": "capability_blocked:voucher",
                }
            )
            continue

        template = "voucher_shop_event_observed"
        params = {
            "strategy": "add_voucher_then_buy_then_wait",
            "voucher_key_candidates": key_cands,
            "allow_shop_fallback": True,
        }
        mapping.append(
            {
                "set_type": "voucher",
                "item_name": name,
                "item_key": slug,
                "template": template,
                "confidence": 0.9,
                "matched_rules": ["voucher:shop_event_observed"],
                "params": params,
                "target": target,
                "raw_fields_used": row,
            }
        )

    return mapping, unsupported


def classify_packs(mechanics_root: Path, capability: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    csv_path = mechanics_root / "booster_packs.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"missing CSV: {csv_path}")

    rows = _read_csv(csv_path)
    mapping: list[dict[str, Any]] = []
    unsupported: list[dict[str, Any]] = []

    pack_strategy = str(((capability.get("pack") or {}).get("recommended_strategy") or "unsupported")).strip().lower()

    for row in rows:
        name = row.get("pack_name", "").strip()
        slug = _slug(name)
        target = f"p5_pack_{slug}"
        pack_type = row.get("pack_type", "").strip()
        size_variant = row.get("size_variant", "").strip()
        key = _pack_key(pack_type, size_variant)

        if pack_strategy == "unsupported":
            mapping.append(
                {
                    "set_type": "pack",
                    "item_name": name,
                    "item_key": key,
                    "template": None,
                    "confidence": 0.0,
                    "matched_rules": [],
                    "params": {},
                    "target": target,
                    "raw_fields_used": row,
                }
            )
            unsupported.append(
                {
                    "set_type": "pack",
                    "item_name": name,
                    "item_key": key,
                    "target": target,
                    "reason": "capability_blocked:pack",
                }
            )
            continue

        # Conservative: exclude Mega by default due multi-pick complexity.
        if _slug(size_variant) == "mega":
            mapping.append(
                {
                    "set_type": "pack",
                    "item_name": name,
                    "item_key": key,
                    "template": None,
                    "confidence": 0.0,
                    "matched_rules": [],
                    "params": {},
                    "target": target,
                    "raw_fields_used": row,
                }
            )
            unsupported.append(
                {
                    "set_type": "pack",
                    "item_name": name,
                    "item_key": key,
                    "target": target,
                    "reason": "multi_pick_complex:mega_pack",
                }
            )
            continue

        template = "pack_open_pick_first_observed"
        params = {
            "strategy": "add_pack_then_buy_then_pack_then_wait",
            "pack_key": key,
            "allow_shop_fallback": True,
            "pack_select_policy": "first_playable",
        }
        mapping.append(
            {
                "set_type": "pack",
                "item_name": name,
                "item_key": key,
                "template": template,
                "confidence": 0.9,
                "matched_rules": ["pack:open_pick_first_observed"],
                "params": params,
                "target": target,
                "raw_fields_used": row,
            }
        )

    return mapping, unsupported


def _load_capability(project_root: Path) -> dict[str, Any]:
    cap_path = project_root / "balatro_mechanics" / "derived" / "p5_capabilities.json"
    if not cap_path.exists():
        return {}
    try:
        return json.loads(cap_path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def write_outputs(
    project_root: Path,
    mechanics_root: Path,
    mapping: list[dict[str, Any]],
    unsupported: list[dict[str, Any]],
    out_derived: Path | None = None,
    export_supported_targets: Path | None = None,
) -> dict[str, Any]:
    derived_dir = out_derived if out_derived is not None else (mechanics_root / "derived")
    derived_dir.mkdir(parents=True, exist_ok=True)

    map_path = derived_dir / "p5_template_map.json"
    unsupported_path = derived_dir / "p5_template_unsupported.json"
    supported_targets_path = export_supported_targets if export_supported_targets is not None else (derived_dir / "p5_supported_targets.txt")

    if not supported_targets_path.is_absolute():
        supported_targets_path = project_root / supported_targets_path
    supported_targets_path.parent.mkdir(parents=True, exist_ok=True)

    map_path.write_text(json.dumps(mapping, ensure_ascii=False, indent=2), encoding="utf-8")
    unsupported_path.write_text(json.dumps(unsupported, ensure_ascii=False, indent=2), encoding="utf-8")

    total = len(mapping)
    supported_entries = [x for x in mapping if isinstance(x.get("template"), str) and x.get("template") in SUPPORTED_TEMPLATES]
    supported = len(supported_entries)

    template_counter: Counter[str] = Counter()
    set_counter: Counter[str] = Counter()
    for x in supported_entries:
        template_counter[str(x.get("template"))] += 1
        set_counter[str(x.get("set_type"))] += 1

    unsupported_counter: Counter[str] = Counter(str(x.get("reason") or "unknown") for x in unsupported)

    supported_targets = sorted(set(str(x.get("target") or "") for x in supported_entries if str(x.get("target") or "")))
    supported_targets_path.write_text("\n".join(supported_targets) + "\n", encoding="utf-8")

    coverage_doc = project_root / "docs" / "COVERAGE_P5_VOUCHERS_PACKS.md"
    coverage_doc.parent.mkdir(parents=True, exist_ok=True)

    vouchers_total = len([x for x in mapping if x.get("set_type") == "voucher"])
    vouchers_supported = len([x for x in supported_entries if x.get("set_type") == "voucher"])
    packs_total = len([x for x in mapping if x.get("set_type") == "pack"])
    packs_supported = len([x for x in supported_entries if x.get("set_type") == "pack"])

    voucher_threshold = min(8, max(1, int(round(vouchers_total * 0.25))))
    pack_threshold = min(6, max(1, int(round(packs_total * 0.35))))

    lines: list[str] = []
    lines.append("# P5 Vouchers + Booster Packs Coverage")
    lines.append("")
    lines.append(f"- Generated at: `{datetime.now(timezone.utc).isoformat()}`")
    lines.append(f"- Total targets: **{total}**")
    lines.append(f"- Supported (conservative): **{supported}**")
    lines.append(f"- Unsupported: **{len(unsupported)}**")
    lines.append("")
    lines.append("## Gate Thresholds")
    lines.append(f"- Voucher supported: **{vouchers_supported}/{vouchers_total}** (threshold >= {voucher_threshold})")
    lines.append(f"- Pack supported: **{packs_supported}/{packs_total}** (threshold >= {pack_threshold})")
    lines.append("")
    lines.append("## Supported Template Counts")
    for tpl in sorted(SUPPORTED_TEMPLATES):
        lines.append(f"- `{tpl}`: {template_counter.get(tpl, 0)}")
    lines.append("")
    lines.append("## Supported By Set")
    lines.append(f"- `voucher`: {set_counter.get('voucher', 0)}")
    lines.append(f"- `pack`: {set_counter.get('pack', 0)}")
    lines.append("")
    lines.append("## Unsupported Top Reasons")
    if unsupported_counter:
        for reason, count in unsupported_counter.most_common(20):
            lines.append(f"- `{reason}`: {count}")
    else:
        lines.append("- none")

    coverage_doc.write_text("\n".join(lines) + "\n", encoding="utf-8")

    status_doc = project_root / "docs" / "COVERAGE_P5_STATUS.md"
    status_doc.write_text(
        "\n".join(
            [
                "# P5 Voucher/Pack Coverage Status",
                "",
                f"- total: **{total}**",
                f"- supported: **{supported}**",
                f"- unsupported: **{len(unsupported)}**",
                f"- vouchers_supported: **{vouchers_supported}/{vouchers_total}**",
                f"- packs_supported: **{packs_supported}/{packs_total}**",
                f"- supported_targets_file: `{supported_targets_path}`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    return {
        "total": total,
        "supported": supported,
        "unsupported": len(unsupported),
        "vouchers_total": vouchers_total,
        "vouchers_supported": vouchers_supported,
        "packs_total": packs_total,
        "packs_supported": packs_supported,
        "map_path": map_path,
        "unsupported_path": unsupported_path,
        "supported_targets_path": supported_targets_path,
        "coverage_doc": coverage_doc,
        "status_doc": status_doc,
        "template_counts": dict(template_counter),
        "unsupported_reasons": dict(unsupported_counter),
    }


def build_and_write(
    project_root: Path,
    out_derived: Path | None = None,
    export_supported_targets: Path | None = None,
) -> dict[str, Any]:
    mechanics_root = ensure_balatro_mechanics_root(project_root)
    capability = _load_capability(project_root)
    vouchers_map, vouchers_unsupported = classify_vouchers(mechanics_root, capability)
    packs_map, packs_unsupported = classify_packs(mechanics_root, capability)

    mapping = vouchers_map + packs_map
    unsupported = vouchers_unsupported + packs_unsupported

    summary = write_outputs(
        project_root=project_root,
        mechanics_root=mechanics_root,
        mapping=mapping,
        unsupported=unsupported,
        out_derived=out_derived,
        export_supported_targets=export_supported_targets,
    )
    summary["mechanics_root"] = mechanics_root
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build conservative P5 voucher/pack template map and coverage docs.")
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
    print(f"vouchers={summary['vouchers_supported']}/{summary['vouchers_total']} packs={summary['packs_supported']}/{summary['packs_total']}")
    print(f"map={summary['map_path']}")
    print(f"unsupported={summary['unsupported_path']}")
    print(f"supported_targets={summary['supported_targets_path']}")
    print(f"coverage_doc={summary['coverage_doc']}")
    print(f"status_doc={summary['status_doc']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
