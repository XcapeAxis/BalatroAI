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

MIN_PROB = 10
MIN_ECON = 10

PROB_PATTERNS = [
    r"\bchance\b",
    r"\brandom\b",
    r"\b1 in \d+\b",
    r"\bprobabil",
    r"\bmay\b.*\bcreate\b",
    r"\bretrigger\b",
    r"\broll\b",
]

ECON_PATTERNS = [
    r"\bmoney\b",
    r"\bshop\b",
    r"\bbuy\b",
    r"\bsell\b",
    r"\breroll\b",
    r"\bvoucher\b",
    r"\bpack\b",
    r"\bbooster\b",
    r"\$\d+",
    r"\bfree\b",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pick P11 probabilistic/economy joker targets from derived unsupported + map.")
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--out-derived", default="balatro_mechanics/derived")
    parser.add_argument("--out-docs", default="docs")
    return parser.parse_args()


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _safe_key(text: str) -> str:
    low = str(text or "").strip().lower()
    low = re.sub(r"[^a-z0-9]+", "_", low)
    low = re.sub(r"_+", "_", low).strip("_")
    return low


def _row_text(row: dict[str, Any]) -> str:
    raw = row.get("raw_fields_used") if isinstance(row.get("raw_fields_used"), dict) else {}
    parts = [
        str(row.get("joker_name") or ""),
        str(row.get("joker_key") or ""),
        str(raw.get("effect_summary") or ""),
        str(raw.get("key_mechanics") or ""),
        str(raw.get("trigger_timing") or ""),
        str(raw.get("constraints_or_conditions") or ""),
        str((row.get("params") or {}).get("promoted_reason") or ""),
        str(row.get("reason") or ""),
    ]
    return " | ".join(parts).lower()


def _match_patterns(text: str, patterns: list[str]) -> list[str]:
    matched: list[str] = []
    for p in patterns:
        if re.search(p, text):
            matched.append(p)
    return matched


def _candidate_info(row: dict[str, Any]) -> dict[str, Any]:
    text = _row_text(row)
    prob_hits = _match_patterns(text, PROB_PATTERNS)
    econ_hits = _match_patterns(text, ECON_PATTERNS)
    return {
        "joker_name": str(row.get("joker_name") or ""),
        "joker_key": str(row.get("joker_key") or ""),
        "template": row.get("template"),
        "target": str(row.get("target") or ""),
        "prob_hits": prob_hits,
        "econ_hits": econ_hits,
        "source_reason": str(row.get("reason") or str((row.get("params") or {}).get("promoted_reason") or "")),
        "raw_fields_used": row.get("raw_fields_used") if isinstance(row.get("raw_fields_used"), dict) else {},
    }


def _pick_with_fallback(candidates: list[dict[str, Any]], want: int, hit_field: str) -> list[dict[str, Any]]:
    with_hits = [c for c in candidates if c.get(hit_field)]
    no_hits = [c for c in candidates if not c.get(hit_field)]
    ordered = sorted(with_hits, key=lambda x: (-len(x.get(hit_field) or []), x.get("joker_key") or ""))
    if len(ordered) < want:
        ordered.extend(sorted(no_hits, key=lambda x: x.get("joker_key") or ""))
    return ordered[:want]


def _unique_by_key(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for r in rows:
        key = str(r.get("joker_key") or "")
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(r)
    return out


def main() -> int:
    args = parse_args()
    project_root = Path(args.project_root).resolve()
    out_derived = Path(args.out_derived)
    if not out_derived.is_absolute():
        out_derived = project_root / out_derived
    out_docs = Path(args.out_docs)
    if not out_docs.is_absolute():
        out_docs = project_root / out_docs

    mechanics_root = ensure_balatro_mechanics_root(project_root)
    derived = mechanics_root / "derived"
    unsupported_path = derived / "joker_template_unsupported.json"
    map_path = derived / "joker_template_map.json"
    if not unsupported_path.exists():
        raise FileNotFoundError(f"missing: {unsupported_path}")
    if not map_path.exists():
        raise FileNotFoundError(f"missing: {map_path}")

    unsupported_raw = _read_json(unsupported_path)
    map_raw = _read_json(map_path)
    unsupported_rows = [x for x in unsupported_raw if isinstance(x, dict)]
    map_rows = [x for x in map_raw if isinstance(x, dict)]

    unsupported_keys = {str(x.get("joker_key") or "").strip().lower() for x in unsupported_rows}
    unsupported_keys = {k for k in unsupported_keys if k}

    infos_all = [_candidate_info(x) for x in map_rows]
    infos_all = [x for x in infos_all if x.get("joker_key")]
    infos_all = _unique_by_key(infos_all)

    infos_unsupported_first = [x for x in infos_all if str(x.get("joker_key")).lower() in unsupported_keys]
    infos_other = [x for x in infos_all if str(x.get("joker_key")).lower() not in unsupported_keys]

    prob_selected = _pick_with_fallback(infos_unsupported_first, MIN_PROB, "prob_hits")
    if len(prob_selected) < MIN_PROB:
        fill = _pick_with_fallback(infos_other, MIN_PROB - len(prob_selected), "prob_hits")
        prob_selected.extend(fill)
    prob_selected = _unique_by_key(prob_selected)[:MIN_PROB]

    econ_selected = _pick_with_fallback(infos_unsupported_first, MIN_ECON, "econ_hits")
    if len(econ_selected) < MIN_ECON:
        fill = _pick_with_fallback(infos_other, MIN_ECON - len(econ_selected), "econ_hits")
        econ_selected.extend(fill)
    econ_selected = _unique_by_key(econ_selected)[:MIN_ECON]

    # Build target names.
    prob_targets: list[str] = [f"p11_prob_{_safe_key(x['joker_key'])}" for x in prob_selected]
    econ_targets: list[str] = [f"p11_econ_{_safe_key(x['joker_key'])}" for x in econ_selected]
    supported_targets = sorted(set(prob_targets + econ_targets))

    out_derived.mkdir(parents=True, exist_ok=True)
    prob_file = out_derived / "p11_prob_targets.txt"
    econ_file = out_derived / "p11_econ_targets.txt"
    supported_file = out_derived / "p11_supported_targets.txt"
    prob_file.write_text("\n".join(prob_targets) + "\n", encoding="utf-8")
    econ_file.write_text("\n".join(econ_targets) + "\n", encoding="utf-8")
    supported_file.write_text("\n".join(supported_targets) + "\n", encoding="utf-8")

    pick_payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_unsupported": str(unsupported_path),
        "source_map": str(map_path),
        "prob_targets": [
            {
                "target": f"p11_prob_{_safe_key(x['joker_key'])}",
                "joker_key": x["joker_key"],
                "joker_name": x["joker_name"],
                "hits": x.get("prob_hits") or [],
                "source_reason": x.get("source_reason"),
            }
            for x in prob_selected
        ],
        "econ_targets": [
            {
                "target": f"p11_econ_{_safe_key(x['joker_key'])}",
                "joker_key": x["joker_key"],
                "joker_name": x["joker_name"],
                "hits": x.get("econ_hits") or [],
                "source_reason": x.get("source_reason"),
            }
            for x in econ_selected
        ],
    }
    pick_json = out_derived / "p11_pick_payload.json"
    pick_json.write_text(json.dumps(pick_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    docs_file = out_docs / "COVERAGE_P11_PICK.md"
    docs_file.parent.mkdir(parents=True, exist_ok=True)
    prob_hit_counter = Counter(hit for row in prob_selected for hit in (row.get("prob_hits") or []))
    econ_hit_counter = Counter(hit for row in econ_selected for hit in (row.get("econ_hits") or []))

    lines: list[str] = []
    lines.append("# P11 Pick Coverage")
    lines.append("")
    lines.append(f"- Generated at: `{pick_payload['generated_at']}`")
    lines.append(f"- Prob targets selected: **{len(prob_targets)}**")
    lines.append(f"- Econ targets selected: **{len(econ_targets)}**")
    lines.append(f"- Supported union targets: **{len(supported_targets)}**")
    lines.append("")
    lines.append("## Prob Targets")
    for row in pick_payload["prob_targets"]:
        lines.append(
            f"- `{row['target']}` <- `{row['joker_key']}` (`{row['joker_name']}`), hits={row['hits']}, reason={row.get('source_reason')}"
        )
    lines.append("")
    lines.append("## Econ Targets")
    for row in pick_payload["econ_targets"]:
        lines.append(
            f"- `{row['target']}` <- `{row['joker_key']}` (`{row['joker_name']}`), hits={row['hits']}, reason={row.get('source_reason')}"
        )
    lines.append("")
    lines.append("## Hit Summary")
    lines.append("- Prob regex hits:")
    for key, count in prob_hit_counter.most_common():
        lines.append(f"  - `{key}`: {count}")
    lines.append("- Econ regex hits:")
    for key, count in econ_hit_counter.most_common():
        lines.append(f"  - `{key}`: {count}")
    docs_file.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"mechanics_root={mechanics_root}")
    print(f"unsupported_input={len(unsupported_rows)}")
    print(f"prob_targets={len(prob_targets)} file={prob_file}")
    print(f"econ_targets={len(econ_targets)} file={econ_file}")
    print(f"supported_targets={len(supported_targets)} file={supported_file}")
    print(f"coverage_doc={docs_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

