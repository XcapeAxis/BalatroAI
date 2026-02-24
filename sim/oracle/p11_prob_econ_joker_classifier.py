from __future__ import annotations

if __package__ is None or __package__ == "":
    import sys
    from pathlib import Path as _Path

    sys.path.append(str(_Path(__file__).resolve().parent.parent.parent))

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sim.oracle.p3_joker_classifier import ensure_balatro_mechanics_root

SUPPORTED_TEMPLATES = {"prob_trigger_observed", "econ_shop_observed"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build P11 probabilistic/economy joker template map.")
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--out-derived", default="balatro_mechanics/derived")
    parser.add_argument("--export-supported-targets", default=None)
    return parser.parse_args()


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _safe_key(text: str) -> str:
    return str(text or "").strip().lower()


def _build_items(
    pick_payload: dict[str, Any],
    base_map_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    base_by_key = {_safe_key(x.get("joker_key")): x for x in base_map_rows if isinstance(x, dict)}
    mapping: list[dict[str, Any]] = []
    unsupported: list[dict[str, Any]] = []

    def add_rows(rows: list[dict[str, Any]], category: str) -> None:
        for row in rows:
            target = str(row.get("target") or "").strip()
            joker_key = _safe_key(row.get("joker_key"))
            if not target or not joker_key:
                continue

            base = base_by_key.get(joker_key, {})
            raw_fields = base.get("raw_fields_used") if isinstance(base.get("raw_fields_used"), dict) else {}
            template = "prob_trigger_observed" if category == "prob" else "econ_shop_observed"
            mode = "prob" if category == "prob" else "econ"

            mapping.append(
                {
                    "target": target,
                    "joker_key": joker_key,
                    "joker_name": str(row.get("joker_name") or base.get("joker_name") or joker_key),
                    "template": template,
                    "confidence": 0.76,
                    "matched_rules": list(row.get("hits") or []),
                    "raw_fields_used": {
                        "effect_summary": raw_fields.get("effect_summary"),
                        "trigger_timing": raw_fields.get("trigger_timing"),
                        "key_mechanics": raw_fields.get("key_mechanics"),
                        "source_reason": row.get("source_reason"),
                    },
                    "params": {
                        "category": category,
                        "mode": mode,
                        "strategy": "p11_prob_econ_multistep",
                        "min_actions": 6,
                        "jokers": [
                            {
                                "key": joker_key,
                                "mode": mode,
                                "kind": f"{category}_joker",
                                "tags": [category],
                            }
                        ],
                    },
                }
            )

    add_rows(list(pick_payload.get("prob_targets") or []), "prob")
    add_rows(list(pick_payload.get("econ_targets") or []), "econ")

    seen: set[str] = set()
    dedup: list[dict[str, Any]] = []
    for item in mapping:
        t = str(item.get("target") or "")
        if not t or t in seen:
            continue
        seen.add(t)
        dedup.append(item)

    return dedup, unsupported


def build_and_write(
    project_root: Path,
    *,
    out_derived: Path | None = None,
    export_supported_targets: Path | None = None,
) -> dict[str, Any]:
    mechanics_root = ensure_balatro_mechanics_root(project_root)
    derived = mechanics_root / "derived"
    pick_path = derived / "p11_pick_payload.json"
    base_map_path = derived / "joker_template_map.json"
    if not pick_path.exists():
        raise FileNotFoundError(f"missing picker output: {pick_path}")
    if not base_map_path.exists():
        raise FileNotFoundError(f"missing base map: {base_map_path}")

    pick_payload = _read_json(pick_path)
    base_map_rows = [x for x in _read_json(base_map_path) if isinstance(x, dict)]
    mapping, unsupported = _build_items(pick_payload, base_map_rows)

    out_dir = out_derived if out_derived is not None else derived
    out_dir.mkdir(parents=True, exist_ok=True)
    map_path = out_dir / "p11_template_map.json"
    unsupported_path = out_dir / "p11_template_unsupported.json"
    supported_targets_path = export_supported_targets if export_supported_targets is not None else (out_dir / "p11_supported_targets.txt")
    if not supported_targets_path.is_absolute():
        supported_targets_path = project_root / supported_targets_path
    supported_targets_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "metadata": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "source_pick": str(pick_path),
            "source_map": str(base_map_path),
        },
        "items": mapping,
    }
    map_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    unsupported_path.write_text(json.dumps(unsupported, ensure_ascii=False, indent=2), encoding="utf-8")

    supported_targets = sorted(str(x.get("target") or "") for x in mapping if str(x.get("template") or "") in SUPPORTED_TEMPLATES)
    supported_targets = [x for x in supported_targets if x]
    supported_targets_path.write_text("\n".join(supported_targets) + "\n", encoding="utf-8")

    tpl_counter = Counter(str(x.get("template") or "") for x in mapping)
    coverage_doc = project_root / "docs" / "COVERAGE_P11_PROB_ECON.md"
    status_doc = project_root / "docs" / "COVERAGE_P11_STATUS.md"
    coverage_doc.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = []
    lines.append("# P11 Prob/Econ Joker Coverage")
    lines.append("")
    lines.append(f"- generated_at: `{payload['metadata']['generated_at']}`")
    lines.append(f"- prob targets: **{len([x for x in mapping if str((x.get('params') or {}).get('category') or '') == 'prob'])}**")
    lines.append(f"- econ targets: **{len([x for x in mapping if str((x.get('params') or {}).get('category') or '') == 'econ'])}**")
    lines.append(f"- supported total: **{len(supported_targets)}**")
    lines.append(f"- unsupported total: **{len(unsupported)}**")
    lines.append("")
    lines.append("## Template Counts")
    for k, v in sorted(tpl_counter.items()):
        lines.append(f"- `{k}`: {v}")
    lines.append("")
    lines.append("## Supported Targets")
    for t in supported_targets:
        lines.append(f"- `{t}`")
    coverage_doc.write_text("\n".join(lines) + "\n", encoding="utf-8")

    status_doc.write_text(
        "\n".join(
            [
                "# P11 Status",
                "",
                f"- supported: **{len(supported_targets)}**",
                f"- unsupported: **{len(unsupported)}**",
                f"- supported_targets_file: `{supported_targets_path}`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    return {
        "mechanics_root": mechanics_root,
        "map_path": map_path,
        "unsupported_path": unsupported_path,
        "supported_targets_path": supported_targets_path,
        "coverage_doc": coverage_doc,
        "status_doc": status_doc,
        "supported": len(supported_targets),
        "unsupported": len(unsupported),
    }


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
    print(f"supported={summary['supported']} unsupported={summary['unsupported']}")
    print(f"map={summary['map_path']}")
    print(f"unsupported={summary['unsupported_path']}")
    print(f"supported_targets={summary['supported_targets_path']}")
    print(f"coverage_doc={summary['coverage_doc']}")
    print(f"status_doc={summary['status_doc']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

