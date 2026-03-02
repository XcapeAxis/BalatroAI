from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import requests  # type: ignore
except Exception:  # pragma: no cover - optional dependency at runtime
    requests = None


TEXT_FIELDS_BY_FILE: dict[str, list[str]] = {
    "jokers.csv": [
        "joker_name",
        "effect_summary",
        "trigger_timing",
        "key_mechanics",
        "constraints_or_conditions",
        "stacking_rules_or_notes",
    ],
    "tarot_cards.csv": [
        "card_name",
        "effect_summary",
        "primary_impact",
        "constraints_or_costs",
        "synergies_or_notes",
    ],
    "spectral_cards.csv": [
        "card_name",
        "effect_summary",
        "primary_impact",
        "constraints_or_costs",
        "synergies_or_notes",
    ],
    "vouchers.csv": [
        "voucher_name",
        "effect_summary",
        "constraints",
        "notes",
    ],
    "booster_packs.csv": [
        "pack_name",
        "pack_type",
        "offered_items_summary",
        "shop_availability_notes",
        "notes",
    ],
}


HAND_ORDER_PATTERNS = [
    r"\bleft card\b",
    r"\bright card\b",
    r"\bleftmost\b",
    r"\brightmost\b",
    r"\bfirst scoring\b",
    r"\blast scoring\b",
    r"\bfirst played\b",
    r"\blast played\b",
    r"\bdrag to rearrange\b",
    r"\brearrange\b",
    r"\bhand order\b",
]

JOKER_ORDER_PATTERNS = [
    r"\bjoker slot\b",
    r"\bjoker slots\b",
    r"\bleftmost joker\b",
    r"\brightmost joker\b",
    r"\badjacent joker\b",
    r"\bto the left\b",
    r"\bto the right\b",
]

POSITION_TARGET_PATTERNS = [
    r"\bselect 2 cards\b",
    r"\bselected card\b",
    r"\bselected joker\b",
    r"\bchoose .* card\b",
    r"\bchoose .* joker\b",
    r"\bslot\b",
]


def _compile(patterns: list[str]) -> list[re.Pattern[str]]:
    return [re.compile(p, re.IGNORECASE) for p in patterns]


HAND_RE = _compile(HAND_ORDER_PATTERNS)
JOKER_RE = _compile(JOKER_ORDER_PATTERNS)
TARGET_RE = _compile(POSITION_TARGET_PATTERNS)


@dataclass
class Hit:
    source_file: str
    entity_name: str
    category: str
    text: str
    matched_patterns: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_file": self.source_file,
            "entity_name": self.entity_name,
            "category": self.category,
            "text": self.text,
            "matched_patterns": self.matched_patterns,
        }


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        return [dict(row) for row in reader]


def _entity_name(row: dict[str, str]) -> str:
    for key in ("joker_name", "card_name", "voucher_name", "pack_name"):
        val = str(row.get(key) or "").strip()
        if val:
            return val
    return "unknown"


def _collect_text(row: dict[str, str], fields: list[str]) -> str:
    parts: list[str] = []
    for field in fields:
        val = str(row.get(field) or "").strip()
        if val:
            parts.append(val)
    return " | ".join(parts)


def _matched(text: str, patterns: list[re.Pattern[str]]) -> list[str]:
    out: list[str] = []
    for p in patterns:
        if p.search(text):
            out.append(p.pattern)
    return out


def _wiki_probe(urls: list[str], timeout_sec: float = 8.0) -> dict[str, Any]:
    probe: dict[str, Any] = {"enabled": bool(urls), "pages": []}
    if not urls:
        return probe
    if requests is None:
        probe["error"] = "requests_unavailable"
        return probe

    for url in urls:
        page: dict[str, Any] = {"url": url}
        try:
            resp = requests.get(url, timeout=timeout_sec)
            page["status_code"] = int(resp.status_code)
            text = resp.text if resp.status_code == 200 else ""
            page["hand_keyword_hits"] = len(_matched(text, HAND_RE))
            page["joker_keyword_hits"] = len(_matched(text, JOKER_RE))
            page["target_keyword_hits"] = len(_matched(text, TARGET_RE))
        except Exception as exc:
            page["error"] = str(exc)
        probe["pages"].append(page)
    return probe


def derive_position_sensitive_effects(mechanics_root: Path, wiki_urls: list[str] | None = None) -> dict[str, Any]:
    hits: list[Hit] = []
    scanned_rows = 0

    for file_name, fields in TEXT_FIELDS_BY_FILE.items():
        src = mechanics_root / file_name
        if not src.exists():
            continue
        rows = _read_csv(src)
        scanned_rows += len(rows)
        for row in rows:
            text = _collect_text(row, fields)
            if not text:
                continue
            hand_patterns = _matched(text, HAND_RE)
            joker_patterns = _matched(text, JOKER_RE)
            target_patterns = _matched(text, TARGET_RE)
            entity = _entity_name(row)

            if hand_patterns:
                hits.append(
                    Hit(
                        source_file=file_name,
                        entity_name=entity,
                        category="hand_order_sensitive",
                        text=text,
                        matched_patterns=hand_patterns,
                    )
                )
            if joker_patterns:
                hits.append(
                    Hit(
                        source_file=file_name,
                        entity_name=entity,
                        category="joker_slot_sensitive",
                        text=text,
                        matched_patterns=joker_patterns,
                    )
                )
            if target_patterns and (file_name in {"tarot_cards.csv", "spectral_cards.csv", "jokers.csv"}):
                hits.append(
                    Hit(
                        source_file=file_name,
                        entity_name=entity,
                        category="position_targeted_effect",
                        text=text,
                        matched_patterns=target_patterns,
                    )
                )

    by_category: dict[str, list[dict[str, Any]]] = {
        "hand_order_sensitive": [],
        "joker_slot_sensitive": [],
        "position_targeted_effect": [],
    }
    for hit in hits:
        by_category.setdefault(hit.category, []).append(hit.to_dict())

    # De-duplicate by (category, file, entity).
    for cat, items in list(by_category.items()):
        seen: set[tuple[str, str, str]] = set()
        deduped: list[dict[str, Any]] = []
        for item in items:
            key = (str(item.get("category")), str(item.get("source_file")), str(item.get("entity_name")))
            if key in seen:
                continue
            seen.add(key)
            deduped.append(item)
        by_category[cat] = sorted(
            deduped,
            key=lambda d: (str(d.get("source_file") or ""), str(d.get("entity_name") or "")),
        )

    summary = {
        "scanned_rows": scanned_rows,
        "hand_order_sensitive_count": len(by_category.get("hand_order_sensitive") or []),
        "joker_slot_sensitive_count": len(by_category.get("joker_slot_sensitive") or []),
        "position_targeted_effect_count": len(by_category.get("position_targeted_effect") or []),
    }

    payload = {
        "schema": "p32_position_sensitive_effects_v1",
        "mechanics_root": str(mechanics_root),
        "summary": summary,
        "categories": by_category,
    }
    if wiki_urls:
        payload["wiki_probe"] = _wiki_probe(wiki_urls)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Derive position-sensitive effects from local Balatro mechanics tables.")
    parser.add_argument(
        "--mechanics-root",
        default="balatro_mechanics",
        help="Path to mechanics CSV root.",
    )
    parser.add_argument(
        "--out",
        default="balatro_mechanics/derived/position_sensitive_effects.json",
        help="Output json path.",
    )
    parser.add_argument(
        "--fetch-wiki",
        action="store_true",
        help="Also probe a few wiki pages for keyword coverage signals (non-blocking).",
    )
    parser.add_argument(
        "--wiki-urls",
        default="https://balatrogame.fandom.com/wiki/Jokers,https://balatrogame.fandom.com/wiki/Tarot_Cards,https://balatrogame.fandom.com/wiki/Spectral_Cards",
        help="Comma-separated wiki URLs for optional probing.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    mechanics_root = Path(args.mechanics_root).resolve()
    out_path = Path(args.out).resolve()

    wiki_urls: list[str] = []
    if bool(args.fetch_wiki):
        wiki_urls = [u.strip() for u in str(args.wiki_urls or "").split(",") if u.strip()]
    payload = derive_position_sensitive_effects(mechanics_root=mechanics_root, wiki_urls=wiki_urls)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload.get("summary") or {}, ensure_ascii=False, indent=2))
    print(f"wrote: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
