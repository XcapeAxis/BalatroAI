from __future__ import annotations

if __package__ is None or __package__ == "":
    import sys
    from pathlib import Path as _Path

    sys.path.append(str(_Path(__file__).resolve().parent.parent.parent))

import argparse
import csv
import json
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_STAKES = [
    "white",
    "red",
    "green",
    "black",
    "blue",
    "purple",
    "orange",
    "gold",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract stake rule hints from local mechanics CSVs.")
    parser.add_argument("--mech-root", default="balatro_mechanics", help="Mechanics root directory.")
    parser.add_argument("--out-derived", default="balatro_mechanics/derived", help="Output derived directory.")
    return parser.parse_args()


def _iter_csv_paths(mech_root: Path) -> list[Path]:
    return sorted(p for p in mech_root.glob("*.csv") if p.is_file())


def _load_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as fp:
        reader = csv.DictReader(fp)
        fieldnames = [str(x) for x in (reader.fieldnames or [])]
        rows: list[dict[str, str]] = []
        for row in reader:
            if isinstance(row, dict):
                rows.append({str(k): str(v or "") for k, v in row.items()})
    return fieldnames, rows


def _contains_stake_token(text: str) -> bool:
    low = text.lower()
    if "stake" in low:
        return True
    for name in DEFAULT_STAKES:
        if re.search(rf"\b{name}\s+stake\b", low):
            return True
    return False


def _detect_modifier_kind(text: str) -> str:
    low = text.lower()
    if any(tok in low for tok in ("money", "$", "interest", "cost", "sell")):
        return "economy"
    if any(tok in low for tok in ("hand size", "hands", "discard", "resource")):
        return "resource"
    if any(tok in low for tok in ("chips", "mult", "score", "blind")):
        return "score"
    if any(tok in low for tok in ("shop", "pack", "voucher", "tag", "boss")):
        return "shop_meta"
    return "unknown"


def _extract_stake_name(text: str) -> str | None:
    low = text.lower()
    m = re.search(r"\b(white|red|green|black|blue|purple|orange|gold)\s+stake\b", low)
    if m:
        return m.group(1)
    if "stake" in low:
        return "unknown"
    return None


def _confidence(entry_count: int, unknown_count: int) -> str:
    if entry_count == 0:
        return "low"
    ratio_unknown = unknown_count / max(1, entry_count)
    if ratio_unknown <= 0.25:
        return "high"
    if ratio_unknown <= 0.6:
        return "med"
    return "low"


def extract_stake_rules(mech_root: Path) -> dict[str, Any]:
    stakes: dict[str, dict[str, Any]] = {
        stake: {
            "stake_name": stake,
            "modifiers": [],
            "source_fields": [],
            "confidence": "low",
            "missing": False,
        }
        for stake in DEFAULT_STAKES
    }
    stakes["unknown"] = {
        "stake_name": "unknown",
        "modifiers": [],
        "source_fields": [],
        "confidence": "low",
        "missing": False,
    }

    found_any = False
    csv_paths = _iter_csv_paths(mech_root)
    for csv_path in csv_paths:
        fields, rows = _load_rows(csv_path)
        if not fields:
            continue

        for row in rows:
            blob_parts: list[str] = []
            for key in fields:
                value = str(row.get(key) or "")
                if value:
                    blob_parts.append(f"{key}:{value}")
            blob = " | ".join(blob_parts)
            if not blob or not _contains_stake_token(blob):
                continue

            found_any = True
            stake_name = _extract_stake_name(blob) or "unknown"
            if stake_name not in stakes:
                stake_name = "unknown"

            kind = _detect_modifier_kind(blob)
            entry = {
                "kind": kind,
                "description": blob[:400],
                "source_csv": csv_path.name,
            }
            stakes[stake_name]["modifiers"].append(entry)
            for f in fields:
                if "stake" in f.lower() or "difficulty" in f.lower() or "notes" in f.lower():
                    stakes[stake_name]["source_fields"].append(f"{csv_path.name}:{f}")

    if not found_any:
        for stake_name in DEFAULT_STAKES:
            stakes[stake_name]["missing"] = True
            stakes[stake_name]["modifiers"] = [{"kind": "unknown", "description": "No stake rules found in local mechanics CSVs.", "source_csv": ""}]
            stakes[stake_name]["source_fields"] = []
            stakes[stake_name]["confidence"] = "low"
    else:
        for stake_name, payload in stakes.items():
            modifiers = payload.get("modifiers") if isinstance(payload.get("modifiers"), list) else []
            unknown_count = sum(1 for x in modifiers if str(x.get("kind") or "") == "unknown")
            payload["confidence"] = _confidence(len(modifiers), unknown_count)
            payload["source_fields"] = sorted(set(str(x) for x in payload.get("source_fields") or []))
            if not modifiers and stake_name != "unknown":
                payload["missing"] = True
                payload["modifiers"] = [{"kind": "unknown", "description": "No direct rule match found; treated as degraded.", "source_csv": ""}]
                payload["confidence"] = "low"

    by_kind: dict[str, int] = defaultdict(int)
    for stake in stakes.values():
        for mod in stake.get("modifiers") or []:
            by_kind[str(mod.get("kind") or "unknown")] += 1

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "mech_root": str(mech_root),
        "stakes": stakes,
        "summary": {
            "stakes_total": len(DEFAULT_STAKES),
            "stakes_with_missing_rules": sum(1 for s in DEFAULT_STAKES if bool(stakes[s].get("missing"))),
            "modifier_kind_counts": dict(sorted(by_kind.items())),
        },
    }


def main() -> int:
    args = parse_args()
    mech_root = Path(args.mech_root)
    if not mech_root.is_absolute():
        mech_root = (Path(__file__).resolve().parent.parent.parent / mech_root).resolve()
    if not mech_root.exists():
        print(f"ERROR: mechanics root not found: {mech_root}")
        return 2

    out_dir = Path(args.out_derived)
    if not out_dir.is_absolute():
        out_dir = (Path(__file__).resolve().parent.parent.parent / out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    payload = extract_stake_rules(mech_root)
    out_path = out_dir / "stakes.json"
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"wrote: {out_path}")
    print(
        "summary: stakes_total={0} missing={1}".format(
            payload["summary"]["stakes_total"],
            payload["summary"]["stakes_with_missing_rules"],
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
