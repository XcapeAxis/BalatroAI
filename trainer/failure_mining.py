from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    out: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except Exception:
            continue
    return out


def _bucket(row: dict[str, Any]) -> str:
    reason = str(row.get("failure_reason") or "").lower()
    phase = str(row.get("failure_phase") or "").upper()
    boss = str(row.get("boss_blind_id") or "")
    if reason in {"", "none"} and str(row.get("result") or "").lower() == "win":
        return "win"
    if "blind" in reason or "shortfall" in reason:
        return "blind_score_shortfall"
    if "resource" in reason or "hands_left" in reason or "discards_left" in reason:
        return "resource_exhausted"
    if "econ" in reason or "money" in reason:
        return "econ_collapse"
    if "shop" in reason or phase in {"SHOP", "PACK_CHOICE"}:
        return "shop_misplay"
    if boss and boss.lower() not in {"", "none"}:
        return "boss_blind_specific"
    if "stall" in reason or phase in {"MENU", "UNKNOWN"}:
        return "phase_stall"
    return "other"


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="P17 failure bucket mining from episode logs.")
    p.add_argument("--episode-logs", required=True, help="Path to eval_long_horizon episode logs jsonl")
    p.add_argument("--baseline-episode-logs", default="", help="Optional baseline logs for delta comparison")
    p.add_argument("--out-dir", required=True)
    return p


def _build_bucket_payload(rows: list[dict[str, Any]], source: str) -> dict[str, Any]:
    bucket_counter: Counter[str] = Counter()
    phase_counter: Counter[str] = Counter()
    boss_counter: Counter[str] = Counter()
    bucket_examples: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for row in rows:
        b = _bucket(row)
        bucket_counter[b] += 1
        phase_counter[str(row.get("failure_phase") or "").upper()] += 1
        boss = str(row.get("boss_blind_id") or "")
        if boss:
            boss_counter[boss] += 1
        if len(bucket_examples[b]) < 10:
            bucket_examples[b].append(
                {
                    "episode_id": row.get("episode_id"),
                    "seed": row.get("seed"),
                    "failure_reason": row.get("failure_reason"),
                    "phase": row.get("failure_phase"),
                    "final_ante": row.get("final_ante"),
                }
            )

    return {
        "schema": "p17_failure_buckets_v1",
        "generated_at": _now_iso(),
        "source": source,
        "counts": dict(sorted(bucket_counter.items())),
        "phase_counts": dict(sorted(phase_counter.items())),
        "boss_counts": dict(sorted(boss_counter.items())),
        "examples": bucket_examples,
        "total": len(rows),
    }


def _pct_map(payload: dict[str, Any]) -> dict[str, float]:
    total = max(1.0, float(payload.get("total") or 0.0))
    c = payload.get("counts") if isinstance(payload.get("counts"), dict) else {}
    out: dict[str, float] = {}
    for k, v in c.items():
        try:
            out[str(k)] = (float(v) / total) * 100.0
        except Exception:
            continue
    return out


def main() -> int:
    args = _build_parser().parse_args()
    rows = _read_jsonl(Path(args.episode_logs))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    buckets = _build_bucket_payload(rows, source=str(Path(args.episode_logs)))
    (out_dir / "failure_buckets_latest.json").write_text(json.dumps(buckets, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (out_dir / "failure_buckets_challenger.json").write_text(json.dumps(buckets, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    md_lines = [
        "# P17 Failure Buckets",
        "",
        f"- source: {args.episode_logs}",
        f"- total: {len(rows)}",
        "",
        "## Buckets",
    ] + [f"- {k}: {v}" for k, v in sorted((buckets.get("counts") or {}).items())]

    if args.baseline_episode_logs:
        base_rows = _read_jsonl(Path(args.baseline_episode_logs))
        base_payload = _build_bucket_payload(base_rows, source=str(Path(args.baseline_episode_logs)))
        (out_dir / "failure_buckets_champion.json").write_text(json.dumps(base_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        b_pct = _pct_map(base_payload)
        c_pct = _pct_map(buckets)
        all_keys = sorted(set(b_pct.keys()) | set(c_pct.keys()))
        delta = {k: float(c_pct.get(k, 0.0) - b_pct.get(k, 0.0)) for k in all_keys}
        (out_dir / "failure_delta_summary.json").write_text(
            json.dumps(
                {
                    "schema": "p17_failure_delta_v1",
                    "generated_at": _now_iso(),
                    "baseline": str(Path(args.baseline_episode_logs)),
                    "candidate": str(Path(args.episode_logs)),
                    "delta_pct_points": delta,
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        md_lines += ["", "## Delta vs Baseline (pct points)"] + [f"- {k}: {delta[k]:.3f}" for k in all_keys]

    (out_dir / "failure_delta_summary.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": "ok", "out_dir": str(out_dir), "total": len(rows)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
