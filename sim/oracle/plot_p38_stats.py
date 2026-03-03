from __future__ import annotations

if __package__ is None or __package__ == "":
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

import argparse
import json
from pathlib import Path
from typing import Any


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot P38 long-episode score/round distributions.")
    parser.add_argument("--fixtures-dir", required=True)
    parser.add_argument("--out-dir", default="docs/artifacts/p38/plots")
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any] | None:
    try:
        obj = json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception:
        return None
    return obj if isinstance(obj, dict) else None


def _safe_float(v: Any) -> float | None:
    try:
        return float(v)
    except Exception:
        return None


def main() -> int:
    args = _parse_args()
    repo_root = Path(__file__).resolve().parents[2]

    fixtures_dir = Path(args.fixtures_dir)
    if not fixtures_dir.is_absolute():
        fixtures_dir = (repo_root / fixtures_dir).resolve()
    if not fixtures_dir.exists():
        raise SystemExit(f"fixtures dir not found: {fixtures_dir}")

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = (repo_root / out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    episode_files = sorted(fixtures_dir.glob("episode_*.json"))
    episodes: list[dict[str, Any]] = []
    for path in episode_files:
        obj = _load_json(path)
        if obj is not None:
            episodes.append(obj)
    if not episodes:
        raise SystemExit(f"no episode_*.json found in {fixtures_dir}")

    oracle_scores: list[float] = []
    sim_scores: list[float] = []
    oracle_rounds: list[float] = []
    sim_rounds: list[float] = []

    for ep in episodes:
        oracle_metrics = ((ep.get("oracle") or {}).get("metrics") or {}) if isinstance(ep.get("oracle"), dict) else {}
        sim_metrics = ((ep.get("sim") or {}).get("metrics") or {}) if isinstance(ep.get("sim"), dict) else {}

        oscore = _safe_float(oracle_metrics.get("total_score"))
        sscore = _safe_float(sim_metrics.get("total_score"))
        orounds = _safe_float(oracle_metrics.get("rounds_survived"))
        srounds = _safe_float(sim_metrics.get("rounds_survived"))

        if oscore is not None:
            oracle_scores.append(float(oscore))
        if sscore is not None:
            sim_scores.append(float(sscore))
        if orounds is not None:
            oracle_rounds.append(float(orounds))
        if srounds is not None:
            sim_rounds.append(float(srounds))

    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception as exc:
        note = out_dir / "plot_generation_skipped.txt"
        note.write_text(f"matplotlib unavailable: {exc}\n", encoding="utf-8")
        print(json.dumps({"status": "SKIPPED", "reason": "matplotlib_unavailable", "note": str(note)}, ensure_ascii=False))
        return 0

    score_png = out_dir / "score_distribution.png"
    rounds_png = out_dir / "rounds_distribution.png"

    plt.figure(figsize=(10, 6))
    if oracle_scores:
        plt.hist(oracle_scores, bins=min(20, max(5, len(oracle_scores))), alpha=0.55, label="oracle")
    if sim_scores:
        plt.hist(sim_scores, bins=min(20, max(5, len(sim_scores))), alpha=0.55, label="sim")
    plt.title("P38 Score Distribution (Oracle vs Sim)")
    plt.xlabel("total_score")
    plt.ylabel("count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(score_png, dpi=140)
    plt.close()

    plt.figure(figsize=(10, 6))
    if oracle_rounds:
        plt.hist(oracle_rounds, bins=min(20, max(5, len(oracle_rounds))), alpha=0.55, label="oracle")
    if sim_rounds:
        plt.hist(sim_rounds, bins=min(20, max(5, len(sim_rounds))), alpha=0.55, label="sim")
    plt.title("P38 Rounds Survived Distribution (Oracle vs Sim)")
    plt.xlabel("rounds_survived")
    plt.ylabel("count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(rounds_png, dpi=140)
    plt.close()

    print(
        json.dumps(
            {
                "status": "PASS",
                "score_distribution": str(score_png),
                "rounds_distribution": str(rounds_png),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
