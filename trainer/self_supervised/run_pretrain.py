from __future__ import annotations

if __package__ is None or __package__ == "":
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

import argparse
import json
import math
import random
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:  # pragma: no cover
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None

from trainer.self_supervised.datasets import (
    build_dataset_rows,
    collect_trajectories,
    write_dataset_jsonl,
    write_dataset_stats,
)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def _read_yaml_or_json(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    suffix = path.suffix.lower()
    if suffix in {".yaml", ".yml"}:
        if yaml is not None:
            payload = yaml.safe_load(text)
        else:
            try:
                payload = json.loads(text)
            except Exception:
                sidecar = path.with_suffix(".json")
                if not sidecar.exists():
                    raise RuntimeError(f"PyYAML unavailable and no JSON sidecar found for {path}")
                payload = json.loads(sidecar.read_text(encoding="utf-8"))
    else:
        payload = json.loads(text)
    if not isinstance(payload, dict):
        raise ValueError(f"config must be mapping: {path}")
    return payload


def run_p32_pretrain_stub(
    *,
    config_path: str | Path,
    out_dir: str | Path | None = None,
    seed_override: int | None = None,
    max_samples_override: int | None = None,
    quiet: bool = False,
) -> dict[str, Any]:
    repo_root = Path(__file__).resolve().parents[2]
    cfg_path = (repo_root / str(config_path)).resolve() if not Path(config_path).is_absolute() else Path(config_path)
    cfg = _read_yaml_or_json(cfg_path)

    data_cfg = cfg.get("data") if isinstance(cfg.get("data"), dict) else {}
    task_cfg = cfg.get("task") if isinstance(cfg.get("task"), dict) else {}
    train_cfg = cfg.get("training") if isinstance(cfg.get("training"), dict) else {}
    out_cfg = cfg.get("output") if isinstance(cfg.get("output"), dict) else {}

    seed = int(seed_override if seed_override is not None else int(train_cfg.get("seed") or 3201))
    rng = random.Random(seed)

    thresholds = tuple(float(x) for x in (task_cfg.get("bucket_thresholds") or [0.0, 120.0])[:2])
    if len(thresholds) != 2:
        thresholds = (0.0, 120.0)
    horizon_steps = int(task_cfg.get("horizon_steps") or 3)

    max_samples = int(max_samples_override if max_samples_override is not None else int(data_cfg.get("max_samples") or 0))
    trajectories, source_summaries = collect_trajectories(repo_root=repo_root, data_cfg=data_cfg)
    rows = build_dataset_rows(
        trajectories,
        bucket_thresholds=(float(thresholds[0]), float(thresholds[1])),
        horizon_steps=horizon_steps,
        max_samples=max_samples,
    )
    if not rows:
        raise RuntimeError("p32 self-supervised stub dataset is empty")

    dataset_out = (repo_root / str(data_cfg.get("dataset_out") or "trainer_data/p32/selfsup_stub_dataset.jsonl")).resolve()
    stats_out = (repo_root / str(data_cfg.get("stats_out") or "docs/artifacts/p32/selfsup_stub_dataset_stats.json")).resolve()
    write_dataset_jsonl(dataset_out, rows)
    dataset_stats = write_dataset_stats(
        out_path=stats_out,
        rows=rows,
        source_summaries=source_summaries,
        dataset_path=dataset_out,
        horizon_steps=horizon_steps,
        bucket_thresholds=(float(thresholds[0]), float(thresholds[1])),
    )

    if out_dir is not None:
        run_dir = Path(out_dir).resolve()
    else:
        run_root = str(out_cfg.get("run_root") or "trainer_runs/p32_selfsup_stub")
        run_dir = (repo_root / run_root / _now_stamp()).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    progress_path = run_dir / "progress.jsonl"

    labels = [int(row.target_next_score_bucket) for row in rows]
    label_counter = Counter(labels)
    major_label, major_count = label_counter.most_common(1)[0]
    major_acc = float(major_count / max(1, len(labels)))
    val_loss = float(-math.log(max(1e-9, major_acc)))

    indices = list(range(len(rows)))
    rng.shuffle(indices)
    sample_preview = [rows[i].to_dict() for i in indices[: min(5, len(indices))]]

    telemetry_events = [
        {
            "schema": "p32_selfsup_progress_v1",
            "ts": _now_iso(),
            "phase": "dataset",
            "status": "ok",
            "seed": seed,
            "num_samples": len(rows),
            "source_count": len(source_summaries),
        },
        {
            "schema": "p32_selfsup_progress_v1",
            "ts": _now_iso(),
            "phase": "stub_train",
            "status": "ok",
            "seed": seed,
            "majority_label": int(major_label),
            "majority_acc": major_acc,
            "val_loss": val_loss,
        },
    ]
    with progress_path.open("w", encoding="utf-8", newline="\n") as fp:
        for event in telemetry_events:
            fp.write(json.dumps(event, ensure_ascii=False) + "\n")

    summary = {
        "schema": "p32_selfsup_stub_summary_v1",
        "status": "ok",
        "generated_at": _now_iso(),
        "config_path": str(cfg_path),
        "run_dir": str(run_dir),
        "dataset_path": str(dataset_out),
        "dataset_stats_path": str(stats_out),
        "seed": seed,
        "task_name": str(task_cfg.get("name") or "pretrain_repr_stub"),
        "dataset_stats": dataset_stats,
        "train_metrics": {
            "majority_label": int(major_label),
            "majority_acc": major_acc,
        },
        "final_metrics": {
            "val_loss": val_loss,
            "val_acc": major_acc,
            "val_count": len(rows),
        },
        "sample_preview": sample_preview,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    summary_out = str(out_cfg.get("summary_out") or "").strip()
    if summary_out:
        summary_out_path = (repo_root / summary_out).resolve()
        summary_out_path.parent.mkdir(parents=True, exist_ok=True)
        summary_out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    if not quiet:
        print(
            "[p32-selfsup-stub] samples={samples} val_acc={acc:.4f} val_loss={loss:.4f} run_dir={run_dir}".format(
                samples=len(rows),
                acc=major_acc,
                loss=val_loss,
                run_dir=run_dir,
            )
        )
    return summary


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run P32 self-supervised pretrain stub.")
    p.add_argument("--config", default="configs/experiments/p32_self_supervised.yaml")
    p.add_argument("--out-dir", default="")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-samples", type=int, default=0)
    p.add_argument("--quiet", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    summary = run_p32_pretrain_stub(
        config_path=args.config,
        out_dir=(args.out_dir if args.out_dir else None),
        seed_override=(args.seed if args.seed > 0 else None),
        max_samples_override=(args.max_samples if args.max_samples > 0 else None),
        quiet=bool(args.quiet),
    )
    print(json.dumps({"status": summary.get("status"), "run_dir": summary.get("run_dir"), "summary_schema": summary.get("schema")}))
    return 0 if str(summary.get("status") or "") == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
