from __future__ import annotations

if __package__ is None or __package__ == "":
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

import argparse
import json
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:  # pragma: no cover - optional in some local environments
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None

from trainer.selfsup.data import (
    build_samples_from_trajectories,
    load_trajectories_from_sources,
    parse_source_tokens,
)


def _require_torch():
    try:
        import torch
        from torch.utils.data import DataLoader, Dataset
    except Exception as exc:  # pragma: no cover - runtime dependency
        raise RuntimeError("PyTorch is required for trainer.experiments.ssl_dataset") from exc
    return torch, Dataset, DataLoader


@dataclass(frozen=True)
class SSLPairSample:
    obs: list[float]
    next_obs: list[float]
    reward_bucket: int
    delta_chips: float
    meta: dict[str, Any]


def _read_yaml_or_json(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in {".yaml", ".yml"}:
        if yaml is not None:
            payload = yaml.safe_load(text)
        else:
            try:
                payload = json.loads(text)
            except Exception:
                sidecar = path.with_suffix(".json")
                if sidecar.exists():
                    payload = json.loads(sidecar.read_text(encoding="utf-8"))
                else:
                    raise
    else:
        payload = json.loads(text)
    if not isinstance(payload, dict):
        raise ValueError(f"config must be mapping: {path}")
    return payload


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as fp:
        for row in rows:
            fp.write(json.dumps(row, ensure_ascii=False) + "\n")


def _load_dataset_rows(path: Path, *, max_samples: int = 0) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fp:
        for line in fp:
            text = line.strip().lstrip("\ufeff")
            if not text:
                continue
            row = json.loads(text)
            if isinstance(row, dict):
                rows.append(row)
            if max_samples > 0 and len(rows) >= int(max_samples):
                break
    return rows


def _parse_sources_from_cfg(data_cfg: dict[str, Any]) -> list[str]:
    out: list[str] = []
    raw = data_cfg.get("sources")
    if not isinstance(raw, list):
        return out
    for item in raw:
        if isinstance(item, dict):
            kind = str(item.get("type") or item.get("kind") or "auto").strip().lower()
            path = str(item.get("path") or "").strip()
            if path:
                out.append(f"{kind}:{path}")
        else:
            token = str(item).strip()
            if token:
                out.append(token)
    return out


def _reward_bucket(delta_chips: float) -> int:
    if delta_chips > 1e-6:
        return 2
    if delta_chips < -1e-6:
        return 0
    return 1


def _row_state_vector(row: dict[str, Any]) -> list[float]:
    state = row.get("state") if isinstance(row.get("state"), dict) else {}
    aux = row.get("aux") if isinstance(row.get("aux"), dict) else {}
    base = state.get("vector") if isinstance(state.get("vector"), list) else []
    vec = [float(x) for x in base]
    vec.extend(
        [
            float(aux.get("score_delta_t") or 0.0) / 250.0,
            float(aux.get("reward_t") or 0.0) / 250.0,
        ]
    )
    return vec


def _row_next_state_vector(row: dict[str, Any], default_dim: int) -> list[float]:
    future = row.get("future") if isinstance(row.get("future"), dict) else {}
    raw = future.get("next_state_vector") if isinstance(future.get("next_state_vector"), list) else []
    vec = [float(x) for x in raw]
    if len(vec) < int(default_dim):
        vec.extend([0.0 for _ in range(int(default_dim) - len(vec))])
    elif len(vec) > int(default_dim):
        vec = vec[: int(default_dim)]
    return vec


def materialize_ssl_rows(
    *,
    repo_root: Path,
    data_cfg: dict[str, Any],
    max_samples_override: int | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    dataset_path_raw = str(data_cfg.get("dataset_path") or data_cfg.get("dataset_out") or "").strip()
    max_samples = int(
        max_samples_override if max_samples_override is not None else int(data_cfg.get("max_samples") or 0)
    )
    if dataset_path_raw:
        dataset_path = (
            (repo_root / dataset_path_raw).resolve()
            if not Path(dataset_path_raw).is_absolute()
            else Path(dataset_path_raw)
        )
        if dataset_path.exists():
            rows = _load_dataset_rows(dataset_path, max_samples=max_samples)
            return rows, {"source": "existing_dataset", "dataset_path": str(dataset_path)}

    source_tokens = _parse_sources_from_cfg(data_cfg)
    if not source_tokens:
        raise RuntimeError("ssl dataset requires data.dataset_path or data.sources")

    specs = parse_source_tokens(source_tokens)
    max_episodes = int(
        data_cfg.get("max_episodes")
        or data_cfg.get("max_trajectories_per_source")
        or 20
    )
    min_episode_length = int(data_cfg.get("min_episode_length") or 2)
    window_size = max(2, int(data_cfg.get("window_size") or 2))
    lookahead_k = max(1, window_size - 1)
    trajectories, source_stats = load_trajectories_from_sources(
        repo_root=repo_root,
        sources=specs,
        max_trajectories_per_source=max(1, max_episodes),
        require_steps=True,
    )
    filtered = [t for t in trajectories if len(t.steps) >= max(2, min_episode_length)]
    samples = build_samples_from_trajectories(
        filtered,
        lookahead_k=lookahead_k,
        max_samples=max_samples,
    )
    rows = [s.to_dict() for s in samples]

    if dataset_path_raw:
        dataset_path = (
            (repo_root / dataset_path_raw).resolve()
            if not Path(dataset_path_raw).is_absolute()
            else Path(dataset_path_raw)
        )
        _write_jsonl(dataset_path, rows)

    return rows, {
        "source": "materialized_from_traces",
        "dataset_path": dataset_path_raw,
        "source_stats": source_stats,
        "trajectory_count": len(trajectories),
        "trajectory_count_after_filter": len(filtered),
        "lookahead_k": lookahead_k,
    }


def build_ssl_pair_samples(
    rows: list[dict[str, Any]],
    *,
    max_samples: int = 0,
) -> tuple[list[SSLPairSample], dict[str, Any]]:
    pair_rows: list[SSLPairSample] = []
    dropped_rows = 0
    bucket_hist: Counter[int] = Counter()
    for row in rows:
        obs = _row_state_vector(row)
        if not obs:
            dropped_rows += 1
            continue
        future = row.get("future") if isinstance(row.get("future"), dict) else {}
        delta = float(future.get("delta_chips_k") or 0.0)
        bucket = _reward_bucket(delta)
        next_obs = _row_next_state_vector(row, len(obs))
        pair_rows.append(
            SSLPairSample(
                obs=obs,
                next_obs=next_obs,
                reward_bucket=bucket,
                delta_chips=delta,
                meta={
                    "step_idx": (
                        int((row.get("meta") or {}).get("step_idx"))
                        if isinstance(row.get("meta"), dict) and (row.get("meta") or {}).get("step_idx") is not None
                        else None
                    ),
                    "source": str((row.get("aux") or {}).get("source") or ""),
                },
            )
        )
        bucket_hist[bucket] += 1
        if max_samples > 0 and len(pair_rows) >= int(max_samples):
            break
    input_dim = len(pair_rows[0].obs) if pair_rows else 0
    return pair_rows, {
        "sample_count": len(pair_rows),
        "dropped_rows": dropped_rows,
        "input_dim": input_dim,
        "reward_bucket_hist": {str(k): int(v) for k, v in sorted(bucket_hist.items())},
    }


def split_pair_samples(
    rows: list[SSLPairSample],
    *,
    train_ratio: float,
    seed: int,
) -> tuple[list[SSLPairSample], list[SSLPairSample]]:
    if not rows:
        return [], []
    if len(rows) == 1:
        return list(rows), list(rows)
    ratio = max(0.5, min(0.95, float(train_ratio)))
    indices = list(range(len(rows)))
    random.Random(seed).shuffle(indices)
    cut = int(len(indices) * ratio)
    cut = max(1, min(cut, len(indices) - 1))
    train_rows = [rows[i] for i in indices[:cut]]
    val_rows = [rows[i] for i in indices[cut:]]
    return train_rows, val_rows


def build_ssl_dataloaders(
    *,
    config: dict[str, Any],
    seed: int,
    repo_root: Path | None = None,
    max_samples_override: int | None = None,
) -> dict[str, Any]:
    torch, Dataset, DataLoader = _require_torch()
    root = repo_root or Path(__file__).resolve().parents[2]
    data_cfg = config.get("data") if isinstance(config.get("data"), dict) else {}
    train_cfg = config.get("training") if isinstance(config.get("training"), dict) else {}
    train_ratio = float(
        data_cfg.get("train_ratio")
        or train_cfg.get("train_ratio")
        or (1.0 - float(train_cfg.get("val_ratio") or 0.2))
    )
    batch_size = int(train_cfg.get("batch_size") or data_cfg.get("batch_size") or 32)
    num_workers = int(data_cfg.get("num_workers") or 0)

    rows, materialize_meta = materialize_ssl_rows(
        repo_root=root,
        data_cfg=data_cfg,
        max_samples_override=max_samples_override,
    )
    max_samples = int(
        max_samples_override
        if max_samples_override is not None
        else int(data_cfg.get("max_samples") or 0)
    )
    pair_rows, pair_meta = build_ssl_pair_samples(rows, max_samples=max_samples)
    if not pair_rows:
        raise RuntimeError("ssl dataset builder produced zero pair samples")

    train_rows, val_rows = split_pair_samples(pair_rows, train_ratio=train_ratio, seed=seed)

    class _PairDataset(Dataset):
        def __init__(self, items: list[SSLPairSample]) -> None:
            self.items = items

        def __len__(self) -> int:
            return len(self.items)

        def __getitem__(self, idx: int) -> SSLPairSample:
            return self.items[idx]

    def _collate(batch: list[SSLPairSample]) -> dict[str, Any]:
        return {
            "obs": torch.tensor([item.obs for item in batch], dtype=torch.float32),
            "next_obs": torch.tensor([item.next_obs for item in batch], dtype=torch.float32),
            "reward_bucket": torch.tensor([item.reward_bucket for item in batch], dtype=torch.long),
            "delta_chips": torch.tensor([item.delta_chips for item in batch], dtype=torch.float32),
        }

    train_loader = DataLoader(
        _PairDataset(train_rows),
        batch_size=max(1, batch_size),
        shuffle=True,
        num_workers=max(0, num_workers),
        collate_fn=_collate,
    )
    val_loader = DataLoader(
        _PairDataset(val_rows),
        batch_size=max(1, batch_size),
        shuffle=False,
        num_workers=max(0, num_workers),
        collate_fn=_collate,
    )

    return {
        "train_loader": train_loader,
        "val_loader": val_loader,
        "input_dim": int(pair_meta.get("input_dim") or 0),
        "num_classes": 3,
        "sample_count": int(pair_meta.get("sample_count") or 0),
        "train_count": len(train_rows),
        "val_count": len(val_rows),
        "materialize_meta": materialize_meta,
        "dataset_meta": pair_meta,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="P37 SSL dataset dry-run builder.")
    p.add_argument("--config", default="configs/experiments/p37_ssl_pretrain.yaml")
    p.add_argument("--seed", type=int, default=3701)
    p.add_argument("--max-samples", type=int, default=0)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    cfg_path = (repo_root / str(args.config)).resolve() if not Path(args.config).is_absolute() else Path(args.config)
    cfg = _read_yaml_or_json(cfg_path)
    built = build_ssl_dataloaders(
        config=cfg,
        seed=int(args.seed),
        repo_root=repo_root,
        max_samples_override=(int(args.max_samples) if int(args.max_samples) > 0 else None),
    )
    train_loader = built["train_loader"]
    batch = next(iter(train_loader))
    print(
        json.dumps(
            {
                "status": "ok",
                "config": str(cfg_path),
                "sample_count": built.get("sample_count"),
                "train_count": built.get("train_count"),
                "val_count": built.get("val_count"),
                "input_dim": built.get("input_dim"),
                "batch_obs_shape": list(batch["obs"].shape),
                "batch_next_obs_shape": list(batch["next_obs"].shape),
                "batch_reward_shape": list(batch["reward_bucket"].shape),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
