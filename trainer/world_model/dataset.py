from __future__ import annotations

if __package__ is None or __package__ == "":
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

try:  # pragma: no cover - optional dependency
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None

from trainer.closed_loop.replay_manifest import build_seeds_payload, now_iso, now_stamp, write_json
from trainer.models.ssl_state_encoder import StateEncoder, StateEncoderConfig
from trainer.world_model.sample_builder import build_samples_from_source_config, summarize_dataset_samples
from trainer.world_model.schema import WorldModelSample, fit_vector


def _read_yaml_or_json(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8-sig")
    if path.suffix.lower() in {".yml", ".yaml"}:
        if yaml is not None:
            payload = yaml.safe_load(text)
        else:
            try:
                payload = json.loads(text)
            except Exception:
                sidecar = path.with_suffix(".json")
                if sidecar.exists():
                    payload = json.loads(sidecar.read_text(encoding="utf-8-sig"))
                else:
                    raise RuntimeError(f"PyYAML unavailable and no sidecar JSON for: {path}")
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


def _write_markdown(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def _resolve_repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _dataset_cfg(cfg: dict[str, Any]) -> dict[str, Any]:
    block = cfg.get("dataset") if isinstance(cfg.get("dataset"), dict) else {}
    return block if block else cfg


def _output_cfg(cfg: dict[str, Any]) -> dict[str, Any]:
    return cfg.get("output") if isinstance(cfg.get("output"), dict) else {}


def _require_torch():
    try:
        import torch
    except Exception as exc:  # pragma: no cover - runtime guarded
        raise RuntimeError("PyTorch is required for world model latent caching") from exc
    return torch


def _extract_state_dict(payload: Any) -> dict[str, Any]:
    if isinstance(payload, dict):
        for key in ("model_state_dict", "state_dict", "model"):
            if isinstance(payload.get(key), dict):
                return payload.get(key)  # type: ignore[return-value]
        if all(isinstance(key, str) for key in payload.keys()):
            return payload
    return {}


def _best_effort_load_encoder(encoder: StateEncoder, checkpoint_path: Path) -> str:
    torch = _require_torch()
    if not checkpoint_path.exists():
        return f"checkpoint_missing:{checkpoint_path}"
    try:
        payload = torch.load(str(checkpoint_path), map_location="cpu")
    except Exception as exc:
        return f"checkpoint_load_failed:{exc}"
    source_state = _extract_state_dict(payload)
    if not source_state:
        return "checkpoint_missing_state_dict"
    target_state = encoder.state_dict()
    patched: dict[str, Any] = {}
    for target_key in target_state.keys():
        if target_key in source_state:
            patched[target_key] = source_state[target_key]
            continue
        for source_key, source_value in source_state.items():
            if str(source_key).endswith(target_key):
                patched[target_key] = source_value
                break
    if not patched:
        return "checkpoint_no_matching_encoder_keys"
    encoder.load_state_dict(patched, strict=False)
    return f"checkpoint_loaded_partial:{len(patched)}"


def _cache_latents(
    samples: list[WorldModelSample],
    *,
    feature_dim: int,
    representation_cfg: dict[str, Any],
) -> tuple[list[WorldModelSample], dict[str, Any]]:
    if not bool(representation_cfg.get("cache_latents", False)):
        return samples, {"enabled": False, "status": "disabled"}

    torch = _require_torch()
    latent_dim = max(8, int(representation_cfg.get("latent_dim") or 32))
    seed = int(representation_cfg.get("seed") or 4501)
    batch_size = max(8, int(representation_cfg.get("batch_size") or 128))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    encoder = StateEncoder(
        StateEncoderConfig(
            input_dim=int(feature_dim),
            latent_dim=int(latent_dim),
            hidden_dim=max(32, int(representation_cfg.get("hidden_dim") or 96)),
            dropout=float(representation_cfg.get("dropout") or 0.0),
        )
    )
    checkpoint_status = "random_init"
    checkpoint_raw = str(representation_cfg.get("encoder_checkpoint") or "").strip()
    if checkpoint_raw:
        checkpoint_status = _best_effort_load_encoder(encoder, Path(checkpoint_raw))
    encoder.eval()

    with torch.no_grad():
        obs_t = torch.tensor([fit_vector(list(sample.obs_t), feature_dim) for sample in samples], dtype=torch.float32)
        obs_t1 = torch.tensor([fit_vector(list(sample.obs_t1), feature_dim) for sample in samples], dtype=torch.float32)
        latent_rows_t: list[list[float]] = []
        latent_rows_t1: list[list[float]] = []
        for start in range(0, obs_t.shape[0], batch_size):
            chunk_t = encoder(obs_t[start : start + batch_size])
            chunk_t1 = encoder(obs_t1[start : start + batch_size])
            latent_rows_t.extend(chunk_t.detach().cpu().tolist())
            latent_rows_t1.extend(chunk_t1.detach().cpu().tolist())

    out = [
        replace(sample, latent_t=list(latent_rows_t[idx]), latent_t1=list(latent_rows_t1[idx]), feature_mode="latent_cached")
        for idx, sample in enumerate(samples)
    ]
    return out, {
        "enabled": True,
        "status": "ok",
        "checkpoint_status": checkpoint_status,
        "latent_dim": latent_dim,
        "seed": seed,
    }


def _stats_markdown(
    *,
    run_id: str,
    manifest_path: Path,
    dataset_path: Path | None,
    summary: dict[str, Any],
    source_stats: list[dict[str, Any]],
    warnings: list[str],
    latent_cache: dict[str, Any],
) -> list[str]:
    lines = [
        f"# P45 World Model Dataset ({run_id})",
        "",
        f"- generated_at: {now_iso()}",
        f"- manifest_json: `{manifest_path}`",
        f"- dataset_jsonl: `{dataset_path}`" if dataset_path else "- dataset_jsonl: `dry_run_only`",
        f"- sample_count: {int(summary.get('sample_count') or 0)}",
        f"- seed_count: {int(summary.get('seed_count') or 0)}",
        f"- episode_count: {int(summary.get('episode_count') or 0)}",
        f"- invalid_ratio: {float(summary.get('invalid_ratio') or 0.0):.4f}",
        f"- latent_cache: {latent_cache.get('status')}",
        "",
        "## Source Stats",
        "",
        "| source_id | source_type | file_count | sample_count | train | val | seed_count | warnings |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in source_stats:
        lines.append(
            "| {source_id} | {source_type} | {file_count} | {sample_count} | {train_samples} | {val_samples} | {seed_count} | {warnings_count} |".format(
                source_id=row.get("source_id"),
                source_type=row.get("source_type"),
                file_count=int(row.get("file_count") or 0),
                sample_count=int(row.get("sample_count") or 0),
                train_samples=int(row.get("train_samples") or 0),
                val_samples=int(row.get("val_samples") or 0),
                seed_count=int(row.get("seed_count") or 0),
                warnings_count=int(row.get("warnings_count") or 0),
            )
        )
    lines.extend(["", "## Split Distribution", "", "| split | count | ratio |", "|---|---:|---:|"])
    for row in summary.get("split_distribution") if isinstance(summary.get("split_distribution"), list) else []:
        if not isinstance(row, dict):
            continue
        lines.append(
            "| {label} | {count} | {ratio:.3f} |".format(
                label=row.get("label"),
                count=int(row.get("count") or 0),
                ratio=float(row.get("ratio") or 0.0),
            )
        )
    lines.extend(["", "## Source Distribution", "", "| source_type | count | ratio |", "|---|---:|---:|"])
    for row in summary.get("source_distribution") if isinstance(summary.get("source_distribution"), list) else []:
        if not isinstance(row, dict):
            continue
        lines.append(
            "| {label} | {count} | {ratio:.3f} |".format(
                label=row.get("label"),
                count=int(row.get("count") or 0),
                ratio=float(row.get("ratio") or 0.0),
            )
        )
    if warnings:
        lines.extend(["", "## Warnings"])
        lines.extend([f"- {warning}" for warning in warnings])
    return lines


def build_world_model_dataset(
    *,
    config_path: str | Path,
    out_dir: str | Path | None = None,
    run_id: str = "",
    quick: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    repo_root = _resolve_repo_root()
    cfg_path = Path(config_path)
    if not cfg_path.is_absolute():
        cfg_path = (repo_root / cfg_path).resolve()
    cfg = _read_yaml_or_json(cfg_path)
    dataset_cfg = _dataset_cfg(cfg)
    output_cfg = _output_cfg(dataset_cfg)

    chosen_run_id = str(run_id or output_cfg.get("run_id") or now_stamp())
    feature_dim = max(8, int(dataset_cfg.get("feature_dim") or 48))
    action_vocab_size = max(32, int(dataset_cfg.get("action_vocab_size") or 4096))
    train_ratio = float(((dataset_cfg.get("split") or {}) if isinstance(dataset_cfg.get("split"), dict) else {}).get("train") or 0.85)
    representation_cfg = dataset_cfg.get("representation") if isinstance(dataset_cfg.get("representation"), dict) else {}
    sources_cfg = dataset_cfg.get("sources") if isinstance(dataset_cfg.get("sources"), list) else []
    if not sources_cfg:
        sources_cfg = [
            {"id": "p42_rollout", "type": "rl_rollout"},
            {"id": "p36_dataset", "type": "selfsup_dataset"},
            {"id": "p41_replay_mix", "type": "replay_manifest"},
        ]

    artifacts_root = str(output_cfg.get("artifacts_root") or "docs/artifacts/p45/wm_dataset")
    run_dir = (repo_root / artifacts_root / chosen_run_id).resolve() if out_dir is None else (Path(out_dir) if Path(out_dir).is_absolute() else (repo_root / out_dir).resolve())
    run_dir.mkdir(parents=True, exist_ok=True)

    all_samples: list[WorldModelSample] = []
    source_stats: list[dict[str, Any]] = []
    warnings: list[str] = []
    for source_cfg in sources_cfg:
        if not isinstance(source_cfg, dict):
            continue
        built, stats, build_warnings = build_samples_from_source_config(
            repo_root=repo_root,
            source_cfg=source_cfg,
            feature_dim=feature_dim,
            action_vocab_size=action_vocab_size,
            train_ratio=train_ratio,
            quick=quick,
        )
        all_samples.extend(built)
        source_stats.append(stats)
        warnings.extend(build_warnings)

    all_samples = sorted(
        all_samples,
        key=lambda sample: (sample.source_type, sample.episode_id, int(sample.step_id)),
    )
    latent_cache = {"enabled": False, "status": "disabled"}
    if all_samples and not dry_run:
        try:
            all_samples, latent_cache = _cache_latents(
                all_samples,
                feature_dim=feature_dim,
                representation_cfg=representation_cfg,
            )
        except Exception as exc:
            warnings.append(f"latent_cache_failed:{exc}")
            latent_cache = {"enabled": bool(representation_cfg.get("cache_latents", False)), "status": f"failed:{exc}"}

    summary = summarize_dataset_samples(all_samples)
    dataset_path = run_dir / "dataset.jsonl" if not dry_run else None
    if dataset_path is not None:
        _write_jsonl(dataset_path, [sample.to_dict() for sample in all_samples])

    discovered_seeds = sorted({sample.seed for sample in all_samples if sample.seed})
    seeds_payload = build_seeds_payload(discovered_seeds, seed_policy_version="p45.world_model.dataset")
    seeds_payload["metadata"] = {
        "source_seed_count": len(discovered_seeds),
        "source_ids": sorted({sample.source_id for sample in all_samples}),
    }

    stats_payload = {
        "schema": "p45_world_model_dataset_stats_v1",
        "generated_at": now_iso(),
        "run_id": chosen_run_id,
        "config_path": str(cfg_path),
        "feature_dim": feature_dim,
        "action_vocab_size": action_vocab_size,
        "summary": summary,
        "source_stats": source_stats,
        "latent_cache": latent_cache,
        "warnings": warnings,
    }
    manifest_payload = {
        "schema": "p45_world_model_dataset_manifest_v1",
        "generated_at": now_iso(),
        "run_id": chosen_run_id,
        "status": "ok" if all_samples else "stub",
        "config_path": str(cfg_path),
        "run_dir": str(run_dir),
        "dataset_jsonl": str(dataset_path) if dataset_path is not None else "",
        "dataset_stats_json": str(run_dir / "dataset_stats.json"),
        "dataset_stats_md": str(run_dir / "dataset_stats.md"),
        "seeds_used_json": str(run_dir / "seeds_used.json"),
        "feature_dim": feature_dim,
        "action_vocab_size": action_vocab_size,
        "sample_count": int(summary.get("sample_count") or 0),
        "train_samples": int(next((row.get("count") for row in summary.get("split_distribution", []) if isinstance(row, dict) and row.get("label") == "train"), 0) or 0),
        "val_samples": int(next((row.get("count") for row in summary.get("split_distribution", []) if isinstance(row, dict) and row.get("label") == "val"), 0) or 0),
        "quick": bool(quick),
        "dry_run": bool(dry_run),
        "representation": representation_cfg,
        "latent_cache": latent_cache,
        "sources": source_stats,
        "warnings": warnings,
    }

    write_json(run_dir / "dataset_manifest.json", manifest_payload)
    write_json(run_dir / "dataset_stats.json", stats_payload)
    write_json(run_dir / "seeds_used.json", seeds_payload)
    _write_markdown(
        run_dir / "dataset_stats.md",
        _stats_markdown(
            run_id=chosen_run_id,
            manifest_path=run_dir / "dataset_manifest.json",
            dataset_path=dataset_path,
            summary=summary,
            source_stats=source_stats,
            warnings=warnings,
            latent_cache=latent_cache,
        ),
    )

    return {
        "status": "ok" if all_samples else "stub",
        "run_id": chosen_run_id,
        "run_dir": str(run_dir),
        "dataset_manifest_json": str(run_dir / "dataset_manifest.json"),
        "dataset_stats_json": str(run_dir / "dataset_stats.json"),
        "dataset_stats_md": str(run_dir / "dataset_stats.md"),
        "dataset_jsonl": str(dataset_path) if dataset_path is not None else "",
        "seeds_used_json": str(run_dir / "seeds_used.json"),
        "sample_count": int(summary.get("sample_count") or 0),
        "invalid_ratio": float(summary.get("invalid_ratio") or 0.0),
        "source_stats": source_stats,
        "latent_cache": latent_cache,
        "warnings": warnings,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a P45 world model dataset from replay and rollout artifacts.")
    parser.add_argument("--config", default="configs/experiments/p45_world_model_smoke.yaml")
    parser.add_argument("--out-dir", default="", help="Optional explicit output directory")
    parser.add_argument("--run-id", default="", help="Optional explicit run_id")
    parser.add_argument("--quick", action="store_true", help="Use smaller source caps for smoke runs")
    parser.add_argument("--dry-run", action="store_true", help="Resolve sources and stats without writing dataset.jsonl")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = build_world_model_dataset(
        config_path=args.config,
        out_dir=(args.out_dir if str(args.out_dir).strip() else None),
        run_id=str(args.run_id or ""),
        quick=bool(args.quick),
        dry_run=bool(args.dry_run),
    )
    print(json.dumps(summary, ensure_ascii=False))
    return 0 if str(summary.get("status")) in {"ok", "stub"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
