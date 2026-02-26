"""Export a deployable model package (RC or stable) with weights, config, metrics, and checksums."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def _build_checksums(root: Path) -> dict[str, Any]:
    entries: dict[str, Any] = {}
    for p in sorted(root.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(root).as_posix()
        if rel == "checksums.json":
            continue
        entries[rel] = {"size_bytes": p.stat().st_size, "sha256": _sha256(p)}
    return entries


def _collect_metrics(project_root: Path, strategy: str) -> dict[str, Any]:
    """Try to find existing eval summaries for inclusion."""
    metrics: dict[str, Any] = {}
    for n in [100, 500, 1000, 2000]:
        candidates = list(project_root.glob(f"docs/artifacts/p*/*/ablation_{n}/eval_gold_{strategy}.json"))
        candidates += list(project_root.glob(f"docs/artifacts/p*/*/ablation_{n}/eval_gold_champion.json"))
        candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        if candidates:
            try:
                metrics[f"eval_{n}_seeds"] = json.loads(candidates[0].read_text(encoding="utf-8"))
            except Exception:
                pass
    return metrics


def export_package(
    *,
    model_path: str,
    strategy: str,
    risk_config: str | None,
    out_dir: str,
    package_id: str | None = None,
    model_id: str | None = None,
    project_root: Path | None = None,
) -> Path:
    proj = project_root or Path.cwd()
    model_p = Path(model_path)
    if not model_p.exists():
        raise FileNotFoundError(f"model not found: {model_p}")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    pkg_id = package_id or f"rc_{strategy}_{stamp}"
    m_id = model_id or f"{strategy}_{stamp}"
    pkg_dir = Path(out_dir)
    pkg_dir.mkdir(parents=True, exist_ok=True)

    # model/
    model_dir = pkg_dir / "model"
    model_dir.mkdir(exist_ok=True)
    if model_p.is_file():
        shutil.copy2(model_p, model_dir / model_p.name)
    manifest = {
        "source_path": str(model_p.resolve()),
        "copied_name": model_p.name,
        "size_bytes": model_p.stat().st_size if model_p.is_file() else 0,
    }
    (model_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    # config/
    cfg_dir = pkg_dir / "config"
    cfg_dir.mkdir(exist_ok=True)
    inference_cfg: dict[str, Any] = {
        "policy_type": strategy,
        "search_budget_ms": 10.0,
    }
    if risk_config and Path(risk_config).exists():
        try:
            import yaml  # type: ignore
            rc_data = yaml.safe_load(Path(risk_config).read_text(encoding="utf-8"))
            inference_cfg["risk_thresholds"] = rc_data
        except Exception:
            inference_cfg["risk_config_path"] = risk_config
        shutil.copy2(risk_config, cfg_dir / Path(risk_config).name)
    (cfg_dir / "inference_config.json").write_text(json.dumps(inference_cfg, indent=2) + "\n", encoding="utf-8")

    # metrics/
    metrics_dir = pkg_dir / "metrics"
    metrics_dir.mkdir(exist_ok=True)
    collected = _collect_metrics(proj, strategy)
    (metrics_dir / "eval_summary.json").write_text(
        json.dumps({"collected_metrics": collected, "generated_at": _now_iso()}, indent=2) + "\n", encoding="utf-8"
    )

    # metadata.json
    seed_refs = [f"eval_seeds_{n}.txt" for n in [20, 100, 500, 1000, 2000]]
    metadata = {
        "schema": "deploy_package_v1",
        "package_id": pkg_id,
        "model_id": m_id,
        "source_strategy": strategy,
        "git_commit": _git_commit(),
        "created_at": _now_iso(),
        "compatibility": {"sim_version": "p20", "schema_version": "v1"},
        "seed_files": seed_refs,
    }
    (pkg_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    # README.md
    readme = f"""# Deploy Package: {pkg_id}

## Loading
```python
import torch
model = torch.load("{model_dir.name}/{model_p.name}", map_location="cpu")
```

## Verification
```
python trainer/package/verify_model_package.py --package-dir {pkg_dir}
```

## Contents
- `model/` - Model weights and manifest
- `config/` - Inference and risk configuration
- `metrics/` - Evaluation summaries
- `metadata.json` - Package metadata
- `checksums.json` - File integrity checksums
"""
    (pkg_dir / "README.md").write_text(readme, encoding="utf-8")

    # checksums.json (must be last)
    checksums = _build_checksums(pkg_dir)
    (pkg_dir / "checksums.json").write_text(json.dumps(checksums, indent=2) + "\n", encoding="utf-8")

    print(json.dumps({"status": "ok", "package_id": pkg_id, "package_dir": str(pkg_dir)}, ensure_ascii=False))
    return pkg_dir


def main() -> int:
    p = argparse.ArgumentParser(description="Export a deployable model package.")
    p.add_argument("--model", required=True, help="Path to model weights (best.pt).")
    p.add_argument("--strategy", required=True, choices=["pv", "hybrid", "rl", "risk_aware", "deploy_student"])
    p.add_argument("--risk-config", default="")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--package-id", default="")
    p.add_argument("--model-id", default="")
    args = p.parse_args()

    export_package(
        model_path=args.model,
        strategy=args.strategy,
        risk_config=args.risk_config or None,
        out_dir=args.out_dir,
        package_id=args.package_id or None,
        model_id=args.model_id or None,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
