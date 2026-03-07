"""YAML → JSON sidecar sync and consistency checker.

P55 tool — run before any P22/nightly run, or as a pre-commit check.

Usage:
    python -m trainer.experiments.config_sidecar_sync --check
        Scan all configs/experiments/**/*.yaml, report drift. Exit 1 if any drift.

    python -m trainer.experiments.config_sidecar_sync --sync
        Regenerate all JSON sidecars from YAML source. Exit 0.

    python -m trainer.experiments.config_sidecar_sync --check --path configs/experiments/p22.yaml
        Check a single file only.

Artefacts:
    docs/artifacts/p55/config_sidecar_sync/<timestamp>/sidecar_sync_report.{json,md}
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None  # type: ignore

from trainer.experiments.config_loader import SIDECAR_HASH_KEY


def _now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _yaml_to_dict(yaml_text: str, source_path: Path) -> dict[str, Any]:
    if yaml is None:
        raise RuntimeError(
            f"PyYAML is required to regenerate sidecars but is not installed.\n"
            f"Install with: pip install pyyaml\n"
            f"Source: {source_path}"
        )
    obj = yaml.safe_load(yaml_text)
    if not isinstance(obj, dict):
        raise ValueError(f"YAML root must be a mapping: {source_path}")
    return obj


def check_file(yaml_path: Path) -> dict[str, Any]:
    """Check sidecar parity for a single YAML file.

    Returns a record dict with keys:
        source_path, sidecar_path, source_hash, sidecar_hash, status
    where status in {in_sync, refreshed, missing, drifted, error}.
    """
    yaml_path = yaml_path.resolve()
    sidecar = yaml_path.with_suffix(".json")

    try:
        yaml_text = yaml_path.read_text(encoding="utf-8")
    except Exception as exc:
        return {
            "source_path": str(yaml_path),
            "sidecar_path": str(sidecar),
            "source_hash": "",
            "sidecar_hash": "",
            "status": "error",
            "error": str(exc),
        }

    source_hash = _sha256_text(yaml_text)

    if not sidecar.exists():
        return {
            "source_path": str(yaml_path),
            "sidecar_path": str(sidecar),
            "source_hash": source_hash,
            "sidecar_hash": "",
            "status": "missing",
        }

    try:
        sidecar_data = json.loads(sidecar.read_text(encoding="utf-8"))
        recorded_hash = sidecar_data.get(SIDECAR_HASH_KEY, "")
    except Exception as exc:
        return {
            "source_path": str(yaml_path),
            "sidecar_path": str(sidecar),
            "source_hash": source_hash,
            "sidecar_hash": "",
            "status": "error",
            "error": f"failed to parse sidecar: {exc}",
        }

    if recorded_hash == source_hash:
        return {
            "source_path": str(yaml_path),
            "sidecar_path": str(sidecar),
            "source_hash": source_hash,
            "sidecar_hash": recorded_hash,
            "status": "in_sync",
        }
    else:
        return {
            "source_path": str(yaml_path),
            "sidecar_path": str(sidecar),
            "source_hash": source_hash,
            "sidecar_hash": recorded_hash,
            "status": "drifted",
        }


def sync_file(yaml_path: Path) -> dict[str, Any]:
    """Generate / refresh the JSON sidecar for a YAML file.

    Injects SIDECAR_HASH_KEY so the loader can verify freshness at runtime.
    Returns a record dict; status is 'refreshed' on success, 'error' on failure.
    """
    yaml_path = yaml_path.resolve()
    sidecar = yaml_path.with_suffix(".json")

    try:
        yaml_text = yaml_path.read_text(encoding="utf-8")
        source_hash = _sha256_text(yaml_text)
        data = _yaml_to_dict(yaml_text, yaml_path)
        data[SIDECAR_HASH_KEY] = source_hash
        sidecar.write_text(
            json.dumps(data, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return {
            "source_path": str(yaml_path),
            "sidecar_path": str(sidecar),
            "source_hash": source_hash,
            "sidecar_hash": source_hash,
            "status": "refreshed",
        }
    except Exception as exc:
        return {
            "source_path": str(yaml_path),
            "sidecar_path": str(sidecar),
            "source_hash": "",
            "sidecar_hash": "",
            "status": "error",
            "error": str(exc),
        }


def discover_yaml_files(root: Path) -> list[Path]:
    return sorted(root.rglob("*.yaml")) + sorted(root.rglob("*.yml"))


def write_report(
    records: list[dict[str, Any]],
    out_dir: Path,
    mode: str,
    timestamp: str,
) -> dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "sidecar_sync_report.json"
    md_path = out_dir / "sidecar_sync_report.md"

    in_sync = [r for r in records if r["status"] in ("in_sync", "refreshed")]
    drifted = [r for r in records if r["status"] == "drifted"]
    missing = [r for r in records if r["status"] == "missing"]
    errors = [r for r in records if r["status"] == "error"]

    report = {
        "schema": "p55_config_sidecar_sync_report_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "timestamp": timestamp,
        "mode": mode,
        "total": len(records),
        "in_sync": len(in_sync),
        "drifted": len(drifted),
        "missing": len(missing),
        "errors": len(errors),
        "overall_status": "clean" if not drifted and not missing and not errors else "drift_detected",
        "records": records,
    }

    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# Config Sidecar Sync Report",
        "",
        f"- generated_at: {report['generated_at']}",
        f"- mode: `{mode}`",
        f"- overall_status: **{report['overall_status']}**",
        "",
        f"| Metric | Count |",
        f"|---|---|",
        f"| Total YAML files | {report['total']} |",
        f"| In sync / refreshed | {report['in_sync']} |",
        f"| Drifted (stale sidecar) | {report['drifted']} |",
        f"| Missing sidecar | {report['missing']} |",
        f"| Errors | {report['errors']} |",
    ]

    if drifted:
        lines += ["", "## Drifted (stale sidecar)"]
        for r in drifted:
            lines.append(f"- `{r['source_path']}`  yaml_hash=`{r['source_hash'][:12]}` sidecar_hash=`{r['sidecar_hash'][:12] if r['sidecar_hash'] else 'none'}`")

    if missing:
        lines += ["", "## Missing sidecar"]
        for r in missing:
            lines.append(f"- `{r['source_path']}`")

    if errors:
        lines += ["", "## Errors"]
        for r in errors:
            lines.append(f"- `{r['source_path']}`: {r.get('error', '?')}")

    lines += [
        "",
        "## Fix command",
        "```bash",
        "python -m trainer.experiments.config_sidecar_sync --sync",
        "# or",
        "powershell -ExecutionPolicy Bypass -File scripts/sync_config_sidecars.ps1",
        "```",
    ]

    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    return {"json": str(json_path), "md": str(md_path)}


def run(
    *,
    mode: str,
    paths: list[Path] | None = None,
    config_root: Path | None = None,
    out_root: Path | None = None,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    """Run check or sync over YAML files.

    Args:
        mode: 'check' (read-only, exit 1 on drift) or 'sync' (generate/update sidecars).
        paths: explicit list of YAML paths to process (if None, discovers all under config_root).
        config_root: root directory to scan (default: repo_root/configs/experiments).
        out_root: root for artefact output (default: repo_root/docs/artifacts/p55/config_sidecar_sync).
        repo_root: repo root (auto-detected if None).

    Returns:
        dict with keys: records, overall_status, report_json, report_md, timestamp.
    """
    if repo_root is None:
        repo_root = Path(__file__).resolve().parent.parent.parent
    if config_root is None:
        config_root = repo_root / "configs" / "experiments"
    if out_root is None:
        out_root = repo_root / "docs" / "artifacts" / "p55" / "config_sidecar_sync"

    timestamp = _now_stamp()
    run_out = out_root / timestamp

    if paths is not None:
        yaml_files = [Path(p).resolve() for p in paths]
    else:
        yaml_files = discover_yaml_files(config_root)

    records: list[dict[str, Any]] = []
    if mode == "sync":
        for yf in yaml_files:
            records.append(sync_file(yf))
    else:
        for yf in yaml_files:
            records.append(check_file(yf))

    paths_written = write_report(records, run_out, mode, timestamp)

    drifted = [r for r in records if r["status"] == "drifted"]
    missing = [r for r in records if r["status"] == "missing"]
    errors = [r for r in records if r["status"] == "error"]
    overall = "clean" if not drifted and not missing and not errors else "drift_detected"

    return {
        "records": records,
        "overall_status": overall,
        "report_json": paths_written["json"],
        "report_md": paths_written["md"],
        "timestamp": timestamp,
        "drifted_count": len(drifted),
        "missing_count": len(missing),
        "error_count": len(errors),
        "total": len(records),
    }


def _main() -> int:
    parser = argparse.ArgumentParser(
        description="P55 YAML/JSON sidecar consistency checker and sync tool."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--check", action="store_true", help="Check parity only (exit 1 if drift).")
    group.add_argument("--sync", action="store_true", help="Regenerate all JSON sidecars from YAML.")
    parser.add_argument("--path", nargs="*", help="Specific YAML file(s) to check/sync.")
    parser.add_argument("--config-root", default=None, help="Directory to scan for YAML files.")
    parser.add_argument("--out-root", default=None, help="Directory for artefact output.")
    parser.add_argument("--quiet", action="store_true", help="Suppress per-file output.")
    args = parser.parse_args()

    requested_mode = "check" if args.check else "sync"
    explicit_paths = [Path(p) for p in args.path] if args.path else None
    config_root = Path(args.config_root) if args.config_root else None
    out_root = Path(args.out_root) if args.out_root else None

    # When --sync is requested but PyYAML is unavailable, downgrade to --check.
    # This makes the CUDA env safe: it can't regenerate sidecars, but can detect
    # drift and fail fast with a clear message rather than silently getting errors.
    if requested_mode == "sync" and yaml is None:
        print(
            "[sidecar-sync] WARNING: PyYAML unavailable in this Python env; "
            "downgrading --sync to --check. "
            "To regenerate sidecars, run from an env with PyYAML installed: "
            "python -m trainer.experiments.config_sidecar_sync --sync",
            flush=True,
        )
        mode = "check"
    else:
        mode = requested_mode

    result = run(mode=mode, paths=explicit_paths, config_root=config_root, out_root=out_root)

    if not args.quiet:
        print(f"[sidecar-sync] mode={mode} total={result['total']} "
              f"drifted={result['drifted_count']} missing={result['missing_count']} "
              f"errors={result['error_count']} overall={result['overall_status']}")
        print(f"[sidecar-sync] report_json={result['report_json']}")
        print(f"[sidecar-sync] report_md={result['report_md']}")

    if result["overall_status"] != "clean":
        if mode == "check":
            print(f"[sidecar-sync] FAIL: sidecar drift detected. Run with --sync to fix.", file=sys.stderr)
        else:
            print(f"[sidecar-sync] FAIL: sync completed with {result['error_count']} error(s).", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(_main())
