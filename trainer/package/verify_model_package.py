"""Verify a deploy model package: schema, file existence, checksums, optional infer smoke."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


REQUIRED_ENTRIES = ["metadata.json", "checksums.json", "model", "config"]
DEFAULT_METADATA_REQUIRED_KEYS = ["package_id", "model_id", "source_strategy", "git_commit", "created_at"]
DEFAULT_ALLOWED_STRATEGIES = ["pv", "hybrid", "rl", "risk_aware", "deploy_student"]


def _load_schema(schema_path: Path | None = None) -> dict[str, Any]:
    path = schema_path or (Path(__file__).resolve().parent / "package_schema_v1.json")
    if not path.exists():
        return {
            "schema_version": "fallback_v1",
            "required_entries": REQUIRED_ENTRIES,
            "metadata_required_keys": DEFAULT_METADATA_REQUIRED_KEYS,
            "allowed_strategies": DEFAULT_ALLOWED_STRATEGIES,
        }
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {
            "schema_version": "fallback_v1",
            "required_entries": REQUIRED_ENTRIES,
            "metadata_required_keys": DEFAULT_METADATA_REQUIRED_KEYS,
            "allowed_strategies": DEFAULT_ALLOWED_STRATEGIES,
        }
    return payload if isinstance(payload, dict) else {}


def verify_package(pkg_dir: Path) -> dict[str, Any]:
    issues: list[str] = []
    schema = _load_schema()
    required_entries = list(schema.get("required_entries") or schema.get("required") or REQUIRED_ENTRIES)
    metadata_required_keys = list(schema.get("metadata_required_keys") or DEFAULT_METADATA_REQUIRED_KEYS)
    allowed_strategies = [str(x) for x in (schema.get("allowed_strategies") or DEFAULT_ALLOWED_STRATEGIES)]

    # 1) Required entries
    for name in required_entries:
        target = pkg_dir / name
        if not target.exists():
            issues.append(f"missing required entry: {name}")

    # 2) metadata.json schema
    meta_path = pkg_dir / "metadata.json"
    metadata: dict[str, Any] = {}
    if meta_path.exists():
        try:
            metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception as e:
            issues.append(f"metadata.json parse error: {e}")
        for k in metadata_required_keys:
            if k not in metadata:
                issues.append(f"metadata.json missing key: {k}")
        strategy = str(metadata.get("source_strategy") or "")
        if strategy and strategy not in allowed_strategies:
            issues.append(f"metadata.json source_strategy unsupported: {strategy}")
        meta_schema = str(metadata.get("schema") or "")
        if meta_schema and meta_schema != "deploy_package_v1":
            issues.append(f"metadata.json schema mismatch: {meta_schema}")

    # 3) checksums
    ck_path = pkg_dir / "checksums.json"
    if ck_path.exists():
        try:
            checksums = json.loads(ck_path.read_text(encoding="utf-8"))
        except Exception as e:
            issues.append(f"checksums.json parse error: {e}")
            checksums = {}
        for rel, info in checksums.items():
            fp = pkg_dir / rel
            if not fp.exists():
                issues.append(f"checksum file missing: {rel}")
                continue
            expected_sha = info.get("sha256", "")
            actual_sha = _sha256(fp)
            if expected_sha and actual_sha != expected_sha:
                issues.append(f"checksum mismatch: {rel} expected={expected_sha[:12]}... got={actual_sha[:12]}...")
            expected_size = info.get("size_bytes")
            if expected_size is not None and fp.stat().st_size != expected_size:
                issues.append(f"size mismatch: {rel} expected={expected_size} got={fp.stat().st_size}")

    # 4) model dir has at least one file
    model_dir = pkg_dir / "model"
    if model_dir.is_dir():
        files = [f for f in model_dir.iterdir() if f.is_file()]
        if not files:
            issues.append("model/ directory is empty")

    passed = len(issues) == 0
    return {
        "schema": "package_verify_report_v1",
        "spec_schema": str(schema.get("schema_version") or schema.get("title") or "package_schema_v1"),
        "package_dir": str(pkg_dir),
        "package_id": metadata.get("package_id", "unknown"),
        "passed": passed,
        "issues": issues,
        "checked_at": _now_iso(),
    }


def run_infer_smoke(pkg_dir: Path, backend: str = "sim") -> dict[str, Any]:
    """Optional: run a single infer step using the packaged model."""
    meta_path = pkg_dir / "metadata.json"
    if not meta_path.exists():
        return {"infer_smoke": "SKIP", "reason": "no metadata.json"}
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    strategy = metadata.get("source_strategy", "pv")
    model_dir = pkg_dir / "model"
    model_files = sorted(model_dir.glob("*.pt"))
    if not model_files:
        return {"infer_smoke": "SKIP", "reason": "no .pt file in model/"}
    model_path = model_files[0]

    cmd = [
        sys.executable, "-B", "trainer/infer_assistant.py",
        "--backend", backend,
        "--policy", strategy if strategy in ("pv", "risk_aware") else "pv",
        "--model", str(model_path),
        "--once",
    ]
    risk_cfg = pkg_dir / "config" / "p19_risk_controller.yaml"
    if risk_cfg.exists() and strategy == "risk_aware":
        cmd += ["--risk-config", str(risk_cfg)]

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        return {
            "infer_smoke": "PASS" if proc.returncode == 0 else "FAIL",
            "returncode": proc.returncode,
            "stdout_tail": (proc.stdout or "")[-500:],
            "stderr_tail": (proc.stderr or "")[-500:],
        }
    except Exception as e:
        return {"infer_smoke": "ERROR", "reason": str(e)}


def main() -> int:
    p = argparse.ArgumentParser(description="Verify a deploy model package.")
    p.add_argument("--package-dir", required=True)
    p.add_argument("--backend", default="sim")
    p.add_argument("--once", action="store_true", help="Run infer smoke test.")
    p.add_argument("--out", default="", help="Output report path (defaults to <pkg>/package_verify_report.json).")
    args = p.parse_args()

    pkg_dir = Path(args.package_dir)
    report = verify_package(pkg_dir)

    if args.once:
        report["infer_smoke_result"] = run_infer_smoke(pkg_dir, args.backend)

    out_path = Path(args.out) if args.out else pkg_dir / "package_verify_report.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"verify_pass": report["passed"], "issues": len(report["issues"]), "report": str(out_path)}))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
