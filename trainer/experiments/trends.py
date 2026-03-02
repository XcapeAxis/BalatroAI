from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .trend_schema import TREND_COLUMNS, coerce_float, normalize_status


_MILESTONE_RE = re.compile(r"^p(\d+)$", re.IGNORECASE)
_RUN_ID_RE = re.compile(r"(\d{8}-\d{6})")
_STAMP_DIR_RE = re.compile(r"^\d{8}-\d{6}$")


@dataclass
class ScanStats:
    source_files_scanned: int = 0
    source_files_indexed: int = 0
    source_files_skipped: int = 0
    rows_emitted: int = 0


def _safe_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _iso(value: Any, source_path: Path) -> str:
    token = str(value or "").strip()
    if "T" in token:
        return token
    m = _RUN_ID_RE.search(token)
    if m:
        try:
            return datetime.strptime(m.group(1), "%Y%m%d-%H%M%S").replace(tzinfo=timezone.utc).isoformat()
        except Exception:
            pass
    return datetime.fromtimestamp(source_path.stat().st_mtime, tz=timezone.utc).isoformat()


def _infer_milestone(path: Path, scan_root: Path) -> str:
    try:
        rel = path.relative_to(scan_root)
    except Exception:
        return "UNKNOWN"
    for part in rel.parts:
        m = _MILESTONE_RE.match(part)
        if m:
            return f"P{int(m.group(1))}"
    return "UNKNOWN"


def _infer_run_id(payload: Any, path: Path) -> str:
    if isinstance(payload, dict):
        for key in ("run_id", "updated_by_run"):
            token = str(payload.get(key) or "").strip()
            if token:
                return token
    m = _RUN_ID_RE.search(str(path))
    return m.group(1) if m else "unknown"


def _infer_gate_name(path: Path) -> str:
    stem = path.stem.lower()
    if stem.startswith("report_p") and stem.endswith("_gate"):
        m = re.match(r"report_(p\d+)_gate$", stem)
        if m:
            return f"Run{m.group(1).upper()}"
    if stem.startswith("report_p"):
        m = re.match(r"report_(p\d+).*$", stem)
        if m:
            return f"Run{m.group(1).upper()}"
    if stem.startswith("gate_"):
        return stem
    if stem == "summary_table":
        return "benchmark_summary"
    if stem == "ranking_summary":
        return "ranking"
    if stem == "flake_report":
        return "flake"
    if stem == "nightly_decision":
        return "nightly_decision"
    if stem == "coverage_summary":
        return "coverage"
    return stem


def _risk_status(raw: Any) -> str:
    token = str(raw or "").strip().lower()
    if token in {"promote", "improvement"}:
        return "improvement"
    if token in {"hold", "stable"}:
        return "stable"
    if token in {"regression", "demote", "fail"}:
        return "regression"
    return token


def _flake_status(raw: Any) -> str:
    token = normalize_status(raw)
    if token == "pass":
        return "stable"
    if token == "fail":
        return "flake_fail"
    return ""


def _rel(path: Path, base: Path) -> str:
    try:
        return path.relative_to(base).as_posix()
    except Exception:
        return str(path).replace("\\", "/")


def _ctx_from_run_dir(run_dir: str | Path | None) -> dict[str, str]:
    if not run_dir:
        return {}
    base = Path(str(run_dir))
    manifest = base / "run_manifest.json"
    if not manifest.exists():
        return {}
    payload = _safe_json(manifest)
    if not isinstance(payload, dict):
        return {}
    exp = payload.get("experiment")
    strategy = ""
    if isinstance(exp, dict):
        strategy = str(exp.get("policy_type") or exp.get("policy") or "").strip()
    return {
        "strategy": strategy,
        "seed_set_name": str(payload.get("seed_set_name") or "").strip(),
        "git_commit": str(payload.get("git_commit") or "").strip(),
        "run_id": str(payload.get("run_id") or "").strip(),
    }


def _emit(
    rows: list[dict[str, Any]],
    *,
    timestamp: str,
    milestone: str,
    run_id: str,
    artifact_path: str,
    gate_name: str,
    strategy: str,
    seed_set_name: str,
    metric_name: str,
    metric_value: float,
    unit: str,
    status: str,
    source_file: str,
    git_commit: str,
    flake_status: str,
    risk_status: str,
) -> None:
    rows.append(
        {
            "timestamp": timestamp,
            "milestone": milestone,
            "run_id": run_id,
            "artifact_path": artifact_path,
            "gate_name": gate_name,
            "strategy": strategy,
            "seed_set_name": seed_set_name,
            "metric_name": metric_name,
            "metric_value": float(metric_value),
            "unit": unit,
            "status": normalize_status(status),
            "source_file": source_file,
            "git_commit": git_commit,
            "flake_status": flake_status,
            "risk_status": risk_status,
        }
    )


def _metric_unit(metric_name: str) -> str:
    if metric_name in {"avg_ante_reached", "median_ante_reached", "median_ante"}:
        return "ante"
    if metric_name == "win_rate":
        return "ratio"
    if metric_name in {"runtime_seconds", "elapsed_sec"}:
        return "seconds"
    if metric_name in {"seed_count", "catastrophic_failure_count", "stage_pass_count", "experiment_count"}:
        return "count"
    return "number"


def _metric_alias(metric_name: str) -> str:
    if metric_name == "median_ante":
        return "median_ante_reached"
    if metric_name in {"elapsed_sec", "runtime_sec"}:
        return "runtime_seconds"
    if metric_name == "mean":
        return "avg_ante_reached"
    return metric_name


def _extract_summary_rows(
    table_rows: list[Any],
    *,
    source_path: Path,
    source_rel: str,
    scan_root: Path,
    milestone: str,
    gate_name: str,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in table_rows:
        if not isinstance(row, dict):
            continue
        ctx = _ctx_from_run_dir(row.get("run_dir"))
        timestamp = _iso(row.get("generated_at"), source_path)
        run_id = ctx.get("run_id") or _infer_run_id(row, source_path)
        status = row.get("status")
        strategy = ctx.get("strategy") or str(row.get("exp_id") or row.get("strategy") or "")
        seed_set_name = ctx.get("seed_set_name") or str(row.get("seed_set_name") or "")
        git_commit = ctx.get("git_commit") or ""
        for name in (
            "avg_ante_reached",
            "median_ante",
            "win_rate",
            "elapsed_sec",
            "seed_count",
            "catastrophic_failure_count",
            "std",
            "mean",
        ):
            value = coerce_float(row.get(name))
            if value is None:
                continue
            canonical = _metric_alias(name)
            _emit(
                out,
                timestamp=timestamp,
                milestone=milestone,
                run_id=run_id,
                artifact_path=_rel(source_path.parent, scan_root),
                gate_name=gate_name,
                strategy=strategy,
                seed_set_name=seed_set_name,
                metric_name=canonical,
                metric_value=value,
                unit=_metric_unit(canonical),
                status=str(status),
                source_file=source_rel,
                git_commit=git_commit,
                flake_status="",
                risk_status="",
            )
    return out


def extract_rows_from_file(source_path: Path, scan_root: Path) -> list[dict[str, Any]]:
    payload = _safe_json(source_path)
    if payload is None:
        return []
    source_rel = _rel(source_path, scan_root)
    milestone = _infer_milestone(source_path, scan_root)
    gate_name = _infer_gate_name(source_path)
    name = source_path.name.lower()

    if name == "summary_table.json" and isinstance(payload, list):
        return _extract_summary_rows(payload, source_path=source_path, source_rel=source_rel, scan_root=scan_root, milestone=milestone, gate_name=gate_name)
    if name.startswith("report_p") and name.endswith(".json") and isinstance(payload, dict) and isinstance(payload.get("rows"), list):
        return _extract_summary_rows(payload.get("rows"), source_path=source_path, source_rel=source_rel, scan_root=scan_root, milestone=milestone, gate_name=gate_name)

    rows: list[dict[str, Any]] = []
    if not isinstance(payload, dict):
        return rows
    timestamp = _iso(payload.get("generated_at") or payload.get("updated_at"), source_path)
    run_id = _infer_run_id(payload, source_path)
    artifact_path = _rel(source_path.parent, scan_root)

    if name.startswith("gate_") or (name.startswith("report_p") and name.endswith("_gate.json")):
        status = normalize_status(payload.get("status") if "status" in payload else payload.get("pass"))
        if "status" in payload:
            _emit(
                rows,
                timestamp=timestamp,
                milestone=milestone,
                run_id=run_id,
                artifact_path=artifact_path,
                gate_name=gate_name,
                strategy="",
                seed_set_name="",
                metric_name="gate_overall_pass",
                metric_value=1.0 if status == "pass" else 0.0,
                unit="ratio",
                status=status,
                source_file=source_rel,
                git_commit="",
                flake_status="",
                risk_status="",
            )
        for key, value_raw in payload.items():
            if key in {"schema", "generated_at", "status"}:
                continue
            if isinstance(value_raw, dict):
                section = value_raw
                section_status = normalize_status(section.get("pass")) if "pass" in section else status
                if "pass" in section:
                    _emit(
                        rows,
                        timestamp=timestamp,
                        milestone=milestone,
                        run_id=run_id,
                        artifact_path=artifact_path,
                        gate_name=gate_name,
                        strategy="",
                        seed_set_name="",
                        metric_name=f"{key}_pass",
                        metric_value=1.0 if section_status == "pass" else 0.0,
                        unit="ratio",
                        status=section_status,
                        source_file=source_rel,
                        git_commit="",
                        flake_status=_flake_status(section.get("flake_status")),
                        risk_status=_risk_status(section.get("risk_status")),
                    )
                for sk, sv_raw in section.items():
                    if sk in {"schema", "generated_at", "pass"}:
                        continue
                    sv = coerce_float(sv_raw)
                    if sv is None:
                        continue
                    _emit(
                        rows,
                        timestamp=timestamp,
                        milestone=milestone,
                        run_id=run_id,
                        artifact_path=artifact_path,
                        gate_name=gate_name,
                        strategy="",
                        seed_set_name="",
                        metric_name=f"{key}_{sk}",
                        metric_value=sv,
                        unit=_metric_unit(sk),
                        status=section_status,
                        source_file=source_rel,
                        git_commit="",
                        flake_status=_flake_status(section.get("flake_status")),
                        risk_status=_risk_status(section.get("risk_status")),
                    )
                continue
            value = coerce_float(value_raw)
            if value is None:
                continue
            _emit(
                rows,
                timestamp=timestamp,
                milestone=milestone,
                run_id=run_id,
                artifact_path=artifact_path,
                gate_name=gate_name,
                strategy="",
                seed_set_name="",
                metric_name=_metric_alias(key),
                metric_value=value,
                unit=_metric_unit(_metric_alias(key)),
                status=status,
                source_file=source_rel,
                git_commit="",
                flake_status=_flake_status(payload.get("flake_status")),
                risk_status=_risk_status(payload.get("risk_status")),
            )
        return rows

    if name == "status.json":
        ctx = _ctx_from_run_dir(source_path.parent)
        for metric_name in ("avg_ante_reached", "median_ante", "win_rate", "elapsed_sec", "seed_count", "catastrophic_failure_count"):
            value = coerce_float(payload.get(metric_name))
            if value is None:
                continue
            canonical = _metric_alias(metric_name)
            _emit(
                rows,
                timestamp=timestamp,
                milestone=milestone,
                run_id=ctx.get("run_id") or run_id,
                artifact_path=artifact_path,
                gate_name=gate_name,
                strategy=ctx.get("strategy", ""),
                seed_set_name=ctx.get("seed_set_name", ""),
                metric_name=canonical,
                metric_value=value,
                unit=_metric_unit(canonical),
                status=str(payload.get("status")),
                source_file=source_rel,
                git_commit=ctx.get("git_commit", ""),
                flake_status="",
                risk_status="",
            )
        return rows

    if name == "ranking_summary.json" and isinstance(payload.get("rows"), list):
        global_flake = _flake_status(payload.get("global_flake_pass"))
        for item in payload.get("rows", []):
            if not isinstance(item, dict):
                continue
            strategy = str(item.get("exp_id") or "")
            risk = "high_risk" if float(item.get("risk_score") or 0.0) > 0 else "low_risk"
            for metric_name in ("avg_ante_reached", "median_ante", "win_rate", "elapsed_sec", "weighted_score", "risk_score"):
                value = coerce_float(item.get(metric_name))
                if value is None:
                    continue
                canonical = _metric_alias(metric_name)
                _emit(
                    rows,
                    timestamp=timestamp,
                    milestone=milestone,
                    run_id=run_id,
                    artifact_path=artifact_path,
                    gate_name=gate_name,
                    strategy=strategy,
                    seed_set_name="",
                    metric_name=canonical,
                    metric_value=value,
                    unit=_metric_unit(canonical),
                    status=str(item.get("status")),
                    source_file=source_rel,
                    git_commit="",
                    flake_status=global_flake,
                    risk_status=risk,
                )
        return rows

    if name == "flake_report.json":
        stats = payload.get("stats") if isinstance(payload.get("stats"), dict) else {}
        metrics = {"flake_score": 0.0 if normalize_status(payload.get("status")) == "pass" else 1.0}
        for key in ("avg_ante_std", "median_ante_span", "win_rate_std", "trace_mismatch"):
            value = coerce_float(stats.get(key))
            if value is not None:
                metrics[key] = value
        for metric_name, metric_value in metrics.items():
            _emit(
                rows,
                timestamp=timestamp,
                milestone=milestone,
                run_id=run_id,
                artifact_path=artifact_path,
                gate_name=gate_name,
                strategy=str(payload.get("exp_id") or ""),
                seed_set_name="",
                metric_name=metric_name,
                metric_value=metric_value,
                unit=_metric_unit(metric_name),
                status=str(payload.get("status")),
                source_file=source_rel,
                git_commit="",
                flake_status=_flake_status(payload.get("status")),
                risk_status="",
            )
        return rows

    if name in {"nightly_decision.json", "candidate.json", "champion.json"}:
        decision = _risk_status(payload.get("decision") or payload.get("status"))
        for key in ("candidate", "top_candidate", "conservative_candidate", "exploratory_candidate"):
            candidate = payload.get(key)
            if not isinstance(candidate, dict):
                continue
            for metric_name in ("avg_ante_reached", "median_ante", "win_rate", "elapsed_sec", "weighted_score", "risk_score"):
                value = coerce_float(candidate.get(metric_name))
                if value is None:
                    continue
                canonical = _metric_alias(metric_name)
                _emit(
                    rows,
                    timestamp=timestamp,
                    milestone=milestone,
                    run_id=run_id,
                    artifact_path=artifact_path,
                    gate_name=gate_name,
                    strategy=str(candidate.get("exp_id") or key),
                    seed_set_name="",
                    metric_name=canonical,
                    metric_value=value,
                    unit=_metric_unit(canonical),
                    status=str(candidate.get("status") or payload.get("status") or "passed"),
                    source_file=source_rel,
                    git_commit="",
                    flake_status="",
                    risk_status=decision,
                )
        if name == "champion.json":
            for metric_name in ("avg_ante_reached", "median_ante", "win_rate", "weighted_score"):
                value = coerce_float(payload.get(metric_name))
                if value is None:
                    continue
                canonical = _metric_alias(metric_name)
                _emit(
                    rows,
                    timestamp=timestamp,
                    milestone=milestone,
                    run_id=run_id,
                    artifact_path=artifact_path,
                    gate_name=gate_name,
                    strategy=str(payload.get("exp_id") or "champion"),
                    seed_set_name="",
                    metric_name=canonical,
                    metric_value=value,
                    unit=_metric_unit(canonical),
                    status=str(payload.get("status") or "passed"),
                    source_file=source_rel,
                    git_commit="",
                    flake_status="",
                    risk_status=decision,
                )
        return rows

    if name == "coverage_summary.json":
        value = coerce_float(payload.get("experiments"))
        if value is not None:
            _emit(
                rows,
                timestamp=timestamp,
                milestone=milestone,
                run_id=run_id,
                artifact_path=artifact_path,
                gate_name=gate_name,
                strategy="",
                seed_set_name="",
                metric_name="experiment_count",
                metric_value=value,
                unit="count",
                status="passed",
                source_file=source_rel,
                git_commit="",
                flake_status="",
                risk_status="",
            )
        seed_cov = payload.get("seed_coverage")
        if isinstance(seed_cov, dict):
            for key in ("unique_seed_count", "total_seed_observations"):
                value = coerce_float(seed_cov.get(key))
                if value is None:
                    continue
                _emit(
                    rows,
                    timestamp=timestamp,
                    milestone=milestone,
                    run_id=run_id,
                    artifact_path=artifact_path,
                    gate_name=gate_name,
                    strategy="",
                    seed_set_name="",
                    metric_name=key,
                    metric_value=value,
                    unit="count",
                    status="passed",
                    source_file=source_rel,
                    git_commit="",
                    flake_status="",
                    risk_status="",
                )
        return rows

    if name == "readme_status_generation.json":
        _emit(
            rows,
            timestamp=timestamp,
            milestone=milestone,
            run_id=run_id,
            artifact_path=artifact_path,
            gate_name=gate_name,
            strategy="",
            seed_set_name="",
            metric_name="readme_status_pass",
            metric_value=1.0 if normalize_status(payload.get("pass")) == "pass" else 0.0,
            unit="ratio",
            status=str(payload.get("pass")),
            source_file=source_rel,
            git_commit="",
            flake_status="",
            risk_status="",
        )
        return rows

    return rows


def _latest_timestamp_dir(milestone_dir: Path) -> Path | None:
    dirs = [d for d in milestone_dir.iterdir() if d.is_dir() and _STAMP_DIR_RE.match(d.name)]
    if not dirs:
        return None
    return sorted(dirs, key=lambda p: p.name)[-1]


def _iter_source_json(scan_root: Path, latest_only: bool) -> list[Path]:
    files: set[Path] = set()
    for milestone_dir in sorted(scan_root.iterdir(), key=lambda p: p.name):
        if not milestone_dir.is_dir() or _MILESTONE_RE.match(milestone_dir.name) is None:
            continue
        if not latest_only:
            files.update(p.resolve() for p in milestone_dir.rglob("*.json"))
            continue
        latest_dir = _latest_timestamp_dir(milestone_dir)
        for p in milestone_dir.rglob("*.json"):
            rel = p.relative_to(milestone_dir)
            include = False
            if latest_dir and str(p).startswith(str(latest_dir)):
                include = True
            elif len(rel.parts) >= 2 and rel.parts[0] == "runs" and rel.parts[1] == "latest":
                include = True
            elif len(rel.parts) == 1 and rel.parts[0] in {"candidate.json", "champion.json", "nightly_decision.json", "flake_report.json"}:
                include = True
            if include:
                files.add(p.resolve())
    return sorted(files)


def dedupe_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    key_map: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in rows:
        key = tuple(row.get(k) for k in TREND_COLUMNS)
        key_map[key] = row
    out = list(key_map.values())
    out.sort(
        key=lambda r: (
            str(r.get("timestamp")),
            str(r.get("milestone")),
            str(r.get("run_id")),
            str(r.get("gate_name")),
            str(r.get("strategy")),
            str(r.get("metric_name")),
            str(r.get("source_file")),
        )
    )
    return out


def index_artifacts(scan_root: Path, *, latest_only: bool = False) -> tuple[list[dict[str, Any]], ScanStats]:
    rows: list[dict[str, Any]] = []
    stats = ScanStats()
    for source_path in _iter_source_json(scan_root, latest_only):
        stats.source_files_scanned += 1
        extracted = extract_rows_from_file(source_path, scan_root)
        if extracted:
            stats.source_files_indexed += 1
            stats.rows_emitted += len(extracted)
            rows.extend(extracted)
        else:
            stats.source_files_skipped += 1
    return dedupe_rows(rows), stats


def load_trend_rows(out_root: Path) -> list[dict[str, Any]]:
    path = out_root / "trend_rows.jsonl"
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        token = line.strip()
        if not token:
            continue
        try:
            payload = json.loads(token)
        except Exception:
            continue
        if isinstance(payload, dict):
            rows.append(payload)
    return dedupe_rows(rows)


def write_trend_outputs(out_root: Path, rows: list[dict[str, Any]], summary: dict[str, Any]) -> dict[str, str]:
    out_root.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_root / "trend_rows.jsonl"
    csv_path = out_root / "trend_rows.csv"
    summary_path = out_root / "trend_index_summary.json"

    with jsonl_path.open("w", encoding="utf-8") as fp:
        for row in rows:
            fp.write(json.dumps(row, ensure_ascii=False) + "\n")

    with csv_path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=TREND_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in TREND_COLUMNS})

    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {
        "trend_rows_jsonl": str(jsonl_path),
        "trend_rows_csv": str(csv_path),
        "trend_index_summary_json": str(summary_path),
    }


def build_index_summary(
    *,
    scan_root: Path,
    out_root: Path,
    mode: str,
    latest_only: bool,
    stats: ScanStats,
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "schema": "p26_trend_index_summary_v1",
        "generated_at": _now_iso(),
        "scan_root": str(scan_root),
        "out_root": str(out_root),
        "mode": mode,
        "latest_only": bool(latest_only),
        "source_files_scanned": stats.source_files_scanned,
        "source_files_indexed": stats.source_files_indexed,
        "source_files_skipped": stats.source_files_skipped,
        "rows_total": len(rows),
        "runs_indexed": len({str(r.get("run_id") or "") for r in rows}),
        "milestones": sorted({str(r.get("milestone") or "UNKNOWN") for r in rows}),
    }


def query_rows(
    rows: list[dict[str, Any]],
    *,
    milestone: str = "",
    strategy: str = "",
    gate_name: str = "",
    run_id: str = "",
) -> list[dict[str, Any]]:
    m = milestone.strip().lower()
    s = strategy.strip().lower()
    g = gate_name.strip().lower()
    r = run_id.strip().lower()
    out: list[dict[str, Any]] = []
    for row in rows:
        if m and str(row.get("milestone") or "").lower() != m:
            continue
        if s and s not in str(row.get("strategy") or "").lower():
            continue
        if g and g not in str(row.get("gate_name") or "").lower():
            continue
        if r and str(row.get("run_id") or "").lower() != r:
            continue
        out.append(row)
    return out
