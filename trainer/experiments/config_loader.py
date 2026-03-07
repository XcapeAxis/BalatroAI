"""Hardened experiment config loader with provenance tracking.

P55 strategy (chosen, authoritative):
- YAML is the sole source of truth.
- JSON sidecars are generated cache artefacts produced by config_sidecar_sync.
- At runtime, when PyYAML is available: always read YAML directly.
- When PyYAML is unavailable (e.g. .venv_trainer_cuda): fall back to the JSON
  sidecar **only if** its sha256 matches the sha256 of the YAML source text.
  If the sidecar is stale (hash mismatch) or missing, raise fast-fail RuntimeError
  with an explicit command to fix it.
- Every load returns a ConfigLoadResult carrying full provenance so summaries,
  dashboards and ops-UI can surface "what was actually read and from where".

Callers that only need the raw dict can call load_config_dict() which wraps
load_config() and returns payload only.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None  # type: ignore


SIDECAR_HASH_KEY = "__yaml_source_sha256__"
"""Key injected into JSON sidecar to record the hash of the YAML source at generation time."""


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass
class ConfigLoadResult:
    """Provenance-rich wrapper returned by load_config()."""

    payload: dict[str, Any]
    """The parsed config mapping."""

    config_source_path: str
    """Absolute path of the primary config file (YAML or JSON)."""

    config_source_type: str
    """'yaml' if parsed from YAML, 'json_sidecar' if parsed from JSON sidecar fallback."""

    config_hash: str
    """sha256 of the YAML source text (always computed from YAML, even on sidecar fallback)."""

    sidecar_path: str
    """Absolute path of the .json sidecar (empty string if no sidecar exists)."""

    sidecar_used: bool
    """True if the JSON sidecar was what was actually parsed (not YAML)."""

    sidecar_in_sync: bool
    """True if the sidecar hash matches the YAML source hash (or if sidecar was not used)."""

    generated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d.pop("payload", None)
        return d

    def summary_line(self) -> str:
        src = "yaml" if not self.sidecar_used else "json_sidecar"
        sync = "in_sync" if self.sidecar_in_sync else "DRIFT_DETECTED"
        return (
            f"config_source={src} hash={self.config_hash[:12]} "
            f"sidecar_in_sync={sync} path={self.config_source_path}"
        )


def load_config(path: Path) -> ConfigLoadResult:
    """Load an experiment config file and return a provenance-rich result.

    Args:
        path: Path to a .yaml/.yml or .json config file.

    Returns:
        ConfigLoadResult with payload + provenance.

    Raises:
        RuntimeError: If PyYAML is unavailable and the JSON sidecar is missing or stale.
        ValueError: If the loaded payload is not a dict.
        FileNotFoundError: If the path does not exist.
    """
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")

    path = path.resolve()
    suffix = path.suffix.lower()

    if suffix in {".yaml", ".yml"}:
        yaml_text = path.read_text(encoding="utf-8")
        yaml_hash = _sha256_text(yaml_text)
        sidecar_path = path.with_suffix(".json")
        sidecar_path_str = str(sidecar_path) if sidecar_path.exists() else ""

        if yaml is not None:
            payload = yaml.safe_load(yaml_text)
            if not isinstance(payload, dict):
                raise ValueError(f"Config payload must be a mapping: {path}")
            sidecar_in_sync = _check_sidecar_in_sync(sidecar_path, yaml_hash)
            return ConfigLoadResult(
                payload=payload,
                config_source_path=str(path),
                config_source_type="yaml",
                config_hash=yaml_hash,
                sidecar_path=sidecar_path_str,
                sidecar_used=False,
                sidecar_in_sync=sidecar_in_sync,
            )
        else:
            # PyYAML unavailable — must use sidecar fallback (P55 policy)
            if not sidecar_path.exists():
                raise RuntimeError(
                    f"PyYAML is unavailable and no JSON sidecar exists for: {path}\n"
                    f"Fix: python -m trainer.experiments.config_sidecar_sync --sync\n"
                    f"Or install PyYAML: pip install pyyaml"
                )
            sidecar_text = sidecar_path.read_text(encoding="utf-8")
            sidecar_data = json.loads(sidecar_text)
            if not isinstance(sidecar_data, dict):
                raise ValueError(f"Sidecar payload must be a mapping: {sidecar_path}")
            recorded_hash = sidecar_data.get(SIDECAR_HASH_KEY, "")
            if recorded_hash != yaml_hash:
                raise RuntimeError(
                    f"JSON sidecar is STALE for: {path}\n"
                    f"  YAML sha256  : {yaml_hash}\n"
                    f"  Sidecar hash : {recorded_hash}\n"
                    f"Fix: python -m trainer.experiments.config_sidecar_sync --sync\n"
                    f"     (or run: scripts/sync_config_sidecars.ps1)"
                )
            # Strip the internal hash key before returning
            payload = {k: v for k, v in sidecar_data.items() if k != SIDECAR_HASH_KEY}
            return ConfigLoadResult(
                payload=payload,
                config_source_path=str(path),
                config_source_type="json_sidecar",
                config_hash=yaml_hash,
                sidecar_path=str(sidecar_path),
                sidecar_used=True,
                sidecar_in_sync=True,
            )

    else:
        # Direct JSON file (e.g. ranking_p24.json that has no YAML counterpart)
        text = path.read_text(encoding="utf-8")
        payload = json.loads(text)
        if not isinstance(payload, dict):
            raise ValueError(f"Config payload must be a mapping: {path}")
        return ConfigLoadResult(
            payload=payload,
            config_source_path=str(path),
            config_source_type="json",
            config_hash=_sha256_text(text),
            sidecar_path="",
            sidecar_used=False,
            sidecar_in_sync=True,
        )


def load_config_dict(path: Path) -> dict[str, Any]:
    """Convenience wrapper — returns only the payload dict."""
    return load_config(path).payload


def _check_sidecar_in_sync(sidecar_path: Path, yaml_hash: str) -> bool:
    """Return True if the sidecar exists and records the correct YAML hash."""
    if not sidecar_path.exists():
        return True  # no sidecar → nothing to drift
    try:
        data = json.loads(sidecar_path.read_text(encoding="utf-8"))
        return data.get(SIDECAR_HASH_KEY, "") == yaml_hash
    except Exception:
        return False
