from __future__ import annotations

import copy
import itertools
import re
from typing import Any


def _slug(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", value.strip())
    return cleaned.strip("-").lower() or "exp"


def build_matrix(config: dict[str, Any]) -> list[dict[str, Any]]:
    """Build experiment list from config matrix with optional param grid expansion."""
    raw = config.get("matrix") or []
    if not isinstance(raw, list):
        raise ValueError("config.matrix must be a list")

    out: list[dict[str, Any]] = []
    for idx, item in enumerate(raw):
        if not isinstance(item, dict):
            raise ValueError(f"matrix[{idx}] must be a mapping")

        base = copy.deepcopy(item)
        exp_id = _slug(str(base.get("id") or base.get("name") or f"exp-{idx+1}"))
        base["id"] = exp_id
        base.setdefault("name", exp_id)
        base.setdefault("seed_mode", "regression_fixed")
        base.setdefault("stages", {})
        base.setdefault("parameters", {})

        grid = base.pop("grid", None)
        if not grid:
            out.append(base)
            continue
        if not isinstance(grid, dict):
            raise ValueError(f"matrix[{idx}].grid must be a mapping")

        keys = sorted(grid.keys())
        values_list = []
        for key in keys:
            vals = grid[key]
            if not isinstance(vals, list) or not vals:
                raise ValueError(f"matrix[{idx}].grid.{key} must be non-empty list")
            values_list.append(vals)

        for combo_idx, combo in enumerate(itertools.product(*values_list), start=1):
            variant = copy.deepcopy(base)
            variant_params = dict(variant.get("parameters") or {})
            name_parts = [exp_id]
            for k, v in zip(keys, combo):
                variant_params[k] = v
                name_parts.append(f"{_slug(k)}-{_slug(str(v))}")
            variant["parameters"] = variant_params
            variant["id"] = _slug("__".join(name_parts))
            variant["name"] = variant["id"]
            variant["variant_index"] = combo_idx
            out.append(variant)
    return out

