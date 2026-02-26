from __future__ import annotations

from typing import Any


def compute_ece(rows: list[dict[str, Any]], bins: int = 10) -> dict[str, Any]:
    if not rows:
        return {"ece": 0.0, "bins": []}
    total = len(rows)
    out_bins = []
    ece = 0.0
    for b in range(int(bins)):
        lo = b / bins
        hi = (b + 1) / bins
        chunk = [r for r in rows if float(r.get("conf") or 0.0) >= lo and (float(r.get("conf") or 0.0) < hi or b == bins - 1)]
        if not chunk:
            out_bins.append({"bin": b, "lo": lo, "hi": hi, "count": 0, "acc": 0.0, "conf": 0.0})
            continue
        acc = sum(1 for r in chunk if bool(r.get("ok"))) / len(chunk)
        conf = sum(float(r.get("conf") or 0.0) for r in chunk) / len(chunk)
        ece += (len(chunk) / total) * abs(acc - conf)
        out_bins.append({"bin": b, "lo": lo, "hi": hi, "count": len(chunk), "acc": acc, "conf": conf})
    return {"ece": ece, "bins": out_bins}
