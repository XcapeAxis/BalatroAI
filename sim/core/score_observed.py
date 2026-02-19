from typing import Any


OBSERVED_TOTAL_PATHS = (
    "$.round.chips",
    "$.score.chips",
    "$.round.score",
    "$.chips",
)


def _parse_path_tokens(path: str) -> list[str | int]:
    s = str(path or "").strip()
    if s.startswith("$"):
        s = s[1:]
    if s.startswith("."):
        s = s[1:]
    tokens: list[str | int] = []
    i = 0
    while i < len(s):
        if s[i] == ".":
            i += 1
            continue
        if s[i] == "[":
            j = s.find("]", i)
            if j == -1:
                break
            idx = s[i + 1 : j].strip()
            if idx.isdigit() or (idx.startswith("-") and idx[1:].isdigit()):
                tokens.append(int(idx))
            else:
                tokens.append(idx)
            i = j + 1
            continue
        j = i
        while j < len(s) and s[j] not in ".[":
            j += 1
        if j > i:
            tokens.append(s[i:j])
        i = j
    return tokens


def _get_by_path(obj: Any, path: str) -> tuple[Any, bool]:
    cur = obj
    for tok in _parse_path_tokens(path):
        if isinstance(tok, int):
            if not isinstance(cur, list) or tok < 0 or tok >= len(cur):
                return None, False
            cur = cur[tok]
        else:
            if not isinstance(cur, dict) or tok not in cur:
                return None, False
            cur = cur[tok]
    return cur, True


def _pick_observed_total(state: dict[str, Any]) -> tuple[str, float]:
    for path in OBSERVED_TOTAL_PATHS:
        value, ok = _get_by_path(state, path)
        if ok and isinstance(value, (int, float)) and not isinstance(value, bool):
            return path, float(value)
    return "none", 0.0


def compute_score_observed(pre_state: dict[str, Any], post_state: dict[str, Any]) -> dict[str, Any]:
    pre_field, pre_total = _pick_observed_total(pre_state)
    post_field, post_total = _pick_observed_total(post_state)
    source = post_field if post_field != "none" else pre_field
    return {
        "source_field": source,
        "pre_total": float(pre_total),
        "total": float(post_total),
        "delta": float(post_total - pre_total),
    }
