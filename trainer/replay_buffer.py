from __future__ import annotations

import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any


@dataclass
class ReplayItem:
    phase: str
    source: str
    stake: str
    action_id: int
    legal_ids: list[int]
    value_target: float
    reward: float
    payload: dict[str, Any]
    tags: list[str]


class ReplayBuffer:
    def __init__(self) -> None:
        self._rows: list[ReplayItem] = []

    def add(self, row: ReplayItem) -> None:
        self._rows.append(row)

    def extend(self, rows: list[ReplayItem]) -> None:
        self._rows.extend(rows)

    def __len__(self) -> int:
        return len(self._rows)

    def sample(self, n: int, *, phase: str | None = None, rng: random.Random | None = None) -> list[ReplayItem]:
        r = rng or random.Random(7)
        pool = self._rows if phase is None else [x for x in self._rows if x.phase == phase]
        if not pool:
            return []
        if n >= len(pool):
            return list(pool)
        idx = list(range(len(pool)))
        r.shuffle(idx)
        return [pool[i] for i in idx[:n]]

    def composition_stats(self) -> dict[str, Any]:
        src = Counter()
        stake = Counter()
        phase = Counter()
        tags = Counter()
        invalid = 0
        illegal = 0
        for row in self._rows:
            src[row.source] += 1
            stake[row.stake] += 1
            phase[row.phase] += 1
            if not row.legal_ids:
                invalid += 1
            if row.action_id not in row.legal_ids:
                illegal += 1
            for t in row.tags:
                tags[t] += 1
        total = max(1, len(self._rows))
        return {
            "total": len(self._rows),
            "source_composition": dict(sorted(src.items())),
            "stake_composition": dict(sorted(stake.items())),
            "phase_composition": dict(sorted(phase.items())),
            "tag_composition": dict(sorted(tags.items())),
            "invalid_rows": invalid,
            "illegal_action_rate": illegal / total,
        }

    def grouped_by_phase(self) -> dict[str, list[ReplayItem]]:
        out: dict[str, list[ReplayItem]] = defaultdict(list)
        for row in self._rows:
            out[row.phase].append(row)
        return dict(out)
