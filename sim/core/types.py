from dataclasses import dataclass, field
from typing import Any


@dataclass
class Card:
    uid: str
    rank: str
    suit: str
    key: str
    modifier: list[str] = field(default_factory=list)
    state: list[str] = field(default_factory=list)


@dataclass
class RoundState:
    hands_left: int
    discards_left: int
    ante: int
    round_num: int
    blind: str


@dataclass
class ScoreState:
    chips: float
    mult: float
    target_chips: float
    last_hand_type: str
    last_base_chips: float
    last_base_mult: float


@dataclass
class CanonicalState:
    schema_version: str
    phase: str
    zones: dict[str, list[Card]]
    round: RoundState
    score: ScoreState
    economy: dict[str, Any]
    jokers: list[dict[str, Any]]
    rng: dict[str, Any]
    flags: dict[str, Any]
