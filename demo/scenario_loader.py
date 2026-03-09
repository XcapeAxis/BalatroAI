from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class DemoScenario:
    scenario_id: str
    name: str
    summary: str
    focus: str
    talk_track: str
    snapshot: dict[str, Any]
    path: Path

    def to_summary(self) -> dict[str, Any]:
        round_info = self.snapshot.get("round") if isinstance(self.snapshot.get("round"), dict) else {}
        score_info = self.snapshot.get("score") if isinstance(self.snapshot.get("score"), dict) else {}
        return {
            "id": self.scenario_id,
            "name": self.name,
            "summary": self.summary,
            "focus": self.focus,
            "talk_track": self.talk_track,
            "phase": str(self.snapshot.get("phase") or "UNKNOWN"),
            "hands_left": int(round_info.get("hands_left") or 0),
            "discards_left": int(round_info.get("discards_left") or 0),
            "chips": float(score_info.get("chips") or 0.0),
            "target_chips": float(score_info.get("target_chips") or 0.0),
            "has_jokers": bool(self.snapshot.get("jokers")),
            "path": str(self.path),
        }


def _default_scenarios_root() -> Path:
    return Path(__file__).resolve().parent / "scenarios"


def _normalize_scenario_payload(payload: dict[str, Any], path: Path) -> DemoScenario:
    snapshot = payload.get("snapshot") if isinstance(payload.get("snapshot"), dict) else {}
    scenario_id = str(payload.get("id") or path.stem).strip()
    if not scenario_id:
        raise ValueError(f"scenario at {path} is missing id")
    return DemoScenario(
        scenario_id=scenario_id,
        name=str(payload.get("name") or scenario_id).strip(),
        summary=str(payload.get("summary") or "").strip(),
        focus=str(payload.get("focus") or "").strip(),
        talk_track=str(payload.get("talk_track") or "").strip(),
        snapshot=snapshot,
        path=path,
    )


def load_scenarios(root: Path | None = None) -> dict[str, DemoScenario]:
    scenario_root = root or _default_scenarios_root()
    scenarios: dict[str, DemoScenario] = {}
    for path in sorted(scenario_root.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        scenario = _normalize_scenario_payload(payload, path)
        scenarios[scenario.scenario_id] = scenario
    if not scenarios:
        raise FileNotFoundError(f"no scenarios found under {scenario_root}")
    return scenarios


def load_scenario(scenario_id: str, root: Path | None = None) -> DemoScenario:
    scenarios = load_scenarios(root=root)
    try:
        return scenarios[str(scenario_id)]
    except KeyError as exc:
        known = ", ".join(sorted(scenarios))
        raise KeyError(f"unknown scenario '{scenario_id}', available: {known}") from exc

