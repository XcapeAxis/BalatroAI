from __future__ import annotations

if __package__ is None or __package__ == "":
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).resolve().parent.parent))

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import requests


def now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a smoke test against the MVP demo API.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8050")
    parser.add_argument("--out", default="", help="Optional output json path")
    return parser.parse_args()


def _request(session: requests.Session, method: str, url: str, payload: dict[str, Any] | None = None) -> Any:
    response = session.request(method=method, url=url, json=payload, timeout=15)
    response.raise_for_status()
    return response.json()


def main() -> int:
    args = parse_args()
    base_url = str(args.base_url).rstrip("/")
    out_path = Path(args.out).resolve() if args.out else (Path(__file__).resolve().parent.parent / "docs" / "artifacts" / "mvp" / f"api_smoke_{now_stamp()}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with requests.Session() as session:
        health = _request(session, "GET", f"{base_url}/api/health")
        scenarios = _request(session, "GET", f"{base_url}/api/scenarios")
        model_info = _request(session, "GET", f"{base_url}/api/model_info")
        initial_state = _request(session, "GET", f"{base_url}/api/state")

        recommend_basic = _request(session, "POST", f"{base_url}/api/recommend", {"policy": "model", "topk": 3})
        step_basic = _request(session, "POST", f"{base_url}/api/step", {"policy": "model"})
        load_risk = _request(session, "POST", f"{base_url}/api/scenario/load", {"scenario_id": "high_risk_discard"})
        autoplay_risk = _request(session, "POST", f"{base_url}/api/autoplay", {"policy": "model", "steps": 2})
        load_joker = _request(session, "POST", f"{base_url}/api/scenario/load", {"scenario_id": "joker_synergy"})
        recommend_joker = _request(session, "POST", f"{base_url}/api/recommend", {"policy": "model", "topk": 3})

    payload = {
        "schema": "mvp_api_smoke_v1",
        "base_url": base_url,
        "health": health,
        "scenario_count": len(scenarios.get("scenarios") or []),
        "model_loaded": bool(model_info.get("loaded")),
        "initial_phase": initial_state.get("phase"),
        "basic_recommendation": (recommend_basic.get("recommendations") or [None])[0],
        "basic_step_phase_after": ((step_basic.get("state") or {}).get("phase")),
        "risk_phase_after_autoplay": ((autoplay_risk.get("state") or {}).get("phase")),
        "risk_steps_executed": autoplay_risk.get("steps_executed"),
        "joker_recommendation": (recommend_joker.get("recommendations") or [None])[0],
        "loaded_scenarios": {
            "risk": (load_risk.get("scenario") or {}).get("id"),
            "joker": (load_joker.get("scenario") or {}).get("id"),
        },
    }
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"out": str(out_path), "scenario_count": payload["scenario_count"], "model_loaded": payload["model_loaded"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

