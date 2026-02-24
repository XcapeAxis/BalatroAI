from __future__ import annotations

if __package__ is None or __package__ == "":
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).resolve().parent.parent))

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import streamlit as st

from trainer import action_space_shop
from trainer.env_client import create_backend
from trainer.infer_assistant_real import _heuristic_hand_rankings, _heuristic_shop_rankings, _latest_model
from trainer.real_observer import build_observation


def _ui_log_path() -> Path:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    path = Path("docs/artifacts/p12/ui_logs") / stamp / "ui.log"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _write_log(path: Path, payload: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


st.set_page_config(page_title="Balatro Assistant", layout="wide")
st.title("Balatro Real Assistant")

if "ui_log_path" not in st.session_state:
    st.session_state["ui_log_path"] = str(_ui_log_path())

default_model = _latest_model()
base_url = st.sidebar.text_input("Base URL", value="http://127.0.0.1:12346")
model_path = st.sidebar.text_input("Model Path", value=str(default_model) if default_model else "")
topk = int(st.sidebar.number_input("Top-K", min_value=1, max_value=10, value=3))
enable_execute = bool(st.sidebar.checkbox("Enable Execute", value=False))

refresh = st.button("Refresh State")
execute_top1 = st.button("Execute Top-1")

backend = create_backend("real", base_url=base_url, timeout_sec=8.0, seed="AAAAAAA")

try:
    state = backend.get_state()
    obs = build_observation(state)
    phase = str(state.get("state") or "UNKNOWN")
except Exception as exc:
    st.error(f"Failed to fetch state from {base_url}: {exc}")
    st.stop()

st.subheader("Current State")
left, right = st.columns(2)
with left:
    st.json(
        {
            "phase": obs["phase"],
            "hands_left": obs["resources"]["hands_left"],
            "discards_left": obs["resources"]["discards_left"],
            "money": obs["resources"]["money"],
            "ante": obs["resources"]["ante"],
            "round_num": obs["resources"]["round_num"],
        }
    )
with right:
    st.write(f"Hand size: {obs['hand']['hand_size']}")
    st.dataframe(obs["hand"]["cards"], use_container_width=True)

if phase in action_space_shop.SHOP_PHASES:
    st.subheader("Shop Offers")
    st.write("Shop")
    st.dataframe(obs["shop"]["shop"]["cards"], use_container_width=True)
    st.write("Vouchers")
    st.dataframe(obs["shop"]["vouchers"]["cards"], use_container_width=True)
    st.write("Packs")
    st.dataframe(obs["shop"]["packs"]["cards"], use_container_width=True)
    st.write("Consumables")
    st.dataframe(obs["shop"]["consumables"]["cards"], use_container_width=True)

st.subheader("Top-K Suggestions")
if phase == "SELECTING_HAND":
    ranked = _heuristic_hand_rankings(state, topk=topk)
    st.dataframe(ranked, use_container_width=True)
    top_action = None
    if ranked:
        top_action = {"action_type": str(ranked[0]["action_type"]), "indices": list(ranked[0]["indices"])}
elif phase in action_space_shop.SHOP_PHASES:
    ranked_shop = _heuristic_shop_rankings(state, topk=topk)
    st.dataframe(ranked_shop, use_container_width=True)
    top_action = dict(ranked_shop[0]["action"]) if ranked_shop else None
else:
    st.info(f"Phase {phase} currently does not produce executable suggestions.")
    top_action = None

st.caption(f"Model path (display only in this UI build): {model_path or 'heuristic-only'}")
st.caption(f"UI log: {st.session_state['ui_log_path']}")

if refresh:
    _write_log(
        Path(st.session_state["ui_log_path"]),
        {"event": "refresh", "phase": phase, "base_url": base_url, "top_action": top_action},
    )
    st.rerun()

if execute_top1:
    if not enable_execute:
        st.warning("Enable Execute is OFF. Toggle it on to send actions.")
    elif not top_action:
        st.warning("No executable top action for current phase.")
    else:
        try:
            after, reward, done, info = backend.step(top_action)
            st.success(f"Executed: {top_action} reward={reward:.4f} done={done}")
            st.json({"after_phase": str(after.get("state") or "UNKNOWN"), "info": info})
            _write_log(
                Path(st.session_state["ui_log_path"]),
                {
                    "event": "execute",
                    "phase_before": phase,
                    "action": top_action,
                    "reward": reward,
                    "done": done,
                    "after_phase": str(after.get("state") or "UNKNOWN"),
                },
            )
        except Exception as exc:
            st.error(f"Execute failed: {exc}")
            _write_log(Path(st.session_state["ui_log_path"]), {"event": "execute_error", "error": str(exc)})

backend.close()
