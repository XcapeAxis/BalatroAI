from __future__ import annotations

if __package__ is None or __package__ == "":
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sim.pybind.sim_env import SimEnvBackend
from trainer import action_space, action_space_shop
from trainer.closed_loop.replay_manifest import write_json
from trainer.policy_arena.adapters.heuristic_adapter import HeuristicAdapter
from trainer.policy_arena.adapters.model_adapter import ModelAdapter
from trainer.policy_arena.adapters.search_adapter import SearchAdapter
from trainer.policy_arena.policy_adapter import BasePolicyAdapter, normalize_action, phase_default_action, phase_from_obs
from trainer.world_model.schema import action_token_from_parts, make_sample_id, stable_hash_int


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _parse_csv(text: str) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for part in str(text or "").split(","):
        token = part.strip()
        if not token or token in seen:
            continue
        seen.add(token)
        out.append(token)
    return out


def _legal_actions_hint(state: dict[str, Any]) -> list[dict[str, Any]] | None:
    phase = phase_from_obs(state)
    if phase == "SELECTING_HAND":
        hand_cards = (state.get("hand") or {}).get("cards") if isinstance(state.get("hand"), dict) else []
        hand_size = min(len(hand_cards or []), action_space.MAX_HAND)
        if hand_size <= 0:
            return []
        out: list[dict[str, Any]] = []
        for aid in action_space.legal_action_ids(hand_size)[:64]:
            action_type, mask = action_space.decode(hand_size, int(aid))
            out.append({"action_type": action_type, "indices": action_space.mask_to_indices(mask, hand_size), "id": int(aid)})
        return out
    if phase in action_space_shop.SHOP_PHASES:
        out = []
        for aid in action_space_shop.legal_action_ids(state)[:32]:
            out.append({"id": int(aid), "action": action_space_shop.action_from_id(state, int(aid))})
        return out
    return None


def resolve_candidate_source(source: str) -> str:
    token = str(source or "heuristic_candidates").strip().lower()
    if token in {"heuristic", "heuristic_candidates", "rule"}:
        return "heuristic_candidates"
    if token in {"search", "search_candidates", "search_expert"}:
        return "search_candidates"
    if token in {"policy", "policy_topk", "model", "model_policy"}:
        return "policy_topk"
    return "heuristic_candidates"


def _build_adapter(
    source: str,
    *,
    model_path: str = "",
    search_max_branch: int = 80,
    search_max_depth: int = 2,
    search_time_budget_ms: float = 15.0,
) -> BasePolicyAdapter:
    resolved = resolve_candidate_source(source)
    if resolved == "search_candidates":
        return SearchAdapter(
            name="p47_search_candidates",
            max_branch=int(search_max_branch),
            max_depth=int(search_max_depth),
            time_budget_ms=float(search_time_budget_ms),
        )
    if resolved == "policy_topk":
        return ModelAdapter(name="p47_policy_topk", model_path=str(model_path or ""))
    return HeuristicAdapter(name="p47_heuristic_candidates")


def capture_sample_state(
    *,
    seed: str = "AAAAAAA",
    target_phase: str = "SELECTING_HAND",
    max_steps: int = 12,
) -> dict[str, Any]:
    backend = SimEnvBackend(seed=seed)
    state = backend.reset(seed=seed)
    if phase_from_obs(state) == target_phase:
        return state
    for _idx in range(max(1, int(max_steps))):
        phase = phase_from_obs(state)
        if phase == target_phase:
            return state
        action = phase_default_action(state, seed=seed)
        state, _reward, done, _info = backend.step(action)
        if bool(done):
            break
    return state


def _normalize_candidate_rows(
    obs: dict[str, Any],
    rows: list[dict[str, Any]],
    *,
    source: str,
    top_k: int,
    dedupe: bool,
) -> list[dict[str, Any]]:
    phase = phase_from_obs(obs)
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in rows:
        if not isinstance(row, dict):
            continue
        action = normalize_action(row.get("action") if isinstance(row.get("action"), dict) else {}, phase=phase)
        action_numeric = _safe_int(action.get("id"), -1) if action.get("id") is not None else None
        token = action_token_from_parts(
            phase=phase,
            action_type=str(action.get("action_type") or "OTHER"),
            action_payload=action,
            numeric_action=action_numeric if action_numeric is not None and action_numeric >= 0 else None,
        )
        if dedupe and token in seen:
            continue
        seen.add(token)
        candidate_id = make_sample_id([source, phase, token, row.get("source_rank"), row.get("source_score")])
        out.append(
            {
                "candidate_id": candidate_id,
                "action": action,
                "action_token": token,
                "action_id": stable_hash_int(token, 2**31 - 1),
                "source": str(row.get("source") or source),
                "source_rank": _safe_int(row.get("source_rank"), len(out) + 1),
                "source_score": float(row.get("source_score") or 0.0),
                "legal": bool(row.get("legal", True)),
                "metadata": dict(row.get("metadata") or {}),
            }
        )
        if len(out) >= max(1, int(top_k)):
            break
    return out


def generate_candidate_actions(
    *,
    obs: dict[str, Any],
    source: str,
    top_k: int = 4,
    legal_actions: list[dict[str, Any]] | None = None,
    seed: str = "AAAAAAA",
    model_path: str = "",
    search_max_branch: int = 80,
    search_max_depth: int = 2,
    search_time_budget_ms: float = 15.0,
    dedupe: bool = True,
) -> dict[str, Any]:
    adapter = _build_adapter(
        source,
        model_path=model_path,
        search_max_branch=search_max_branch,
        search_max_depth=search_max_depth,
        search_time_budget_ms=search_time_budget_ms,
    )
    adapter.reset(seed=seed)
    try:
        raw_rows = adapter.candidate_actions(obs, legal_actions=legal_actions, top_k=max(1, int(top_k)))
    finally:
        adapter.close()
    candidates = _normalize_candidate_rows(
        obs,
        raw_rows if isinstance(raw_rows, list) else [],
        source=resolve_candidate_source(source),
        top_k=top_k,
        dedupe=dedupe,
    )
    return {
        "schema": "p47_candidate_action_set_v1",
        "generated_at": _now_iso(),
        "phase": phase_from_obs(obs),
        "seed": str(seed),
        "source": resolve_candidate_source(source),
        "top_k": max(1, int(top_k)),
        "candidate_count": len(candidates),
        "candidates": candidates,
    }


def _discover_model_path(repo_root: Path) -> str:
    candidates: list[Path] = []
    for root in (
        repo_root / "docs/artifacts/p46/imagination_pipeline",
        repo_root / "docs/artifacts/p22/runs",
        repo_root / "docs/artifacts/p41/candidate_train",
    ):
        if not root.exists():
            continue
        candidates.extend(root.glob("**/best.pt"))
    ordered = sorted({path.resolve() for path in candidates}, key=lambda path: str(path))
    return str(ordered[-1]) if ordered else ""


def run_candidate_smoke(
    *,
    out_dir: str | Path | None = None,
    seed: str = "AAAAAAA",
    top_k: int = 4,
    sources: list[str] | None = None,
    model_path: str = "",
) -> dict[str, Any]:
    repo_root = Path(__file__).resolve().parents[2]
    output_root = (
        (repo_root / "docs/artifacts/p47").resolve()
        if out_dir is None
        else (Path(out_dir) if Path(out_dir).is_absolute() else (repo_root / out_dir).resolve())
    )
    output_root.mkdir(parents=True, exist_ok=True)
    resolved_sources = sources or ["heuristic_candidates", "search_candidates"]
    resolved_model_path = str(model_path or _discover_model_path(repo_root))
    state = capture_sample_state(seed=seed, target_phase="SELECTING_HAND", max_steps=12)
    legal_actions = _legal_actions_hint(state)

    rows: list[dict[str, Any]] = []
    for source in resolved_sources:
        payload = generate_candidate_actions(
            obs=state,
            source=source,
            top_k=top_k,
            legal_actions=legal_actions,
            seed=seed,
            model_path=resolved_model_path,
        )
        rows.append(payload)

    output = {
        "schema": "p47_candidate_action_smoke_v1",
        "generated_at": _now_iso(),
        "seed": seed,
        "phase": phase_from_obs(state),
        "state_summary": {
            "round_num": _safe_int(state.get("round_num"), 0),
            "ante_num": _safe_int(state.get("ante_num"), 0),
            "money": _safe_int(state.get("money"), 0),
        },
        "sources": rows,
        "model_path": resolved_model_path,
        "status": "ok" if rows else "stub",
    }
    out_path = output_root / f"candidate_smoke_{_now_stamp()}.json"
    write_json(out_path, output)
    return {"status": str(output.get("status") or "stub"), "out": str(out_path), "payload": output}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="P47 candidate action smoke and helper.")
    parser.add_argument("--seed", default="AAAAAAA")
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--sources", default="heuristic_candidates,search_candidates")
    parser.add_argument("--model-path", default="")
    parser.add_argument("--out-dir", default="docs/artifacts/p47")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = run_candidate_smoke(
        out_dir=args.out_dir,
        seed=str(args.seed or "AAAAAAA"),
        top_k=max(1, int(args.top_k)),
        sources=_parse_csv(args.sources),
        model_path=str(args.model_path or ""),
    )
    print(json.dumps(summary, ensure_ascii=False))
    return 0 if str(summary.get("status")) in {"ok", "stub"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
