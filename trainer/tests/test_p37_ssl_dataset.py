from __future__ import annotations

from trainer.experiments.ssl_dataset import build_ssl_pair_samples, split_pair_samples


def _sample_rows() -> list[dict]:
    base_state = {"phase": "SELECTING_HAND", "action_type": "PLAY", "vector": [0.2] * 13}
    aux = {"score_delta_t": 12.0, "reward_t": 6.0, "stake": "white", "source": "oracle_trace"}
    rows = []
    for idx, delta in enumerate([20.0, 0.0, -10.0]):
        rows.append(
            {
                "state": base_state,
                "aux": aux,
                "future": {
                    "delta_chips_k": delta,
                    "terminal_within_k": 0,
                    "next_state_vector": [0.3] * 13,
                    "next_action_type": "PLAY",
                },
                "meta": {"step_idx": idx},
            }
        )
    return rows


def test_build_ssl_pair_samples_shape() -> None:
    rows = _sample_rows()
    pair_rows, meta = build_ssl_pair_samples(rows)
    assert len(pair_rows) == 3
    assert int(meta["input_dim"]) == 15
    assert len(pair_rows[0].obs) == 15
    assert len(pair_rows[0].next_obs) == 15
    assert sorted([int(x.reward_bucket) for x in pair_rows]) == [0, 1, 2]


def test_split_pair_samples_non_empty() -> None:
    rows = _sample_rows()
    pair_rows, _ = build_ssl_pair_samples(rows)
    train_rows, val_rows = split_pair_samples(pair_rows, train_ratio=0.8, seed=3701)
    assert train_rows
    assert val_rows
