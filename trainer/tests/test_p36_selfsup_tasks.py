from __future__ import annotations

from trainer.selfsup.tasks import (
    SelfSupActionTypeTask,
    SelfSupFutureValueTask,
    build_action_type_batch,
    build_future_value_batch,
)


def _sample_rows() -> list[dict]:
    base_state = {"phase": "SELECTING_HAND", "action_type": "PLAY", "vector": [0.1] * 13}
    aux = {"score_delta_t": 10.0, "reward_t": 5.0, "stake": "white", "source": "oracle_trace"}
    rows = []
    for idx, next_action in enumerate(["PLAY", "DISCARD", "SHOP_BUY"]):
        rows.append(
            {
                "state": base_state,
                "aux": aux,
                "future": {
                    "delta_chips_k": float(idx + 1),
                    "terminal_within_k": 0,
                    "next_state_vector": [0.2] * 13,
                    "next_action_type": next_action,
                },
                "meta": {"step_idx": idx},
            }
        )
    return rows


def test_future_value_task_shape() -> None:
    rows = _sample_rows()
    batch = build_future_value_batch(rows)
    assert len(batch.features) == 3
    assert len(batch.features[0]) == 15

    import torch

    task = SelfSupFutureValueTask(input_dim=len(batch.features[0]), latent_dim=16, hidden_dim=32, dropout=0.0)
    x = torch.tensor(batch.features, dtype=torch.float32)
    y = task.forward(x)
    assert list(y.shape) == [3]


def test_action_type_task_shape() -> None:
    rows = _sample_rows()
    batch = build_action_type_batch(rows)
    assert len(batch.features_t) == 3
    assert len(batch.features_tp1[0]) == len(batch.features_t[0])

    import torch

    task = SelfSupActionTypeTask(
        input_dim=len(batch.features_t[0]),
        num_classes=len(batch.label_vocab),
        latent_dim=16,
        hidden_dim=32,
        dropout=0.0,
    )
    x_t = torch.tensor(batch.features_t, dtype=torch.float32)
    x_tp1 = torch.tensor(batch.features_tp1, dtype=torch.float32)
    logits = task.forward(x_t, x_tp1)
    assert list(logits.shape) == [3, len(batch.label_vocab)]

