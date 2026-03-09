from __future__ import annotations

if __package__ is None or __package__ == "":
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).resolve().parent.parent))

import argparse
import csv
import json
import random
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from demo.model_inference import MAX_ACTIONS, build_policy_model, load_latest_bundle, recommend_actions
from demo.scenario_loader import load_scenarios
from sim.core.engine import SimEnv


ProgressCallback = Callable[[dict[str, Any]], None]


def _require_torch():
    try:
        import torch
        import torch.nn.functional as F
        from torch.utils.data import DataLoader, Dataset
    except Exception as exc:
        raise RuntimeError("MVP Demo 训练需要 PyTorch") from exc
    return torch, F, Dataset, DataLoader


def now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def default_run_dir() -> Path:
    run_id = now_stamp()
    return Path(__file__).resolve().parent.parent / "docs" / "artifacts" / "mvp" / "model_train" / run_id


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="训练更强的 MVP Demo 手牌策略模型。")
    parser.add_argument("--dataset", default="", help="由 build_mvp_dataset.py 生成的 dataset.jsonl 路径")
    parser.add_argument("--run-dir", default="", help="输出目录，默认写入 docs/artifacts/mvp/model_train/<run_id>")
    parser.add_argument("--epochs", type=int, default=14)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=8e-4)
    parser.add_argument("--weight-decay", type=float, default=2e-4)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--rank-dim", type=int, default=24)
    parser.add_argument("--suit-dim", type=int, default=10)
    parser.add_argument("--card-hidden", type=int, default=128)
    parser.add_argument("--context-hidden", type=int, default=96)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--min-delta", type=float, default=1e-3)
    parser.add_argument("--max-train-samples", type=int, default=0, help="可选。训练前先截断样本数，适合 sweep。")
    parser.add_argument("--progress-json", default="", help="可选。持续写入训练进度 JSON。")
    parser.add_argument("--no-update-latest", action="store_true", help="训练完成后不刷新 latest_run.txt。")
    return parser.parse_args()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fp:
        for line in fp:
            text = line.strip()
            if not text:
                continue
            rows.append(json.loads(text))
    return rows


def _resolve_dataset_path(dataset_arg: str, run_dir: Path) -> Path:
    if dataset_arg:
        return Path(dataset_arg).resolve()
    candidate = run_dir / "dataset.jsonl"
    if candidate.exists():
        return candidate
    latest_runs = sorted(
        [path for path in run_dir.parent.iterdir() if path.is_dir() and (path / "dataset.jsonl").exists()],
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if latest_runs:
        return latest_runs[0] / "dataset.jsonl"
    raise FileNotFoundError("未找到 dataset.jsonl，请先运行 demo/build_mvp_dataset.py")


@dataclass
class Sample:
    rank: list[int]
    suit: list[int]
    chip: list[int]
    enh: list[int]
    edt: list[int]
    seal: list[int]
    pad: list[int]
    context: list[int]
    target: int
    legal_action_ids: list[int]


@dataclass
class TrainConfig:
    dataset: Path | None = None
    run_dir: Path | None = None
    epochs: int = 14
    batch_size: int = 256
    lr: float = 8e-4
    weight_decay: float = 2e-4
    val_ratio: float = 0.15
    seed: int = 17
    device: str = "auto"
    rank_dim: int = 24
    suit_dim: int = 10
    card_hidden: int = 128
    context_hidden: int = 96
    hidden_dim: int = 256
    dropout: float = 0.1
    patience: int = 4
    min_delta: float = 1e-3
    max_train_samples: int = 0
    progress_json: Path | None = None
    update_latest_hint: bool = True
    model_name: str = "mvp_hand_policy_v2"


def _sample_from_record(record: dict[str, Any]) -> Sample:
    features = record["features"]
    return Sample(
        rank=list(features["card_rank_ids"]),
        suit=list(features["card_suit_ids"]),
        chip=list(features.get("card_chip_hint") or [0] * 10),
        enh=list(features["card_has_enhancement"]),
        edt=list(features["card_has_edition"]),
        seal=list(features["card_has_seal"]),
        pad=list(features["hand_pad_mask"]),
        context=list(features["context"]),
        target=int(record["expert_action_id"]),
        legal_action_ids=[int(aid) for aid in record["legal_action_ids"]],
    )


def _split_samples(samples: list[Sample], val_ratio: float, seed: int) -> tuple[list[Sample], list[Sample]]:
    rng = random.Random(seed)
    shuffled = list(samples)
    rng.shuffle(shuffled)
    if len(shuffled) <= 1:
        return shuffled, shuffled
    cut = int(len(shuffled) * (1.0 - val_ratio))
    cut = max(1, min(cut, len(shuffled) - 1))
    return shuffled[:cut], shuffled[cut:]


def _write_progress(path: Path | None, payload: dict[str, Any]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def evaluate_demo_scenarios(run_dir: Path) -> dict[str, Any]:
    bundle = load_latest_bundle(run_dir=run_dir)
    scenarios = load_scenarios()
    rows: list[dict[str, Any]] = []
    for scenario in scenarios.values():
        env = SimEnv(seed=scenario.scenario_id.upper())
        env.reset(from_snapshot=scenario.snapshot)
        payload = recommend_actions(env.get_state(), env=env, policy="model", topk=3, bundle=bundle)
        top = (payload.get("recommendations") or [{}])[0]
        rows.append(
            {
                "scenario_id": scenario.scenario_id,
                "scenario_name": scenario.name,
                "top_label": str(top.get("label") or ""),
                "source": str(top.get("source") or ""),
                "score": float(top.get("score") or 0.0),
                "confidence": float(top.get("confidence") or 0.0),
                "teacher_agrees": bool(top.get("teacher_agrees")),
                "expected_score": float(((top.get("preview") or {}).get("expected_score") or 0.0)),
                "reason": str(top.get("reason") or ""),
            }
        )
    report = {"schema": "mvp_demo_scenario_eval_v1", "run_dir": str(run_dir), "scenarios": rows}
    (run_dir / "scenario_eval.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def train_model(config: TrainConfig, progress_callback: ProgressCallback | None = None) -> dict[str, Any]:
    run_dir = Path(config.run_dir).resolve() if config.run_dir else default_run_dir()
    run_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = _resolve_dataset_path(str(config.dataset or ""), run_dir)
    dataset_stats_source = dataset_path.with_name("dataset_stats.json")
    progress_json = Path(config.progress_json).resolve() if config.progress_json else None

    torch, F, Dataset, DataLoader = _require_torch()
    raw_records = _read_jsonl(dataset_path)
    if not raw_records:
        raise RuntimeError(f"dataset is empty: {dataset_path}")

    if int(config.max_train_samples) > 0 and len(raw_records) > int(config.max_train_samples):
        rng = random.Random(int(config.seed))
        raw_records = rng.sample(raw_records, int(config.max_train_samples))

    samples = [_sample_from_record(record) for record in raw_records]
    train_samples, val_samples = _split_samples(samples, val_ratio=float(config.val_ratio), seed=int(config.seed))

    random.seed(int(config.seed))
    torch.manual_seed(int(config.seed))

    class _SamplesDataset(Dataset):
        def __init__(self, rows: list[Sample]):
            self.rows = rows

        def __len__(self) -> int:
            return len(self.rows)

        def __getitem__(self, index: int) -> Sample:
            return self.rows[index]

    def _collate(batch: list[Sample]) -> dict[str, Any]:
        legal_mask = torch.zeros((len(batch), MAX_ACTIONS), dtype=torch.float32)
        for row_idx, sample in enumerate(batch):
            for action_id in sample.legal_action_ids:
                if 0 <= int(action_id) < MAX_ACTIONS:
                    legal_mask[row_idx, int(action_id)] = 1.0
        return {
            "rank": torch.tensor([sample.rank for sample in batch], dtype=torch.long),
            "suit": torch.tensor([sample.suit for sample in batch], dtype=torch.long),
            "chip": torch.tensor([sample.chip for sample in batch], dtype=torch.float32),
            "enh": torch.tensor([sample.enh for sample in batch], dtype=torch.float32),
            "edt": torch.tensor([sample.edt for sample in batch], dtype=torch.float32),
            "seal": torch.tensor([sample.seal for sample in batch], dtype=torch.float32),
            "pad": torch.tensor([sample.pad for sample in batch], dtype=torch.float32),
            "context": torch.tensor([sample.context for sample in batch], dtype=torch.float32),
            "targets": torch.tensor([sample.target for sample in batch], dtype=torch.long),
            "legal_mask": legal_mask,
        }

    train_loader = DataLoader(_SamplesDataset(train_samples), batch_size=max(8, int(config.batch_size)), shuffle=True, collate_fn=_collate)
    val_loader = DataLoader(_SamplesDataset(val_samples), batch_size=max(8, int(config.batch_size)), shuffle=False, collate_fn=_collate)

    if config.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(config.device)

    import torch.nn as nn

    model = build_policy_model(
        nn,
        {
            "arch": "mlp_v2",
            "max_actions": MAX_ACTIONS,
            "rank_dim": int(config.rank_dim),
            "suit_dim": int(config.suit_dim),
            "card_hidden": int(config.card_hidden),
            "context_hidden": int(config.context_hidden),
            "hidden_dim": int(config.hidden_dim),
            "dropout": float(config.dropout),
        },
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config.lr), weight_decay=float(config.weight_decay))

    def emit(update: dict[str, Any]) -> None:
        payload = {
            "schema": "mvp_train_progress_v2",
            "stage": "training",
            "status": "running",
            "run_dir": str(run_dir),
            "dataset_path": str(dataset_path),
            "device": str(device),
            "train_samples": len(train_samples),
            "val_samples": len(val_samples),
            **update,
        }
        _write_progress(progress_json, payload)
        if progress_callback is not None:
            progress_callback(payload)

    def _masked_loss(logits, targets, legal_mask):
        masked_logits = torch.where(legal_mask > 0, logits, torch.full_like(logits, -1e9))
        return F.cross_entropy(masked_logits, targets), masked_logits

    def _evaluate(loader) -> dict[str, float]:
        model.eval()
        total = 0
        total_loss = 0.0
        correct1 = 0
        correct3 = 0
        with torch.no_grad():
            for batch in loader:
                batch = {key: value.to(device) for key, value in batch.items()}
                logits = model(batch)
                loss, masked_logits = _masked_loss(logits, batch["targets"], batch["legal_mask"])
                total += int(batch["targets"].shape[0])
                total_loss += float(loss.item()) * int(batch["targets"].shape[0])
                top1 = masked_logits.argmax(dim=1)
                correct1 += int((top1 == batch["targets"]).sum().item())
                topk = torch.topk(masked_logits, k=min(3, masked_logits.shape[1]), dim=1).indices
                for idx in range(int(topk.shape[0])):
                    if int(batch["targets"][idx].item()) in topk[idx].tolist():
                        correct3 += 1
        return {
            "loss": total_loss / max(1, total),
            "acc1": correct1 / max(1, total),
            "acc3": correct3 / max(1, total),
        }

    history: list[dict[str, Any]] = []
    best_val = None
    best_epoch = 0
    bad_epochs = 0
    early_stopped = False
    best_path = run_dir / "mvp_policy.pt"
    started_at = time.time()
    emit({"message": "开始训练模型。", "epoch": 0, "epochs_total": int(config.epochs), "progress": 0.0})

    for epoch in range(1, max(1, int(config.epochs)) + 1):
        epoch_started = time.time()
        model.train()
        train_loss_sum = 0.0
        train_items = 0
        online_correct1 = 0
        online_correct3 = 0
        for batch in train_loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            logits = model(batch)
            loss, masked_logits = _masked_loss(logits, batch["targets"], batch["legal_mask"])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_items += int(batch["targets"].shape[0])
            train_loss_sum += float(loss.item()) * int(batch["targets"].shape[0])
            top1 = masked_logits.argmax(dim=1)
            online_correct1 += int((top1 == batch["targets"]).sum().item())
            topk = torch.topk(masked_logits, k=min(3, masked_logits.shape[1]), dim=1).indices
            for idx in range(int(topk.shape[0])):
                if int(batch["targets"][idx].item()) in topk[idx].tolist():
                    online_correct3 += 1

        val_metrics = _evaluate(val_loader)
        row = {
            "epoch": epoch,
            "train_loss": train_loss_sum / max(1, train_items),
            "train_acc1": online_correct1 / max(1, train_items),
            "train_acc3": online_correct3 / max(1, train_items),
            "val_loss": val_metrics["loss"],
            "val_acc1": val_metrics["acc1"],
            "val_acc3": val_metrics["acc3"],
            "epoch_sec": time.time() - epoch_started,
        }
        history.append(row)
        if best_val is None or float(val_metrics["loss"]) < float(best_val) - float(config.min_delta):
            best_val = float(val_metrics["loss"])
            best_epoch = epoch
            bad_epochs = 0
            torch.save(model.state_dict(), best_path)
        else:
            bad_epochs += 1

        elapsed = time.time() - started_at
        avg_epoch_sec = elapsed / max(1, epoch)
        eta_sec = max(0.0, avg_epoch_sec * (int(config.epochs) - epoch))
        emit(
            {
                "message": f"第 {epoch} 轮完成。",
                "epoch": epoch,
                "epochs_total": int(config.epochs),
                "progress": round(epoch / max(1, int(config.epochs)), 4),
                "elapsed_sec": round(elapsed, 2),
                "eta_sec": round(eta_sec, 2),
                "best_val_loss": best_val,
                "best_epoch": best_epoch,
                "bad_epochs": bad_epochs,
                "history": history,
                "metrics": row,
            }
        )

        if bad_epochs >= max(1, int(config.patience)):
            early_stopped = True
            break

    metrics = {
        "schema": "mvp_demo_train_metrics_v2",
        "dataset_path": str(dataset_path),
        "train_samples": len(train_samples),
        "val_samples": len(val_samples),
        "best_val_loss": best_val,
        "best_epoch": best_epoch,
        "early_stopped": early_stopped,
        "history": history,
        "final": history[-1],
        "duration_sec": round(time.time() - started_at, 2),
    }
    (run_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    with (run_dir / "loss_curve.csv").open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=["epoch", "train_loss", "val_loss", "val_acc1", "val_acc3"])
        writer.writeheader()
        for row in history:
            writer.writerow(
                {
                    "epoch": row["epoch"],
                    "train_loss": row["train_loss"],
                    "val_loss": row["val_loss"],
                    "val_acc1": row["val_acc1"],
                    "val_acc3": row["val_acc3"],
                }
            )

    model_config = {
        "schema": "mvp_demo_model_config_v2",
        "model_name": config.model_name,
        "checkpoint_name": "mvp_policy.pt",
        "arch": "mlp_v2",
        "max_actions": MAX_ACTIONS,
        "dataset_path": str(dataset_path),
        "train_samples": len(train_samples),
        "val_samples": len(val_samples),
        "device_used": str(device),
        "rank_dim": int(config.rank_dim),
        "suit_dim": int(config.suit_dim),
        "card_hidden": int(config.card_hidden),
        "context_hidden": int(config.context_hidden),
        "hidden_dim": int(config.hidden_dim),
        "dropout": float(config.dropout),
        "epochs_requested": int(config.epochs),
        "epochs_completed": len(history),
        "best_epoch": best_epoch,
        "patience": int(config.patience),
        "lr": float(config.lr),
        "weight_decay": float(config.weight_decay),
        "seed": int(config.seed),
        "max_train_samples": int(config.max_train_samples),
        "architecture": {
            "max_actions": MAX_ACTIONS,
            "rank_dim": int(config.rank_dim),
            "suit_dim": int(config.suit_dim),
            "numeric_dim": 24,
            "card_hidden": int(config.card_hidden),
            "context_hidden": int(config.context_hidden),
            "hidden_dim": int(config.hidden_dim),
            "dropout": float(config.dropout),
        },
    }
    (run_dir / "model_config.json").write_text(json.dumps(model_config, ensure_ascii=False, indent=2), encoding="utf-8")
    if dataset_stats_source.exists():
        (run_dir / "dataset_stats.json").write_text(dataset_stats_source.read_text(encoding="utf-8"), encoding="utf-8")
    (run_dir / "checkpoint_path.txt").write_text(str(best_path), encoding="utf-8")

    summary = "\n".join(
        [
            f"# MVP 模型训练 {run_dir.name}",
            "",
            f"- 数据集：`{dataset_path}`",
            f"- 训练样本：`{len(train_samples)}`",
            f"- 验证样本：`{len(val_samples)}`",
            f"- 最优验证损失：`{best_val:.4f}`" if best_val is not None else "- 最优验证损失：`n/a`",
            f"- 最优轮次：`{best_epoch}`",
            f"- 最终 Top-1：`{history[-1]['val_acc1']:.4f}`",
            f"- 最终 Top-3：`{history[-1]['val_acc3']:.4f}`",
            f"- 设备：`{device}`",
            f"- 早停：`{'是' if early_stopped else '否'}`",
            f"- Checkpoint：`{best_path}`",
        ]
    )
    (run_dir / "training_summary.md").write_text(summary + "\n", encoding="utf-8")

    scenario_eval = evaluate_demo_scenarios(run_dir)

    if bool(config.update_latest_hint):
        latest_hint = run_dir.parent / "latest_run.txt"
        latest_hint.write_text(run_dir.name, encoding="utf-8")

    result = {
        "run_dir": str(run_dir),
        "checkpoint": str(best_path),
        "best_val_loss": best_val,
        "best_epoch": best_epoch,
        "history": history,
        "scenario_eval": scenario_eval,
    }
    emit(
        {
            "status": "finished",
            "message": "训练完成，已生成最佳 checkpoint 与场景评估结果。",
            "epoch": len(history),
            "epochs_total": int(config.epochs),
            "progress": 1.0,
            "best_val_loss": best_val,
            "best_epoch": best_epoch,
            "history": history,
            "scenario_eval": scenario_eval,
            "result": result,
        }
    )
    return result


def main() -> int:
    args = parse_args()
    config = TrainConfig(
        dataset=Path(args.dataset).resolve() if args.dataset else None,
        run_dir=Path(args.run_dir).resolve() if args.run_dir else None,
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
        val_ratio=float(args.val_ratio),
        seed=int(args.seed),
        device=str(args.device),
        rank_dim=int(args.rank_dim),
        suit_dim=int(args.suit_dim),
        card_hidden=int(args.card_hidden),
        context_hidden=int(args.context_hidden),
        hidden_dim=int(args.hidden_dim),
        dropout=float(args.dropout),
        patience=int(args.patience),
        min_delta=float(args.min_delta),
        max_train_samples=int(args.max_train_samples),
        progress_json=Path(args.progress_json).resolve() if args.progress_json else None,
        update_latest_hint=not bool(args.no_update_latest),
    )
    result = train_model(config)
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
