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

from demo.build_mvp_dataset import BuildDatasetConfig, build_dataset
from demo.model_inference import load_latest_bundle, recommend_actions
from demo.scenario_loader import load_scenarios
from demo.train_mvp_model import TrainConfig, train_model
from demo.training_status import infer_profile, patch_status, read_status, write_status
from sim.core.engine import SimEnv


def now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="运行 MVP-S2 训练流水线。")
    parser.add_argument("--status-path", default="", help="训练状态 JSON 输出路径。")
    parser.add_argument("--job-id", default="", help="训练任务标识。")
    parser.add_argument("--profile", default="", help="训练配置名称，例如 smoke / fast / standard。")
    parser.add_argument("--budget-minutes", type=int, default=120)
    parser.add_argument("--episodes", type=int, default=1200)
    parser.add_argument("--max-steps", type=int, default=40)
    parser.add_argument("--scenario-copies", type=int, default=192)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--final-epochs", type=int, default=16)
    parser.add_argument("--sweep-epochs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--run-root", default="", help="模型产物根目录。")
    return parser.parse_args()


def _scenario_eval(run_dir: Path) -> dict[str, Any]:
    scenarios = load_scenarios()
    bundle = load_latest_bundle(run_dir=run_dir)
    results: list[dict[str, Any]] = []
    agree_count = 0
    for scenario in scenarios.values():
        env = SimEnv(seed=scenario.scenario_id.upper())
        env.reset(from_snapshot=scenario.snapshot)
        payload = recommend_actions(env.get_state(), env=env, policy="model", topk=3, bundle=bundle)
        top = (payload.get("recommendations") or [{}])[0]
        agree = bool(top.get("teacher_agrees"))
        if agree:
            agree_count += 1
        results.append(
            {
                "scenario_id": scenario.scenario_id,
                "scenario_name": scenario.name,
                "top1_label": top.get("label") or "",
                "teacher_agrees": agree,
                "confidence": float(top.get("confidence") or 0.0),
                "expected_score": float(((top.get("preview") or {}).get("expected_score")) or 0.0),
                "reason": top.get("reason") or "",
            }
        )
    return {
        "schema": "mvp_demo_scenario_eval_v1",
        "run_dir": str(run_dir),
        "scenario_count": len(results),
        "teacher_agreement_count": agree_count,
        "results": results,
    }


def main() -> int:
    args = parse_args()
    project_root = Path(__file__).resolve().parent.parent
    model_root = project_root / "docs" / "artifacts" / "mvp" / "model_train"
    status_path = Path(args.status_path).resolve() if args.status_path else (project_root / "docs" / "artifacts" / "mvp" / "training_status" / "latest.json")
    status_path.parent.mkdir(parents=True, exist_ok=True)

    previous_bundle = load_latest_bundle()
    previous_run_dir = str(previous_bundle.run_dir) if previous_bundle.run_dir else ""
    previous_final = dict(previous_bundle.metrics.get("final") or {})
    existing_status = read_status(status_path)

    job_id = str(args.job_id or f"s2_{now_stamp()}")
    run_root = Path(args.run_root).resolve() if args.run_root else model_root
    run_root.mkdir(parents=True, exist_ok=True)
    dataset_run_dir = run_root / f"{job_id}_dataset"
    final_dir = run_root / f"{job_id}_final"
    sweep_root = run_root / f"{job_id}_sweep"
    progress_events: list[dict[str, Any]] = []

    def remember_progress(payload: dict[str, Any]) -> list[dict[str, Any]]:
        progress_events.append(dict(payload))
        return progress_events[-80:]

    write_status(
        {
            "job_id": job_id,
            "profile": str(args.profile or infer_profile(existing_status)),
            "status": "building_dataset",
            "status_label": "构建数据集",
            "message": "正在基于模拟器和 teacher 策略构建监督数据集。",
            "started_at": datetime.now().astimezone().isoformat(),
            "budget_minutes": int(args.budget_minutes),
            "device": str(args.device),
            "run_dir": str(dataset_run_dir),
            "final_run_dir": str(final_dir),
            "dataset": {
                "episodes_target": int(args.episodes),
                "max_steps": int(args.max_steps),
                "scenario_copies": int(args.scenario_copies),
            },
            "training": {},
            "evaluation": {},
            "sweep": {"candidates": []},
            "progress": [],
            "artifacts": {
                "previous_run_dir": previous_run_dir,
                "previous_final_metrics": previous_final,
            },
        },
        path=status_path,
    )

    dataset_cfg = BuildDatasetConfig(
        episodes=int(args.episodes),
        max_steps=int(args.max_steps),
        explore_prob=0.12,
        scenario_copies=int(args.scenario_copies),
        run_dir=dataset_run_dir,
        progress_json=status_path.parent / f"{job_id}_dataset_progress.json",
        progress_every=12,
    )
    dataset_result = build_dataset(
        dataset_cfg,
        progress_callback=lambda payload: patch_status(
            status_path,
            status="building_dataset",
            status_label="构建数据集",
            message=str(payload.get("message") or "正在构建训练数据集。"),
            dataset=payload,
            progress=remember_progress(payload),
        ),
    )

    sweep_candidates = [
        TrainConfig(
            dataset=Path(dataset_result["dataset_path"]),
            run_dir=sweep_root / "trial_a",
            epochs=int(args.sweep_epochs),
            batch_size=int(args.batch_size),
            lr=9e-4,
            weight_decay=2e-4,
            device=str(args.device),
            hidden_dim=192,
            card_hidden=96,
            context_hidden=80,
            dropout=0.08,
            patience=2,
            max_train_samples=18000,
            progress_json=status_path.parent / f"{job_id}_trial_a_progress.json",
            update_latest_hint=False,
            model_name="mvp_hand_policy_sweep_a",
        ),
        TrainConfig(
            dataset=Path(dataset_result["dataset_path"]),
            run_dir=sweep_root / "trial_b",
            epochs=int(args.sweep_epochs),
            batch_size=int(args.batch_size),
            lr=8e-4,
            weight_decay=2e-4,
            device=str(args.device),
            hidden_dim=256,
            card_hidden=128,
            context_hidden=96,
            dropout=0.1,
            patience=2,
            max_train_samples=18000,
            progress_json=status_path.parent / f"{job_id}_trial_b_progress.json",
            update_latest_hint=False,
            model_name="mvp_hand_policy_sweep_b",
        ),
    ]
    sweep_results: list[dict[str, Any]] = []
    best_trial: dict[str, Any] | None = None
    for index, candidate in enumerate(sweep_candidates, start=1):
        trial_result = train_model(
            candidate,
            progress_callback=lambda payload, trial_index=index: patch_status(
                status_path,
                status="training",
                status_label="超参试跑",
                message=f"正在试跑候选配置 {trial_index}/{len(sweep_candidates)}。",
                training=payload,
                progress=remember_progress(payload),
            ),
        )
        metrics = json.loads((Path(trial_result["run_dir"]) / "metrics.json").read_text(encoding="utf-8"))
        final = dict(metrics.get("final") or {})
        trial_summary = {
            "name": f"trial_{index}",
            "run_dir": str(trial_result["run_dir"]),
            "best_val_loss": float(metrics.get("best_val_loss") or 0.0),
            "val_acc1": float(final.get("val_acc1") or 0.0),
            "val_acc3": float(final.get("val_acc3") or 0.0),
            "config": {
                "hidden_dim": candidate.hidden_dim,
                "card_hidden": candidate.card_hidden,
                "context_hidden": candidate.context_hidden,
                "dropout": candidate.dropout,
                "lr": candidate.lr,
            },
        }
        sweep_results.append(trial_summary)
        if best_trial is None or trial_summary["best_val_loss"] < best_trial["best_val_loss"]:
            best_trial = trial_summary
        patch_status(status_path, sweep={"candidates": sweep_results})

    assert best_trial is not None
    best_cfg = best_trial["config"]
    patch_status(
        status_path,
        status="training",
        status_label="正式训练",
        message=f"候选试跑完成，选择 {best_trial['name']} 作为正式训练配置。",
        sweep={"candidates": sweep_results, "selected": best_trial["name"]},
    )
    final_cfg = TrainConfig(
        dataset=Path(dataset_result["dataset_path"]),
        run_dir=final_dir,
        epochs=int(args.final_epochs),
        batch_size=int(args.batch_size),
        lr=float(best_cfg["lr"]),
        weight_decay=2e-4,
        device=str(args.device),
        hidden_dim=int(best_cfg["hidden_dim"]),
        card_hidden=int(best_cfg["card_hidden"]),
        context_hidden=int(best_cfg["context_hidden"]),
        dropout=float(best_cfg["dropout"]),
        patience=4,
        progress_json=status_path.parent / f"{job_id}_final_progress.json",
        update_latest_hint=True,
        model_name="mvp_hand_policy_v2",
    )
    train_model(
        final_cfg,
        progress_callback=lambda payload: patch_status(
            status_path,
            status="training",
            status_label="正式训练",
            message="正在基于优胜配置做正式训练。",
            training=payload,
            progress=remember_progress(payload),
        ),
    )

    patch_status(
        status_path,
        status="evaluating",
        status_label="评估新模型",
        message="正式训练完成，正在评估 3 个演示场景。",
    )
    scenario_eval = _scenario_eval(final_dir)
    (final_dir / "demo_scenario_eval.json").write_text(json.dumps(scenario_eval, ensure_ascii=False, indent=2), encoding="utf-8")

    final_metrics = json.loads((final_dir / "metrics.json").read_text(encoding="utf-8"))
    final_summary = dict(final_metrics.get("final") or {})
    prev_val_loss = float(previous_bundle.metrics.get("best_val_loss") or 0.0) if previous_bundle.metrics else 0.0
    new_val_loss = float(final_metrics.get("best_val_loss") or 0.0)
    improved = prev_val_loss <= 0.0 or (new_val_loss > 0.0 and new_val_loss < prev_val_loss)

    verdict = {
        "schema": "mvp_training_verdict_v1",
        "job_id": job_id,
        "previous_run_dir": previous_run_dir,
        "previous_best_val_loss": prev_val_loss,
        "new_run_dir": str(final_dir),
        "new_best_val_loss": new_val_loss,
        "improved_over_previous": improved,
        "scenario_eval": scenario_eval,
    }
    verdict_path = final_dir / "training_verdict.json"
    verdict_path.write_text(json.dumps(verdict, ensure_ascii=False, indent=2), encoding="utf-8")

    patch_status(
        status_path,
        status="finished",
        status_label="训练完成",
        message="新模型训练完成，已写入最新默认模型。",
        finished_at=datetime.now().astimezone().isoformat(),
        final_run_dir=str(final_dir),
        training={
            "run_dir": str(final_dir),
            "best_checkpoint": str(final_dir / "mvp_policy.pt"),
            "best_val_loss": new_val_loss,
            "val_acc1": float(final_summary.get("val_acc1") or 0.0),
            "val_acc3": float(final_summary.get("val_acc3") or 0.0),
            "epoch": int(final_summary.get("epoch") or 0),
            "epochs": int(args.final_epochs),
        },
        progress=progress_events[-80:],
        evaluation={
            "improved_over_previous": improved,
            "scenario_count": int(scenario_eval.get("scenario_count") or 0),
            "teacher_agreement_count": int(scenario_eval.get("teacher_agreement_count") or 0),
        },
        artifacts={
            "dataset_run_dir": str(dataset_run_dir),
            "scenario_eval_path": str(final_dir / "demo_scenario_eval.json"),
            "verdict_path": str(verdict_path),
            "previous_run_dir": previous_run_dir,
        },
    )
    print(json.dumps({"job_id": job_id, "final_run_dir": str(final_dir), "improved": improved}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        try:
            args = parse_args()
            status_path = Path(args.status_path).resolve() if args.status_path else None
            patch_status(
                status_path,
                status="failed",
                status_label="训练失败",
                message=f"训练流水线失败：{exc}",
                finished_at=datetime.now().astimezone().isoformat(),
            )
        except Exception:
            pass
        raise
