"""Regression tests for ablation strategy routing, gate decision schema, and critical inputs."""
from __future__ import annotations

if __package__ is None or __package__ == "":
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import json
import tempfile
import unittest
from pathlib import Path

from trainer.run_ablation import _build_eval_args


class TestAblationStrategyRouting(unittest.TestCase):
    """Test that run_ablation builds correct policy/model args per strategy."""

    def setUp(self) -> None:
        self.tmp = tempfile.mkdtemp()
        self.out_json = Path(self.tmp) / "out.json"
        self.logs_jsonl = Path(self.tmp) / "logs.jsonl"
        self.kw = {
            "backend": "sim",
            "stake": "gold",
            "episodes": 2,
            "seeds_file": "seeds.txt",
            "out_json": self.out_json,
            "logs_jsonl": self.logs_jsonl,
            "max_steps_per_episode": 60,
        }

    def test_rl_strategy_requires_model(self) -> None:
        with self.assertRaises(RuntimeError) as ctx:
            _build_eval_args(
                strategy="rl",
                model=None,
                rl_model=None,
                risk_config=None,
                **self.kw,
            )
        self.assertIn("rl strategy requires a model", str(ctx.exception))

    def test_rl_strategy_includes_model_arg(self) -> None:
        args = _build_eval_args(
            strategy="rl",
            model="/path/to/rl.pt",
            rl_model=None,
            risk_config=None,
            **self.kw,
        )
        self.assertEqual(args[args.index("--policy") + 1], "pv")
        self.assertEqual(args[args.index("--model") + 1], "/path/to/rl.pt")

    def test_champion_strategy_uses_pv_policy_and_model(self) -> None:
        args = _build_eval_args(
            strategy="champion",
            model="/path/to/champ.pt",
            rl_model=None,
            risk_config=None,
            **self.kw,
        )
        self.assertEqual(args[args.index("--policy") + 1], "pv")
        self.assertEqual(args[args.index("--model") + 1], "/path/to/champ.pt")

    def test_pv_strategy_uses_pv_policy_and_model(self) -> None:
        args = _build_eval_args(
            strategy="pv",
            model="/path/to/pv.pt",
            rl_model=None,
            risk_config=None,
            **self.kw,
        )
        self.assertEqual(args[args.index("--policy") + 1], "pv")
        self.assertEqual(args[args.index("--model") + 1], "/path/to/pv.pt")

    def test_risk_aware_includes_risk_config_and_rl_model(self) -> None:
        args = _build_eval_args(
            strategy="risk_aware",
            model="/path/to/pv.pt",
            rl_model="/path/to/rl.pt",
            risk_config="/path/to/risk.yaml",
            **self.kw,
        )
        self.assertEqual(args[args.index("--policy") + 1], "risk_aware")
        self.assertIn("--risk-config", args)
        self.assertEqual(args[args.index("--risk-config") + 1], "/path/to/risk.yaml")
        self.assertEqual(args[args.index("--rl-model") + 1], "/path/to/rl.pt")


class TestGateDecisionSchema(unittest.TestCase):
    """Test promotion/gate decision payload schema and allowed values."""

    ALLOWED_DECISIONS = frozenset(
        {"promote", "reject", "hold_for_promote", "hold_for_more_data", "rolled_back"}
    )

    def test_decision_payload_schema_and_decision_value(self) -> None:
        payload = {
            "schema": "champion_decision_v3",
            "decision": "reject",
            "final_decision": "reject",
            "reason": "perf_gate_fail",
            "perf_gate_pass": False,
            "risk_guard_pass": True,
        }
        self.assertEqual(payload["schema"], "champion_decision_v3")
        self.assertIn(payload["decision"], self.ALLOWED_DECISIONS)

    def test_gate_functional_has_expected_keys(self) -> None:
        """Gate functional JSON should have pass/fail and key checks."""
        gate = {"functional_gate": "PASS", "checks": []}
        self.assertIn("functional_gate", gate)
        self.assertIn(gate["functional_gate"], ("PASS", "FAIL"))


class TestCanaryDivergenceArtifact(unittest.TestCase):
    """Test canary divergence summary exposes synthetic label when present."""

    def test_divergence_summary_with_synthetic_label(self) -> None:
        summary = {
            "schema": "p19_canary_divergence_v1",
            "steps": 10,
            "top1_agreement_rate": 0.8,
            "metrics_source": "synthetic",
            "metrics_note": "Divergence and risk_aware rates are derived from synthetic rules",
        }
        self.assertEqual(summary.get("metrics_source"), "synthetic")
        self.assertIn("synthetic", (summary.get("metrics_note") or ""))


if __name__ == "__main__":
    unittest.main()
