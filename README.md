# BalatroAI

> Language: [Chinese (Simplified)](README.zh-CN.md) | [English](README.en.md)

<!-- BADGES:START -->
[![Latest Gate](https://img.shields.io/badge/Latest_Gate-RunP29_PASS-2EA44F)](scripts/run_regressions.ps1)
[![Workflow](https://img.shields.io/badge/Workflow-mainline--only-2EA44F)](scripts/git_sync.ps1)
[![Seed Governance](https://img.shields.io/badge/Seed_Governance-P23%2B_enabled-0E8A16)](configs/experiments/seeds_p23.yaml)
[![Experiment Orchestrator](https://img.shields.io/badge/Experiment_Orchestrator-P22%2B_enabled-1F6FEB)](scripts/run_p22.ps1)
[![Trend Warehouse](https://img.shields.io/badge/Trend_Warehouse-P26%2B_enabled-0E8A16)](docs/TREND_WAREHOUSE_P26.md)
[![Docs Coverage](https://img.shields.io/badge/Docs_Coverage-P15--P33-6E7781)](docs/)
[![Platform](https://img.shields.io/badge/Platform-Windows-0078D6)](USAGE_GUIDE.md)
[![Python](https://img.shields.io/badge/Python-3.12%2B-3776AB)](trainer/requirements.txt)
[![License](https://img.shields.io/badge/License-Not_Specified-6E7781)](#license-and-contributing)
[![CI Smoke](https://github.com/XcapeAxis/BalatroAI/actions/workflows/ci-smoke.yml/badge.svg)](https://github.com/XcapeAxis/BalatroAI/actions/workflows/ci-smoke.yml)
[![GitHub Stars](https://img.shields.io/github/stars/XcapeAxis/BalatroAI?style=social)](https://github.com/XcapeAxis/BalatroAI/stargazers)
[![GitHub Issues](https://img.shields.io/github/issues/XcapeAxis/BalatroAI)](https://github.com/XcapeAxis/BalatroAI/issues)
<!-- BADGES:END -->

BalatroAI is a local engineering stack for Balatro-focused simulation, training, evaluation, operations, and demo delivery.

## Choose Your Language

- Chinese (Simplified): [README.zh-CN.md](README.zh-CN.md)
- English: [README.en.md](README.en.md)

## Key Documents

| Document | Chinese | English |
|---|---|---|
| Project overview | [README.zh-CN.md](README.zh-CN.md) | [README.en.md](README.en.md) |
| Demo guide | [DEMO_README.zh-CN.md](DEMO_README.zh-CN.md) | [DEMO_README.en.md](DEMO_README.en.md) |
| Usage guide | [USAGE_GUIDE.zh-CN.md](USAGE_GUIDE.zh-CN.md) | [USAGE_GUIDE.en.md](USAGE_GUIDE.en.md) |
| Demo script | [docs/MVP_DEMO_SCRIPT.zh-CN.md](docs/MVP_DEMO_SCRIPT.zh-CN.md) | [docs/MVP_DEMO_SCRIPT.en.md](docs/MVP_DEMO_SCRIPT.en.md) |
| Roadmap | [docs/ROADMAP.zh-CN.md](docs/ROADMAP.zh-CN.md) | [docs/ROADMAP.en.md](docs/ROADMAP.en.md) |
| Architecture | [docs/ARCHITECTURE.zh-CN.md](docs/ARCHITECTURE.zh-CN.md) | [docs/ARCHITECTURE.en.md](docs/ARCHITECTURE.en.md) |

## Quick Start

- Chinese quick start: [README.zh-CN.md](README.zh-CN.md#快速开始)
- English quick start: [README.en.md](README.en.md#quick-start)

## What This Project Is

- Chinese overview: [README.zh-CN.md](README.zh-CN.md#项目定位)
- English overview: [README.en.md](README.en.md#project-positioning)

## Scope and Boundaries

- Chinese scope: [README.zh-CN.md](README.zh-CN.md#项目边界)
- English scope: [README.en.md](README.en.md#scope-and-boundaries)

## Architecture Overview

- Chinese architecture: [docs/ARCHITECTURE.zh-CN.md](docs/ARCHITECTURE.zh-CN.md)
- English architecture: [docs/ARCHITECTURE.en.md](docs/ARCHITECTURE.en.md)

## Reproducibility

- Main reference: [docs/REPRODUCIBILITY_P25.md](docs/REPRODUCIBILITY_P25.md)
- Seed and run tracking: [docs/SEEDS_AND_REPRODUCIBILITY.md](docs/SEEDS_AND_REPRODUCIBILITY.md)

## Example Outputs

- Demo API smoke: `docs/artifacts/mvp/api_smoke_20260309_154858.json`
- Demo training status: `docs/artifacts/mvp/training_status/latest.json`
- P22 summary example: `docs/artifacts/p22/runs/<run_id>/summary_table.json`

## Roadmap

- Chinese roadmap: [docs/ROADMAP.zh-CN.md](docs/ROADMAP.zh-CN.md)
- English roadmap: [docs/ROADMAP.en.md](docs/ROADMAP.en.md)

## Known Limitations

- Deep milestone docs still keep their original repository writing style.
- The bilingual split currently focuses on public-facing overview and demo-entry documents.

## Repository Status

<!-- STATUS:START -->
<!-- README_STATUS:BEGIN -->
### Repository Status (Auto-generated)

- branch: main
- latest_gate: RunP29 (PASS)
- recent_trend_signal: regression
- trend_warehouse_last_updated: 2026-03-02T17:16:27.519440+00:00
- trend_rows_count: 20115
- champion: quick_risk_aware (champion)
- candidate:  (decision: hold)
- docs_coverage: P15-P33
<!-- README_STATUS:END -->
<!-- STATUS:END -->

## License and Contributing

- License: no top-level `LICENSE` file is currently present.
- Contributions: stay on `main`, keep changes auditable, and run the relevant gates before changing operational defaults.
