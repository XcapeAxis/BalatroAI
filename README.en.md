# BalatroAI

> Chinese homepage: [README.md](README.md)

BalatroAI is a local engineering stack for Balatro-focused simulation, training, evaluation, overnight execution, and demo delivery. The repository is organized so another engineer can reproduce runs, inspect artifacts, and continue the project on a Windows machine without rebuilding the workflow from scratch.

## Project Positioning

The repo currently serves two visible purposes:

- A local web demo that is suitable for interview walkthroughs
- A simulator-first research and operations stack covering experiment orchestration, validation, local ops, autonomy, and Windows handoff

## MVP Demo

The repository already includes a local web demo that can be opened directly in a browser. It is not a research console. It is a scenario-driven AI decision sandbox that shows the board state, compares model vs heuristic recommendations, executes one step, and surfaces the training status of a real checkpoint.

What you can show in about two minutes:

- Local browser UI driven by the simulator rather than static mock data
- `3` built-in scenarios: strong opening hand, high-risk discard pivot, and Joker synergy burst
- Side-by-side model vs heuristic recommendations
- Linked UI updates when a recommendation is selected
- A genuinely trained minimum viable model plus visible training progress

Launch the demo:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_mvp_demo.ps1 -OpenBrowser
```

Bootstrap a usable model first, then launch:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\bootstrap_mvp_demo.ps1 -OpenBrowser
```

Default URL:

```text
http://127.0.0.1:8050/
```

## Quick Start

Recommended handoff flow for a Windows workstation:

1. Clone the repository.

```powershell
git clone https://github.com/XcapeAxis/BalatroAI.git
cd BalatroAI
```

2. Bootstrap the local Windows environment.

```powershell
powershell -ExecutionPolicy Bypass -File scripts\setup_windows.ps1 -Mode auto -SkipSmoke
```

3. Run the doctor check.

```powershell
powershell -ExecutionPolicy Bypass -File scripts\doctor.ps1
```

4. Run the default quick matrix.

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_p22.ps1 -Quick
```

If you only want the interview demo, run `scripts\run_mvp_demo.ps1` directly.

## Scope And Boundaries

Good fit for:

- Balatro-style AI decision visualization
- Simulator parity and replay validation
- Seed-governed training, evaluation, and comparison
- Local dashboard, Ops UI, attention queue, and morning summary workflows
- Reproducible Windows handoff and environment checks

Not intended for:

- Direct control of a commercial game client
- High-risk promotion without review gates
- Untracked environment changes
- Claiming the current model is a universal optimal agent

## Architecture Snapshot

The main flow is:

1. Real runtime or `balatrobot` produces oracle traces.
2. The simulator validates parity through scoped replay checks.
3. Training and evaluation consume those traces and replay artifacts.
4. `P22` aggregates experiment rows, seeds, artifacts, and summaries.
5. Dashboard, Ops UI, attention queue, and morning summary expose those artifacts locally.

Key reading:

- Architecture: [docs/ARCHITECTURE.en.md](docs/ARCHITECTURE.en.md)
- Roadmap: [docs/ROADMAP.en.md](docs/ROADMAP.en.md)
- Demo guide: [DEMO_README.en.md](DEMO_README.en.md)
- Demo script: [docs/MVP_DEMO_SCRIPT.en.md](docs/MVP_DEMO_SCRIPT.en.md)

## Common Entrypoints

| Goal | Command | Main output |
|---|---|---|
| Launch demo | `powershell -ExecutionPolicy Bypass -File scripts\run_mvp_demo.ps1 -OpenBrowser` | `http://127.0.0.1:8050/` |
| Run doctor | `powershell -ExecutionPolicy Bypass -File scripts\doctor.ps1` | `docs/artifacts/p58/latest_doctor.json` |
| Run quick matrix | `powershell -ExecutionPolicy Bypass -File scripts\run_p22.ps1 -Quick` | `summary_table.json`, dashboard |
| Run RunP22 gate | `powershell -ExecutionPolicy Bypass -File scripts\run_regressions.ps1 -RunP22` | regression artifacts |
| Start Ops UI | `powershell -ExecutionPolicy Bypass -File scripts\run_ops_ui.ps1` | `http://127.0.0.1:8765/` |
| Run autonomy entry | `powershell -ExecutionPolicy Bypass -File scripts\run_autonomy.ps1 -Quick` | `docs/artifacts/p60/latest_autonomy_entry.json` |

## Further Reading

- Demo guide: [DEMO_README.en.md](DEMO_README.en.md)
- Usage guide: [USAGE_GUIDE.en.md](USAGE_GUIDE.en.md)
- Architecture: [docs/ARCHITECTURE.en.md](docs/ARCHITECTURE.en.md)
- Roadmap: [docs/ROADMAP.en.md](docs/ROADMAP.en.md)
- P22 orchestrator: [docs/EXPERIMENTS_P22.md](docs/EXPERIMENTS_P22.md)
- Windows bootstrap: [docs/P58_WINDOWS_BOOTSTRAP.md](docs/P58_WINDOWS_BOOTSTRAP.md)

## License and Contributing

- License: no top-level `LICENSE` file is currently present.
- Contributions: stay on `main`, keep changes auditable, and run the relevant gates before changing operational defaults.
