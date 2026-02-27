# Project and Commands Usage Guide

This guide is for the project root directory and summarizes available features and runnable commands.

## 0. Conventions and Directory

- Always run commands from the **project root directory**.
- Main modules:
  - `benchmark_balatrobot.py`: Single/multi-instance throughput and latency benchmarks
  - `sweep_throughput.py`: Batch instance sweep and curves
  - `trainer/`: Data collection, BC training, evaluation, inference assistant
  - `sim/`: Simulator + oracle alignment + diff regression

## 1. BalatroBot Service Startup

Confirm game paths first:

- Balatro: `D:\SteamLibrary\steamapps\common\Balatro\Balatro.exe`
- Lovely: `D:\SteamLibrary\steamapps\common\Balatro\version.dll`

### 1.1 Manual start (recommended: direct)

```powershell
balatrobot serve --headless --fast --port 12346 --love-path "D:\SteamLibrary\steamapps\common\Balatro\Balatro.exe" --lovely-path "D:\SteamLibrary\steamapps\common\Balatro\version.dll"
```

### 1.2 Manual start (uvx)

```powershell
uvx balatrobot serve --headless --fast --port 12346 --love-path "D:\SteamLibrary\steamapps\common\Balatro\Balatro.exe" --lovely-path "D:\SteamLibrary\steamapps\common\Balatro\version.dll"
```

### 1.3 Health check

```powershell
python -c "import requests;print(requests.post('http://127.0.0.1:12346',json={'id':1,'jsonrpc':'2.0','method':'health','params':{}},timeout=5).text)"
```

## 2. Benchmark (throughput and latency)

### 2.1 Single instance (basic)

```powershell
python benchmark_balatrobot.py --instances 1 --steps-per-instance 100 --mode action_only
```

### 2.2 RL semantic mode

```powershell
python benchmark_balatrobot.py --instances 1 --steps-per-instance 100 --mode rl_step
```

### 2.3 Multi-instance (instances already started manually)

```powershell
python benchmark_balatrobot.py --instances 4 --ports 12346,12347,12348,12349 --steps-per-instance 100 --mode action_only
```

### 2.4 Multi-instance auto-launch (direct)

```powershell
python benchmark_balatrobot.py --launch-instances --launcher direct --balatrobot-cmd balatrobot --instances 2 --steps-per-instance 100 --base-url http://127.0.0.1 --love-path "D:\SteamLibrary\steamapps\common\Balatro\Balatro.exe" --lovely-path "D:\SteamLibrary\steamapps\common\Balatro\version.dll"
```

### 2.5 Multi-instance auto-launch (uvx)

```powershell
python benchmark_balatrobot.py --launch-instances --launcher uvx --uvx-path uvx --instances 2 --steps-per-instance 100 --base-url http://127.0.0.1 --love-path "D:\SteamLibrary\steamapps\common\Balatro\Balatro.exe" --lovely-path "D:\SteamLibrary\steamapps\common\Balatro\version.dll"
```

## 3. Sweep (instance scan)

### 3.1 Quick acceptance sweep (low cost)

```powershell
python sweep_throughput.py --instances 1,3,5 --repeats 1 --steps-per-instance 50 --mode action_only --launch-instances --launcher direct --balatrobot-cmd balatrobot --love-path "D:\SteamLibrary\steamapps\common\Balatro\Balatro.exe" --lovely-path "D:\SteamLibrary\steamapps\common\Balatro\version.dll" --raw-dir sweep_raw_quick
```

### 3.2 Full sweep (including rl_step)

```powershell
python sweep_throughput.py --instances 1,2,4,8 --repeats 3 --steps-per-instance 200 --mode both --launch-instances --launcher direct --balatrobot-cmd balatrobot --love-path "D:\SteamLibrary\steamapps\common\Balatro\Balatro.exe" --lovely-path "D:\SteamLibrary\steamapps\common\Balatro\version.dll" --csv-out sweep_results.csv --raw-dir sweep_raw_full
```

Output highlights:

- Detail CSV: `sweep_results.csv`
- Summary CSV: `sweep_results_summary.csv`

## 4. Trainer (rollout -> train -> eval -> infer)

### 4.1 One-click environment setup (dedicated venv)

```powershell
.\trainer\scripts\setup_trainer_env.ps1
```

### 4.2 One-click E2E smoke

```powershell
.\trainer\scripts\smoke_e2e.ps1 --base-urls "http://127.0.0.1:12346"
```

### 4.3 Manual steps (real backend)

```powershell
.venv_trainer\Scripts\python.exe trainer\rollout.py --backend real --base-urls http://127.0.0.1:12346 --episodes 20 --restart-on-fail --out trainer_data\dataset_real.jsonl
```

```powershell
.venv_trainer\Scripts\python.exe trainer\dataset.py --path trainer_data\dataset_real.jsonl --summary --validate
```

```powershell
.venv_trainer\Scripts\python.exe trainer\train_bc.py --train-jsonl trainer_data\dataset_real.jsonl --epochs 8 --batch-size 64 --out-dir trainer_runs\bc_v1
```

```powershell
.venv_trainer\Scripts\python.exe trainer\eval.py --offline --model trainer_runs\bc_v1\best.pt --dataset trainer_data\dataset_real.jsonl
```

```powershell
.venv_trainer\Scripts\python.exe trainer\infer_assistant.py --backend real --base-url http://127.0.0.1:12346 --model trainer_runs\bc_v1\best.pt --topk 3 --once
```

Offline evaluation metrics of interest:

- `top1`
- `top3`
- `illegal_rate`
- `random_top3`
- `top3_lift`

## 5. Simulator + Oracle (sim)

### 5.1 20-step oracle/sim alignment demo

```powershell
.venv_trainer\Scripts\python.exe sim\oracle\run_oracle_trace.py --base-url http://127.0.0.1:12346 --seed AAAAAAA --action-trace sim\tests\fixtures\action_trace_20.jsonl --out sim\runtime\oracle_trace.jsonl --snapshot-every 10
```

```powershell
.venv_trainer\Scripts\python.exe sim\tests\run_sim_trace.py --action-trace sim\tests\fixtures\action_trace_20.jsonl --out sim\runtime\sim_trace.jsonl --seed AAAAAAA
```

```powershell
.venv_trainer\Scripts\python.exe sim\tests\test_oracle_diff.py --oracle-trace sim\runtime\oracle_trace.jsonl --sim-trace sim\runtime\sim_trace.jsonl --scope hand_core --fail-fast
```

### 5.2 Directed fixture (replay from snapshot)

```powershell
.venv_trainer\Scripts\python.exe sim\tests\run_directed_fixture.py --oracle-snapshot sim\tests\fixtures_runtime\start_snapshot_p0_01_pair_play.json --action-trace sim\tests\fixtures_directed\action_trace_p0_01_pair_play.jsonl --out-trace sim\runtime\directed_p0_01_sim_trace.jsonl
```

### 5.3 Auto-build P0 oracle fixtures + diff report

```powershell
.venv_trainer\Scripts\python.exe sim\oracle\batch_build_p0_oracle_fixtures.py --base-url http://127.0.0.1:12346 --out-dir sim\tests\fixtures_runtime\oracle_p0 --scope score_core --max-steps 80 --seed AAAAAAA
```

Output highlights:

- `oracle_start_snapshot_*.json`
- `action_trace_*.jsonl`
- `oracle_trace_*.jsonl`
- `sim_trace_*.jsonl`
- `report_p0.json`

## 6. Ready-to-run command sets

### 6.1 Minimal full pipeline (recommended first run)

```powershell
python benchmark_balatrobot.py --instances 1 --steps-per-instance 30 --mode action_only
python sweep_throughput.py --instances 1,3 --repeats 1 --steps-per-instance 30 --mode action_only --launch-instances --launcher direct --balatrobot-cmd balatrobot --love-path "D:\SteamLibrary\steamapps\common\Balatro\Balatro.exe" --lovely-path "D:\SteamLibrary\steamapps\common\Balatro\version.dll"
.\trainer\scripts\smoke_e2e.ps1 --base-urls "http://127.0.0.1:12346"
```

### 6.2 sim/oracle regression set

```powershell
.venv_trainer\Scripts\python.exe sim\oracle\generate_p0_trace.py --base-url http://127.0.0.1:12346 --target p0_01_straight --out-dir sim\tests\fixtures_runtime\oracle_p0
.venv_trainer\Scripts\python.exe sim\oracle\run_oracle_trace.py --base-url http://127.0.0.1:12346 --seed AAAAAAA --action-trace sim\tests\fixtures_runtime\oracle_p0\action_trace_p0_01_straight.jsonl --out sim\tests\fixtures_runtime\oracle_p0\oracle_trace_p0_01_straight.jsonl --snapshot-every 1
.venv_trainer\Scripts\python.exe sim\tests\run_directed_fixture.py --oracle-snapshot sim\tests\fixtures_runtime\oracle_p0\oracle_start_snapshot_p0_01_straight.json --action-trace sim\tests\fixtures_runtime\oracle_p0\action_trace_p0_01_straight.jsonl --oracle-trace sim\tests\fixtures_runtime\oracle_p0\oracle_trace_p0_01_straight.jsonl --scope score_core --fail-fast --out-trace sim\tests\fixtures_runtime\oracle_p0\sim_trace_p0_01_straight.jsonl
```

## 7. Troubleshooting

### 7.1 `WinError 10061` / connection refused

- Instance crashed or port not ready.
- Reduce concurrency `--instances`, add `--stagger-start`.
- Check logs under `logs\` for the corresponding port.

### 7.2 torch install failure on Python 3.15 alpha

- Use: `.\trainer\scripts\setup_trainer_env.ps1`
- The script will try `py -3.12` / `py -3.14` first to create `.venv_trainer`.

### 7.3 Too many leftover windows after auto-launch

- Prefer script-managed benchmark/rollout and ensure commands exit cleanly.
- If needed, clean up manually:

```powershell
taskkill /F /IM Balatro.exe
taskkill /F /IM balatrobot.exe
```

### 7.4 Paths and data layout

- Project data and artifacts are under the project root (e.g. `logs/`, `trainer_data/`, `trainer_runs/`, `sim/runtime/`).

---

For more detail, see:

- `trainer/README.md`
- `sim/README.md`
