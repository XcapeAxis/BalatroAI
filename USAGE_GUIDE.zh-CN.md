# 项目与命令使用指南

> 语言切换： [简体中文](USAGE_GUIDE.zh-CN.md) | [English](USAGE_GUIDE.en.md)

这份指南面向项目根目录，汇总当前仓库里最常用的功能入口和可直接运行的命令。

## 0. 约定与目录

- 默认都在**项目根目录**执行命令。
- 主要模块：
  - `benchmark_balatrobot.py`：单实例 / 多实例吞吐与延迟基准
  - `sweep_throughput.py`：批量实例扫描与曲线输出
  - `trainer/`：数据采集、行为克隆训练、评测、推理助手
  - `sim/`：模拟器、oracle 对齐和 diff 回归

## 1. BalatroBot 服务启动

先确认游戏路径：

- Balatro：`D:\SteamLibrary\steamapps\common\Balatro\Balatro.exe`
- Lovely：`D:\SteamLibrary\steamapps\common\Balatro\version.dll`

### 1.1 手动启动（推荐：direct）

```powershell
balatrobot serve --headless --fast --port 12346 --love-path "D:\SteamLibrary\steamapps\common\Balatro\Balatro.exe" --lovely-path "D:\SteamLibrary\steamapps\common\Balatro\version.dll"
```

### 1.2 手动启动（uvx）

```powershell
uvx balatrobot serve --headless --fast --port 12346 --love-path "D:\SteamLibrary\steamapps\common\Balatro\Balatro.exe" --lovely-path "D:\SteamLibrary\steamapps\common\Balatro\version.dll"
```

### 1.3 健康检查

```powershell
python -c "import requests;print(requests.post('http://127.0.0.1:12346',json={'id':1,'jsonrpc':'2.0','method':'health','params':{}},timeout=5).text)"
```

## 2. 基准测试（吞吐与延迟）

### 2.1 单实例基础测试

```powershell
python benchmark_balatrobot.py --instances 1 --steps-per-instance 100 --mode action_only
```

### 2.2 RL 语义模式

```powershell
python benchmark_balatrobot.py --instances 1 --steps-per-instance 100 --mode rl_step
```

### 2.3 多实例测试（实例已手动启动）

```powershell
python benchmark_balatrobot.py --instances 4 --ports 12346,12347,12348,12349 --steps-per-instance 100 --mode action_only
```

### 2.4 多实例自动拉起（direct）

```powershell
python benchmark_balatrobot.py --launch-instances --launcher direct --balatrobot-cmd balatrobot --instances 2 --steps-per-instance 100 --base-url http://127.0.0.1 --love-path "D:\SteamLibrary\steamapps\common\Balatro\Balatro.exe" --lovely-path "D:\SteamLibrary\steamapps\common\Balatro\version.dll"
```

### 2.5 多实例自动拉起（uvx）

```powershell
python benchmark_balatrobot.py --launch-instances --launcher uvx --uvx-path uvx --instances 2 --steps-per-instance 100 --base-url http://127.0.0.1 --love-path "D:\SteamLibrary\steamapps\common\Balatro\Balatro.exe" --lovely-path "D:\SteamLibrary\steamapps\common\Balatro\version.dll"
```

## 3. 扫描测试（实例 sweep）

### 3.1 快速验收 sweep（低成本）

```powershell
python sweep_throughput.py --instances 1,3,5 --repeats 1 --steps-per-instance 50 --mode action_only --launch-instances --launcher direct --balatrobot-cmd balatrobot --love-path "D:\SteamLibrary\steamapps\common\Balatro\Balatro.exe" --lovely-path "D:\SteamLibrary\steamapps\common\Balatro\version.dll" --raw-dir sweep_raw_quick
```

### 3.2 完整 sweep（包含 `rl_step`）

```powershell
python sweep_throughput.py --instances 1,2,4,8 --repeats 3 --steps-per-instance 200 --mode both --launch-instances --launcher direct --balatrobot-cmd balatrobot --love-path "D:\SteamLibrary\steamapps\common\Balatro\Balatro.exe" --lovely-path "D:\SteamLibrary\steamapps\common\Balatro\version.dll" --csv-out sweep_results.csv --raw-dir sweep_raw_full
```

主要输出：

- 详情 CSV：`sweep_results.csv`
- 汇总 CSV：`sweep_results_summary.csv`

## 4. Trainer 流程（rollout -> train -> eval -> infer）

### 4.1 一键环境准备（独立虚拟环境）

```powershell
.\trainer\scripts\setup_trainer_env.ps1
```

### 4.2 一键 E2E smoke

```powershell
.\trainer\scripts\smoke_e2e.ps1 --base-urls "http://127.0.0.1:12346"
```

### 4.3 手动分步执行（真实后端）

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

离线评测重点指标：

- `top1`
- `top3`
- `illegal_rate`
- `random_top3`
- `top3_lift`

## 5. 模拟器与 Oracle（sim）

### 5.1 20 步 oracle / sim 对齐演示

```powershell
.venv_trainer\Scripts\python.exe sim\oracle\run_oracle_trace.py --base-url http://127.0.0.1:12346 --seed AAAAAAA --action-trace sim\tests\fixtures\action_trace_20.jsonl --out sim\runtime\oracle_trace.jsonl --snapshot-every 10
```

```powershell
.venv_trainer\Scripts\python.exe sim\tests\run_sim_trace.py --action-trace sim\tests\fixtures\action_trace_20.jsonl --out sim\runtime\sim_trace.jsonl --seed AAAAAAA
```

```powershell
.venv_trainer\Scripts\python.exe sim\tests\test_oracle_diff.py --oracle-trace sim\runtime\oracle_trace.jsonl --sim-trace sim\runtime\sim_trace.jsonl --scope hand_core --fail-fast
```

### 5.2 定向 fixture（从 snapshot 回放）

```powershell
.venv_trainer\Scripts\python.exe sim\tests\run_directed_fixture.py --oracle-snapshot sim\tests\fixtures_runtime\start_snapshot_p0_01_pair_play.json --action-trace sim\tests\fixtures_directed\action_trace_p0_01_pair_play.jsonl --out-trace sim\runtime\directed_p0_01_sim_trace.jsonl
```

### 5.3 自动构建 P0 oracle fixtures + diff 报告

```powershell
.venv_trainer\Scripts\python.exe sim\oracle\batch_build_p0_oracle_fixtures.py --base-url http://127.0.0.1:12346 --out-dir sim\tests\fixtures_runtime\oracle_p0 --scope score_core --max-steps 80 --seed AAAAAAA
```

主要输出：

- `oracle_start_snapshot_*.json`
- `action_trace_*.jsonl`
- `oracle_trace_*.jsonl`
- `sim_trace_*.jsonl`
- `report_p0.json`

## 6. 可直接运行的命令组合

### 6.1 最小全流程（推荐第一次执行）

```powershell
python benchmark_balatrobot.py --instances 1 --steps-per-instance 30 --mode action_only
python sweep_throughput.py --instances 1,3 --repeats 1 --steps-per-instance 30 --mode action_only --launch-instances --launcher direct --balatrobot-cmd balatrobot --love-path "D:\SteamLibrary\steamapps\common\Balatro\Balatro.exe" --lovely-path "D:\SteamLibrary\steamapps\common\Balatro\version.dll"
.\trainer\scripts\smoke_e2e.ps1 --base-urls "http://127.0.0.1:12346"
```

### 6.2 sim / oracle 回归组合

```powershell
.venv_trainer\Scripts\python.exe sim\oracle\generate_p0_trace.py --base-url http://127.0.0.1:12346 --target p0_01_straight --out-dir sim\tests\fixtures_runtime\oracle_p0
.venv_trainer\Scripts\python.exe sim\oracle\run_oracle_trace.py --base-url http://127.0.0.1:12346 --seed AAAAAAA --action-trace sim\tests\fixtures_runtime\oracle_p0\action_trace_p0_01_straight.jsonl --out sim\tests\fixtures_runtime\oracle_p0\oracle_trace_p0_01_straight.jsonl --snapshot-every 1
.venv_trainer\Scripts\python.exe sim\tests\run_directed_fixture.py --oracle-snapshot sim\tests\fixtures_runtime\oracle_p0\oracle_start_snapshot_p0_01_straight.json --action-trace sim\tests\fixtures_runtime\oracle_p0\action_trace_p0_01_straight.jsonl --oracle-trace sim\tests\fixtures_runtime\oracle_p0\oracle_trace_p0_01_straight.jsonl --scope score_core --fail-fast --out-trace sim\tests\fixtures_runtime\oracle_p0\sim_trace_p0_01_straight.jsonl
```

## 7. 常见问题排查

### 7.1 `WinError 10061` / connection refused

- 实例崩溃或端口还没准备好。
- 降低并发 `--instances`，并增加 `--stagger-start`。
- 查看 `logs\` 下对应端口的日志。

### 7.2 Python 3.15 alpha 下 torch 安装失败

- 使用：`.\trainer\scripts\setup_trainer_env.ps1`
- 该脚本会优先尝试 `py -3.12` / `py -3.14` 来创建 `.venv_trainer`。

### 7.3 自动拉起后遗留太多窗口

- 优先使用脚本托管的 benchmark / rollout，并确保命令正常退出。
- 如有需要，可手动清理：

```powershell
taskkill /F /IM Balatro.exe
taskkill /F /IM balatrobot.exe
```

### 7.4 路径与数据布局

- 项目数据与产物默认都在项目根目录下，例如 `logs/`、`trainer_data/`、`trainer_runs/`、`sim/runtime/`。

---

更多说明请查看：

- `trainer/README.md`
- `sim/README.md`
