# Simulator + Oracle (Dual Track)

This folder provides a Python-first clean-room simulator and an oracle pipeline that queries local balatrobot service.

## Directory Layout

- `core/`: simulator engine, canonical serialization, hashing.
- `pybind/`: backend adapter (`SimEnvBackend`), reserved for future Rust migration.
- `oracle/`: canonicalization from real gamestate and oracle trace runner.
- `spec/`: `state_v1`, `action_v1`, `trace_v1` schemas.
- `tests/`: sim trace runner, directed fixture runner, and oracle/sim diff tool.
- `runtime/`: generated traces (local runtime output).

## Demo (20-step)

1. Oracle trace:

```powershell
.venv_trainer\Scripts\python.exe sim\oracle\run_oracle_trace.py --base-url http://127.0.0.1:12346 --seed AAAAAAA --action-trace sim\tests\fixtures\action_trace_20.jsonl --out sim\runtime\oracle_trace.jsonl --snapshot-every 10
```

2. Sim trace:

```powershell
.venv_trainer\Scripts\python.exe sim\tests\run_sim_trace.py --action-trace sim\tests\fixtures\action_trace_20.jsonl --out sim\runtime\sim_trace.jsonl --seed AAAAAAA
```

3. Diff:

```powershell
.venv_trainer\Scripts\python.exe sim\tests\test_oracle_diff.py --oracle-trace sim\runtime\oracle_trace.jsonl --sim-trace sim\runtime\sim_trace.jsonl --scope hand_core --fail-fast
```

## Directed Fixtures

### Fixture files

- Directed action/meta fixtures: `sim/tests/fixtures_directed/`
- Local runtime snapshots: `sim/tests/fixtures_runtime/` (gitignored)

Each directed case has:

- `action_trace_p0_XX_*.jsonl`
- `meta_p0_XX_*.json`
- local snapshot (not committed), for example `sim/tests/fixtures_runtime/start_snapshot_p0_XX.json`

### Generate a local oracle snapshot

Run oracle trace with snapshots and extract one step snapshot:

```powershell
.venv_trainer\Scripts\python.exe sim\oracle\run_oracle_trace.py --base-url http://127.0.0.1:12346 --seed AAAAAAA --action-trace sim\tests\fixtures_directed\action_trace_p0_01_pair_play.jsonl --out sim\runtime\oracle_p0_01_trace.jsonl --snapshot-every 1
```

Then copy one `canonical_state_snapshot` JSON object into:

`sim/tests/fixtures_runtime/start_snapshot_p0_01_pair_play.json`

### Run directed fixture (sim only)

```powershell
.venv_trainer\Scripts\python.exe sim\tests\run_directed_fixture.py --oracle-snapshot sim\tests\fixtures_runtime\start_snapshot_p0_01_pair_play.json --action-trace sim\tests\fixtures_directed\action_trace_p0_01_pair_play.jsonl --out-trace sim\runtime\directed_p0_01_sim_trace.jsonl
```

### Run directed fixture with oracle diff

```powershell
.venv_trainer\Scripts\python.exe sim\tests\run_directed_fixture.py --oracle-snapshot sim\tests\fixtures_runtime\start_snapshot_p0_01_pair_play.json --action-trace sim\tests\fixtures_directed\action_trace_p0_01_pair_play.jsonl --oracle-trace sim\runtime\oracle_p0_01_trace.jsonl --scope hand_core --fail-fast --snapshot-every 1 --out-trace sim\runtime\directed_p0_01_sim_trace.jsonl
```

### Batch run all P0 directed traces (sim only)

```powershell
Get-ChildItem sim\tests\fixtures_directed\action_trace_p0_*.jsonl | ForEach-Object {
  $name = $_.BaseName
  $snap = "sim/tests/fixtures_runtime/start_snapshot_$($name -replace '^action_trace_','').json"
  .venv_trainer\Scripts\python.exe sim\tests\run_directed_fixture.py --oracle-snapshot $snap --action-trace $_.FullName --out-trace "sim/runtime/$name.sim_trace.jsonl"
}
```

## Notes

- No proprietary game resources are committed to this repository.
- Oracle runner only talks to local balatrobot RPC endpoint.
- Early stage parity target is `hand_core`; `full` parity is expected to diverge before Joker/macro milestones are completed.
- Runtime snapshots in `sim/tests/fixtures_runtime/` are local artifacts and should not be committed.
