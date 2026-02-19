# pybind Layer (Python-first)

Current implementation uses a pure Python adapter (`sim_env.py`) to keep the interface stable:

- `reset(seed) -> state`
- `get_state() -> state`
- `step(action) -> (state, reward, done, info)`
- `health() -> bool`
- `close()`

Future Rust migration can replace this file with a pyo3/maturin-backed module while keeping the same API.
