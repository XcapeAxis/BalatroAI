# Agent Task Template

Copy this template when giving a task to the Cursor Agent. Fill in the sections that apply.

---

## Fields

- **Task name**: (short title)
- **Background / context**: (why, current state)
- **Goal**: (verifiable outcome; what “done” looks like)
- **Out of scope**: (what not to do; avoid scope creep)
- **Impact**: (files / modules likely to change)
- **Risks**: (what could go wrong or break)
- **Acceptance criteria**: (including gates: e.g. “pytest X passes”, “script Y runs without error”)
- **Commands / entry points**: (e.g. `pytest trainer/tests/`, `./safe_run.sh ...`)
- **Output format**: (e.g. list changed files, commands run, verification result, risks)
- **Allow git commit**: (Y / N)

### If this is an experiment / evaluation task, also add:

- **Fixed variables**: (what stays the same across runs)
- **Varying variables**: (what you change)
- **Metrics**: (what you measure)
- **Sample scope**: (e.g. N runs, subset of data) if applicable
- **Result location**: (path or artifact where results are stored)

---

## Example 1: Ordinary feature / fix

- **Task name**: Fix calibration lookup when deck is empty
- **Background**: In trainer calibration, an empty deck causes an index error.
- **Goal**: No crash when deck is empty; calibration returns a defined value (e.g. default or skip).
- **Out of scope**: Changing calibration algorithm or other modules.
- **Impact**: `trainer/calibration.py` (and possibly tests).
- **Risks**: Edge cases for other “empty” inputs.
- **Acceptance**: `pytest trainer/tests/ -k calibration` passes; manual run with empty deck does not crash.
- **Commands**: `./safe_run.sh --timeout 180 pytest trainer/tests/ -k calibration -q`
- **Output**: Changed files, commands run, pytest result, any new risks.
- **Allow git commit**: N

---

## Example 2: Experiment / evaluation

- **Task name**: Ablation – disable feature X and compare win rate
- **Background**: We want to see the impact of feature X on win rate.
- **Goal**: One run with X enabled, one with X disabled; report win rate (and optionally variance) for both.
- **Out of scope**: Changing other features or training code beyond the single flag.
- **Impact**: Config or script that toggles X; run/result paths.
- **Risks**: Different seeds or env could affect comparison.
- **Acceptance**: Two result summaries (or files) produced; metrics clearly labeled.
- **Fixed variables**: Seed, data path, model size, other flags.
- **Varying variables**: X on vs off.
- **Metrics**: Win rate (and optionally std).
- **Sample scope**: e.g. 3 runs per setting.
- **Result location**: e.g. `trainer_runs/ablation_x_YYYYMMDD/`.
- **Output**: Changed files, commands, where results are, short facts (numbers) vs inference vs recommendations.
- **Allow git commit**: N
