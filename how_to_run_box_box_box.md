# Box Box Box — How to Run

## Prerequisites

Make sure you have Python 3 installed.

Check it with:

```bash
python3 --version
```

Install the required packages:

```bash
python3 -m pip install numpy scikit-learn
```

If you are using a virtual environment, activate it first.

---

## Project Structure Used by the Solution

This guide assumes these files exist:

- `solution/solve.py`
- `solution/train_or_fit.py`
- `solution/evaluate_local.py`
- `solution/formula_model.py`
- `solution/run_command.txt`

And these data folders exist:

- `data/historical_races/`
- `data/test_cases/inputs/`
- `data/test_cases/expected_outputs/`

---

## Run from the Repository Root

All commands below should be run from the repository root folder.

Example:

```bash
cd /path/to/box-box-box
```

---

## Step 1 — Fit the Formula Model

This reads all historical race JSON files and creates:

- `solution/fitted_formula.json`

Run:

```bash
python3 solution/train_or_fit.py
```

What this does:

- loads historical races from `data/historical_races/`
- fits the deterministic formula model
- saves the fitted parameters to `solution/fitted_formula.json`

---

## Step 2 — Evaluate on Local Test Cases

This runs the solver against:

- `data/test_cases/inputs/*.json`

and compares predictions with:

- `data/test_cases/expected_outputs/*.json`

Run:

```bash
python3 solution/evaluate_local.py
```

This prints:

- which test cases matched exactly
- which ones missed
- final exact-match accuracy

---

## Step 3 — Run the Final Solver on One Input File

The challenge expects the solver to read a single JSON race from `stdin` and print one JSON result to `stdout`.

Example:

```bash
cat data/test_cases/inputs/test_001.json | python3 solution/solve.py
```

Expected output format:

```json
{
  "race_id": "TEST_001",
  "finishing_positions": ["D001", "D002", "D003", "..."]
}
```

---

## Step 4 — Submission Command

Your `solution/run_command.txt` should contain:

```txt
python3 solution/solve.py
```

This allows the challenge runner to do something like:

```bash
cat data/test_cases/inputs/test_001.json | python3 solution/solve.py
```

---

## Recommended Full Workflow

Run these in order:

```bash
python3 -m pip install numpy scikit-learn
python3 solution/train_or_fit.py
python3 solution/evaluate_local.py
cat data/test_cases/inputs/test_001.json | python3 solution/solve.py
```

---

## If `fitted_formula.json` Already Exists

If `solution/fitted_formula.json` is already present, `solution/solve.py` will use it directly.

You only need to rerun training if:

- you changed `formula_model.py`
- you changed the fitting logic
- you want to regenerate the model from the historical races

To refit, just run:

```bash
python3 solution/train_or_fit.py
```

---

## Common Issues

### 1. `python: command not found`

Use `python3` instead of `python`:

```bash
python3 solution/train_or_fit.py
```

### 2. Missing packages

Install them with:

```bash
python3 -m pip install numpy scikit-learn
```

### 3. No historical data found

Check that the folder exists and contains JSON files:

```bash
ls data/historical_races
```

### 4. No test files found

Check:

```bash
ls data/test_cases/inputs
ls data/test_cases/expected_outputs
```

---

## Clean Re-run

If you want to force a fresh fit, delete the old fitted model first:

```bash
rm -f solution/fitted_formula.json
python3 solution/train_or_fit.py
```

Then evaluate again:

```bash
python3 solution/evaluate_local.py
```

---

## Final Notes

- Run everything from the repo root.
- Do not print extra logs from `solution/solve.py` to `stdout`.
- The final submission should only output the required JSON.

