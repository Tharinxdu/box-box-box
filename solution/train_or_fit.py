#!/usr/bin/env python3
"""
Fit the analytical formula model from historical races and save it.

Run this locally before submission so that solution/fitted_formula.json
is created and committed.
"""

from __future__ import annotations

import sys
from pathlib import Path

from formula_model import load_historical_races, fit_best_model, save_model

REPO_ROOT = Path(__file__).resolve().parent.parent
HISTORICAL_DIR = REPO_ROOT / "data" / "historical_races"
MODEL_PATH = Path(__file__).resolve().parent / "fitted_formula.json"


def main() -> int:
    races = load_historical_races(HISTORICAL_DIR)
    if not races:
        print("No historical races found.", file=sys.stderr)
        return 1

    print(f"Loaded {len(races)} historical races", file=sys.stderr)
    model = fit_best_model(races)
    save_model(MODEL_PATH, model)

    print(f"Saved fitted formula to: {MODEL_PATH}", file=sys.stderr)
    print(f"Chosen family: {model['family']}", file=sys.stderr)
    print(
        f"Validation exact-match during selection: "
        f"{model.get('selection_validation_exact_match', 0.0):.4f}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())