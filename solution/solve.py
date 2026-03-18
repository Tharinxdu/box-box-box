#!/usr/bin/env python3
"""
Final stdin -> stdout solver for submission.

Behavior:
- Reads one race JSON from stdin
- Loads solution/fitted_formula.json if it exists
- If missing, silently auto-fits a default analytical model from data/historical_races/
- Outputs:
    {
      "race_id": "...",
      "finishing_positions": [...]
    }

No extra stdout logging.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from formula_model import (
    load_historical_races,
    load_model,
    save_model,
    fit_default_model,
    predict_finishing_positions,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = Path(__file__).resolve().parent / "fitted_formula.json"
HISTORICAL_DIR = REPO_ROOT / "data" / "historical_races"


def ensure_model() -> dict:
    if MODEL_PATH.exists():
        return load_model(MODEL_PATH)

    races = load_historical_races(HISTORICAL_DIR)
    if not races:
        raise FileNotFoundError(
            "solution/fitted_formula.json not found and no historical races available for auto-fit"
        )

    model = fit_default_model(races)
    save_model(MODEL_PATH, model)
    return model


def main() -> int:
    test_case = json.load(sys.stdin)
    model = ensure_model()

    finishing_positions = predict_finishing_positions(test_case, model)

    output = {
        "race_id": test_case["race_id"],
        "finishing_positions": finishing_positions,
    }

    print(json.dumps(output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())