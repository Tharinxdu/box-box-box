#!/usr/bin/env python3
"""
Evaluate the deterministic simulator against local test cases.

Usage:
    python3 solution/evaluate_local.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from formula_model import (
    load_historical_races,
    load_model,
    save_model,
    fit_best_model,
    predict_finishing_positions,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = Path(__file__).resolve().parent / "fitted_formula.json"
HISTORICAL_DIR = REPO_ROOT / "data" / "historical_races"
TEST_INPUT_DIR = REPO_ROOT / "data" / "test_cases" / "inputs"
TEST_EXPECTED_DIR = REPO_ROOT / "data" / "test_cases" / "expected_outputs"


def ensure_model() -> dict:
    if MODEL_PATH.exists():
        return load_model(MODEL_PATH)

    races = load_historical_races(HISTORICAL_DIR)
    if not races:
        raise FileNotFoundError("No historical races found for fitting")

    model = fit_best_model(races)
    save_model(MODEL_PATH, model)
    return model


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main() -> int:
    model = ensure_model()

    input_files = sorted(TEST_INPUT_DIR.glob("*.json"))
    if not input_files:
        print("No local test inputs found.", file=sys.stderr)
        return 1

    total = 0
    exact = 0

    for input_path in input_files:
        expected_path = TEST_EXPECTED_DIR / input_path.name
        if not expected_path.exists():
            print(f"Missing expected output for {input_path.name}", file=sys.stderr)
            continue

        test_case = load_json(input_path)
        expected = load_json(expected_path)

        pred = predict_finishing_positions(test_case, model)
        truth = expected["finishing_positions"]

        ok = pred == truth
        total += 1
        exact += int(ok)

        print(f"{input_path.name}: {'OK' if ok else 'MISS'}")
        if not ok:
            print("  predicted:", pred)
            print("  expected :", truth)

    if total == 0:
        print("No comparable test cases found.", file=sys.stderr)
        return 1

    print()
    print(f"Exact-match accuracy: {exact}/{total} = {exact / total:.2%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())