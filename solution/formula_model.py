#!/usr/bin/env python3
"""
Deterministic formula-based F1 strategy model.

This version improves on the earlier polynomial approximation by fitting a
per-compound / per-age score basis from historical races.

Why this helps:
- The hidden simulator is deterministic and lap-based.
- A tire's effect is much more naturally represented by how many laps were run
  at age 1, age 2, age 3, ... on each compound than by only age_sum/age_sq_sum.
- The uploaded historical races also show exact tie-breaks by grid position,
  so prediction sorts by (score, grid_position).

Interpretation:
- We fit a deterministic score:
    score(strategy) =
        pit_term
      + track bias
      + track-specific compound usage bias
      + sum over compound/age bins of learned lap-cost contributions
      + temperature/base interactions on those bins

- Lower score = faster.
- Final ranking = ascending score, tie-break by grid position.

Dependencies:
- numpy
- scikit-learn
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from sklearn.linear_model import SGDClassifier

COMPOUNDS = ("SOFT", "MEDIUM", "HARD")
PAIR_I, PAIR_J = np.triu_indices(20, 1)


def load_json(path: str | Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: str | Path, payload) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def load_historical_races(historical_dir: str | Path) -> List[dict]:
    historical_dir = Path(historical_dir)
    races: List[dict] = []
    for path in sorted(historical_dir.glob("*.json")):
        data = load_json(path)
        if isinstance(data, list):
            races.extend(data)
    return races


def get_stints(strategy: dict, total_laps: int) -> List[Tuple[str, int, int, int]]:
    """
    Returns a list of:
        (compound, length, start_lap, end_lap)

    Pit stops happen at the END of the specified lap.
    """
    pits = sorted(strategy.get("pit_stops", []), key=lambda x: int(x["lap"]))
    stints: List[Tuple[str, int, int, int]] = []

    current_tire = strategy["starting_tire"]
    start_lap = 1

    for pit in pits:
        pit_lap = int(pit["lap"])
        length = pit_lap - start_lap + 1
        stints.append((current_tire, length, start_lap, pit_lap))
        current_tire = pit["to_tire"]
        start_lap = pit_lap + 1

    if start_lap <= total_laps:
        stints.append((current_tire, total_laps - start_lap + 1, start_lap, total_laps))

    return stints


def split_races(
    races: List[dict],
    valid_fraction: float = 0.2,
    random_state: int = 42,
) -> Tuple[List[dict], List[dict]]:
    ids = [race["race_id"] for race in races]
    rng = random.Random(random_state)
    rng.shuffle(ids)

    cut = int(round(len(ids) * (1.0 - valid_fraction)))
    train_ids = set(ids[:cut])
    valid_ids = set(ids[cut:])

    train_races = [race for race in races if race["race_id"] in train_ids]
    valid_races = [race for race in races if race["race_id"] in valid_ids]
    return train_races, valid_races


def build_context(races: List[dict]) -> dict:
    tracks = sorted({race["race_config"]["track"] for race in races})
    max_laps = max(int(race["race_config"]["total_laps"]) for race in races)
    max_age = max(80, max_laps)  # safe ceiling
    base_ref = float(np.mean([float(r["race_config"]["base_lap_time"]) for r in races]))
    return {
        "tracks": tracks,
        "max_age": int(max_age),
        "base_ref": base_ref,
    }


def feature_names_for_family(family: str, ctx: dict) -> List[str]:
    tracks = ctx["tracks"]
    max_age = ctx["max_age"]

    if family == "age_hist_track_v1":
        names = ["pit_time_total"]

        for track in tracks:
            names.append(f"track_bias::{track}")

        for track in tracks:
            for compound in COMPOUNDS:
                names.append(f"track_compound_laps::{track}::{compound}")

        for compound in COMPOUNDS:
            for age in range(1, max_age + 1):
                names.append(f"age_count::{compound}::{age}")
                names.append(f"temp_x_age_count::{compound}::{age}")
                names.append(f"base_x_age_count::{compound}::{age}")

        return names

    if family == "age_hist_v1":
        names = ["pit_time_total"]
        for compound in COMPOUNDS:
            for age in range(1, max_age + 1):
                names.append(f"age_count::{compound}::{age}")
                names.append(f"temp_x_age_count::{compound}::{age}")
                names.append(f"base_x_age_count::{compound}::{age}")
        return names

    raise ValueError(f"Unknown family: {family}")


def build_age_hist_features(race: dict, strategy: dict, ctx: dict, family: str) -> np.ndarray:
    rc = race["race_config"]
    total_laps = int(rc["total_laps"])
    track = rc["track"]
    track_temp = float(rc["track_temp"])
    base_lap_time = float(rc["base_lap_time"])
    pit_lane_time = float(rc["pit_lane_time"])

    temp_dev = track_temp - 30.0
    base_dev = base_lap_time - float(ctx["base_ref"])

    tracks = ctx["tracks"]
    track_to_idx = {t: i for i, t in enumerate(tracks)}
    track_idx = track_to_idx[track]
    max_age = ctx["max_age"]

    num_pits = len(strategy.get("pit_stops", []))
    pit_total = num_pits * pit_lane_time

    # Count how many laps are run at each age on each compound
    age_count = {
        compound: np.zeros(max_age, dtype=float)
        for compound in COMPOUNDS
    }
    compound_laps = {compound: 0.0 for compound in COMPOUNDS}

    stints = get_stints(strategy, total_laps)
    for compound, length, _, _ in stints:
        compound_laps[compound] += length
        capped = min(length, max_age)
        for age in range(1, capped + 1):
            age_count[compound][age - 1] += 1.0

    values: List[float] = [pit_total]

    if family == "age_hist_track_v1":
        # Track bias
        track_bias = np.zeros(len(tracks), dtype=float)
        track_bias[track_idx] = 1.0
        values.extend(track_bias.tolist())

        # Track-specific compound usage
        for t in tracks:
            for compound in COMPOUNDS:
                values.append(compound_laps[compound] if t == track else 0.0)

    for compound in COMPOUNDS:
        counts = age_count[compound]
        for count in counts:
            values.append(count)
            values.append(temp_dev * count)
            values.append(base_dev * count)

    return np.asarray(values, dtype=float)


def feature_vector_for_strategy(race: dict, strategy: dict, model_or_ctx: dict, family: str | None = None) -> np.ndarray:
    """
    Compatible with both:
    - fitting time: feature_vector_for_strategy(race, strategy, ctx, family)
    - prediction time: feature_vector_for_strategy(race, strategy, model)
    """
    if family is None:
        family = model_or_ctx["family"]
        ctx = {
            "tracks": model_or_ctx["tracks"],
            "max_age": model_or_ctx["max_age"],
            "base_ref": model_or_ctx["base_ref"],
        }
    else:
        ctx = model_or_ctx

    return build_age_hist_features(race, strategy, ctx, family)


def race_feature_matrix(
    race: dict,
    model_or_ctx: dict,
    family: str | None = None,
    order: str = "finish",
) -> Tuple[np.ndarray, List[Tuple[int, str]]]:
    """
    order='finish' -> rows follow true finishing order (for fitting)
    order='grid'   -> rows follow grid position ascending (for prediction/tie-break)
    """
    driver_lookup = {}
    for pos_key, strategy in race["strategies"].items():
        pos = int(pos_key.replace("pos", ""))
        driver_lookup[strategy["driver_id"]] = (pos, strategy)

    rows = []
    meta: List[Tuple[int, str]] = []

    if order == "finish":
        for driver_id in race["finishing_positions"]:
            pos, strategy = driver_lookup[driver_id]
            rows.append(feature_vector_for_strategy(race, strategy, model_or_ctx, family))
            meta.append((pos, driver_id))
        return np.vstack(rows), meta

    if order == "grid":
        for pos_key, strategy in sorted(race["strategies"].items(), key=lambda kv: int(kv[0].replace("pos", ""))):
            pos = int(pos_key.replace("pos", ""))
            rows.append(feature_vector_for_strategy(race, strategy, model_or_ctx, family))
            meta.append((pos, strategy["driver_id"]))
        return np.vstack(rows), meta

    raise ValueError(f"Unknown order: {order}")


def compute_feature_scale(races: List[dict], ctx: dict, family: str) -> np.ndarray:
    """
    Pairwise diffs cancel means, so we only scale by std for SGD stability.
    """
    count = 0
    sum_sq = None

    for race in races:
        for strategy in race["strategies"].values():
            vec = feature_vector_for_strategy(race, strategy, ctx, family)
            if sum_sq is None:
                sum_sq = np.zeros_like(vec)
            sum_sq += vec * vec
            count += 1

    if sum_sq is None or count == 0:
        raise ValueError("No races found while computing feature scale")

    std = np.sqrt(sum_sq / float(count))
    std[std < 1e-8] = 1.0
    return std


def build_pair_examples_for_race(
    race: dict,
    ctx: dict,
    family: str,
    scale: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Rows are ordered by true finish:
        faster rows first, slower rows later

    For each pair (i, j) with i < j:
        slower_minus_faster = arr[j] - arr[i] -> class 1
        faster_minus_slower = arr[i] - arr[j] -> class 0

    Learned linear score is interpreted as:
        larger score = slower
        lower score  = faster
    """
    arr, _ = race_feature_matrix(race, ctx, family, order="finish")
    diffs = arr[PAIR_J] - arr[PAIR_I]
    diffs = diffs / scale

    x = np.vstack([diffs, -diffs])
    y = np.concatenate(
        [
            np.ones(len(diffs), dtype=int),
            np.zeros(len(diffs), dtype=int),
        ]
    )
    return x, y


def predict_scores_for_race(race: dict, model: dict) -> List[Tuple[float, int, str]]:
    family = model["family"]
    coef = np.asarray(model["coef"], dtype=float)

    arr, meta = race_feature_matrix(race, model, None, order="grid")
    scores = arr.dot(coef)

    out = []
    for score, (pos, driver_id) in zip(scores, meta):
        out.append((float(score), pos, driver_id))
    return out


def predict_finishing_positions(race: dict, model: dict) -> List[str]:
    """
    Lower score = faster.
    Tie-break = lower grid position.

    The uploaded historical data shows this exact tie-break behavior.
    """
    scored = predict_scores_for_race(race, model)
    scored.sort(key=lambda x: (x[0], x[1]))
    return [driver_id for _, _, driver_id in scored]


def exact_match_accuracy(races: List[dict], model: dict) -> float:
    if not races:
        return 0.0

    correct = 0
    for race in races:
        pred = predict_finishing_positions(race, model)
        if pred == race["finishing_positions"]:
            correct += 1
    return correct / len(races)


def fit_family_model(
    train_races: List[dict],
    valid_races: List[dict],
    ctx: dict,
    family: str,
    alpha: float,
    epochs: int = 2,
    random_state: int = 42,
) -> dict:
    """
    Fits a linear pairwise classifier over analytical formula features.
    Converts scaled coefficients back to raw closed-form coefficients.
    """
    scale = compute_feature_scale(train_races, ctx, family)

    clf = SGDClassifier(
        loss="log_loss",
        penalty="l2",
        alpha=alpha,
        fit_intercept=False,
        random_state=random_state,
        learning_rate="optimal",
        max_iter=1,
        tol=None,
        shuffle=False,
    )

    classes = np.array([0, 1], dtype=int)
    train_order = list(train_races)
    first_call = True

    for epoch in range(epochs):
        rng = random.Random(random_state + epoch)
        rng.shuffle(train_order)

        for race in train_order:
            x, y = build_pair_examples_for_race(race, ctx, family, scale)
            if first_call:
                clf.partial_fit(x, y, classes=classes)
                first_call = False
            else:
                clf.partial_fit(x, y)

    raw_coef = (clf.coef_[0] / scale).astype(float)

    model = {
        "family": family,
        "coef": raw_coef.tolist(),
        "feature_names": feature_names_for_family(family, ctx),
        "alpha": alpha,
        "epochs": epochs,
        "tracks": ctx["tracks"],
        "max_age": ctx["max_age"],
        "base_ref": ctx["base_ref"],
    }

    model["validation_exact_match"] = exact_match_accuracy(valid_races, model)
    return model


def fit_default_model(
    races: List[dict],
    random_state: int = 42,
) -> dict:
    """
    Fast fallback used by solve.py if no fitted formula file is present.
    """
    ctx = build_context(races)
    family = "age_hist_track_v1"
    alpha = 1e-5
    scale = compute_feature_scale(races, ctx, family)

    clf = SGDClassifier(
        loss="log_loss",
        penalty="l2",
        alpha=alpha,
        fit_intercept=False,
        random_state=random_state,
        learning_rate="optimal",
        max_iter=1,
        tol=None,
        shuffle=False,
    )

    classes = np.array([0, 1], dtype=int)
    race_order = list(races)
    first_call = True

    for epoch in range(2):
        rng = random.Random(random_state + epoch)
        rng.shuffle(race_order)

        for race in race_order:
            x, y = build_pair_examples_for_race(race, ctx, family, scale)
            if first_call:
                clf.partial_fit(x, y, classes=classes)
                first_call = False
            else:
                clf.partial_fit(x, y)

    raw_coef = (clf.coef_[0] / scale).astype(float)

    return {
        "family": family,
        "coef": raw_coef.tolist(),
        "feature_names": feature_names_for_family(family, ctx),
        "alpha": alpha,
        "epochs": 2,
        "tracks": ctx["tracks"],
        "max_age": ctx["max_age"],
        "base_ref": ctx["base_ref"],
        "note": "Auto-fitted default model",
    }


def fit_best_model(
    races: List[dict],
    random_state: int = 42,
) -> dict:
    """
    Model selection on a validation split, then final refit on all races.

    The key improvement over the older version is the age-histogram family,
    which lets the fit learn a much more detailed degradation curve than
    a quadratic summary.
    """
    train_races, valid_races = split_races(races, valid_fraction=0.2, random_state=random_state)
    ctx = build_context(races)

    candidates = [
        ("age_hist_track_v1", 3e-6),
        ("age_hist_track_v1", 1e-5),
        ("age_hist_v1", 1e-5),
        ("age_hist_v1", 3e-5),
    ]

    best_model = None
    best_key = None

    for family, alpha in candidates:
        model = fit_family_model(
            train_races=train_races,
            valid_races=valid_races,
            ctx=ctx,
            family=family,
            alpha=alpha,
            epochs=2,
            random_state=random_state,
        )

        # Prefer higher exact-match. If tied, prefer the track-aware family.
        key = (
            model["validation_exact_match"],
            1.0 if family == "age_hist_track_v1" else 0.0,
            -alpha,
        )

        if best_model is None or key > best_key:
            best_model = model
            best_key = key

    chosen_family = best_model["family"]
    chosen_alpha = best_model["alpha"]

    scale = compute_feature_scale(races, ctx, chosen_family)

    clf = SGDClassifier(
        loss="log_loss",
        penalty="l2",
        alpha=chosen_alpha,
        fit_intercept=False,
        random_state=random_state,
        learning_rate="optimal",
        max_iter=1,
        tol=None,
        shuffle=False,
    )

    classes = np.array([0, 1], dtype=int)
    race_order = list(races)
    first_call = True

    for epoch in range(3):
        rng = random.Random(random_state + epoch)
        rng.shuffle(race_order)

        for race in race_order:
            x, y = build_pair_examples_for_race(race, ctx, chosen_family, scale)
            if first_call:
                clf.partial_fit(x, y, classes=classes)
                first_call = False
            else:
                clf.partial_fit(x, y)

    raw_coef = (clf.coef_[0] / scale).astype(float)

    final_model = {
        "family": chosen_family,
        "coef": raw_coef.tolist(),
        "feature_names": feature_names_for_family(chosen_family, ctx),
        "alpha": chosen_alpha,
        "epochs": 3,
        "tracks": ctx["tracks"],
        "max_age": ctx["max_age"],
        "base_ref": ctx["base_ref"],
        "selection_validation_exact_match": best_model["validation_exact_match"],
    }
    return final_model


def load_model(path: str | Path) -> dict:
    return load_json(path)


def save_model(path: str | Path, model: dict) -> None:
    save_json(path, model)