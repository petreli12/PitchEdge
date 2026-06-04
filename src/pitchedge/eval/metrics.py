"""Proper scoring rules for 3-way (H/D/A) match predictions.

All probabilities are in [0, 1]. Outcomes are encoded 0=home, 1=draw, 2=away.
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np

OUTCOME_HOME = 0
OUTCOME_DRAW = 1
OUTCOME_AWAY = 2

_EPS = 1e-15


def outcome_from_goals(home_goals: int, away_goals: int) -> int:
    """Return outcome code for a finished match."""
    if home_goals > away_goals:
        return OUTCOME_HOME
    if home_goals < away_goals:
        return OUTCOME_AWAY
    return OUTCOME_DRAW


def _as_triple(probs: Sequence[float]) -> tuple[float, float, float]:
    p = (float(probs[0]), float(probs[1]), float(probs[2]))
    total = sum(p)
    if total <= 0.0:
        raise ValueError("probability triple must have positive sum")
    return (p[0] / total, p[1] / total, p[2] / total)


def multiclass_brier(probs: Sequence[float], outcome: int) -> float:
    """Multi-class Brier score for one row (lower is better)."""
    p = _as_triple(probs)
    y = [0.0, 0.0, 0.0]
    y[outcome] = 1.0
    return sum((p[i] - y[i]) ** 2 for i in range(3))


def multiclass_log_loss(probs: Sequence[float], outcome: int) -> float:
    """Multi-class log loss for one row (lower is better)."""
    p = _as_triple(probs)
    return -math.log(max(p[outcome], _EPS))


def mean_brier(probs_list: Sequence[Sequence[float]], outcomes: Sequence[int]) -> float:
    """Mean Brier over rows."""
    if not probs_list:
        return float("nan")
    return float(
        np.mean([multiclass_brier(p, o) for p, o in zip(probs_list, outcomes, strict=True)])
    )


def mean_log_loss(probs_list: Sequence[Sequence[float]], outcomes: Sequence[int]) -> float:
    """Mean log loss over rows."""
    if not probs_list:
        return float("nan")
    return float(
        np.mean(
            [multiclass_log_loss(p, o) for p, o in zip(probs_list, outcomes, strict=True)]
        )
    )


def expected_calibration_error(
    probs_list: Sequence[Sequence[float]],
    outcomes: Sequence[int],
    *,
    n_bins: int = 10,
) -> float:
    """ECE using max-confidence binning (Guo et al., multiclass extension).

    Bin by predicted max probability; compare bin accuracy to mean confidence.
    Probabilities in [0, 1]; lower ECE is better calibrated.
    """
    if not probs_list:
        return float("nan")

    confidences: list[float] = []
    correct: list[float] = []
    for probs, outcome in zip(probs_list, outcomes, strict=True):
        p = _as_triple(probs)
        pred = int(np.argmax(p))
        confidences.append(max(p))
        correct.append(1.0 if pred == outcome else 0.0)

    conf = np.asarray(confidences, dtype=float)
    acc = np.asarray(correct, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    n = len(conf)
    ece = 0.0

    for i in range(n_bins):
        lo_edge = edges[i]
        hi_edge = edges[i + 1]
        if i < n_bins - 1:
            mask = (conf >= lo_edge) & (conf < hi_edge)
        else:
            mask = (conf >= lo_edge) & (conf <= hi_edge)
        count = int(mask.sum())
        if count == 0:
            continue
        bin_conf = float(conf[mask].mean())
        bin_acc = float(acc[mask].mean())
        ece += (count / n) * abs(bin_acc - bin_conf)

    return float(ece)


def home_win_ece(
    probs_list: Sequence[Sequence[float]],
    outcomes: Sequence[int],
    *,
    n_bins: int = 10,
) -> float:
    """ECE for P(home win) vs home-win indicator (aligns with reliability plots)."""
    if not probs_list:
        return float("nan")

    predicted = np.array([_as_triple(p)[0] for p in probs_list], dtype=float)
    observed = (np.asarray(outcomes) == OUTCOME_HOME).astype(float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    n = len(predicted)
    ece = 0.0

    for i in range(n_bins):
        lo_edge = edges[i]
        hi_edge = edges[i + 1]
        if i < n_bins - 1:
            mask = (predicted >= lo_edge) & (predicted < hi_edge)
        else:
            mask = (predicted >= lo_edge) & (predicted <= hi_edge)
        count = int(mask.sum())
        if count == 0:
            continue
        bin_conf = float(predicted[mask].mean())
        bin_acc = float(observed[mask].mean())
        ece += (count / n) * abs(bin_acc - bin_conf)

    return float(ece)


def paired_log_loss_difference(
    model_probs: Sequence[Sequence[float]],
    market_probs: Sequence[Sequence[float]],
    outcomes: Sequence[int],
) -> dict[str, float]:
    """Paired model-minus-market log loss per row and summary statistics.

    Returns mean difference, sample std, standard error (ddof=1), and n.
    """
    if len(model_probs) != len(market_probs) or len(model_probs) != len(outcomes):
        raise ValueError("model_probs, market_probs, and outcomes must align")
    n = len(outcomes)
    if n == 0:
        return {
            "mean_diff": float("nan"),
            "std_diff": float("nan"),
            "se_diff": float("nan"),
            "n": 0.0,
        }

    diffs = [
        multiclass_log_loss(m, o) - multiclass_log_loss(k, o)
        for m, k, o in zip(model_probs, market_probs, outcomes, strict=True)
    ]
    arr = np.asarray(diffs, dtype=float)
    std = float(arr.std(ddof=1)) if n > 1 else 0.0
    se = std / math.sqrt(n) if n > 0 else float("nan")
    return {
        "mean_diff": float(arr.mean()),
        "std_diff": std,
        "se_diff": se,
        "n": float(n),
    }
