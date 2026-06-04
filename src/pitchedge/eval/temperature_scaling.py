"""Single-parameter temperature scaling for 3-way match probabilities.

Probabilities are in [0, 1] and renormalized after scaling. Temperature T > 1
softens (less confident); T < 1 sharpens. T = 1 leaves probabilities unchanged.
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np
from scipy.optimize import minimize_scalar

from pitchedge.eval.metrics import multiclass_log_loss

_EPS = 1e-15


def _as_array(probs: Sequence[float]) -> np.ndarray:
    p = np.asarray(probs, dtype=float)
    total = p.sum()
    if total <= 0.0:
        raise ValueError("probability triple must have positive sum")
    return p / total


def apply_temperature(
    probs: Sequence[float],
    temperature: float,
) -> tuple[float, float, float]:
    """Scale logits by ``temperature`` and return a normalized triple."""
    if temperature <= 0.0:
        raise ValueError("temperature must be positive")
    if abs(temperature - 1.0) < 1e-12:
        p = _as_array(probs)
        return (float(p[0]), float(p[1]), float(p[2]))

    p = _as_array(probs)
    logits = np.log(np.clip(p, _EPS, 1.0))
    scaled = logits / temperature
    scaled -= scaled.max()
    exp = np.exp(scaled)
    out = exp / exp.sum()
    return (float(out[0]), float(out[1]), float(out[2]))


def fit_temperature(
    probs_list: Sequence[Sequence[float]],
    outcomes: Sequence[int],
    *,
    t_min: float = 0.25,
    t_max: float = 5.0,
) -> float:
    """Find T minimizing mean log loss on ``probs_list`` / ``outcomes``."""

    def objective(t: float) -> float:
        scaled = [apply_temperature(p, t) for p in probs_list]
        return float(
            np.mean(
                [
                    multiclass_log_loss(s, o)
                    for s, o in zip(scaled, outcomes, strict=True)
                ]
            )
        )

    result = minimize_scalar(
        objective,
        bounds=(t_min, t_max),
        method="bounded",
    )
    if not result.success:
        return 1.0
    return float(result.x)


def scale_prob_list(
    probs_list: Sequence[Sequence[float]],
    temperature: float,
) -> list[tuple[float, float, float]]:
    """Apply ``temperature`` to each probability triple."""
    return [apply_temperature(p, temperature) for p in probs_list]
