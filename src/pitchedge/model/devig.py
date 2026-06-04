"""De-vig decimal 3-way (1X2) odds into calibrated probabilities.

All returned probabilities are in [0, 1] and sum to 1 (within floating tolerance).

Methods (see docs/ARCHITECTURE.md section 3.1):
  * ``proportional`` (default): p_i = (1/d_i) / sum(1/d_j). MVP calibration floor.
  * ``shin``: Shin's method for favorite-longshot bias (slower; optional).
"""

from __future__ import annotations

import math
from typing import Literal

import numpy as np

DevigMethod = Literal["proportional", "shin"]

_SUM_TOL = 1e-9


def _validate_odds(home_odds: float, draw_odds: float, away_odds: float) -> None:
    for label, value in (
        ("home_odds", home_odds),
        ("draw_odds", draw_odds),
        ("away_odds", away_odds),
    ):
        if value <= 1.0:
            raise ValueError(f"{label} must be decimal odds > 1.0, got {value}")


def implied_raw(home_odds: float, draw_odds: float, away_odds: float) -> tuple[float, float, float]:
    """Return raw implied probabilities ``(r_home, r_draw, r_away)`` before de-vig.

    Sum is the bookmaker overround (typically > 1).
    """
    _validate_odds(home_odds, draw_odds, away_odds)
    return (1.0 / home_odds, 1.0 / draw_odds, 1.0 / away_odds)


def devig_proportional(
    home_odds: float, draw_odds: float, away_odds: float
) -> tuple[float, float, float]:
    """Proportional normalization: ``p_i = r_i / sum(r)``."""
    rh, rd, ra = implied_raw(home_odds, draw_odds, away_odds)
    total = rh + rd + ra
    return (rh / total, rd / total, ra / total)


def devig_shin(
    home_odds: float, draw_odds: float, away_odds: float, *, max_iter: int = 100
) -> tuple[float, float, float]:
    """Shin's method for a 3-outcome market (iterative z).

    Reference: Shin (1993); implementation follows the standard fixed-point
    approach used for 1X2 de-vigging.
    """
    implied = np.array(implied_raw(home_odds, draw_odds, away_odds), dtype=float)
    n = len(implied)
    z = 0.0
    for _ in range(max_iter):
        denom = 2.0 * (1.0 - z)
        inner = np.sqrt(z * z + 4.0 * (1.0 - z) * (implied * implied) / denom)
        probs = ((inner - z) / denom) * (1.0 - z)
        z_next = float(np.sum(probs * probs) / n)
        if abs(z_next - z) < 1e-10:
            z = z_next
            break
        z = z_next
    else:
        raise RuntimeError("Shin de-vig did not converge")

    denom = 2.0 * (1.0 - z)
    inner = np.sqrt(z * z + 4.0 * (1.0 - z) * (implied * implied) / denom)
    probs = ((inner - z) / denom) * (1.0 - z)
    total = float(probs.sum())
    return (float(probs[0] / total), float(probs[1] / total), float(probs[2] / total))


def devig_three_way(
    home_odds: float,
    draw_odds: float,
    away_odds: float,
    *,
    method: DevigMethod = "proportional",
) -> tuple[float, float, float]:
    """Return de-vigged ``(p_home, p_draw, p_away)`` summing to 1.

    Parameters
    ----------
    home_odds, draw_odds, away_odds:
        Decimal odds strictly greater than 1.0.
    method:
        ``proportional`` (default) or ``shin``.
    """
    if method == "proportional":
        probs = devig_proportional(home_odds, draw_odds, away_odds)
    elif method == "shin":
        probs = devig_shin(home_odds, draw_odds, away_odds)
    else:
        raise ValueError(f"unknown de-vig method: {method!r}")

    total = sum(probs)
    if not math.isclose(total, 1.0, rel_tol=0, abs_tol=_SUM_TOL):
        raise ValueError(f"de-vigged probabilities must sum to 1, got {total}")
    return probs


def assert_normalized(probs: tuple[float, float, float]) -> None:
    """Raise if the vector is not a valid 3-way probability mass."""
    total = sum(probs)
    if not math.isclose(total, 1.0, rel_tol=0, abs_tol=_SUM_TOL):
        raise ValueError(f"probabilities must sum to 1, got {total}")
    for p in probs:
        if p < -1e-12 or p > 1.0 + 1e-12:
            raise ValueError(f"probability out of [0,1]: {p}")
