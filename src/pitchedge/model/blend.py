"""Blend model probabilities with de-vigged market probabilities.

``p_blend = w * p_model + (1 - w) * p_market``, renormalized. ``w`` comes from
config (``BLEND_W``); provenance is the Phase 4 backtest, not tuned inline here.
"""

from __future__ import annotations

import math
from typing import Sequence

from pitchedge import config

_SUM_TOL = 1e-9


def blend(
    p_model: Sequence[float],
    p_market: Sequence[float],
    w: float | None = None,
) -> tuple[float, float, float]:
    """Return renormalized ``(p_home, p_draw, p_away)`` in [0, 1].

    Parameters
    ----------
    p_model, p_market:
        Length-3 sequences ``(p_home, p_draw, p_away)``, each summing to ~1.
    w:
        Weight on the model vector; ``1 - w`` on the market. Defaults to
        ``config.BLEND_W``.
    """
    if w is None:
        w = config.BLEND_W
    if not (0.0 <= w <= 1.0):
        raise ValueError(f"blend weight w must be in [0, 1], got {w}")

    pm = tuple(float(x) for x in p_model)
    pk = tuple(float(x) for x in p_market)
    _assert_prob_triple(pm, label="p_model")
    _assert_prob_triple(pk, label="p_market")

    mixed = tuple(w * m + (1.0 - w) * k for m, k in zip(pm, pk, strict=True))
    total = sum(mixed)
    if total <= 0.0:
        raise ValueError("blended probabilities collapsed to zero; check inputs")
    return (mixed[0] / total, mixed[1] / total, mixed[2] / total)


def _assert_prob_triple(probs: tuple[float, float, float], *, label: str) -> None:
    total = sum(probs)
    if not math.isclose(total, 1.0, rel_tol=0, abs_tol=1e-6):
        raise ValueError(f"{label} must sum to ~1.0, got {total:.6f}")
    for p in probs:
        if p < -1e-12 or p > 1.0 + 1e-12:
            raise ValueError(f"{label} has invalid probability {p}")
