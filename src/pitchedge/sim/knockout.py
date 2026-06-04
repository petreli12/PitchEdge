"""Knockout-stage simulation (extra time + penalties on draws)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from pitchedge.sim.sampling import MatchupPair, sample_one

if TYPE_CHECKING:
    from pitchedge.model.dixon_coles import DixonColesModel


def play_knockout_match(
    model: DixonColesModel | None,
    home_id: int,
    away_id: int,
    wc_to_model_id: dict[int, int],
    rng: np.random.Generator,
    *,
    max_goals: int | None = None,
    catalog: dict[MatchupPair, object] | None = None,
) -> int:
    """Return winning ``team_id`` after 90', optional ET, then penalties.

    Extra time re-samples from the same score matrix (documented simplification).
    Penalties are 50/50 with no Elo nudge (ARCHITECTURE §3.5 default).

    Requires ``catalog`` for the fast path (``model`` may be ``None``).
    """
    if catalog is None:
        from pitchedge.sim.scoreline import sample_scoreline

        hg, ag = sample_scoreline(
            model, home_id, away_id, wc_to_model_id, rng, max_goals=max_goals
        )
    else:
        hg, ag = sample_one(catalog, home_id, away_id, rng)

    if hg > ag:
        return home_id
    if ag > hg:
        return away_id

    if catalog is None:
        from pitchedge.sim.scoreline import sample_scoreline

        ehg, eag = sample_scoreline(
            model, home_id, away_id, wc_to_model_id, rng, max_goals=max_goals
        )
    else:
        ehg, eag = sample_one(catalog, home_id, away_id, rng)

    total_h = hg + ehg
    total_a = ag + eag
    if total_h > total_a:
        return home_id
    if total_a > total_h:
        return away_id

    return home_id if rng.random() < 0.5 else away_id
