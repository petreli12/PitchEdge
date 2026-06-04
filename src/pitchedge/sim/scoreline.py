"""Sample match scorelines from the standalone Dixon-Coles score matrix."""

from __future__ import annotations

import numpy as np

from pitchedge import config
from pitchedge.model.dixon_coles import DixonColesModel, score_matrix
from pitchedge.model.venues import dixon_coles_neutral_for_wc_fixture
from pitchedge.sim.sampling import (
    MatchupPair,
    MatchupCDF,
    build_matchup_catalog,
    sample_from_catalog,
    sample_one,
    build_matchup_cdf,
)

_matrix_cache: dict[tuple[int, int, bool], object] = {}


def clear_score_matrix_cache() -> None:
    """Clear legacy matrix cache (for tests)."""
    _matrix_cache.clear()


def sample_scoreline(
    model: DixonColesModel,
    home_wc_id: int,
    away_wc_id: int,
    wc_to_model_id: dict[int, int],
    rng: np.random.Generator,
    *,
    max_goals: int | None = None,
    host_home: bool | None = None,
    catalog: dict[MatchupPair, object] | None = None,
) -> tuple[int, int]:
    """Sample one scoreline from the published model matrix (not blend / market).

    When ``catalog`` is provided, uses precomputed CDF + ``searchsorted`` (no matrix
    rebuild). Otherwise builds a one-off CDF for this pairing.
    """
    if host_home is not None:
        raise ValueError("host_home override not supported with catalog path; use WC home_id")

    if catalog is not None:
        return sample_one(catalog, home_wc_id, away_wc_id, rng)

    mg = max_goals if max_goals is not None else config.DC_MAX_GOALS
    neutral = dixon_coles_neutral_for_wc_fixture(home_wc_id)
    mat = score_matrix(
        model,
        wc_to_model_id[home_wc_id],
        wc_to_model_id[away_wc_id],
        max_goals=mg,
        neutral=neutral,
    )
    cdf = build_matchup_cdf(mat)
    u = rng.random()
    idx = int(cdf.sample_indices(np.array([u]))[0])
    return divmod(idx, cdf.n_cols)
