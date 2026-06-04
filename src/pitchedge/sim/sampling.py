"""Precomputed score-matrix CDFs and vectorized scoreline sampling."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pitchedge import config
from pitchedge.model.dixon_coles import DixonColesModel, score_matrix
from pitchedge.model.venues import dixon_coles_neutral_for_wc_fixture

MatchupPair = tuple[int, int]


@dataclass(frozen=True)
class MatchupCDF:
    """Flattened cumulative distribution for one home/away WC fixture row.

    ``cdf`` has length ``K + 1`` with ``cdf[0] == 0`` and ``cdf[-1] == 1``.
    Sampling uses ``searchsorted`` on uniform draws (equivalent to ``rng.choice``
    with ``p=flat``).
    """

    cdf: np.ndarray
    n_cols: int
    neutral: bool | None = None

    @property
    def n_cells(self) -> int:
        return self.cdf.size - 1

    @property
    def n_rows(self) -> int:
        return self.n_cells // self.n_cols

    def sample_indices(self, u: np.ndarray) -> np.ndarray:
        """Map uniforms in [0, 1) to flat matrix indices (vectorized)."""
        u = np.asarray(u, dtype=np.float64)
        return np.searchsorted(self.cdf, u, side="right") - 1

    def indices_to_scorelines(
        self, flat_idx: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Convert flat indices to ``(home_goals, away_goals)`` arrays."""
        return np.divmod(flat_idx, self.n_cols)


def build_matchup_cdf(
    matrix: np.ndarray,
    *,
    neutral: bool | None = None,
) -> MatchupCDF:
    flat = np.asarray(matrix, dtype=np.float64).ravel()
    total = flat.sum()
    if total <= 0.0:
        raise ValueError("score matrix must have positive mass")
    cdf = np.empty(flat.size + 1, dtype=np.float64)
    cdf[0] = 0.0
    cdf[1:] = np.cumsum(flat / total)
    cdf[-1] = 1.0
    return MatchupCDF(cdf=cdf, n_cols=matrix.shape[1], neutral=neutral)


def probability_mass_from_cdf(entry: MatchupCDF) -> np.ndarray:
    """Reshape ``np.diff(cdf)`` to the goal grid (rows=home, cols=away)."""
    return np.diff(entry.cdf).reshape(entry.n_rows, entry.n_cols)


def decode_scoreline_from_uniform(entry: MatchupCDF, u: float) -> tuple[int, int]:
    """Return the scoreline for a single ``u`` in ``[0, 1)`` (inverse-CDF decode)."""
    idx = int(entry.sample_indices(np.asarray([u], dtype=np.float64))[0])
    return divmod(idx, entry.n_cols)


def build_matchup_cdf_for_pair(
    model: DixonColesModel,
    home_wc: int,
    away_wc: int,
    wc_to_model_id: dict[int, int],
    *,
    max_goals: int | None = None,
) -> tuple[MatchupCDF, np.ndarray, bool]:
    """Build one catalog entry and return ``(cdf_entry, score_matrix, neutral)``."""
    mg = max_goals if max_goals is not None else config.DC_MAX_GOALS
    neutral = dixon_coles_neutral_for_wc_fixture(home_wc)
    mat = score_matrix(
        model,
        wc_to_model_id[home_wc],
        wc_to_model_id[away_wc],
        max_goals=mg,
        neutral=neutral,
    )
    return build_matchup_cdf(mat, neutral=neutral), mat, neutral


def build_matchup_catalog(
    model: DixonColesModel,
    wc_team_ids: list[int],
    wc_to_model_id: dict[int, int],
    *,
    max_goals: int | None = None,
) -> dict[MatchupPair, MatchupCDF]:
    """Precompute CDFs for every ordered WC pairing (home venue policy from ``home_id``)."""
    mg = max_goals if max_goals is not None else config.DC_MAX_GOALS
    catalog: dict[MatchupPair, MatchupCDF] = {}
    for home_wc in wc_team_ids:
        for away_wc in wc_team_ids:
            if home_wc == away_wc:
                continue
            key = (home_wc, away_wc)
            neutral = dixon_coles_neutral_for_wc_fixture(home_wc)
            mat = score_matrix(
                model,
                wc_to_model_id[home_wc],
                wc_to_model_id[away_wc],
                max_goals=mg,
                neutral=neutral,
            )
            catalog[key] = build_matchup_cdf(mat, neutral=neutral)
    return catalog


def sample_from_catalog(
    catalog: dict[MatchupPair, MatchupCDF],
    home_wc_id: int,
    away_wc_id: int,
    u: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample scorelines for ``u`` (any shape); returns goals with same shape."""
    cdf = catalog[(home_wc_id, away_wc_id)]
    shape = u.shape
    flat_idx = cdf.sample_indices(u.ravel())
    hg, ag = cdf.indices_to_scorelines(flat_idx)
    return hg.reshape(shape), ag.reshape(shape)


def sample_one(
    catalog: dict[MatchupPair, MatchupCDF],
    home_wc_id: int,
    away_wc_id: int,
    rng: np.random.Generator,
) -> tuple[int, int]:
    """Single draw (knockout path)."""
    u = rng.random()
    idx = int(catalog[(home_wc_id, away_wc_id)].sample_indices(np.array([u]))[0])
    cols = catalog[(home_wc_id, away_wc_id)].n_cols
    return divmod(idx, cols)
