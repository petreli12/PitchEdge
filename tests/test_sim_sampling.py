"""Deterministic verification of CDF catalog sampling and fast/legacy alignment.

Distributional equivalence uses **aligned RNG streams** (not 20k Monte Carlo SE):
pre-drawn ``group_uniforms`` of shape ``(n_sims, 72)`` plus fresh
``np.random.default_rng(seed)`` for tiebreaks/knockouts on each path. Fast sim
interleaves group resolve + knockout per replicate (same order as legacy). Bound:
``max_t |ΔP(win)| = 0`` at ``n=200``, ``seed=20260611``.
"""

from __future__ import annotations

import numpy as np
import pytest

from pitchedge.model.dixon_coles import DixonColesModel
from pitchedge.model.venues import dixon_coles_neutral_for_wc_fixture, is_wc_2026_host
from pitchedge.sim.group_batch import (
    accumulate_standings,
    decode_group_scorelines,
    idx_to_team_id_map,
    prepare_group_batch,
    resolve_group_outcomes,
)
from pitchedge.sim.group_stage import (
    GroupStanding,
    _apply_result,
    rank_best_thirds,
    rank_group,
)
from pitchedge.sim.sampling import (
    build_matchup_catalog,
    build_matchup_cdf,
    build_matchup_cdf_for_pair,
    decode_scoreline_from_uniform,
    probability_mass_from_cdf,
    sample_from_catalog,
    sample_one,
)
from pitchedge.sim.tournament import run_monte_carlo_fast, run_monte_carlo_legacy
from pitchedge.sim.wc_teams import load_group_fixtures, load_wc_teams, teams_by_group, wc_id_to_model_id


def _toy_model() -> DixonColesModel:
    return DixonColesModel(
        attack={i: (i % 7) * 0.04 - 0.12 for i in range(1, 200)},
        defense={i: ((i % 5) * 0.03 - 0.06) for i in range(1, 200)},
        home_adv=0.22,
        rho=-0.06,
        xi=0.0,
    )


@pytest.fixture(autouse=True)
def _clear_matrix_cache():
    from pitchedge.sim.scoreline import clear_score_matrix_cache

    clear_score_matrix_cache()
    yield
    clear_score_matrix_cache()


@pytest.fixture
def wc_context():
    teams = load_wc_teams()
    fixtures = load_group_fixtures()
    by_group = teams_by_group(teams)
    team_ids_by_group = {g: [t.team_id for t in ts] for g, ts in by_group.items()}
    wc_to_model = wc_id_to_model_id(teams)
    wc_team_ids = [t.team_id for t in teams]
    return fixtures, team_ids_by_group, wc_to_model, wc_team_ids


@pytest.mark.parametrize(
    "home_wc,away_wc",
    [
        (9, 2),
        (37, 16),
        (1, 2),
        (13, 10),
        (5, 45),
        (29, 48),
    ],
)
def test_catalog_cdf_matches_score_matrix(home_wc: int, away_wc: int, wc_context) -> None:
    """np.diff(cdf) reshaped equals score_matrix; neutral flag matches co-host rule."""
    _, _, wc_to_model, _ = wc_context
    model = _toy_model()
    entry, mat, neutral = build_matchup_cdf_for_pair(
        model, home_wc, away_wc, wc_to_model, max_goals=8
    )
    assert neutral == dixon_coles_neutral_for_wc_fixture(home_wc)
    assert entry.neutral == neutral
    if is_wc_2026_host(home_wc):
        assert neutral is False
    else:
        assert neutral is True

    mass = probability_mass_from_cdf(entry)
    assert mass.shape == mat.shape
    np.testing.assert_allclose(mass, mat, rtol=0.0, atol=1e-9)
    np.testing.assert_allclose(mass.sum(), 1.0, rtol=0.0, atol=1e-12)


def test_hand_checked_uniform_decode() -> None:
    """Fixed 3x3 matrix: decode u into the correct CDF bin."""
    mat = np.array(
        [
            [0.10, 0.05, 0.02],
            [0.08, 0.20, 0.05],
            [0.03, 0.07, 0.40],
        ],
        dtype=np.float64,
    )
    entry = build_matchup_cdf(mat, neutral=True)
    assert entry.n_rows == 3 and entry.n_cols == 3

    cases = [
        (0.0, 0, 0),
        (0.09, 0, 0),
        (0.11, 0, 1),
        (0.16, 0, 2),
        (0.20, 1, 0),
        (0.35, 1, 1),
        (0.55, 2, 1),
        (0.70, 2, 2),
        (0.999, 2, 2),
    ]
    for u, exp_h, exp_a in cases:
        h, a = decode_scoreline_from_uniform(entry, u)
        assert (h, a) == (exp_h, exp_a), f"u={u}"


class _FixedUniformRng:
    """Minimal RNG stub returning one pre-set uniform per call."""

    def __init__(self, u: float) -> None:
        self._u = u

    def random(self) -> float:
        return self._u


@pytest.mark.parametrize("home_wc,away_wc", [(1, 37), (13, 37), (37, 1)])
def test_sample_one_matches_decode(home_wc: int, away_wc: int, wc_context) -> None:
    """sample_one and decode_scoreline_from_uniform agree on fixed uniforms."""
    _, _, wc_to_model, _ = wc_context
    model = _toy_model()
    catalog = build_matchup_catalog(model, [1, 13, 37], wc_to_model, max_goals=8)
    entry = catalog[(home_wc, away_wc)]
    for u in (0.0, 0.17, 0.42, 0.88, 0.999):
        dec = decode_scoreline_from_uniform(entry, u)
        one = sample_one(catalog, home_wc, away_wc, _FixedUniformRng(u))
        assert one == dec


def test_vectorized_decode_matches_per_fixture_uniforms(wc_context) -> None:
    """Batch decode equals sample_from_catalog on each fixture column."""
    fixtures, _, wc_to_model, wc_team_ids = wc_context
    model = _toy_model()
    catalog = build_matchup_catalog(model, wc_team_ids, wc_to_model, max_goals=8)
    u = np.linspace(0.01, 0.99, 72 * 5).reshape(5, 72)
    hg_b, ag_b = decode_group_scorelines(catalog, fixtures, u)
    for s in range(u.shape[0]):
        for col, fix in enumerate(fixtures):
            hg_row, ag_row = sample_from_catalog(
                catalog, fix.home_id, fix.away_id, np.array([u[s, col]])
            )
            assert int(hg_b[s, col]) == int(hg_row[0])
            assert int(ag_b[s, col]) == int(ag_row[0])


def test_accumulated_standings_match_legacy_apply(wc_context) -> None:
    """Vectorized points/GF/GA match sequential _apply_result per sim."""
    fixtures, team_ids_by_group, wc_to_model, wc_team_ids = wc_context
    model = _toy_model()
    catalog = build_matchup_catalog(model, wc_team_ids, wc_to_model, max_goals=8)
    _, layout = prepare_group_batch(
        model, fixtures, team_ids_by_group, wc_team_ids, wc_to_model
    )
    u = np.random.default_rng(123).random((4, len(fixtures)))
    hg, ag = decode_group_scorelines(catalog, fixtures, u)
    points, gf, ga = accumulate_standings(layout, hg, ag)

    for s in range(u.shape[0]):
        standings = {
            tid: GroupStanding(team_id=tid, group_label="")
            for ids in team_ids_by_group.values()
            for tid in ids
        }
        for col, fix in enumerate(fixtures):
            _apply_result(
                standings,
                fix.home_id,
                fix.away_id,
                int(hg[s, col]),
                int(ag[s, col]),
            )
        for tid, idx in layout.team_id_to_idx.items():
            st = standings[tid]
            assert st.points == int(points[s, idx])
            assert st.goals_for == int(gf[s, idx])
            assert st.goals_against == int(ga[s, idx])


def test_rankings_match_resolve_path(wc_context) -> None:
    """Per-sim rank_group on decoded scores matches resolve_group_outcomes rows."""
    fixtures, team_ids_by_group, wc_to_model, wc_team_ids = wc_context
    model = _toy_model()
    catalog, layout = prepare_group_batch(
        model, fixtures, team_ids_by_group, wc_team_ids, wc_to_model
    )
    rng = np.random.default_rng(77)
    u = rng.random((6, len(fixtures)))
    hg, ag = decode_group_scorelines(catalog, fixtures, u)
    points, gf, ga = accumulate_standings(layout, hg, ag)
    idx_map = idx_to_team_id_map(layout)
    outcomes = resolve_group_outcomes(
        layout,
        team_ids_by_group,
        idx_map,
        points,
        gf,
        ga,
        hg,
        ag,
        rng,
    )
    rng2 = np.random.default_rng(77)
    _ = rng2.random((6, len(fixtures)))
    for s in range(6):
        thirds: list[GroupStanding] = []
        for g in layout.group_labels:
            tids = team_ids_by_group[g]
            standings = {
                tid: GroupStanding(
                    team_id=tid,
                    group_label=g,
                    played=3,
                    points=int(points[s, layout.team_id_to_idx[tid]]),
                    goals_for=int(gf[s, layout.team_id_to_idx[tid]]),
                    goals_against=int(ga[s, layout.team_id_to_idx[tid]]),
                )
                for tid in tids
            }
            results = []
            for col in layout.group_match_cols[g]:
                col_i = int(col)
                results.append(
                    (
                        idx_map[int(layout.home_idx[col_i])],
                        idx_map[int(layout.away_idx[col_i])],
                        int(hg[s, col_i]),
                        int(ag[s, col_i]),
                    )
                )
            ranked = rank_group(standings, results, rng2)
            assert ranked == outcomes[s].ranked_by_group[g]
            thirds.append(standings[ranked[2]])
        best = rank_best_thirds(thirds, rng2)
        assert [t.group_label for t in best] == outcomes[
            s
        ].best_eight_third_group_letters


def _aligned_fast_legacy(
    wc_context,
    *,
    n_sims: int,
    seed: int,
) -> tuple[dict, dict, np.ndarray]:
    """Run fast and legacy on shared group uniforms and fresh tiebreak/KO RNGs (same seed)."""
    fixtures, team_ids_by_group, wc_to_model, wc_team_ids = wc_context
    model = _toy_model()
    catalog, layout = prepare_group_batch(
        model, fixtures, team_ids_by_group, wc_team_ids, wc_to_model
    )
    rng_u = np.random.default_rng(seed)
    group_uniforms = rng_u.random((n_sims, len(fixtures)))
    common = dict(
        model=model,
        n_sims=n_sims,
        fixtures=fixtures,
        team_ids_by_group=team_ids_by_group,
        wc_to_model=wc_to_model,
        wc_team_ids=wc_team_ids,
        group_uniforms=group_uniforms,
        catalog=catalog,
    )
    rng_fast = np.random.default_rng(seed)
    rng_legacy = np.random.default_rng(seed)
    fast = run_monte_carlo_fast(**common, rng=rng_fast, layout=layout)
    legacy = run_monte_carlo_legacy(**common, rng=rng_legacy)
    return fast, legacy, group_uniforms


def test_fast_legacy_distributional_equivalence_aligned_rng(wc_context) -> None:
    """Aligned RNG: shared group_uniforms + per-sim tiebreak/KO order => |ΔP(win)| = 0."""
    n = 200
    seed = 20260611
    fast, legacy, _ = _aligned_fast_legacy(wc_context, n_sims=n, seed=seed)
    for tid in fast:
        assert fast[tid].top_two == legacy[tid].top_two
        assert fast[tid].r16 == legacy[tid].r16
        assert fast[tid].qf == legacy[tid].qf
        assert fast[tid].sf == legacy[tid].sf
        assert fast[tid].final == legacy[tid].final
        assert fast[tid].win == legacy[tid].win
    diffs = [abs(fast[tid].win - legacy[tid].win) / n for tid in fast]
    assert max(diffs) == 0.0
