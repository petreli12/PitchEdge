"""Phase 5 Monte Carlo tournament sim property tests."""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from pitchedge.sim.annex_c import load_annex_c, resolve_third_place_slots, third_place_combination_key
from pitchedge.sim.bracket import build_qualifiers
from pitchedge.sim.group_stage import GroupStanding, rank_group, rank_best_thirds
from pitchedge.sim.tournament import run_monte_carlo
from pitchedge.sim.wc_teams import load_group_fixtures, load_wc_teams, teams_by_group, wc_id_to_model_id
from pitchedge.model.dixon_coles import DixonColesModel


def _flat_model() -> DixonColesModel:
    """Equal-strength teams so title odds are diffuse but valid."""
    attack = {i: 0.0 for i in range(1, 200)}
    defense = {i: 0.0 for i in range(1, 200)}
    return DixonColesModel(
        attack=attack,
        defense=defense,
        home_adv=0.0,
        rho=0.0,
        xi=0.0,
    )


@pytest.fixture(autouse=True)
def _clear_matrix_cache():
    from pitchedge.sim.scoreline import clear_score_matrix_cache

    clear_score_matrix_cache()
    yield
    clear_score_matrix_cache()


@pytest.fixture
def wc_layout():
    teams = load_wc_teams()
    fixtures = load_group_fixtures()
    by_group = teams_by_group(teams)
    team_ids_by_group = {g: [t.team_id for t in ts] for g, ts in by_group.items()}
    wc_to_model = wc_id_to_model_id(teams)
    return fixtures, team_ids_by_group, wc_to_model


def test_annex_c_has_495_combinations():
    assert len(load_annex_c()) == 495


def test_annex_c_lookup_example():
    key = third_place_combination_key(list("CDEFGHIJ"))
    mapping = resolve_third_place_slots(list("CDEFGHIJ"))
    assert key == "CDEFGHIJ"
    assert set(mapping.keys()) == {"1A", "1B", "1D", "1E", "1G", "1I", "1K", "1L"}
    assert all(v.startswith("3") for v in mapping.values())


def test_group_advance_probs_sum_to_two(wc_layout):
    model = _flat_model()
    agg = run_monte_carlo(model, n_sims=120, seed=42)
    teams = load_wc_teams()
    by_group = teams_by_group(teams)
    for g, ts in by_group.items():
        total = sum(agg.by_team[t.team_id]["p_advance_group"] for t in ts)
        assert total == pytest.approx(2.0, abs=0.15)


def test_global_win_prob_sums_to_one(wc_layout):
    fixtures, team_ids_by_group, wc_to_model = wc_layout
    model = _flat_model()
    agg = run_monte_carlo(
        model,
        n_sims=120,
        seed=99,
        fixtures_path=None,
        teams_path=None,
    )
    total_win = sum(p["p_win"] for p in agg.by_team.values())
    assert total_win == pytest.approx(1.0, abs=0.05)


def test_monte_carlo_reproducible_with_seed(wc_layout):
    """Same seed => byte-identical probabilities across every stage."""
    model = _flat_model()
    a = run_monte_carlo(model, n_sims=40, seed=20260611)
    b = run_monte_carlo(model, n_sims=40, seed=20260611)
    stages = ("p_advance_group", "p_r16", "p_qf", "p_sf", "p_final", "p_win")
    assert set(a.by_team) == set(b.by_team)
    for tid in a.by_team:
        for stage in stages:
            assert a.by_team[tid][stage] == b.by_team[tid][stage]


def test_sim_invariants_hold_on_production_mapping(wc_layout):
    """ΣP(win)≈1 and per-group ΣP(advance)≈2 under the real WC->model resolver.

    Fast / low-sim structural contract: complements the flat-model checks by
    exercising the same ``wc_id_to_model_id`` path predict.py and the sim share.
    """
    fixtures, team_ids_by_group, wc_to_model = wc_layout
    teams = load_wc_teams()
    # Deterministic strength prior keyed by the production model ids (no DB/fit).
    attack = {wc_to_model[t.team_id]: 0.0 for t in teams}
    defense = {wc_to_model[t.team_id]: 0.0 for t in teams}
    model = DixonColesModel(attack=attack, defense=defense, home_adv=0.0, rho=0.0, xi=0.0)

    agg = run_monte_carlo(model, n_sims=120, seed=2026)

    assert sum(p["p_win"] for p in agg.by_team.values()) == pytest.approx(1.0, abs=0.02)
    by_group = teams_by_group(teams)
    for g, ts in by_group.items():
        total = sum(agg.by_team[t.team_id]["p_advance_group"] for t in ts)
        assert total == pytest.approx(2.0, abs=0.1)


def test_favorites_have_higher_title_prob_than_minnows(wc_layout):
    """Sanity: Argentina/France/Spain should beat minnows on a fitted-ish prior."""
    teams = load_wc_teams()
    wc_to_model = wc_id_to_model_id(teams)
    model = DixonColesModel(
        attack={
            wc_to_model[37]: 0.35,
            wc_to_model[33]: 0.30,
            wc_to_model[29]: 0.28,
            wc_to_model[9]: 0.25,
            wc_to_model[48]: -0.25,
            wc_to_model[12]: -0.30,
            wc_to_model[7]: -0.35,
        },
        defense={
            wc_to_model[37]: -0.35,
            wc_to_model[33]: -0.30,
            wc_to_model[29]: -0.28,
            wc_to_model[9]: -0.25,
            wc_to_model[48]: 0.25,
            wc_to_model[12]: 0.30,
            wc_to_model[7]: 0.35,
        },
        home_adv=0.0,
        rho=-0.05,
        xi=0.0,
    )
    fixtures, team_ids_by_group, _ = wc_layout
    agg = run_monte_carlo(model, n_sims=150, seed=7)
    p_arg = agg.by_team[37]["p_win"]
    p_gha = agg.by_team[48]["p_win"]
    assert p_arg > p_gha * 3
    assert p_arg > 0.05


def test_rank_group_two_way_h2h():
    rng = np.random.default_rng(0)
    standings = {
        1: GroupStanding(1, "A", points=4, goals_for=3, goals_against=3),
        2: GroupStanding(2, "A", points=4, goals_for=3, goals_against=3),
        3: GroupStanding(3, "A", points=3, goals_for=2, goals_against=1),
        4: GroupStanding(4, "A", points=0, goals_for=0, goals_against=4),
    }
    results = [
        (1, 2, 1, 1),
        (2, 1, 2, 0),
        (1, 3, 1, 0),
        (3, 4, 2, 0),
    ]
    ranked = rank_group(standings, results, rng)
    assert ranked[0] == 2
    assert ranked[1] == 1


def test_rank_group_ranks_by_points_not_insertion_order():
    """Regression: ranking must follow simulated results, not group-list order.

    The team listed first (id=1) has the fewest points and must finish last; the
    team listed last (id=4) has the most points and must finish first. A prior
    bug returned teams in insertion order whenever their (points, GD, GF) tuples
    were distinct, pinning the first-listed team to 1st in ~100% of sims.
    """
    standings = {
        1: GroupStanding(1, "A", points=0, goals_for=1, goals_against=9),
        2: GroupStanding(2, "A", points=3, goals_for=3, goals_against=5),
        3: GroupStanding(3, "A", points=6, goals_for=5, goals_against=3),
        4: GroupStanding(4, "A", points=9, goals_for=9, goals_against=1),
    }
    results = [(4, 1, 3, 0), (3, 1, 2, 1), (2, 1, 2, 1),
               (4, 3, 1, 0), (4, 2, 2, 0), (3, 2, 1, 0)]
    expected = [4, 3, 2, 1]
    for perm in itertools.permutations([1, 2, 3, 4]):
        shuffled = {tid: standings[tid] for tid in perm}
        ranked = rank_group(shuffled, results, np.random.default_rng(0))
        assert ranked == expected, f"insertion order {perm} -> {ranked}"


def test_rank_group_gd_breaks_equal_points():
    """Equal points resolved by goal difference then goals-for (not list order)."""
    standings = {
        1: GroupStanding(1, "A", points=6, goals_for=4, goals_against=3),  # GD +1
        2: GroupStanding(2, "A", points=6, goals_for=7, goals_against=2),  # GD +5
    }
    results = [(1, 3, 2, 1), (2, 3, 4, 0)]
    for perm in ([1, 2], [2, 1]):
        shuffled = {tid: standings[tid] for tid in perm}
        ranked = rank_group(shuffled, results, np.random.default_rng(0))
        assert ranked[0] == 2 and ranked[1] == 1


def test_build_qualifiers_resolves_annex():
    ranked = {
        g: [100 + ord(g), 200 + ord(g), 300 + ord(g), 400 + ord(g)]
        for g in "ABCDEFGHIJKL"
    }
    third = {g: ranked[g][2] for g in ranked}
    best = list("ABCDEFGH")
    q = build_qualifiers(ranked, third, best)
    assert len(q.third_slots) == 8
