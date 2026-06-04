"""Tests for Dixon-Coles model and score matrix properties."""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pytest

from pitchedge.model.dixon_coles import (
    DixonColesConfig,
    DixonColesModel,
    apply_elo_shrinkage,
    fit_dixon_coles,
    match_probs,
    score_matrix,
)


def _synthetic_domestic_league() -> list[dict]:
    """Four teams; team 1 is dominant at home."""
    rows: list[dict] = []
    start = date(2023, 1, 1)
    for i in range(30):
        rows.append(
            {
                "date": start + timedelta(days=2 * i),
                "home_id": 1,
                "away_id": 2,
                "home_goals": 3,
                "away_goals": 0,
                "neutral": False,
            }
        )
        rows.append(
            {
                "date": start + timedelta(days=2 * i + 1),
                "home_id": 3,
                "away_id": 4,
                "home_goals": 1,
                "away_goals": 1,
                "neutral": False,
            }
        )
    return rows


@pytest.fixture
def fitted_model() -> DixonColesModel:
    return fit_dixon_coles(
        _synthetic_domestic_league(),
        elo_by_team={1: 1700.0, 2: 1400.0, 3: 1500.0, 4: 1500.0},
        dc_config=DixonColesConfig(xi=0.0, min_matches=20, recency_years=99.0),
    )


def test_score_matrix_sums_to_one(fitted_model: DixonColesModel):
    mat = score_matrix(fitted_model, 1, 2, max_goals=10)
    assert mat.sum() == pytest.approx(1.0, abs=1e-6)


def test_match_probs_sum_to_one(fitted_model: DixonColesModel):
    probs = match_probs(fitted_model, 1, 2, max_goals=10)
    total = probs.p_home + probs.p_draw + probs.p_away
    assert total == pytest.approx(1.0, abs=1e-6)


def test_strong_team_favored_at_home(fitted_model: DixonColesModel):
    probs = match_probs(fitted_model, 1, 2, max_goals=10)
    assert probs.p_home > 0.5
    assert probs.exp_home_goals > probs.exp_away_goals


def test_mean_attack_near_zero_after_fit():
    model = fit_dixon_coles(
        _synthetic_domestic_league(),
        dc_config=DixonColesConfig(xi=0.0, min_matches=100, recency_years=99.0),
    )
    assert np.mean(list(model.attack.values())) == pytest.approx(0.0, abs=1e-6)


def test_low_data_team_triggers_shrinkage_warning(caplog):
    model = DixonColesModel(
        attack={99: 0.5, 100: 0.0},
        defense={99: -0.2, 100: 0.0},
        home_adv=0.2,
        rho=-0.03,
        xi=0.005,
        match_counts={99: 5, 100: 40},
    )
    apply_elo_shrinkage(
        model,
        elo_by_team={99: 1500.0, 100: 1600.0},
        dc_config=DixonColesConfig(min_matches=30),
    )
    assert any("team_id=99" in r.message for r in caplog.records)


def test_optimizer_reports_finite_nll():
    model = fit_dixon_coles(
        _synthetic_domestic_league(),
        dc_config=DixonColesConfig(xi=0.0, min_matches=100, recency_years=99.0),
    )
    assert model.fit_neg_log_likelihood is not None
    assert model.fit_neg_log_likelihood < 1e6


def test_neutral_venue_swapping_home_away_mirrors_probs():
    """At neutral venues, 1X2 must not depend on which team is listed as home."""
    model = DixonColesModel(
        attack={1: 0.35, 2: -0.15, 3: -0.20},
        defense={1: -0.10, 2: 0.20, 3: -0.10},
        home_adv=0.22,
        rho=-0.04,
        xi=0.005,
    )
    home_id, away_id = 1, 2
    a = match_probs(model, home_id, away_id, max_goals=10, neutral=True)
    b = match_probs(model, away_id, home_id, max_goals=10, neutral=True)

    assert a.p_draw == pytest.approx(b.p_draw, abs=1e-6)
    assert a.p_home == pytest.approx(b.p_away, abs=1e-6)
    assert a.p_away == pytest.approx(b.p_home, abs=1e-6)
    assert a.exp_home_goals == pytest.approx(b.exp_away_goals, abs=1e-5)
    assert a.exp_away_goals == pytest.approx(b.exp_home_goals, abs=1e-5)

    # Home advantage applies only when neutral=False.
    with_gamma = match_probs(model, home_id, away_id, max_goals=10, neutral=False)
    assert with_gamma.p_home > a.p_home
