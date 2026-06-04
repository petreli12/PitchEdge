"""Venue policy and WC wrappers for Dixon-Coles predictions."""

from __future__ import annotations

import numpy as np
import pytest

from pitchedge.model.dixon_coles import DixonColesModel, match_probs, wc_match_probs
from pitchedge.model.venues import (
    WC_2026_HOST_TEAM_IDS,
    dixon_coles_neutral_for_tournament_fixture,
    dixon_coles_neutral_for_wc_fixture,
)
from pitchedge.predict import fixture_model_probs
from pitchedge.sim.scoreline import sample_scoreline


@pytest.fixture
def toy_model() -> DixonColesModel:
    return DixonColesModel(
        attack={1: 0.2, 5: 0.0, 13: 0.15, 9: 0.35, 2: -0.15},
        defense={1: -0.1, 5: 0.0, 13: -0.05, 9: -0.2, 2: 0.1},
        home_adv=0.25,
        rho=-0.04,
        xi=0.005,
    )


def test_wc_fixture_defaults_to_neutral(toy_model: DixonColesModel):
    """Non-host listed home -> gamma off."""
    assert dixon_coles_neutral_for_wc_fixture(9) is True
    wc = wc_match_probs(toy_model, 9, 2)  # Brazil home, South Africa away
    raw = match_probs(toy_model, 9, 2, neutral=True)
    assert wc.p_home == pytest.approx(raw.p_home, abs=1e-9)


def test_co_host_listed_home_applies_gamma(toy_model: DixonColesModel):
    """Mexico (1) as home_id -> intentional host-nation exception."""
    assert dixon_coles_neutral_for_wc_fixture(1) is False
    assert 1 in WC_2026_HOST_TEAM_IDS
    wc_auto = wc_match_probs(toy_model, 1, 2)
    with_gamma = match_probs(toy_model, 1, 2, neutral=False)
    neutral = match_probs(toy_model, 1, 2, neutral=True)
    assert wc_auto.p_home == pytest.approx(with_gamma.p_home, abs=1e-9)
    assert wc_auto.p_home > neutral.p_home


def test_host_home_override_forces_gamma(toy_model: DixonColesModel):
    assert dixon_coles_neutral_for_wc_fixture(9, host_home=True) is False
    forced = wc_match_probs(toy_model, 9, 2, host_home=True)
    with_gamma = match_probs(toy_model, 9, 2, neutral=False)
    assert forced.p_home == pytest.approx(with_gamma.p_home, abs=1e-9)


def test_host_home_false_forces_neutral_even_for_mexico(toy_model: DixonColesModel):
    assert dixon_coles_neutral_for_wc_fixture(1, host_home=False) is True
    wc = wc_match_probs(toy_model, 1, 2, host_home=False)
    neutral = match_probs(toy_model, 1, 2, neutral=True)
    assert wc.p_home == pytest.approx(neutral.p_home, abs=1e-9)


def test_tournament_backtest_neutral_by_default():
    assert dixon_coles_neutral_for_tournament_fixture() is True
    assert dixon_coles_neutral_for_tournament_fixture(host_home=True) is False


def test_predict_and_sim_route_through_wc_wrappers(toy_model: DixonColesModel):
    # predict.py translates WC ids -> model ids; identity map mirrors wc_match_probs.
    identity = {9: 9, 2: 2}
    via_predict = fixture_model_probs(toy_model, 9, 2, identity)
    via_wc = wc_match_probs(toy_model, 9, 2)
    assert via_predict.p_draw == pytest.approx(via_wc.p_draw, abs=1e-9)

    rng = np.random.default_rng(42)
    wc_map = {9: 9, 2: 2}
    hg, ag = sample_scoreline(toy_model, 9, 2, wc_map, rng)
    assert hg >= 0 and ag >= 0


def test_fixture_model_probs_translates_wc_ids_not_fallback(toy_model: DixonColesModel):
    """Regression: WC ids differ from model ids; lookup must not hit the 0/0 fallback.

    Feeding raw WC ids to the model keys (which are historical hash ids) misses
    every team and collapses to equal lambdas (~0.34/0.32/0.34). With a proper
    map, a strong favorite must stay a strong favorite.
    """
    # WC ids 100/200 are absent from toy_model; map them to real model keys 9/2.
    wc_to_model = {100: 9, 200: 2}
    probs = fixture_model_probs(toy_model, 100, 200, wc_to_model)
    direct = match_probs(toy_model, 9, 2, neutral=True)
    assert probs.p_home == pytest.approx(direct.p_home, abs=1e-9)
    assert probs.p_home > 0.5  # favorite, not a 3-way coin flip

    with pytest.raises(KeyError):
        fixture_model_probs(toy_model, 100, 200, {100: 9})  # away unmapped
