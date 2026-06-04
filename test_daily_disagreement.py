"""Tests for daily_disagreement — pure ranking math, no DB."""

from pitchedge.content.daily_disagreement import (
    Candidate, score_candidate, rank_disagreements, select_top,
    to_narrative_input, _tvd,
)


def _cand(**over):
    base = dict(
        fixture_id=1, home="Spain", away="Morocco", stage="Round of 16",
        kickoff_local="Jul 5", venue="Atlanta",
        p_home=0.54, p_draw=0.27, p_away=0.19,
        exp_home_goals=1.7, exp_away_goals=0.9,
        market_p_home=0.61, market_p_draw=0.24, market_p_away=0.15,
    )
    base.update(over)
    return Candidate(**base)


def test_tvd_zero_when_identical():
    assert _tvd((0.5, 0.3, 0.2), (0.5, 0.3, 0.2)) == 0.0


def test_tvd_bounds():
    assert 0.0 <= _tvd((1, 0, 0), (0, 0, 1)) <= 1.0


def test_largest_gap_picks_right_outcome_and_sign():
    # model is lower on home (0.54 vs 0.61) -> biggest gap is home, negative
    d = score_candidate(_cand())
    assert d.outcome == "home"
    assert d.delta_pts < 0
    assert "lower on Spain" in d.note


def test_note_is_neutral_no_tip_language():
    d = score_candidate(_cand())
    banned = ["bet", "lock", "value", "edge", "should", "back "]
    assert not any(b in d.note.lower() for b in banned)


def test_ranking_orders_by_score():
    small = _cand(fixture_id=1, market_p_home=0.55, market_p_draw=0.27,
                  market_p_away=0.18)  # tiny gap
    big = _cand(fixture_id=2, home="Brazil", away="Serbia",
                market_p_home=0.40, market_p_draw=0.30, market_p_away=0.30)  # big gap
    ranked = rank_disagreements([small, big])
    assert ranked[0].candidate.fixture_id == 2


def test_salience_breaks_near_ties():
    a = _cand(fixture_id=1, salience=1.0)
    b = _cand(fixture_id=2, salience=2.0)  # same gap, higher salience
    ranked = rank_disagreements([a, b])
    assert ranked[0].candidate.fixture_id == 2


def test_select_top_returns_none_below_threshold():
    weak = _cand(market_p_home=0.55, market_p_draw=0.27, market_p_away=0.18)
    assert select_top([weak], min_tvd=0.10) is None


def test_to_narrative_input_carries_market_and_note():
    d = score_candidate(_cand())
    ni = to_narrative_input(d)
    assert ni.market_p_home == 0.61
    assert ni.disagreement_note == d.note
