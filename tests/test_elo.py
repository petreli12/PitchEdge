"""Tests for Elo rating updates."""

from __future__ import annotations

from datetime import date

import pytest

from pitchedge.model.elo import EloConfig, elo_win_prob, fit_elo


def test_elo_win_prob_stronger_home_favored():
    p = elo_win_prob(1600.0, 1400.0, home_advantage=100.0)
    assert p > 0.5
    p_neutral = elo_win_prob(1600.0, 1400.0, home_advantage=0.0)
    assert p > p_neutral > 0.5


def test_elo_update_is_zero_sum_per_match():
    cfg = EloConfig(initial_rating=1500.0, k_base=20.0, home_advantage=100.0)
    matches = [
        {
            "date": date(2020, 1, 1),
            "home_id": 1,
            "away_id": 2,
            "home_goals": 2,
            "away_goals": 0,
            "neutral": False,
        }
    ]
    ratings_before = {1: 1500.0, 2: 1500.0}
    ratings_after, _ = fit_elo(matches, elo_config=cfg)

    delta_home = ratings_after[1] - ratings_before[1]
    delta_away = ratings_after[2] - ratings_before[2]
    assert delta_home + delta_away == pytest.approx(0.0, abs=1e-9)


def test_draw_is_zero_sum():
    cfg = EloConfig()
    matches = [
        {
            "date": date(2020, 1, 1),
            "home_id": 10,
            "away_id": 20,
            "home_goals": 1,
            "away_goals": 1,
            "neutral": True,
        }
    ]
    ratings, _ = fit_elo(matches, elo_config=cfg)
    # Both started 1500; draw with neutral -> small symmetric move, net zero.
    assert ratings[10] == pytest.approx(ratings[20], abs=1e-6)
    assert ratings[10] + ratings[20] == pytest.approx(3000.0, abs=1e-6)


def test_fit_elo_from_db_writes_snapshots(empty_ingest_tables):
    from pitchedge import db
    from pitchedge.ingest import history
    from pitchedge.model.elo import fit_elo_from_db

    sample = __import__("pathlib").Path(__file__).parent / "fixtures" / "history_sample.csv"
    history.ingest_history(sample)

    n, as_of = fit_elo_from_db()
    assert n > 0
    assert as_of is not None
    count = db.fetch_one("SELECT count(*) AS n FROM team_ratings")["n"]
    assert count == n
