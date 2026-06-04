"""Phase 6 prediction scoring tests."""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

import pytest
from sqlalchemy import text

from pitchedge.ingest.team_ids import team_name_to_id
from pitchedge.model.dixon_coles import DixonColesModel
from pitchedge.predict import INSERT_PREDICTION_SQL, build_prediction_rows
from pitchedge.score import score_finished_matches, score_row

UTC = timezone.utc


def _flat_model() -> DixonColesModel:
    ids = range(1, 200)
    return DixonColesModel(
        attack={i: 0.0 for i in ids},
        defense={i: 0.0 for i in ids},
        home_adv=0.0,
        rho=0.0,
        xi=0.0,
    )


def test_score_row_perfect_home_win():
    pred = {"prediction_id": 1, "p_home": 0.7, "p_draw": 0.2, "p_away": 0.1}
    row = score_row(pred, 2, 0)
    assert row["outcome"] == "H"
    assert row["brier"] == pytest.approx(0.14, abs=1e-6)


def test_score_finished_idempotent(conn, empty_ingest_tables):
    model = _flat_model()
    ko = datetime(2022, 11, 20, 19, 0, tzinfo=UTC)
    pred_utc = ko - timedelta(days=1)
    conn.execute(
        text(
            "INSERT INTO teams (team_id, name) VALUES "
            "(10, 'Argentina'), (11, 'France')"
        )
    )
    conn.execute(
        text(
            "INSERT INTO fixtures (fixture_id, kickoff_utc, home_id, away_id, "
            "stage, group_label, status) "
            "VALUES (8100, :ko, 10, 11, 'Final', NULL, 'final')"
        ),
        {"ko": ko},
    )
    arg_id = team_name_to_id("Argentina")
    fra_id = team_name_to_id("France")
    conn.execute(
        text(
            "INSERT INTO raw_results (date, home_id, away_id, home_goals, "
            "away_goals, competition, neutral) "
            "VALUES (:d, :h, :a, 3, 3, 'World Cup', true)"
        ),
        {
            "d": date(2022, 11, 20),
            "h": arg_id,
            "a": fra_id,
        },
    )
    fix = {
        "fixture_id": 8100,
        "home_id": 10,
        "away_id": 11,
        "kickoff_utc": ko,
        "home_name": "Argentina",
        "away_name": "France",
    }
    rows = build_prediction_rows(
        fix,
        model,
        {10: 10, 11: 11},
        model_version="v1",
        predicted_utc=pred_utc,
        market_probs=(0.33, 0.34, 0.33),
    )
    for row in rows:
        conn.execute(text(INSERT_PREDICTION_SQL), row)

    a1, i1 = score_finished_matches(conn=conn)
    assert a1 == 3
    assert i1 == 3
    a2, i2 = score_finished_matches(conn=conn)
    assert a2 == 0
    assert i2 == 0
