"""Phase 6 prediction logging tests."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import text
from sqlalchemy.exc import DBAPIError

from pitchedge.model.dixon_coles import DixonColesModel
from pitchedge.predict import (
    INSERT_PREDICTION_SQL,
    assert_pre_kickoff,
    build_prediction_rows,
)

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


def test_assert_pre_kickoff_rejects_at_kickoff():
    ko = datetime(2026, 6, 11, 19, 0, tzinfo=UTC)
    with pytest.raises(ValueError, match="before kickoff"):
        assert_pre_kickoff(ko, ko)


def test_build_prediction_rows_normalize(conn, empty_ingest_tables):
    model = _flat_model()
    ko = datetime.now(UTC) + timedelta(days=3)
    pred_utc = datetime.now(UTC)
    conn.execute(
        text(
            "INSERT INTO teams (team_id, name) VALUES (1, 'Spain'), (2, 'Morocco')"
        )
    )
    conn.execute(
        text(
            "INSERT INTO fixtures (fixture_id, kickoff_utc, home_id, away_id, "
            "stage, group_label, status) "
            "VALUES (8001, :ko, 1, 2, 'Group', 'H', 'scheduled')"
        ),
        {"ko": ko},
    )
    fix = {
        "fixture_id": 8001,
        "home_id": 1,
        "away_id": 2,
        "kickoff_utc": ko,
        "home_name": "Spain",
        "away_name": "Morocco",
    }
    rows = build_prediction_rows(
        fix,
        model,
        {1: 1, 2: 2},
        model_version="test_v1",
        predicted_utc=pred_utc,
        market_probs=(0.5, 0.28, 0.22),
    )
    assert len(rows) == 3
    sources = {r["source"] for r in rows}
    assert sources == {"model", "market", "blend"}
    for r in rows:
        total = r["p_home"] + r["p_draw"] + r["p_away"]
        assert total == pytest.approx(1.0, abs=0.01)


def test_insert_rejected_at_kickoff_by_db_trigger(conn, empty_ingest_tables):
    """DB trigger rejects predicted_utc >= kickoff (see test_migrations)."""
    model = _flat_model()
    ko = datetime.now(UTC) + timedelta(days=2)
    pred_utc = datetime.now(UTC)
    conn.execute(
        text("INSERT INTO teams (team_id, name) VALUES (3, 'A'), (4, 'B')")
    )
    conn.execute(
        text(
            "INSERT INTO fixtures (fixture_id, kickoff_utc, home_id, away_id, "
            "stage, group_label, status) "
            "VALUES (8002, :ko, 3, 4, 'Group', 'A', 'scheduled')"
        ),
        {"ko": ko},
    )
    fix = {
        "fixture_id": 8002,
        "home_id": 3,
        "away_id": 4,
        "kickoff_utc": ko,
        "home_name": "A",
        "away_name": "B",
    }
    rows = build_prediction_rows(
        fix,
        model,
        {3: 3, 4: 4},
        model_version="v",
        predicted_utc=pred_utc,
        market_probs=(0.4, 0.3, 0.3),
    )
    conn.execute(text(INSERT_PREDICTION_SQL), rows[0])

    with pytest.raises(DBAPIError):
        conn.execute(
            text(
                "INSERT INTO match_predictions "
                "(fixture_id, model_version, predicted_utc, p_home, p_draw, "
                "p_away, source) VALUES "
                "(8002, 'v2', :pred, 0.4, 0.3, 0.3, 'model')"
            ),
            {"pred": ko},
        )
