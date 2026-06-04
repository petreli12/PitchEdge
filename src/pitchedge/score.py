"""Score finished-match predictions (Brier + log loss); idempotent inserts only."""

from __future__ import annotations

import logging
from datetime import date, datetime, timezone
from typing import Any

from sqlalchemy import text

from pitchedge import db
from pitchedge.eval.metrics import (
    multiclass_brier,
    multiclass_log_loss,
    outcome_from_goals,
)
from pitchedge.ingest.team_ids import team_name_to_id

log = logging.getLogger(__name__)

OUTCOME_CHAR = ("H", "D", "A")

INSERT_SCORE_SQL = """
INSERT INTO prediction_scores (prediction_id, brier, log_loss, outcome, scored_utc)
VALUES (:prediction_id, :brier, :log_loss, :outcome, :scored_utc)
ON CONFLICT (prediction_id) DO NOTHING
"""

UNSCORED_FINAL_SQL = """
WITH latest AS (
    SELECT DISTINCT ON (mp.fixture_id, mp.source)
        mp.id,
        mp.fixture_id,
        mp.p_home,
        mp.p_draw,
        mp.p_away,
        mp.source
    FROM match_predictions mp
    ORDER BY mp.fixture_id, mp.source, mp.predicted_utc DESC, mp.id DESC
)
SELECT
    l.id AS prediction_id,
    l.fixture_id,
    l.p_home,
    l.p_draw,
    l.p_away,
    l.source,
    th.name AS home_name,
    ta.name AS away_name,
    COALESCE(NULLIF(TRIM(th.history_name), ''), th.name) AS home_history_name,
    COALESCE(NULLIF(TRIM(ta.history_name), ''), ta.name) AS away_history_name,
    (f.kickoff_utc AT TIME ZONE 'UTC')::date AS match_date
FROM latest l
JOIN fixtures f ON f.fixture_id = l.fixture_id
JOIN teams th ON th.team_id = f.home_id
JOIN teams ta ON ta.team_id = f.away_id
LEFT JOIN prediction_scores ps ON ps.prediction_id = l.id
WHERE f.status = 'final'
  AND ps.prediction_id IS NULL
  AND f.home_id IS NOT NULL
  AND f.away_id IS NOT NULL
"""

RESULTS_FOR_DATE_SQL = """
SELECT home_id, away_id, home_goals, away_goals
FROM raw_results
WHERE date = :match_date
  AND home_goals IS NOT NULL
  AND away_goals IS NOT NULL
"""


def outcome_char(home_goals: int, away_goals: int) -> str:
    """Return ``H`` / ``D`` / ``A`` for ``prediction_scores.outcome``."""
    code = outcome_from_goals(home_goals, away_goals)
    return OUTCOME_CHAR[code]


def lookup_result(
    conn,
    *,
    home_name: str,
    away_name: str,
    match_date: date,
) -> tuple[int, int] | None:
    """Find goals in ``raw_results`` for a WC fixture (historical nation ids)."""
    home_id = team_name_to_id(home_name)
    away_id = team_name_to_id(away_name)
    rows = conn.execute(
        text(RESULTS_FOR_DATE_SQL),
        {"match_date": match_date},
    ).mappings().all()
    for row in rows:
        if int(row["home_id"]) == home_id and int(row["away_id"]) == away_id:
            return int(row["home_goals"]), int(row["away_goals"])
        if int(row["home_id"]) == away_id and int(row["away_id"]) == home_id:
            return int(row["away_goals"]), int(row["home_goals"])
    return None


def score_row(
    prediction: dict[str, Any],
    home_goals: int,
    away_goals: int,
    *,
    scored_utc: datetime | None = None,
) -> dict[str, Any]:
    """Build one ``prediction_scores`` insert row."""
    probs = (
        float(prediction["p_home"]),
        float(prediction["p_draw"]),
        float(prediction["p_away"]),
    )
    outcome_code = outcome_from_goals(home_goals, away_goals)
    return {
        "prediction_id": int(prediction["prediction_id"]),
        "brier": multiclass_brier(probs, outcome_code),
        "log_loss": multiclass_log_loss(probs, outcome_code),
        "outcome": OUTCOME_CHAR[outcome_code],
        "scored_utc": scored_utc or datetime.now(timezone.utc),
    }


def _score_on_connection(conn) -> tuple[int, int]:
    """Score unscored predictions using an open SQLAlchemy connection."""
    attempted = 0
    inserted = 0
    scored_utc = datetime.now(timezone.utc)
    pending = conn.execute(text(UNSCORED_FINAL_SQL)).mappings().all()
    if not pending:
        log.info("score: no unscored predictions on final fixtures")
        return 0, 0

    by_fixture: dict[int, list[dict[str, Any]]] = {}
    for row in pending:
        pred = dict(row)
        by_fixture.setdefault(int(pred["fixture_id"]), []).append(pred)

    for fixture_id, preds in by_fixture.items():
        sample = preds[0]
        match_date = sample["match_date"]
        if isinstance(match_date, str):
            match_date = date.fromisoformat(match_date)
        goals = lookup_result(
            conn,
            home_name=str(sample["home_history_name"]),
            away_name=str(sample["away_history_name"]),
            match_date=match_date,
        )
        if goals is None:
            log.warning(
                "score: no raw_results for fixture_id=%s %s vs %s on %s",
                fixture_id,
                sample["home_history_name"],
                sample["away_history_name"],
                match_date,
            )
            continue
        hg, ag = goals
        for pred in preds:
            attempted += 1
            row = score_row(pred, hg, ag, scored_utc=scored_utc)
            result = conn.execute(text(INSERT_SCORE_SQL), row)
            if result.rowcount:
                inserted += 1
    return attempted, inserted


def score_finished_matches(
    *,
    db_url: str | None = None,
    conn=None,
) -> tuple[int, int]:
    """Score unscored predictions on final fixtures; idempotent.

    Returns ``(attempted, inserted)``. Pass ``conn`` for tests inside a transaction.
    """
    if conn is not None:
        attempted, inserted = _score_on_connection(conn)
    else:
        with db.connect(db_url) as connection:
            attempted, inserted = _score_on_connection(connection)
    log.info("score: attempted=%d inserted=%d", attempted, inserted)
    return attempted, inserted


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    attempted, inserted = score_finished_matches()
    print(f"scored: attempted={attempted} inserted={inserted}")


if __name__ == "__main__":
    main()
