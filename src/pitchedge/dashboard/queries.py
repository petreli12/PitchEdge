"""Read-only SQL for the Streamlit dashboard (no probability recomputation)."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from sqlalchemy import text

from pitchedge import db

UPCOMING_PROBS_SQL = """
SELECT
    f.fixture_id,
    f.kickoff_utc,
    f.stage,
    f.group_label,
    th.name AS home,
    ta.name AS away,
    m.p_home AS model_p_home,
    m.p_draw AS model_p_draw,
    m.p_away AS model_p_away,
    m.predicted_utc AS model_predicted_utc,
    mk.p_home AS market_p_home,
    mk.p_draw AS market_p_draw,
    mk.p_away AS market_p_away,
    bl.p_home AS blend_p_home,
    bl.p_draw AS blend_p_draw,
    bl.p_away AS blend_p_away
FROM fixtures f
JOIN teams th ON th.team_id = f.home_id
JOIN teams ta ON ta.team_id = f.away_id
LEFT JOIN LATERAL (
    SELECT p_home, p_draw, p_away, predicted_utc
    FROM match_predictions
    WHERE fixture_id = f.fixture_id AND source = 'model'
    ORDER BY predicted_utc DESC
    LIMIT 1
) m ON TRUE
LEFT JOIN LATERAL (
    SELECT p_home, p_draw, p_away
    FROM match_predictions
    WHERE fixture_id = f.fixture_id AND source = 'market'
    ORDER BY predicted_utc DESC
    LIMIT 1
) mk ON TRUE
LEFT JOIN LATERAL (
    SELECT p_home, p_draw, p_away
    FROM match_predictions
    WHERE fixture_id = f.fixture_id AND source = 'blend'
    ORDER BY predicted_utc DESC
    LIMIT 1
) bl ON TRUE
WHERE f.status = 'scheduled'
  AND f.kickoff_utc > :now_utc
  AND f.home_id IS NOT NULL
  AND f.away_id IS NOT NULL
ORDER BY f.kickoff_utc ASC
"""

LATEST_SIM_SQL = """
SELECT
    t.name,
    t.group_label,
    sr.team_id,
    sr.p_win,
    sr.p_final,
    sr.p_sf,
    sr.p_qf,
    sr.p_r16,
    sr.p_advance_group,
    sr.n_sims,
    sr.run_batch_utc
FROM sim_results sr
JOIN teams t ON t.team_id = sr.team_id
WHERE sr.run_batch_utc = (
    SELECT MAX(run_batch_utc) FROM sim_results
)
ORDER BY sr.p_win DESC NULLS LAST, t.name ASC
"""

MISSING_ODDS_SQL = """
SELECT
    f.fixture_id,
    th.name AS home,
    ta.name AS away,
    f.kickoff_utc
FROM fixtures f
JOIN teams th ON th.team_id = f.home_id
JOIN teams ta ON ta.team_id = f.away_id
WHERE f.status = 'scheduled'
  AND f.home_id IS NOT NULL
  AND f.away_id IS NOT NULL
  AND NOT EXISTS (
      SELECT 1 FROM odds_snapshots os WHERE os.fixture_id = f.fixture_id
  )
ORDER BY f.kickoff_utc ASC
"""

RECEIPT_STATUS_SQL = """
SELECT
    COUNT(DISTINCT f.fixture_id) FILTER (
        WHERE f.status = 'scheduled'
          AND f.kickoff_utc > :now_utc
          AND f.home_id IS NOT NULL
          AND f.away_id IS NOT NULL
    ) AS upcoming_fixtures,
    COUNT(DISTINCT mp.fixture_id) FILTER (
        WHERE mp.source = 'model'
          AND EXISTS (
              SELECT 1 FROM fixtures fx
              WHERE fx.fixture_id = mp.fixture_id
                AND fx.status = 'scheduled'
                AND fx.kickoff_utc > :now_utc
          )
    ) AS fixtures_with_model_pred,
    MIN(mp.predicted_utc) FILTER (WHERE mp.source = 'model') AS earliest_model_utc,
    MAX(mp.predicted_utc) FILTER (WHERE mp.source = 'model') AS latest_model_utc
FROM fixtures f
LEFT JOIN match_predictions mp ON mp.fixture_id = f.fixture_id
"""


def fetch_upcoming_match_probs(
    *,
    db_url: str | None = None,
    now_utc: datetime | None = None,
) -> list[dict[str, Any]]:
    """Upcoming fixtures with latest model / market / blend rows from receipts."""
    now = now_utc or datetime.now(timezone.utc)
    with db.connect(db_url) as conn:
        result = conn.execute(text(UPCOMING_PROBS_SQL), {"now_utc": now})
        return [dict(r) for r in result.mappings().all()]


def fetch_latest_sim_results(*, db_url: str | None = None) -> list[dict[str, Any]]:
    """Latest Monte Carlo batch joined to team names."""
    with db.connect(db_url) as conn:
        result = conn.execute(text(LATEST_SIM_SQL))
        return [dict(r) for r in result.mappings().all()]


def fetch_fixtures_missing_odds(*, db_url: str | None = None) -> list[dict[str, Any]]:
    """Scheduled fixtures with home/away set but no odds_snapshots row."""
    with db.connect(db_url) as conn:
        result = conn.execute(text(MISSING_ODDS_SQL))
        return [dict(r) for r in result.mappings().all()]


def fetch_prediction_receipt_status(
    *,
    db_url: str | None = None,
    now_utc: datetime | None = None,
) -> dict[str, Any]:
    """Counts for launch-board baseline verification (persisted model preds)."""
    now = now_utc or datetime.now(timezone.utc)
    with db.connect(db_url) as conn:
        row = conn.execute(
            text(RECEIPT_STATUS_SQL),
            {"now_utc": now},
        ).mappings().one()
        return dict(row)
