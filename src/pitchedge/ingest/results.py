"""Sync finished World Cup match results into ``raw_results`` and ``fixtures``.

During the tournament, scores arrive after kickoff (updated history CSV, manual
entry, or an API adapter). This module upserts goals into ``raw_results`` and
marks the matching WC fixture ``status='final'`` so ``score.py`` can write
``prediction_scores``.

Probabilities in ``match_predictions`` are never mutated; only ground-truth rows
and fixture status are updated.
"""

from __future__ import annotations

import logging
from datetime import date, datetime
from typing import Any

from sqlalchemy import text

from pitchedge import db
from pitchedge.ingest.team_ids import team_name_to_id

log = logging.getLogger(__name__)

UPSERT_RESULT_SQL = """
INSERT INTO raw_results (
    date, home_id, away_id, home_goals, away_goals, competition, neutral
) VALUES (
    :date, :home_id, :away_id, :home_goals, :away_goals, :competition, :neutral
)
ON CONFLICT (date, home_id, away_id, competition) DO UPDATE SET
    home_goals = EXCLUDED.home_goals,
    away_goals = EXCLUDED.away_goals
WHERE EXCLUDED.home_goals IS NOT NULL
  AND EXCLUDED.away_goals IS NOT NULL
"""

MARK_FIXTURE_FINAL_SQL = """
UPDATE fixtures
SET status = 'final'
WHERE fixture_id = :fixture_id
  AND status = 'scheduled'
"""

MATCH_FIXTURE_SQL = """
SELECT f.fixture_id
FROM fixtures f
JOIN teams th ON th.team_id = f.home_id
JOIN teams ta ON ta.team_id = f.away_id
WHERE f.home_id IS NOT NULL
  AND f.away_id IS NOT NULL
  AND (f.kickoff_utc AT TIME ZONE 'UTC')::date = :match_date
  AND th.name = :home_name
  AND ta.name = :away_name
LIMIT 1
"""


def record_result(
    *,
    home_name: str,
    away_name: str,
    match_date: date,
    home_goals: int,
    away_goals: int,
    competition: str = "FIFA World Cup",
    neutral: bool = True,
    fixture_id: int | None = None,
    wc_home_name: str | None = None,
    wc_away_name: str | None = None,
    db_url: str | None = None,
) -> dict[str, Any]:
    """Upsert one final score and mark the WC fixture final.

    ``home_name`` / ``away_name`` are the labels used in ``raw_results`` (Kaggle
    ``history_name`` when set, else WC ``teams.name``). ``wc_*`` override the
    ``teams.name`` lookup when they differ (e.g. Czech Republic vs Czechia).
    Returns a detail dict.
    """
    row = {
        "date": match_date,
        "home_id": team_name_to_id(home_name),
        "away_id": team_name_to_id(away_name),
        "home_goals": home_goals,
        "away_goals": away_goals,
        "competition": competition,
        "neutral": neutral,
    }
    fix_home = (wc_home_name or home_name).strip()
    fix_away = (wc_away_name or away_name).strip()
    with db.connect(db_url) as conn:
        conn.execute(text(UPSERT_RESULT_SQL), row)
        fid = fixture_id
        if fid is None:
            found = conn.execute(
                text(MATCH_FIXTURE_SQL),
                {
                    "match_date": match_date,
                    "home_name": fix_home,
                    "away_name": fix_away,
                },
            ).scalar()
            fid = int(found) if found is not None else None
        marked = 0
        if fid is not None:
            result = conn.execute(text(MARK_FIXTURE_FINAL_SQL), {"fixture_id": fid})
            marked = result.rowcount or 0
    detail = {
        "home": home_name,
        "away": away_name,
        "score": f"{home_goals}-{away_goals}",
        "fixture_id": fid,
        "fixture_marked_final": marked,
    }
    log.info(
        "record_result home=%s away=%s score=%s fixture_id=%s marked=%d",
        home_name,
        away_name,
        detail["score"],
        fid,
        marked,
    )
    return detail


def sync_from_history_csv(
    csv_path: str | None = None,
    *,
    db_url: str | None = None,
) -> tuple[int, int]:
    """Upsert WC rows with non-null scores from the history CSV; mark fixtures final.

    Returns ``(results_upserted, fixtures_marked_final)``.
    """
    from pathlib import Path

    import pandas as pd

    from pitchedge import config
    from pitchedge.ingest.history import load_history_frame

    path = Path(csv_path) if csv_path is not None else Path(config.HISTORY_CSV_PATH)
    frame = load_history_frame(path)
    wc = frame[
        frame["tournament"].astype(str).str.contains("World Cup", case=False, na=False)
    ].copy()
    wc = wc[wc["home_score"].notna() & wc["away_score"].notna()]
    if wc.empty:
        log.info("sync_from_history_csv: no scored WC rows in %s", path)
        return 0, 0

    results_upserted = 0
    fixtures_marked = 0
    with db.connect(db_url) as conn:
        for record in wc.to_dict(orient="records"):
            home_goals = int(record["home_score"])
            away_goals = int(record["away_score"])
            match_date = record["date"]
            if isinstance(match_date, datetime):
                match_date = match_date.date()
            row = {
                "date": match_date,
                "home_id": team_name_to_id(str(record["home_team"])),
                "away_id": team_name_to_id(str(record["away_team"])),
                "home_goals": home_goals,
                "away_goals": away_goals,
                "competition": str(record["tournament"]).strip(),
                "neutral": bool(record["neutral"]) if record["neutral"] is not False else False,
            }
            res = conn.execute(text(UPSERT_RESULT_SQL), row)
            if res.rowcount:
                results_upserted += 1
            home_display = str(record["home_team"]).strip()
            away_display = str(record["away_team"]).strip()
            # Map Kaggle labels back to WC team names where they differ.
            if away_display == "Czech Republic":
                away_display = "Czechia"
            if home_display == "Korea Republic":
                home_display = "South Korea"
            fid = conn.execute(
                text(MATCH_FIXTURE_SQL),
                {
                    "match_date": match_date,
                    "home_name": home_display,
                    "away_name": away_display,
                },
            ).scalar()
            if fid is not None:
                upd = conn.execute(
                    text(MARK_FIXTURE_FINAL_SQL), {"fixture_id": int(fid)}
                )
                fixtures_marked += upd.rowcount or 0

    log.info(
        "sync_from_history_csv: upserted=%d fixtures_marked=%d path=%s",
        results_upserted,
        fixtures_marked,
        path,
    )
    return results_upserted, fixtures_marked


def apply_known_results(*, db_url: str | None = None) -> list[dict[str, Any]]:
    """Apply hard-coded opening scores when the history CSV still has NaN placeholders.

    Source: FIFA match reports (June 11–12 2026 openers). Extend this list or
    replace with an API adapter as the tournament progresses.
    """
    from datetime import date as date_cls

    entries = [
        dict(
            home_name="Mexico",
            away_name="South Africa",
            match_date=date_cls(2026, 6, 11),
            home_goals=2,
            away_goals=0,
            fixture_id=1,
            neutral=True,
        ),
        dict(
            home_name="South Korea",
            away_name="Czech Republic",
            wc_away_name="Czechia",
            match_date=date_cls(2026, 6, 12),
            home_goals=2,
            away_goals=1,
            fixture_id=2,
            neutral=True,
        ),
    ]
    return [record_result(db_url=db_url, **e) for e in entries]
