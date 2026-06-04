"""Load World Cup teams and fixtures from user-provided CSVs.

**teams CSV** (required columns):
  team_id, name, confederation, group_label
  Optional: fifa_code, odds_name, history_name (Kaggle label when != name)

**fixtures CSV** (required columns):
  fixture_id, kickoff_utc, home_id, away_id, stage, group_label
  Optional: status (defaults to ``scheduled``)

Times in ``kickoff_utc`` must be ISO-8601 UTC (e.g. ``2026-06-11T19:00:00+00:00``).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

from pitchedge import config, db

log = logging.getLogger(__name__)

TEAM_COLUMNS = ("team_id", "name", "confederation", "group_label")
FIXTURE_COLUMNS = (
    "fixture_id",
    "kickoff_utc",
    "home_id",
    "away_id",
    "stage",
    "group_label",
)

INSERT_TEAM_SQL = """
INSERT INTO teams (
    team_id, name, fifa_code, confederation, group_label, odds_name, history_name
) VALUES (
    :team_id, :name, :fifa_code, :confederation, :group_label, :odds_name, :history_name
)
ON CONFLICT (team_id) DO UPDATE SET
    name = EXCLUDED.name,
    fifa_code = EXCLUDED.fifa_code,
    confederation = EXCLUDED.confederation,
    group_label = EXCLUDED.group_label,
    odds_name = EXCLUDED.odds_name,
    history_name = EXCLUDED.history_name
"""

INSERT_FIXTURE_SQL = """
INSERT INTO fixtures (
    fixture_id, kickoff_utc, home_id, away_id, stage, group_label, status
) VALUES (
    :fixture_id, :kickoff_utc, :home_id, :away_id, :stage, :group_label, :status
)
ON CONFLICT (fixture_id) DO NOTHING
"""


def load_teams_frame(csv_path: str | Path) -> pd.DataFrame:
    path = Path(csv_path)
    if not path.is_file():
        raise FileNotFoundError(f"teams CSV not found: {path}")

    frame = pd.read_csv(path)
    missing = [c for c in TEAM_COLUMNS if c not in frame.columns]
    if missing:
        raise ValueError(f"teams CSV missing columns {missing}; need {TEAM_COLUMNS}")
    return frame


def load_fixtures_frame(csv_path: str | Path) -> pd.DataFrame:
    path = Path(csv_path)
    if not path.is_file():
        raise FileNotFoundError(f"fixtures CSV not found: {path}")

    frame = pd.read_csv(path)
    missing = [c for c in FIXTURE_COLUMNS if c not in frame.columns]
    if missing:
        raise ValueError(
            f"fixtures CSV missing columns {missing}; need {FIXTURE_COLUMNS}"
        )
    return frame


def teams_frame_to_rows(frame: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in frame.to_dict(orient="records"):
        fifa = record.get("fifa_code")
        if fifa is not None and isinstance(fifa, float) and pd.isna(fifa):
            fifa = None
        odds_name = record.get("odds_name")
        if odds_name is not None and isinstance(odds_name, float) and pd.isna(odds_name):
            odds_name = None
        elif odds_name is not None:
            odds_name = str(odds_name).strip() or None
        history_name = record.get("history_name")
        if history_name is not None and isinstance(history_name, float) and pd.isna(
            history_name
        ):
            history_name = None
        elif history_name is not None:
            history_name = str(history_name).strip() or None
        rows.append(
            {
                "team_id": int(record["team_id"]),
                "name": str(record["name"]).strip(),
                "fifa_code": None if fifa is None else str(fifa).strip(),
                "confederation": str(record["confederation"]).strip(),
                "group_label": str(record["group_label"]).strip(),
                "odds_name": odds_name,
                "history_name": history_name,
            }
        )
    return rows


def fixtures_frame_to_rows(frame: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    kickoffs = pd.to_datetime(frame["kickoff_utc"], utc=True, errors="coerce")
    if kickoffs.isna().any():
        bad = int(kickoffs.isna().sum())
        raise ValueError(f"fixtures CSV has {bad} rows with unparseable kickoff_utc")

    for record, kickoff in zip(frame.to_dict(orient="records"), kickoffs, strict=True):
        status = record.get("status")
        if status is None or (isinstance(status, float) and pd.isna(status)):
            status = "scheduled"
        else:
            status = str(status).strip().lower()
        if status not in ("scheduled", "final"):
            raise ValueError(f"invalid fixture status {status!r}")

        home = record["home_id"]
        away = record["away_id"]
        rows.append(
            {
                "fixture_id": int(record["fixture_id"]),
                "kickoff_utc": kickoff.to_pydatetime(),
                "home_id": None if pd.isna(home) else int(home),
                "away_id": None if pd.isna(away) else int(away),
                "stage": str(record["stage"]).strip(),
                "group_label": str(record["group_label"]).strip(),
                "status": status,
            }
        )
    return rows


def ingest_teams(
    csv_path: str | Path | None = None,
    *,
    db_url: str | None = None,
) -> tuple[int, int]:
    """Load teams idempotently. Returns ``(attempted, inserted)``."""
    path = Path(csv_path) if csv_path is not None else Path(config.TEAMS_CSV_PATH)
    rows = teams_frame_to_rows(load_teams_frame(path))
    attempted = len(rows)
    inserted = db.execute(INSERT_TEAM_SQL, rows, db_url=db_url)
    inserted = max(inserted, 0)
    log.info("teams ingest: path=%s attempted=%d inserted=%d", path, attempted, inserted)
    return attempted, inserted


def ingest_fixtures(
    csv_path: str | Path | None = None,
    *,
    db_url: str | None = None,
) -> tuple[int, int]:
    """Load fixtures idempotently. Returns ``(attempted, inserted)``."""
    path = (
        Path(csv_path) if csv_path is not None else Path(config.FIXTURES_CSV_PATH)
    )
    rows = fixtures_frame_to_rows(load_fixtures_frame(path))
    attempted = len(rows)
    inserted = db.execute(INSERT_FIXTURE_SQL, rows, db_url=db_url)
    inserted = max(inserted, 0)
    log.info(
        "fixtures ingest: path=%s attempted=%d inserted=%d",
        path,
        attempted,
        inserted,
    )
    return attempted, inserted


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    t_attempted, t_inserted = ingest_teams()
    f_attempted, f_inserted = ingest_fixtures()
    print(f"teams: attempted={t_attempted} inserted={t_inserted}")
    print(f"fixtures: attempted={f_attempted} inserted={f_inserted}")


if __name__ == "__main__":
    main()
