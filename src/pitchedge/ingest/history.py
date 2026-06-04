"""Load international match history into ``raw_results``.

Expects the martj42 / Kaggle ``results.csv`` layout (also mirrored on GitHub):
  date, home_team, away_team, home_score, away_score, tournament, city, country,
  neutral

Source: https://www.kaggle.com/datasets/martj42/international-football-results-from-1872-to-2017
Default path: ``data/kaggle/international-results/results.csv`` (download separately).

Dedupes on ``(date, home_id, away_id, competition)`` via ``ON CONFLICT DO NOTHING``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

from pitchedge import config, db
from pitchedge.ingest.team_ids import team_name_to_id

log = logging.getLogger(__name__)

# Kaggle / GitHub column names (user CSV may alias ``tournament`` -> same field).
REQUIRED_COLUMNS = (
    "date",
    "home_team",
    "away_team",
    "home_score",
    "away_score",
    "tournament",
    "neutral",
)

INSERT_SQL = """
INSERT INTO raw_results (
    date, home_id, away_id, home_goals, away_goals, competition, neutral
) VALUES (
    :date, :home_id, :away_id, :home_goals, :away_goals, :competition, :neutral
)
ON CONFLICT (date, home_id, away_id, competition) DO NOTHING
"""

# Batch size for executemany inserts (full history is ~40k+ rows).
_BATCH_SIZE = 2000


def _parse_neutral(value: Any) -> bool:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return False
    if isinstance(value, bool):
        return value
    text = str(value).strip().upper()
    if text in ("TRUE", "T", "1", "YES"):
        return True
    if text in ("FALSE", "F", "0", "NO"):
        return False
    raise ValueError(f"unrecognized neutral flag: {value!r}")


def _optional_int(value: Any) -> int | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    return int(value)


def load_history_frame(csv_path: str | Path) -> pd.DataFrame:
    """Read and validate the historical results CSV without touching the DB."""
    path = Path(csv_path)
    if not path.is_file():
        raise FileNotFoundError(
            f"history CSV not found: {path}. Download from Kaggle "
            "(martj42/international-football-results) and place at "
            f"{config.HISTORY_CSV_PATH}"
        )

    frame = pd.read_csv(path)
    missing = [c for c in REQUIRED_COLUMNS if c not in frame.columns]
    if missing:
        raise ValueError(
            f"history CSV missing columns {missing}; expected {REQUIRED_COLUMNS}"
        )

    frame = frame[list(REQUIRED_COLUMNS)].copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.date
    if frame["date"].isna().any():
        bad = int(frame["date"].isna().sum())
        raise ValueError(f"history CSV has {bad} rows with unparseable date")

    return frame


def frame_to_rows(frame: pd.DataFrame) -> list[dict[str, Any]]:
    """Convert a validated history frame to DB insert parameter dicts."""
    rows: list[dict[str, Any]] = []
    for record in frame.to_dict(orient="records"):
        competition = record["tournament"]
        if competition is None or (
            isinstance(competition, float) and pd.isna(competition)
        ):
            competition = ""
        else:
            competition = str(competition).strip()

        rows.append(
            {
                "date": record["date"],
                "home_id": team_name_to_id(str(record["home_team"])),
                "away_id": team_name_to_id(str(record["away_team"])),
                "home_goals": _optional_int(record["home_score"]),
                "away_goals": _optional_int(record["away_score"]),
                "competition": competition,
                "neutral": _parse_neutral(record["neutral"]),
            }
        )
    return rows


def ingest_history(
    csv_path: str | Path | None = None,
    *,
    db_url: str | None = None,
) -> tuple[int, int]:
    """Load historical results idempotently.

    Returns ``(rows_attempted, rows_inserted)`` where ``rows_inserted`` is the
    sum of ``cursor.rowcount`` from batched inserts (0 on re-run when all exist).
    """
    path = Path(csv_path) if csv_path is not None else Path(config.HISTORY_CSV_PATH)
    frame = load_history_frame(path)
    rows = frame_to_rows(frame)

    attempted = len(rows)
    inserted = 0
    for start in range(0, attempted, _BATCH_SIZE):
        batch = rows[start : start + _BATCH_SIZE]
        count = db.execute(INSERT_SQL, batch, db_url=db_url)
        if count > 0:
            inserted += count

    log.info(
        "history ingest: path=%s attempted=%d inserted=%d",
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
    attempted, inserted = ingest_history()
    print(f"history: attempted={attempted} inserted={inserted}")


if __name__ == "__main__":
    main()
