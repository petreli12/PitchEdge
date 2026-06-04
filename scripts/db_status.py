#!/usr/bin/env python3
"""Print which database PitchEdge is using and row counts for core tables."""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pitchedge import config, db  # noqa: E402

TABLES = (
    "raw_results",
    "teams",
    "fixtures",
    "odds_snapshots",
    "team_ratings",
    "match_predictions",
)


def _redact_url(url: str) -> str:
    return re.sub(r":([^:@/]+)@", r":***@", url)


def main() -> None:
    print(f"DB_URL: {_redact_url(config.DB_URL)}")
    try:
        for table in TABLES:
            row = db.fetch_one(f"SELECT count(*) AS n FROM {table}")
            print(f"  {table}: {row['n']}")
    except Exception as exc:
        print(f"  (could not connect: {exc})")
        sys.exit(1)


if __name__ == "__main__":
    main()
