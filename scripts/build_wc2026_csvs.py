#!/usr/bin/env python3
"""Write data/wc2026/teams.csv and fixtures.csv from confirmed draw + schedule.

Times are converted from US/Eastern (America/New_York) to UTC for ``kickoff_utc``.

Usage (from repo root):
    uv run python scripts/build_wc2026_csvs.py
"""

from __future__ import annotations

import csv
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from wc2026_schedule_data import GROUP_STAGE, KNOCKOUT  # noqa: E402
from wc2026_teams_data import GROUPS, HISTORY_NAME_OVERRIDES  # noqa: E402

OUT_DIR = ROOT / "data" / "wc2026"
EASTERN = ZoneInfo("America/New_York")
UTC = ZoneInfo("UTC")


def _est_to_utc_iso(date_str: str, time_est: str) -> str:
    hour, minute = map(int, time_est.split(":"))
    local = datetime.strptime(date_str, "%Y-%m-%d").replace(
        hour=hour, minute=minute, tzinfo=EASTERN
    )
    return local.astimezone(UTC).isoformat()


def build_teams() -> dict[str, int]:
    """Write teams.csv; return name -> team_id."""
    name_to_id: dict[str, int] = {}
    team_id = 1
    rows: list[dict[str, str]] = []

    for group_label in sorted(GROUPS):
        for name, fifa_code, confederation, odds_name in GROUPS[group_label]:
            rows.append(
                {
                    "team_id": str(team_id),
                    "name": name,
                    "fifa_code": fifa_code,
                    "confederation": confederation,
                    "group_label": group_label,
                    "odds_name": odds_name,
                    "history_name": HISTORY_NAME_OVERRIDES.get(name, ""),
                }
            )
            name_to_id[name] = team_id
            team_id += 1

    path = OUT_DIR / "teams.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "team_id",
                "name",
                "fifa_code",
                "confederation",
                "group_label",
                "odds_name",
                "history_name",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"wrote {path} ({len(rows)} teams)")
    return name_to_id


def build_fixtures(name_to_id: dict[str, int]) -> None:
    rows: list[dict[str, str]] = []

    for fid, date, time_est, home, away, group, stage in GROUP_STAGE:
        rows.append(
            {
                "fixture_id": str(fid),
                "kickoff_utc": _est_to_utc_iso(date, time_est),
                "home_id": str(name_to_id[home]),
                "away_id": str(name_to_id[away]),
                "stage": stage,
                "group_label": group,
                "status": "scheduled",
            }
        )

    for fid, date, time_est, stage in KNOCKOUT:
        rows.append(
            {
                "fixture_id": str(fid),
                "kickoff_utc": _est_to_utc_iso(date, time_est),
                "home_id": "",
                "away_id": "",
                "stage": stage,
                "group_label": "",
                "status": "scheduled",
            }
        )

    path = OUT_DIR / "fixtures.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "fixture_id",
                "kickoff_utc",
                "home_id",
                "away_id",
                "stage",
                "group_label",
                "status",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"wrote {path} ({len(rows)} fixtures)")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    name_to_id = build_teams()
    build_fixtures(name_to_id)
    print("Done. Next: make ingest-fixtures (after db-up && migrate)")


if __name__ == "__main__":
    main()
