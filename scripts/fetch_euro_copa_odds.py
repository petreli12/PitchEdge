#!/usr/bin/env python3
"""Fetch Euro 2024 + Copa 2024 historical odds into a local cache CSV.

Requires an Odds API key on a **paid** plan with ``/v4/historical`` access.
Free plans return 401 (HISTORICAL_UNAVAILABLE_ON_FREE_USAGE_PLAN).

Usage:
    uv run python scripts/fetch_euro_copa_odds.py
    make backtest   # reads data/backtest/euro_copa_odds_cache.csv when present
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pitchedge import config
from pitchedge.eval.backtest import _frame_to_match_rows, eval_matches
from pitchedge.eval.tournaments import HELD_OUT_TOURNAMENTS
from pitchedge.ingest.backtest_odds import (
    enrich_odds_from_odds_api,
    odds_api_historical_available,
)
from pitchedge.ingest.history import load_history_frame

log = logging.getLogger(__name__)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    if not config.ODDS_API_KEY:
        log.error("ODDS_API_KEY is not set in .env")
        sys.exit(1)
    if not odds_api_historical_available():
        log.error(
            "Historical odds are not available on this API key (free plan). "
            "Upgrade at https://the-odds-api.com then re-run this script."
        )
        sys.exit(1)

    frame = load_history_frame(Path(config.HISTORY_CSV_PATH))
    all_matches = _frame_to_match_rows(frame)
    payload: list[dict] = []
    for tournament in HELD_OUT_TOURNAMENTS:
        if not tournament.odds_api_sport_key:
            continue
        for m in eval_matches(all_matches, tournament):
            payload.append(
                {
                    "tournament_slug": tournament.slug,
                    "date": m["date"],
                    "home_team": m["home_team"],
                    "away_team": m["away_team"],
                }
            )

    rows: list[dict] = []
    by_slug = {t.slug: t for t in HELD_OUT_TOURNAMENTS}
    for slug in ("euro_2024", "copa_2024"):
        tourn = by_slug[slug]
        subset = [m for m in payload if m["tournament_slug"] == slug]
        log.info("Fetching %s (%d matches)...", tourn.label, len(subset))
        df = enrich_odds_from_odds_api(
            subset,
            sport_key=tourn.odds_api_sport_key,
            historical_ok=True,
        )
        if not df.empty:
            rows.extend(df.to_dict(orient="records"))

    if not rows:
        log.error("No odds rows fetched; cache not written.")
        sys.exit(1)

    out = Path(config.BACKTEST_EURO_COPA_CACHE)
    out.parent.mkdir(parents=True, exist_ok=True)
    import pandas as pd

    pd.DataFrame(rows)[
        ["date", "home_team", "away_team", "home_odds", "draw_odds", "away_odds"]
    ].to_csv(out, index=False)
    log.info("Wrote %d rows to %s", len(rows), out)


if __name__ == "__main__":
    main()
