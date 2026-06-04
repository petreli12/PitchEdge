#!/usr/bin/env python3
"""Explore the-odds-api.com with your API key (no DB writes).

Helps answer:
  - Is your API key valid?
  - Is ``soccer_fifa_world_cup`` active on your plan?
  - What team name strings does the API use (for matching ``teams.csv``)?

Usage:
    export ODDS_API_KEY=your_key   # or set in .env
    uv run python scripts/probe_odds_api.py

Optional:
    uv run python scripts/probe_odds_api.py --sport soccer_fifa_world_cup --regions us,uk
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pitchedge import config  # noqa: E402
from pitchedge.ingest.odds import fetch_odds  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe the-odds-api.com v4")
    parser.add_argument("--sport", default=config.ODDS_API_SPORT_KEY)
    parser.add_argument("--regions", default=config.ODDS_API_REGIONS)
    parser.add_argument("--limit", type=int, default=8, help="events to print")
    args = parser.parse_args()

    if not config.ODDS_API_KEY:
        print(
            "ODDS_API_KEY is not set.\n"
            "1. Sign up: https://the-odds-api.com/\n"
            "2. Copy your key into .env as ODDS_API_KEY=...\n"
            "3. Re-run this script."
        )
        sys.exit(1)

    print(f"Base URL: {config.ODDS_API_BASE_URL}")
    print(f"Sport key: {args.sport}")
    print(f"Regions: {args.regions}")
    print()

    # List in-season sports (free, no quota).
    import requests

    sports_url = f"{config.ODDS_API_BASE_URL.rstrip('/')}/v4/sports"
    resp = requests.get(
        sports_url, params={"apiKey": config.ODDS_API_KEY}, timeout=30
    )
    resp.raise_for_status()
    sports = resp.json()
    wc = [s for s in sports if "world_cup" in s.get("key", "").lower()]
    print("World Cup related sport keys on your account:")
    for s in wc:
        active = s.get("active")
        print(f"  - {s['key']}: {s.get('title')} (active={active})")
    if not any(s.get("key") == args.sport for s in sports):
        print(
            f"\nWARNING: {args.sport!r} not in your /v4/sports list. "
            "Pick an active key from above and set ODDS_API_SPORT_KEY in .env."
        )
    print()

    try:
        events = fetch_odds(
            sport_key=args.sport,
            regions=args.regions,
            markets="h2h",
            odds_format="decimal",
        )
    except Exception as exc:
        print(f"Odds fetch failed: {exc}")
        sys.exit(1)

    print(f"Fetched {len(events)} events (uses API quota).")
    if not events:
        print(
            "No events returned. Before kickoff this can be normal; try closer to "
            "the tournament or check sport_key / regions."
        )
        return

    print(f"\nFirst {min(args.limit, len(events))} events (team names for CSV matching):")
    for event in events[: args.limit]:
        home = event.get("home_team")
        away = event.get("away_team")
        commence = event.get("commence_time")
        books = [b["key"] for b in event.get("bookmakers", [])]
        print(f"  {commence}  {home} vs {away}  books={books[:4]}")

    # Show one full h2h market as JSON sample for debugging.
    sample = events[0]
    for book in sample.get("bookmakers", [])[:1]:
        for market in book.get("markets", []):
            if market.get("key") == "h2h":
                print("\nSample h2h outcomes (check Draw label and team spelling):")
                print(json.dumps(market.get("outcomes"), indent=2))
                break

    print(
        "\nNext steps:\n"
        "  - Align teams.csv ``odds_name`` (or ``name``) with the home_team/away_team "
        "strings above.\n"
        "  - Run: make ingest-fixtures && make ingest-odds\n"
        "  - Tournament winner odds use a different key: soccer_fifa_world_cup_winner"
    )


if __name__ == "__main__":
    main()
