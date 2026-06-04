#!/usr/bin/env python3
"""List scheduled WC fixtures with no odds_snapshots (Phase 7 gate)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pitchedge.dashboard.queries import fetch_fixtures_missing_odds


def main() -> None:
    missing = fetch_fixtures_missing_odds()
    if not missing:
        print("All scheduled fixtures with known teams have at least one odds snapshot.")
        return
    print(f"fixtures missing odds: {len(missing)}")
    for row in missing:
        print(
            f"  id={row['fixture_id']}: {row['home']} vs {row['away']} "
            f"({row['kickoff_utc']})"
        )
    sys.exit(1 if missing else 0)


if __name__ == "__main__":
    main()
