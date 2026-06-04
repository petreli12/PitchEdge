"""Data ingestion: historical results, WC fixtures/teams, odds snapshots.

All loaders are idempotent (``ON CONFLICT DO NOTHING`` on natural keys).
Teams use upsert so ``odds_name`` can be refreshed without duplicating rows.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

__all__ = [
    "ingest_history",
    "ingest_teams",
    "ingest_fixtures",
    "ingest_odds_snapshots",
]

if TYPE_CHECKING:
    from pitchedge.ingest.fixtures import ingest_fixtures, ingest_teams
    from pitchedge.ingest.history import ingest_history
    from pitchedge.ingest.odds import ingest_odds_snapshots


def __getattr__(name: str):
    if name == "ingest_history":
        from pitchedge.ingest.history import ingest_history

        return ingest_history
    if name in ("ingest_teams", "ingest_fixtures"):
        from pitchedge.ingest import fixtures

        return getattr(fixtures, name)
    if name == "ingest_odds_snapshots":
        from pitchedge.ingest.odds import ingest_odds_snapshots

        return ingest_odds_snapshots
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
