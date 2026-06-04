"""Dashboard data dispatch: live Postgres or bundled read-only snapshot.

The Streamlit app imports these functions instead of the raw query modules so a
single switch (``snapshot.is_snapshot_mode()``) decides the source. In live mode
they delegate to the existing DB query/aggregation functions; in snapshot mode
they read the frozen JSON export. No probabilities are recomputed in either
path.
"""

from __future__ import annotations

from typing import Any

from pitchedge.dashboard import snapshot


def is_snapshot_mode() -> bool:
    return snapshot.is_snapshot_mode()


def dashboard_views() -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Return ``(upcoming, sim_rows, receipt)`` from the active source."""
    if snapshot.is_snapshot_mode():
        return (
            snapshot.load_upcoming(),
            snapshot.load_sim_results(),
            snapshot.load_receipt(),
        )
    from pitchedge.dashboard.queries import (
        fetch_latest_sim_results,
        fetch_prediction_receipt_status,
        fetch_upcoming_match_probs,
    )

    return (
        fetch_upcoming_match_probs(),
        fetch_latest_sim_results(),
        fetch_prediction_receipt_status(),
    )


def calibration_summary() -> dict[str, Any]:
    if snapshot.is_snapshot_mode():
        return snapshot.load_calibration_summary()
    from pitchedge.content.calibration_tracker import load_calibration_summary

    return load_calibration_summary()


def scored_home_win_series() -> dict[str, Any]:
    if snapshot.is_snapshot_mode():
        return snapshot.load_scored_home_win_series()
    from pitchedge.content.calibration_tracker import load_scored_home_win_series

    return load_scored_home_win_series()


def fixtures_missing_odds() -> list[dict[str, Any]]:
    if snapshot.is_snapshot_mode():
        return snapshot.load_missing_odds()
    from pitchedge.dashboard.queries import fetch_fixtures_missing_odds

    return fetch_fixtures_missing_odds()


def ranked_disagreements(limit: int = 8) -> list[Any]:
    """Top model-vs-market disagreements from the active source."""
    from pitchedge.content.daily_disagreement import rank_disagreements

    if snapshot.is_snapshot_mode():
        candidates = snapshot.load_disagreement_candidates()
    else:
        from pitchedge import db
        from pitchedge.content.daily_disagreement import fetch_candidates

        with db.connect() as conn:
            candidates = fetch_candidates(conn, within_hours=24 * 365)
    return rank_disagreements(candidates)[:limit]
