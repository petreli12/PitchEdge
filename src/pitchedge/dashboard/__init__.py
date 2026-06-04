"""Read-only dashboard queries and landing-page helpers (no probability math)."""

from pitchedge.dashboard.queries import (
    fetch_latest_sim_results,
    fetch_prediction_receipt_status,
    fetch_upcoming_match_probs,
)
from pitchedge.dashboard.subscribers import capture_subscriber_email

__all__ = [
    "capture_subscriber_email",
    "fetch_latest_sim_results",
    "fetch_prediction_receipt_status",
    "fetch_upcoming_match_probs",
]
