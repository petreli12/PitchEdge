"""Export a frozen, read-only snapshot of the dashboard's datasets.

Runs against the live Postgres (where the locked pre-kickoff predictions and the
latest sim live) and serializes the exact views the Streamlit app renders into
``data/snapshot/*.json``. The public deploy reads those files instead of the DB
(see ``pitchedge.dashboard.snapshot``). This recomputes nothing — it just freezes
what the live queries already return.

Usage:
    DB_URL=postgresql+psycopg://.../pitchedge \
        uv run python scripts/export_snapshot.py
    # or: make export-snapshot
"""

from __future__ import annotations

import dataclasses
import json
import logging
import math
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any

from pitchedge import config, db
from pitchedge.content.calibration_tracker import (
    load_calibration_summary,
    load_scored_home_win_series,
)
from pitchedge.content.daily_disagreement import fetch_candidates
from pitchedge.dashboard import snapshot
from pitchedge.dashboard.queries import (
    fetch_fixtures_missing_odds,
    fetch_latest_sim_results,
    fetch_prediction_receipt_status,
    fetch_upcoming_match_probs,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
log = logging.getLogger("export_snapshot")

OUT_DIR = Path(config.DASHBOARD_SNAPSHOT_DIR.strip() or (Path("data") / "snapshot"))


def _clean(value: Any) -> Any:
    """JSON-safe coercion: datetimes -> ISO, Decimal -> float, NaN -> None."""
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, Decimal):
        value = float(value)
    if isinstance(value, float):
        return None if math.isnan(value) else value
    if isinstance(value, dict):
        return {k: _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    if hasattr(value, "item") and not isinstance(value, (str, bytes)):
        try:
            return _clean(value.item())  # numpy scalar
        except Exception:  # pragma: no cover - defensive
            return value
    return value


def _write(name: str, payload: Any) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with (OUT_DIR / name).open("w", encoding="utf-8") as fh:
        json.dump(_clean(payload), fh, indent=2, allow_nan=False)
    log.info("wrote %s", OUT_DIR / name)


def main() -> None:
    log.info("exporting dashboard snapshot from DB to %s", OUT_DIR)

    upcoming = fetch_upcoming_match_probs()
    sim_rows = fetch_latest_sim_results()
    receipt = fetch_prediction_receipt_status()
    missing_odds = fetch_fixtures_missing_odds()

    with db.connect() as conn:
        candidates = fetch_candidates(conn, within_hours=24 * 365)
    cand_rows = [dataclasses.asdict(c) for c in candidates]

    calibration = load_calibration_summary()
    calibration_payload = {src: dataclasses.asdict(sc) for src, sc in calibration.items()}

    series = load_scored_home_win_series()
    series_payload = {
        "model_home": [float(x) for x in series["model_home"]],
        "market_home": [float(x) for x in series["market_home"]],
        "outcomes": [int(x) for x in series["outcomes"]],
        "outcomes_home": [float(x) for x in series["outcomes_home"]],
        "n_fixtures": int(series["n_fixtures"]),
    }

    _write(snapshot.UPCOMING_FILE, upcoming)
    _write(snapshot.SIM_FILE, sim_rows)
    _write(snapshot.RECEIPT_FILE, receipt)
    _write(snapshot.MISSING_ODDS_FILE, missing_odds)
    _write(snapshot.DISAGREEMENT_FILE, cand_rows)
    _write(snapshot.CALIBRATION_FILE, calibration_payload)
    _write(snapshot.SCORED_SERIES_FILE, series_payload)

    meta = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        # Database name only — never host/port/credentials (this file is public).
        "source_db": config.DB_URL.rsplit("/", 1)[-1].split("?")[0],
        "n_upcoming_fixtures": len(upcoming),
        "n_sim_teams": len(sim_rows),
        "n_disagreement_candidates": len(cand_rows),
        "n_scored_fixtures": series_payload["n_fixtures"],
        "model_temperature": config.MODEL_TEMPERATURE,
        "blend_w": config.BLEND_W,
    }
    _write(snapshot.META_FILE, meta)

    log.info(
        "snapshot complete: %d fixtures, %d sim teams, %d disagreements, %d scored",
        len(upcoming),
        len(sim_rows),
        len(cand_rows),
        series_payload["n_fixtures"],
    )


if __name__ == "__main__":
    main()
