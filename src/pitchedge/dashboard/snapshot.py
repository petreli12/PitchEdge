"""Read-only snapshot data source for the public dashboard.

The public deploy (e.g. Streamlit Community Cloud) cannot reach the local
Postgres. Instead of recomputing anything, we bundle a frozen export of the
exact datasets the dashboard renders and read them from JSON here. This matches
the project's read-only / "locked receipts baseline" design: the snapshot is
produced once by ``scripts/export_snapshot.py`` against the live DB and never
mutated by the app.

Shapes returned here mirror the live query functions in ``dashboard.queries``
and ``content.calibration_tracker`` so the rendering code is identical in both
modes.
"""

from __future__ import annotations

import json
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any

from pitchedge import config

# Datasets written by scripts/export_snapshot.py.
UPCOMING_FILE = "upcoming.json"
SIM_FILE = "sim_results.json"
RECEIPT_FILE = "receipt.json"
MISSING_ODDS_FILE = "missing_odds.json"
DISAGREEMENT_FILE = "disagreement_candidates.json"
CALIBRATION_FILE = "calibration_summary.json"
SCORED_SERIES_FILE = "scored_home_win_series.json"
META_FILE = "meta.json"

# Datetime-bearing keys that should be parsed back into datetimes on load so the
# display helpers (strftime) work exactly as they do for DB rows.
_DT_KEYS = ("kickoff_utc", "model_predicted_utc", "run_batch_utc", "latest_model_utc", "earliest_model_utc")


def snapshot_dir() -> Path | None:
    """Return the active snapshot directory, or ``None`` for live DB mode.

    Snapshot mode is **opt-in** via ``DASHBOARD_SNAPSHOT_DIR`` so the mere
    presence of a committed ``data/snapshot/`` never hijacks local live-DB runs
    (``make dashboard``). The Streamlit Cloud entrypoint (``streamlit_app.py``)
    sets this for the public deploy; ``make dashboard-snapshot`` sets it locally.
    """
    configured = config.DASHBOARD_SNAPSHOT_DIR.strip()
    if not configured:
        return None
    path = Path(configured)
    return path if (path / META_FILE).exists() else None


def is_snapshot_mode() -> bool:
    """True when the dashboard should serve the bundled snapshot."""
    return snapshot_dir() is not None


def _read(name: str) -> Any:
    directory = snapshot_dir()
    if directory is None:
        raise RuntimeError("snapshot mode is not active; no snapshot directory found")
    with (directory / name).open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _parse_dt(value: Any) -> Any:
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value)
        except ValueError:
            return value
    return value


def _hydrate_dts(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    for row in rows:
        for key in _DT_KEYS:
            if key in row and row[key] is not None:
                row[key] = _parse_dt(row[key])
    return rows


@lru_cache(maxsize=None)
def load_meta() -> dict[str, Any]:
    """Snapshot provenance: when it was generated and from which source."""
    return _read(META_FILE)


def load_upcoming() -> list[dict[str, Any]]:
    return _hydrate_dts(_read(UPCOMING_FILE))


def load_sim_results() -> list[dict[str, Any]]:
    return _hydrate_dts(_read(SIM_FILE))


def load_receipt() -> dict[str, Any]:
    row = _read(RECEIPT_FILE)
    for key in _DT_KEYS:
        if key in row and row[key] is not None:
            row[key] = _parse_dt(row[key])
    return row


def load_missing_odds() -> list[dict[str, Any]]:
    return _hydrate_dts(_read(MISSING_ODDS_FILE))


def load_disagreement_candidates() -> list[Any]:
    """Reconstruct ``daily_disagreement.Candidate`` objects from the snapshot."""
    from pitchedge.content.daily_disagreement import Candidate

    rows = _read(DISAGREEMENT_FILE)
    return [Candidate(**row) for row in rows]


def load_calibration_summary() -> dict[str, Any]:
    """Reconstruct ``{source: SourceCalibration}`` from the snapshot."""
    from pitchedge.content.calibration_tracker import SourceCalibration

    raw = _read(CALIBRATION_FILE)
    out: dict[str, Any] = {}
    for source, payload in raw.items():
        out[source] = SourceCalibration(
            source=payload["source"],
            n=int(payload["n"]),
            mean_brier=_nan(payload["mean_brier"]),
            mean_log_loss=_nan(payload["mean_log_loss"]),
            ece=_nan(payload["ece"]),
        )
    return out


def load_scored_home_win_series() -> dict[str, Any]:
    """Reconstruct the reliability-diagram series (numpy arrays) from snapshot."""
    import numpy as np

    raw = _read(SCORED_SERIES_FILE)
    return {
        "model_home": np.asarray(raw.get("model_home", []), dtype=float),
        "market_home": np.asarray(raw.get("market_home", []), dtype=float),
        "outcomes": np.asarray(raw.get("outcomes", []), dtype=int),
        "outcomes_home": np.asarray(raw.get("outcomes_home", []), dtype=float),
        "n_fixtures": int(raw.get("n_fixtures", 0)),
    }


def _nan(value: Any) -> float:
    return float("nan") if value is None else float(value)
