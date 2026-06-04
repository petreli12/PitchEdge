"""Live receipts scorecard: standalone model vs market (never blend).

Reads ``prediction_scores`` joined to ``match_predictions``. Blend rows are
excluded so the public tracker scores PitchEdge's model, not the de-vigged market
copy we publish when BLEND_W=0.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from sqlalchemy import text

from pitchedge.content.daily_disagreement import published_model_probs
from pitchedge.eval.metrics import expected_calibration_error


@dataclass
class SourceCalibration:
    """Pooled proper scores for one prediction source."""

    source: str
    n: int
    mean_brier: float
    mean_log_loss: float
    ece: float


_SCORED_ROWS_SQL = """
SELECT
    mp.source,
    mp.p_home,
    mp.p_draw,
    mp.p_away,
    ps.brier,
    ps.log_loss,
    ps.outcome
FROM prediction_scores ps
JOIN match_predictions mp ON mp.id = ps.prediction_id
WHERE mp.source IN ('model', 'market')
"""


def _outcome_code(outcome_char: str) -> int:
    return {"H": 0, "D": 1, "A": 2}[outcome_char]


def summarize_scored_rows(rows: list[dict[str, Any]]) -> dict[str, SourceCalibration]:
    """Aggregate Brier, log loss, and ECE per source from scored DB rows."""
    by_source: dict[str, list[dict[str, Any]]] = {"model": [], "market": []}
    for row in rows:
        src = row["source"]
        if src in by_source:
            by_source[src].append(row)

    out: dict[str, SourceCalibration] = {}
    for source, subset in by_source.items():
        if not subset:
            continue
        if source == "model":
            probs = [
                published_model_probs(
                    float(r["p_home"]),
                    float(r["p_draw"]),
                    float(r["p_away"]),
                )
                for r in subset
            ]
        else:
            probs = [
                (float(r["p_home"]), float(r["p_draw"]), float(r["p_away"]))
                for r in subset
            ]
        outcomes = [_outcome_code(str(r["outcome"])) for r in subset]
        briers = [float(r["brier"]) for r in subset if r["brier"] is not None]
        losses = [float(r["log_loss"]) for r in subset if r["log_loss"] is not None]
        out[source] = SourceCalibration(
            source=source,
            n=len(subset),
            mean_brier=float(sum(briers) / len(briers)) if briers else float("nan"),
            mean_log_loss=float(sum(losses) / len(losses)) if losses else float("nan"),
            ece=expected_calibration_error(probs, outcomes),
        )
    return out


def fetch_calibration_summary(conn) -> dict[str, SourceCalibration]:
    """Load model vs market receipts from the database (never blend)."""
    result = conn.execute(text(_SCORED_ROWS_SQL))
    rows = [dict(r) for r in result.mappings().all()]
    return summarize_scored_rows(rows)


def load_calibration_summary(
    *, db_url: str | None = None
) -> dict[str, SourceCalibration]:
    """Open a DB connection and return model vs market calibration (never blend)."""
    from pitchedge import db

    with db.connect(db_url) as conn:
        return fetch_calibration_summary(conn)


_PAIRED_SCORED_SQL = """
SELECT
    mp.fixture_id,
    mp.source,
    mp.p_home,
    mp.p_draw,
    mp.p_away,
    ps.outcome
FROM prediction_scores ps
JOIN match_predictions mp ON mp.id = ps.prediction_id
WHERE mp.source IN ('model', 'market')
"""


def fetch_scored_home_win_series(conn) -> dict[str, Any]:
    """Paired home-win probs and outcomes for reliability diagram (model vs market)."""
    import numpy as np

    from pitchedge.eval.calibration import outcomes_to_home_indicator

    result = conn.execute(text(_PAIRED_SCORED_SQL))
    all_rows = [dict(r) for r in result.mappings().all()]
    by_fixture: dict[int, dict[str, Any]] = {}
    for row in all_rows:
        fid = int(row["fixture_id"])
        slot = by_fixture.setdefault(fid, {})
        slot[str(row["source"])] = row
        slot["outcome"] = row["outcome"]

    m_home: list[float] = []
    k_home: list[float] = []
    out_codes: list[int] = []
    for slot in by_fixture.values():
        if "model" not in slot or "market" not in slot:
            continue
        mr = slot["model"]
        kr = slot["market"]
        ph, _, _ = published_model_probs(
            float(mr["p_home"]),
            float(mr["p_draw"]),
            float(mr["p_away"]),
        )
        m_home.append(ph)
        k_home.append(float(kr["p_home"]))
        out_codes.append({"H": 0, "D": 1, "A": 2}[str(slot["outcome"])])

    outcomes_arr = np.asarray(out_codes, dtype=int)
    return {
        "model_home": np.asarray(m_home, dtype=float),
        "market_home": np.asarray(k_home, dtype=float),
        "outcomes": outcomes_arr,
        "outcomes_home": outcomes_to_home_indicator(outcomes_arr),
        "n_fixtures": len(out_codes),
    }


def load_scored_home_win_series(*, db_url: str | None = None) -> dict[str, Any]:
    """Open DB and return paired home-win series for the calibration diagram."""
    from pitchedge import db

    with db.connect(db_url) as conn:
        return fetch_scored_home_win_series(conn)


def model_vs_market_gap(summary: dict[str, SourceCalibration]) -> dict[str, float | None]:
    """Return paired gaps (model minus market) when both sides exist."""
    model = summary.get("model")
    market = summary.get("market")
    if not model or not market:
        return {"brier_gap": None, "log_loss_gap": None, "ece_gap": None}
    return {
        "brier_gap": model.mean_brier - market.mean_brier,
        "log_loss_gap": model.mean_log_loss - market.mean_log_loss,
        "ece_gap": model.ece - market.ece,
    }
