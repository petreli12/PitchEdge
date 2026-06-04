"""Log pre-kickoff model, market, and blend predictions to ``match_predictions``.

Standalone model probabilities use Dixon-Coles at ``MODEL_TEMPERATURE=1.0`` (raw
``wc_match_probs``), not temperature-scaled display probs. Receipts compare model
vs market, never blend-as-model.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import text

from pitchedge import config, db
from pitchedge.model.blend import blend
from pitchedge.model.devig import devig_three_way
from pitchedge.model.dixon_coles import DixonColesModel, MatchProbs, fit_dixon_coles_from_db
from pitchedge.model.dixon_coles import match_probs
from pitchedge.model.venues import dixon_coles_neutral_for_wc_fixture
from pitchedge.sim.wc_teams import assert_wc_teams_in_model, wc_id_to_model_id

log = logging.getLogger(__name__)

INSERT_PREDICTION_SQL = """
INSERT INTO match_predictions (
    fixture_id, model_version, predicted_utc,
    p_home, p_draw, p_away, exp_home_goals, exp_away_goals, source
) VALUES (
    :fixture_id, :model_version, :predicted_utc,
    :p_home, :p_draw, :p_away, :exp_home_goals, :exp_away_goals, :source
)
"""

UPCOMING_FIXTURES_SQL = """
SELECT
    f.fixture_id,
    f.home_id,
    f.away_id,
    f.kickoff_utc,
    th.name AS home_name,
    ta.name AS away_name
FROM fixtures f
JOIN teams th ON th.team_id = f.home_id
JOIN teams ta ON ta.team_id = f.away_id
WHERE f.status = 'scheduled'
  AND f.kickoff_utc > :now_utc
  AND f.home_id IS NOT NULL
  AND f.away_id IS NOT NULL
ORDER BY f.kickoff_utc ASC
"""

LATEST_ODDS_SQL = """
SELECT DISTINCT ON (fixture_id)
    fixture_id, home_odds, draw_odds, away_odds, book, captured_utc
FROM odds_snapshots
WHERE fixture_id = ANY(:fixture_ids)
ORDER BY fixture_id, captured_utc DESC
"""


def fixture_model_probs(
    model: DixonColesModel,
    home_wc_id: int,
    away_wc_id: int,
    wc_to_model_id: dict[int, int],
    *,
    max_goals: int | None = None,
    host_home: bool | None = None,
) -> MatchProbs:
    """Dixon-Coles probabilities for a ``fixtures`` row (WC ``team_id``s).

    The WC id (``fixtures.home_id``, 1..48) drives the neutral-venue / host
    policy, while the fitted model is keyed by the historical hash id, so the WC
    ids must be translated via ``wc_to_model_id`` before the model lookup. Mixing
    the two id spaces silently collapses to the league-average fallback (equal
    lambdas), so a missing mapping raises rather than logging a bogus prediction.
    """
    try:
        home_model_id = wc_to_model_id[home_wc_id]
        away_model_id = wc_to_model_id[away_wc_id]
    except KeyError as exc:  # pragma: no cover - guarded upstream
        raise KeyError(
            f"WC team_id {exc.args[0]} has no model_team_id mapping; cannot "
            "compute model probabilities"
        ) from exc
    neutral = dixon_coles_neutral_for_wc_fixture(home_wc_id, host_home=host_home)
    return match_probs(
        model,
        home_model_id,
        away_model_id,
        max_goals=max_goals,
        neutral=neutral,
    )


def model_version_label(model: DixonColesModel) -> str:
    """Stable tag for a fitted Dixon-Coles snapshot."""
    nll = model.fit_neg_log_likelihood
    nll_s = f"{nll:.4f}" if nll is not None else "na"
    return f"dc_xi{config.DC_XI:g}_nll{nll_s}"


def assert_pre_kickoff(predicted_utc: datetime, kickoff_utc: datetime) -> None:
    """Application guard: ``predicted_utc`` must be strictly before kickoff."""
    if predicted_utc.tzinfo is None:
        predicted_utc = predicted_utc.replace(tzinfo=timezone.utc)
    if kickoff_utc.tzinfo is None:
        kickoff_utc = kickoff_utc.replace(tzinfo=timezone.utc)
    if predicted_utc >= kickoff_utc:
        raise ValueError(
            f"predicted_utc ({predicted_utc.isoformat()}) must be before "
            f"kickoff_utc ({kickoff_utc.isoformat()})"
        )


def build_prediction_rows(
    fixture: dict[str, Any],
    model: DixonColesModel,
    wc_to_model_id: dict[int, int],
    *,
    model_version: str,
    predicted_utc: datetime,
    market_probs: tuple[float, float, float] | None,
) -> list[dict[str, Any]]:
    """Return up to three insert rows (model, market, blend) for one fixture."""
    kickoff = fixture["kickoff_utc"]
    assert_pre_kickoff(predicted_utc, kickoff)

    home_wc = int(fixture["home_id"])
    away_wc = int(fixture["away_id"])
    probs = fixture_model_probs(model, home_wc, away_wc, wc_to_model_id)
    base = {
        "fixture_id": int(fixture["fixture_id"]),
        "model_version": model_version,
        "predicted_utc": predicted_utc,
        "exp_home_goals": probs.exp_home_goals,
        "exp_away_goals": probs.exp_away_goals,
    }
    rows: list[dict[str, Any]] = [
        {
            **base,
            "source": "model",
            "p_home": probs.p_home,
            "p_draw": probs.p_draw,
            "p_away": probs.p_away,
        }
    ]
    if market_probs is None:
        log.info(
            "predict: no odds for fixture_id=%s (%s vs %s); skipping market/blend",
            fixture["fixture_id"],
            fixture["home_name"],
            fixture["away_name"],
        )
        return rows

    ph, pd, pa = market_probs
    rows.append(
        {**base, "source": "market", "p_home": ph, "p_draw": pd, "p_away": pa}
    )
    bh, bd, ba = blend((probs.p_home, probs.p_draw, probs.p_away), market_probs)
    rows.append(
        {**base, "source": "blend", "p_home": bh, "p_draw": bd, "p_away": ba}
    )
    return rows


def fetch_upcoming_fixtures(conn, *, now_utc: datetime | None = None) -> list[dict[str, Any]]:
    """Scheduled WC fixtures with known home/away and future kickoff."""
    now = now_utc or datetime.now(timezone.utc)
    result = conn.execute(text(UPCOMING_FIXTURES_SQL), {"now_utc": now})
    return [dict(r) for r in result.mappings().all()]


def latest_market_probs_by_fixture(
    conn, fixture_ids: list[int]
) -> dict[int, tuple[float, float, float]]:
    """De-vigged market triple per fixture from the latest odds snapshot."""
    if not fixture_ids:
        return {}
    result = conn.execute(
        text(LATEST_ODDS_SQL),
        {"fixture_ids": fixture_ids},
    )
    out: dict[int, tuple[float, float, float]] = {}
    for row in result.mappings().all():
        fid = int(row["fixture_id"])
        try:
            out[fid] = devig_three_way(
                float(row["home_odds"]),
                float(row["draw_odds"]),
                float(row["away_odds"]),
            )
        except (TypeError, ValueError) as exc:
            log.warning("predict: skip fixture_id=%s bad odds: %s", fid, exc)
    return out


def log_upcoming_predictions(
    model: DixonColesModel | None = None,
    *,
    db_url: str | None = None,
    predicted_utc: datetime | None = None,
) -> int:
    """Log predictions for all upcoming fixtures; return rows inserted."""
    dc = model if model is not None else fit_dixon_coles_from_db(db_url=db_url)
    version = model_version_label(dc)
    batch_utc = predicted_utc or datetime.now(timezone.utc)
    # Single shared resolver (same one the sim uses); fail loud rather than log a
    # coin flip if a WC team is missing from the fitted model.
    wc_to_model = wc_id_to_model_id()
    assert_wc_teams_in_model(wc_to_model, dc.attack.keys())
    written = 0
    with db.connect(db_url) as conn:
        fixtures = fetch_upcoming_fixtures(conn, now_utc=batch_utc)
        if not fixtures:
            log.info("predict: no upcoming fixtures")
            return 0
        fids = [int(f["fixture_id"]) for f in fixtures]
        market = latest_market_probs_by_fixture(conn, fids)
        for fix in fixtures:
            fid = int(fix["fixture_id"])
            home_wc = int(fix["home_id"])
            away_wc = int(fix["away_id"])
            if home_wc not in wc_to_model or away_wc not in wc_to_model:
                log.warning(
                    "predict: fixture_id=%s (%s vs %s) has WC team_id without a "
                    "model mapping; skipping",
                    fid,
                    fix["home_name"],
                    fix["away_name"],
                )
                continue
            rows = build_prediction_rows(
                fix,
                dc,
                wc_to_model,
                model_version=version,
                predicted_utc=batch_utc,
                market_probs=market.get(fid),
            )
            conn.execute(text(INSERT_PREDICTION_SQL), rows)
            written += len(rows)
    log.info("predict: fixtures=%d rows=%d", len(fixtures), written)
    return written


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    n = log_upcoming_predictions()
    print(f"predictions logged: {n} rows")


if __name__ == "__main__":
    main()
