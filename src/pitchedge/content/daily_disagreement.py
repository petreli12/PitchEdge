"""
Pick the fixture where our standalone model most disagrees with the de-vigged
market, and shape it into a ready-to-render post.

Never compare blend vs market: with BLEND_W=0 the blend equals the market, so
disagreement would be zero and receipts would score the market as "us".
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Optional, Sequence

from sqlalchemy import text

from pitchedge import config
from pitchedge.content.narrative import NarrativeInput
from pitchedge.eval.temperature_scaling import apply_temperature

log = logging.getLogger(__name__)

DEFAULT_MIN_TVD = 0.06


@dataclass
class Candidate:
    """One upcoming fixture with standalone model and de-vigged market probs."""

    fixture_id: int
    home: str
    away: str
    stage: str
    kickoff_local: str
    venue: Optional[str]
    p_home: float
    p_draw: float
    p_away: float
    exp_home_goals: float
    exp_away_goals: float
    market_p_home: float
    market_p_draw: float
    market_p_away: float
    salience: float = 1.0


@dataclass
class Disagreement:
    candidate: Candidate
    tvd: float
    score: float
    outcome: str
    delta_pts: float
    note: str


_OUTCOMES = ("home", "draw", "away")


def published_model_probs(
    p_home: float,
    p_draw: float,
    p_away: float,
    *,
    temperature: float | None = None,
) -> tuple[float, float, float]:
    """Model probabilities shown in content (temperature-scaled when configured)."""
    t = temperature if temperature is not None else config.MODEL_TEMPERATURE
    return apply_temperature((p_home, p_draw, p_away), t)


def _tvd(p: tuple[float, float, float], q: tuple[float, float, float]) -> float:
    return 0.5 * sum(abs(pi - qi) for pi, qi in zip(p, q))


def _largest_gap(c: Candidate) -> tuple[str, float]:
    model = {"home": c.p_home, "draw": c.p_draw, "away": c.p_away}
    market = {"home": c.market_p_home, "draw": c.market_p_draw, "away": c.market_p_away}
    deltas = {o: model[o] - market[o] for o in _OUTCOMES}
    outcome = max(deltas, key=lambda o: abs(deltas[o]))
    return outcome, deltas[outcome] * 100.0


def _team_for_outcome(c: Candidate, outcome: str) -> str:
    return {"home": c.home, "away": c.away, "draw": "the draw"}[outcome]


def build_note(c: Candidate, outcome: str, delta_pts: float) -> str:
    subject = _team_for_outcome(c, outcome)
    direction = "higher" if delta_pts > 0 else "lower"
    model_p = {"home": c.p_home, "draw": c.p_draw, "away": c.p_away}[outcome]
    mkt_p = {
        "home": c.market_p_home,
        "draw": c.market_p_draw,
        "away": c.market_p_away,
    }[outcome]
    return (
        f"Our model is {abs(delta_pts):.0f}pts {direction} on {subject} than the "
        f"market (model {model_p:.0%} vs market {mkt_p:.0%})."
    )


def score_candidate(c: Candidate) -> Disagreement:
    p = (c.p_home, c.p_draw, c.p_away)
    q = (c.market_p_home, c.market_p_draw, c.market_p_away)
    tvd = _tvd(p, q)
    outcome, delta_pts = _largest_gap(c)
    return Disagreement(
        candidate=c,
        tvd=tvd,
        score=tvd * c.salience,
        outcome=outcome,
        delta_pts=delta_pts,
        note=build_note(c, outcome, delta_pts),
    )


def rank_disagreements(candidates: Sequence[Candidate]) -> list[Disagreement]:
    return sorted(
        (score_candidate(c) for c in candidates),
        key=lambda d: d.score,
        reverse=True,
    )


def select_top(
    candidates: Sequence[Candidate],
    min_tvd: Optional[float] = None,
) -> Optional[Disagreement]:
    min_tvd = DEFAULT_MIN_TVD if min_tvd is None else min_tvd
    ranked = rank_disagreements(candidates)
    if not ranked or ranked[0].tvd < min_tvd:
        log.info("no fixture clears min_tvd=%.3f; skipping disagreement post", min_tvd)
        return None
    return ranked[0]


def to_narrative_input(d: Disagreement) -> NarrativeInput:
    c = d.candidate
    return NarrativeInput(
        home=c.home,
        away=c.away,
        stage=c.stage,
        kickoff_local=c.kickoff_local,
        venue=c.venue,
        p_home=c.p_home,
        p_draw=c.p_draw,
        p_away=c.p_away,
        exp_home_goals=c.exp_home_goals,
        exp_away_goals=c.exp_away_goals,
        market_p_home=c.market_p_home,
        market_p_draw=c.market_p_draw,
        market_p_away=c.market_p_away,
        disagreement_note=d.note,
    )


_FETCH_SQL = """
SELECT
    f.fixture_id,
    th.name AS home,
    ta.name AS away,
    f.stage,
    f.kickoff_utc,
    mp.p_home AS raw_p_home,
    mp.p_draw AS raw_p_draw,
    mp.p_away AS raw_p_away,
    mp.exp_home_goals,
    mp.exp_away_goals,
    mk.p_home AS market_p_home,
    mk.p_draw AS market_p_draw,
    mk.p_away AS market_p_away
FROM fixtures f
JOIN teams th ON th.team_id = f.home_id
JOIN teams ta ON ta.team_id = f.away_id
JOIN LATERAL (
    SELECT p_home, p_draw, p_away, exp_home_goals, exp_away_goals
    FROM match_predictions
    WHERE fixture_id = f.fixture_id AND source = 'model'
    ORDER BY predicted_utc DESC
    LIMIT 1
) mp ON TRUE
JOIN LATERAL (
    SELECT p_home, p_draw, p_away
    FROM match_predictions
    WHERE fixture_id = f.fixture_id AND source = 'market'
    ORDER BY predicted_utc DESC
    LIMIT 1
) mk ON TRUE
WHERE f.kickoff_utc > now()
  AND f.kickoff_utc <= now() + make_interval(hours => :hours)
"""


def _row_to_candidate(row: dict[str, Any], kickoff_local: str) -> Candidate:
    ph, pd, pa = published_model_probs(
        float(row["raw_p_home"]),
        float(row["raw_p_draw"]),
        float(row["raw_p_away"]),
    )
    return Candidate(
        fixture_id=int(row["fixture_id"]),
        home=str(row["home"]),
        away=str(row["away"]),
        stage=str(row["stage"]),
        kickoff_local=kickoff_local,
        venue=row.get("venue"),
        p_home=ph,
        p_draw=pd,
        p_away=pa,
        exp_home_goals=float(row["exp_home_goals"] or 0.0),
        exp_away_goals=float(row["exp_away_goals"] or 0.0),
        market_p_home=float(row["market_p_home"]),
        market_p_draw=float(row["market_p_draw"]),
        market_p_away=float(row["market_p_away"]),
    )


def load_top_disagreement(
    *,
    within_hours: int = 36,
    min_tvd: float | None = None,
    db_url: str | None = None,
) -> Disagreement | None:
    """Load upcoming fixtures from DB and return the top model-vs-market disagreement."""
    from pitchedge import db

    with db.connect(db_url) as conn:
        candidates = fetch_candidates(conn, within_hours=within_hours)
    return select_top(candidates, min_tvd=min_tvd)


def fetch_candidates(conn, within_hours: int = 36) -> list[Candidate]:
    """Upcoming fixtures with latest ``source='model'`` vs ``source='market'``.

    Model probabilities are temperature-scaled for display when
    ``MODEL_TEMPERATURE`` != 1. Blend rows are never used for disagreement.
    """
    result = conn.execute(
        text(_FETCH_SQL),
        {"hours": within_hours},
    )
    rows = [dict(r) for r in result.mappings().all()]
    candidates: list[Candidate] = []
    for row in rows:
        kickoff = row["kickoff_utc"]
        kickoff_local = (
            kickoff.strftime("%b %d, %H:%M UTC")
            if hasattr(kickoff, "strftime")
            else str(kickoff)
        )
        candidates.append(_row_to_candidate(row, kickoff_local))
    return candidates
