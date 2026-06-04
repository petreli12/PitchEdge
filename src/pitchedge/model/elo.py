"""World-Football-style Elo for international matches.

Ingests ``raw_results`` chronologically, applies goal-difference weighting and a
home-advantage term (skipped for neutral venues), and writes reproducible
snapshots to ``team_ratings``.

Probabilities from Elo are 2-way win expectations (no draw); use Dixon-Coles /
market blend for 3-way match probs.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from datetime import date
from typing import Any

from pitchedge import config, db

log = logging.getLogger(__name__)

INSERT_RATING_SQL = """
INSERT INTO team_ratings (team_id, as_of_date, elo, attack_strength, defense_strength)
VALUES (:team_id, :as_of_date, :elo, NULL, NULL)
ON CONFLICT (team_id, as_of_date) DO UPDATE SET elo = EXCLUDED.elo
"""


@dataclass(frozen=True)
class EloConfig:
    """Elo hyperparameters (units: rating points)."""

    initial_rating: float = 1500.0
    k_base: float = 20.0
    home_advantage: float = 100.0
    rating_divisor: float = 400.0


def elo_win_prob(
    elo_home: float,
    elo_away: float,
    *,
    home_advantage: float = 0.0,
    rating_divisor: float = 400.0,
) -> float:
    """P(home win) from Elo difference (no draw mass).

    ``home_advantage`` is added to the home side's effective rating (0 for neutral).
    Returns a probability in [0, 1].
    """
    exponent = (elo_away - (elo_home + home_advantage)) / rating_divisor
    return 1.0 / (1.0 + math.pow(10.0, exponent))


def _goal_diff_multiplier(goal_diff: int) -> float:
    """World-Football-style margin-of-victory multiplier (goal_diff >= 0)."""
    gd = abs(goal_diff)
    if gd <= 1:
        return 1.0
    if gd == 2:
        return 1.5
    return (11.0 + gd) / 8.0


def _home_result_points(home_goals: int, away_goals: int) -> tuple[float, float]:
    """Actual scores for home/away: 1 win, 0.5 draw, 0 loss."""
    if home_goals > away_goals:
        return 1.0, 0.0
    if home_goals < away_goals:
        return 0.0, 1.0
    return 0.5, 0.5


def load_results_chronological(*, db_url: str | None = None) -> list[dict[str, Any]]:
    """Load dated results with scores for Elo fitting."""
    sql = """
        SELECT date, home_id, away_id, home_goals, away_goals, neutral
        FROM raw_results
        WHERE home_goals IS NOT NULL AND away_goals IS NOT NULL
        ORDER BY date ASC, match_id ASC
    """
    return db.fetch_all(sql, db_url=db_url)


def fit_elo(
    matches: list[dict[str, Any]],
    *,
    elo_config: EloConfig | None = None,
) -> tuple[dict[int, float], date | None]:
    """Run Elo updates over ``matches`` chronologically.

    Returns ``(ratings_by_team_id, last_match_date)``.
    """
    cfg = elo_config or EloConfig(
        initial_rating=config.ELO_INITIAL,
        k_base=config.ELO_K,
        home_advantage=config.ELO_HOME_ADV,
        rating_divisor=config.ELO_RATING_DIVISOR,
    )
    ratings: dict[int, float] = {}
    last_date: date | None = None

    for row in matches:
        home_id = int(row["home_id"])
        away_id = int(row["away_id"])
        home_goals = int(row["home_goals"])
        away_goals = int(row["away_goals"])
        neutral = bool(row["neutral"])
        last_date = row["date"]

        ra = ratings.setdefault(home_id, cfg.initial_rating)
        rb = ratings.setdefault(away_id, cfg.initial_rating)

        ha = 0.0 if neutral else cfg.home_advantage
        ea = elo_win_prob(ra, rb, home_advantage=ha, rating_divisor=cfg.rating_divisor)
        eb = 1.0 - ea

        sa, sb = _home_result_points(home_goals, away_goals)
        mov = _goal_diff_multiplier(home_goals - away_goals)
        k = cfg.k_base * mov

        ratings[home_id] = ra + k * (sa - ea)
        ratings[away_id] = rb + k * (sb - eb)

    return ratings, last_date


def persist_ratings(
    ratings: dict[int, float],
    as_of_date: date,
    *,
    db_url: str | None = None,
) -> int:
    """Write one Elo snapshot row per team for ``as_of_date``. Returns row count."""
    rows = [
        {"team_id": tid, "as_of_date": as_of_date, "elo": float(elo)}
        for tid, elo in ratings.items()
    ]
    if not rows:
        return 0
    db.execute(INSERT_RATING_SQL, rows, db_url=db_url)
    return len(rows)


def fit_elo_from_db(
    *,
    db_url: str | None = None,
    as_of_date: date | None = None,
) -> tuple[int, date | None]:
    """Fit Elo on all scored ``raw_results`` and persist to ``team_ratings``.

    Returns ``(teams_written, as_of_date_used)``.
    """
    matches = load_results_chronological(db_url=db_url)
    if not matches:
        log.warning("no scored raw_results to fit Elo")
        return 0, None

    ratings, last_date = fit_elo(matches)
    snapshot_date = as_of_date or last_date
    if snapshot_date is None:
        return 0, None

    n = persist_ratings(ratings, snapshot_date, db_url=db_url)
    log.info(
        "Elo fit: matches=%d teams=%d as_of_date=%s",
        len(matches),
        n,
        snapshot_date,
    )
    return n, snapshot_date


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    n, as_of = fit_elo_from_db()
    print(f"Elo snapshots written: {n} teams, as_of_date={as_of}")


if __name__ == "__main__":
    main()
