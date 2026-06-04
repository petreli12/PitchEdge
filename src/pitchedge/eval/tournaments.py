"""Held-out international tournaments for Phase 4 backtesting."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date


@dataclass(frozen=True)
class HeldOutTournament:
    """One tournament edition scored out-of-sample."""

    slug: str
    label: str
    competition: str  # ``raw_results.competition`` / Kaggle ``tournament``
    start: date
    end: date
    football_data_sheet: str | None = None  # sheet in football-data xlsx
    odds_api_sport_key: str | None = None  # the-odds-api historical sport key


HELD_OUT_TOURNAMENTS: tuple[HeldOutTournament, ...] = (
    HeldOutTournament(
        slug="wc_2018",
        label="FIFA World Cup 2018",
        competition="FIFA World Cup",
        start=date(2018, 6, 14),
        end=date(2018, 7, 15),
        football_data_sheet="WorldCup2018",
    ),
    HeldOutTournament(
        slug="wc_2022",
        label="FIFA World Cup 2022",
        competition="FIFA World Cup",
        start=date(2022, 11, 20),
        end=date(2022, 12, 18),
        football_data_sheet="WorldCup2022",
    ),
    HeldOutTournament(
        slug="euro_2024",
        label="UEFA Euro 2024",
        competition="UEFA Euro",
        start=date(2024, 6, 14),
        end=date(2024, 7, 14),
        odds_api_sport_key="soccer_uefa_european_championship",
    ),
    HeldOutTournament(
        slug="copa_2024",
        label="Copa América 2024",
        competition="Copa América",
        start=date(2024, 6, 20),
        end=date(2024, 7, 14),
        odds_api_sport_key="soccer_conmebol_copa_america",
    ),
)
