"""Tests for football-data odds loader."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from pitchedge.ingest.backtest_odds import (
    load_football_data_world_cups,
    lookup_market_probs,
    match_key,
    normalize_team_name,
)

XLSX = Path("data/backtest/WorldCup2022.xlsx")


@pytest.mark.skipif(not XLSX.is_file(), reason="download WorldCup2022.xlsx first")
def test_load_world_cup_odds():
    df = load_football_data_world_cups(XLSX)
    assert len(df) == 128
    assert (df["p_home"] + df["p_draw"] + df["p_away"]).apply(
        lambda s: abs(s - 1.0) < 1e-6
    ).all()


@pytest.mark.skipif(not XLSX.is_file(), reason="download WorldCup2022.xlsx first")
def test_lookup_wc_2018_opener():
    df = load_football_data_world_cups(XLSX)
    probs = lookup_market_probs(
        df,
        home_team="Russia",
        away_team="Saudi Arabia",
        match_date=date(2018, 6, 14),
    )
    assert probs is not None
    assert sum(probs) == pytest.approx(1.0, abs=1e-6)


def test_normalize_team_aliases():
    assert normalize_team_name("USA") == normalize_team_name("United States")
