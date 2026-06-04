"""Tests for 3-way de-vig."""

from __future__ import annotations

import math

import pytest

from pitchedge.model.devig import (
    assert_normalized,
    devig_proportional,
    devig_shin,
    devig_three_way,
    implied_raw,
)


@pytest.mark.parametrize(
    "home,draw,away",
    [
        (2.10, 3.40, 3.25),
        (1.44, 4.33, 7.75),
        (1.95, 3.50, 4.20),
    ],
)
def test_proportional_sums_to_one(home, draw, away):
    probs = devig_proportional(home, draw, away)
    assert_normalized(probs)
    assert sum(probs) == pytest.approx(1.0, abs=1e-9)


def test_devigged_below_raw_overround():
    home, draw, away = 2.10, 3.40, 3.25
    raw = sum(implied_raw(home, draw, away))
    devigged = sum(devig_proportional(home, draw, away))
    assert raw > 1.0
    assert devigged == pytest.approx(1.0, abs=1e-9)
    assert devigged < raw


def test_shin_sums_to_one():
    probs = devig_shin(2.10, 3.40, 3.25)
    assert_normalized(probs)


def test_devig_three_way_default_matches_proportional():
    odds = (1.95, 3.50, 4.20)
    assert devig_three_way(*odds) == devig_proportional(*odds)
    assert devig_three_way(*odds, method="shin") == devig_shin(*odds)


def test_invalid_odds_rejected():
    with pytest.raises(ValueError):
        devig_three_way(1.0, 3.0, 4.0)
