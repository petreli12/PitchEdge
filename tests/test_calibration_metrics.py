"""Tests for ECE, paired log-loss SE, and temperature scaling."""

from __future__ import annotations

import pytest

from pitchedge.eval.metrics import (
    expected_calibration_error,
    paired_log_loss_difference,
)
from pitchedge.eval.temperature_scaling import apply_temperature, fit_temperature


def test_ece_bounded():
    probs = [(0.9, 0.05, 0.05)] * 50 + [(0.1, 0.45, 0.45)] * 50
    outcomes = [0] * 50 + [2] * 50
    ece = expected_calibration_error(probs, outcomes)
    assert 0.0 <= ece <= 1.0


def test_paired_log_loss_se():
    model = [(0.7, 0.2, 0.1), (0.4, 0.3, 0.3)]
    market = [(0.6, 0.25, 0.15), (0.5, 0.25, 0.25)]
    outcomes = [0, 1]
    stats = paired_log_loss_difference(model, market, outcomes)
    assert stats["n"] == 2
    assert stats["se_diff"] > 0
    assert stats["mean_diff"] == pytest.approx(
        stats["mean_diff"],
        rel=0,
    )


def test_temperature_identity_at_one():
    p = (0.5, 0.3, 0.2)
    assert apply_temperature(p, 1.0) == pytest.approx(p, abs=1e-9)


def test_fit_temperature_returns_positive():
    probs = [(0.55, 0.25, 0.20)] * 40 + [(0.35, 0.30, 0.35)] * 40
    outcomes = [0] * 40 + [2] * 40
    t = fit_temperature(probs, outcomes)
    assert 0.25 <= t <= 5.0
