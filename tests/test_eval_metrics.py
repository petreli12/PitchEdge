"""Tests for backtest scoring metrics."""

from __future__ import annotations

import pytest

from pitchedge.eval.metrics import (
    mean_brier,
    mean_log_loss,
    multiclass_brier,
    multiclass_log_loss,
    outcome_from_goals,
)


def test_outcome_encoding():
    assert outcome_from_goals(2, 1) == 0
    assert outcome_from_goals(1, 1) == 1
    assert outcome_from_goals(0, 1) == 2


def test_perfect_prediction_zero_brier():
    p = (1.0, 0.0, 0.0)
    assert multiclass_brier(p, 0) == pytest.approx(0.0, abs=1e-9)


def test_log_loss_penalizes_wrong_class():
    p = (0.7, 0.2, 0.1)
    assert multiclass_log_loss(p, 0) < multiclass_log_loss(p, 2)


def test_mean_metrics():
    probs = [(0.6, 0.2, 0.2), (0.3, 0.3, 0.4)]
    outcomes = [0, 2]
    assert mean_brier(probs, outcomes) > 0
    assert mean_log_loss(probs, outcomes) > 0
