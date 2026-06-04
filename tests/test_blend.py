"""Tests for model/market blend."""

from __future__ import annotations

import pytest

from pitchedge.model.blend import blend


def test_blend_w0_equals_market():
    model = (0.5, 0.25, 0.25)
    market = (0.4, 0.3, 0.3)
    assert blend(model, market, w=0.0) == pytest.approx(market, abs=1e-9)


def test_blend_w1_equals_model():
    model = (0.5, 0.25, 0.25)
    market = (0.4, 0.3, 0.3)
    assert blend(model, market, w=1.0) == pytest.approx(model, abs=1e-9)


def test_blend_renormalizes():
    model = (0.6, 0.2, 0.2)
    market = (0.3, 0.3, 0.4)
    probs = blend(model, market, w=0.5)
    assert sum(probs) == pytest.approx(1.0, abs=1e-9)
