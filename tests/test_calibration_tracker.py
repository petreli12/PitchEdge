"""Calibration tracker aggregates model vs market only."""

from pitchedge.content.calibration_tracker import (
    model_vs_market_gap,
    summarize_scored_rows,
)
from pitchedge.content.daily_disagreement import published_model_probs


def test_summarize_excludes_blend():
    rows = [
        {
            "source": "model",
            "p_home": 0.5,
            "p_draw": 0.3,
            "p_away": 0.2,
            "brier": 0.5,
            "log_loss": 0.9,
            "outcome": "H",
        },
        {
            "source": "market",
            "p_home": 0.55,
            "p_draw": 0.25,
            "p_away": 0.2,
            "brier": 0.45,
            "log_loss": 0.85,
            "outcome": "H",
        },
        {
            "source": "blend",
            "p_home": 0.55,
            "p_draw": 0.25,
            "p_away": 0.2,
            "brier": 0.45,
            "log_loss": 0.85,
            "outcome": "H",
        },
    ]
    summary = summarize_scored_rows(rows)
    assert set(summary.keys()) == {"model", "market"}
    gap = model_vs_market_gap(summary)
    assert gap["log_loss_gap"] is not None


def test_tracker_model_ece_uses_published_model_probs(monkeypatch):
    """Model ECE uses ``published_model_probs`` (``config.MODEL_TEMPERATURE``)."""
    monkeypatch.setattr("pitchedge.config.MODEL_TEMPERATURE", 1.5)
    rows = [
        {
            "source": "model",
            "p_home": 0.6,
            "p_draw": 0.25,
            "p_away": 0.15,
            "brier": 0.5,
            "log_loss": 0.9,
            "outcome": "H",
        },
    ]
    scaled = published_model_probs(0.6, 0.25, 0.15)
    assert scaled != (0.6, 0.25, 0.15)
    summary = summarize_scored_rows(rows)
    assert summary["model"].n == 1
