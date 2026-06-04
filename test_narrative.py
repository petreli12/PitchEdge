"""Tests for narrative generation — no live API calls."""

import json
from types import SimpleNamespace

import pytest

from pitchedge.content.narrative import (
    NarrativeInput,
    generate_narrative,
    build_user_message,
    _parse_strict_json,
)


def _mock_client(reply_text: str):
    """Return an object that quacks like anthropic.Anthropic for our usage."""
    block = SimpleNamespace(type="text", text=reply_text)
    message = SimpleNamespace(content=[block])
    messages = SimpleNamespace(create=lambda **kw: message)
    return SimpleNamespace(messages=messages)


def _valid_input(**over):
    base = dict(
        home="Spain", away="Morocco", stage="Round of 16",
        kickoff_local="July 5, 3:00 PM ET", venue="Atlanta",
        p_home=0.54, p_draw=0.27, p_away=0.19,
        exp_home_goals=1.7, exp_away_goals=0.9,
    )
    base.update(over)
    return NarrativeInput(**base)


def test_happy_path_parses_clean_json():
    reply = json.dumps({
        "headline": "Spain favored, Morocco live as a spoiler",
        "preview": "Our model leans Spain in this Round of 16 tie...",
        "model_read": "Our model is a touch lower on Spain than the market.",
        "confidence_note": None,
    })
    out = generate_narrative(_valid_input(), client=_mock_client(reply))
    assert out["headline"].startswith("Spain")
    assert out["model_read"] is not None
    assert out["confidence_note"] is None


def test_tolerates_code_fences():
    raw = "```json\n" + json.dumps({
        "headline": "h", "preview": "p", "model_read": None, "confidence_note": None
    }) + "\n```"
    assert _parse_strict_json(raw)["headline"] == "h"


def test_missing_key_triggers_fallback():
    # model returns JSON missing required keys -> generate_narrative falls back
    bad = json.dumps({"headline": "only this"})
    out = generate_narrative(_valid_input(), client=_mock_client(bad))
    assert out.keys() >= {"headline", "preview", "model_read", "confidence_note"}


def test_garbage_output_triggers_fallback():
    out = generate_narrative(_valid_input(), client=_mock_client("not json at all"))
    assert "Spain" in out["headline"] or "Morocco" in out["headline"]


def test_rejects_unnormalized_probabilities():
    with pytest.raises(ValueError):
        generate_narrative(
            _valid_input(p_home=0.9, p_draw=0.9, p_away=0.9),
            client=_mock_client("{}"),
        )


def test_payload_excludes_none_fields():
    msg = build_user_message(_valid_input(venue=None))
    assert "venue" not in msg
    assert "Spain" in msg


def test_fallback_flags_coin_flip():
    out = generate_narrative(
        _valid_input(p_home=0.36, p_draw=0.30, p_away=0.34),
        client=_mock_client("garbage"),
    )
    assert out["confidence_note"] is not None
