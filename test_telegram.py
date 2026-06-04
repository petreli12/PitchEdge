"""Tests for telegram.py — mocked session + narrative client, no network."""

from types import SimpleNamespace

import pytest

from pitchedge.content import telegram as tg
from pitchedge.content.daily_disagreement import Candidate


# --- fakes ----------------------------------------------------------------- #
class FakeSession:
    """Records the last post() call and returns a Telegram-shaped ok response."""
    def __init__(self, ok=True, description="bad"):
        self.ok, self.description, self.calls = ok, description, []

    def post(self, url, data=None, files=None, timeout=None):
        self.calls.append({"url": url, "data": data, "files": files})
        body = {"ok": True, "result": {"message_id": 42}} if self.ok \
            else {"ok": False, "description": self.description}
        return SimpleNamespace(json=lambda: body, text="err")


def _narrative_client(headline="Brazil favored, market less sure"):
    import json
    reply = json.dumps({
        "headline": headline,
        "preview": "Our model leans Brazil here.",
        "model_read": "Our model is higher on Brazil than the market.",
        "confidence_note": None,
    })
    block = SimpleNamespace(type="text", text=reply)
    msg = SimpleNamespace(content=[block])
    return SimpleNamespace(messages=SimpleNamespace(create=lambda **kw: msg))


def _cand(**over):
    base = dict(
        fixture_id=2, home="Brazil", away="Serbia", stage="Group F",
        kickoff_local="Jun 20", venue="LA",
        p_home=0.62, p_draw=0.22, p_away=0.16,
        exp_home_goals=1.9, exp_away_goals=0.8,
        market_p_home=0.45, market_p_draw=0.28, market_p_away=0.27,
    )
    base.update(over)
    return Candidate(**base)


# --- caption --------------------------------------------------------------- #
def test_caption_has_headline_and_footer():
    cap = tg.build_caption({
        "headline": "Spain favored", "preview": "Body here.",
        "model_read": None, "confidence_note": None,
    })
    assert "<b>Spain favored</b>" in cap
    assert "Logged before kickoff" in cap


def test_caption_escapes_html():
    cap = tg.build_caption({
        "headline": "A & B <test>", "preview": "x", "model_read": None,
        "confidence_note": None,
    })
    assert "&amp;" in cap and "&lt;test&gt;" in cap


def test_caption_truncates_to_limit():
    cap = tg.build_caption({
        "headline": "H", "preview": "word " * 400,
        "model_read": None, "confidence_note": None,
    })
    assert len(cap) <= tg.CAPTION_LIMIT


# --- posting --------------------------------------------------------------- #
def test_dry_run_sends_nothing(tmp_path, monkeypatch):
    monkeypatch.setattr(tg.config, "TELEGRAM_CHAT_ID", "1", raising=False)
    monkeypatch.setattr(tg.config, "TELEGRAM_BOT_TOKEN", "t", raising=False)
    img = tmp_path / "c.png"
    img.write_bytes(b"x")
    sess = FakeSession()
    out = tg.post_photo(str(img), "cap", session=sess, dry_run=True)
    assert out is None and sess.calls == []


def test_api_error_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(tg.config, "TELEGRAM_CHAT_ID", "1", raising=False)
    monkeypatch.setattr(tg.config, "TELEGRAM_BOT_TOKEN", "t", raising=False)
    img = tmp_path / "c.png"
    img.write_bytes(b"x")
    with pytest.raises(RuntimeError):
        tg.post_photo(str(img), "cap", session=FakeSession(ok=False))


# --- orchestration --------------------------------------------------------- #
def test_disagreement_skips_when_below_threshold():
    weak = _cand(market_p_home=0.60, market_p_draw=0.23, market_p_away=0.17)
    out = tg.post_daily_disagreement([weak], min_tvd=0.20,
                                     narrative_client=_narrative_client())
    assert out is None


def test_disagreement_full_flow_posts(tmp_path, monkeypatch):
    monkeypatch.setattr(tg.config, "TELEGRAM_CHAT_ID", "1", raising=False)
    monkeypatch.setattr(tg.config, "TELEGRAM_BOT_TOKEN", "t", raising=False)
    sess = FakeSession()
    out = tg.post_daily_disagreement(
        [_cand()], narrative_client=_narrative_client(),
        session=sess, tmp_dir=str(tmp_path),
    )
    assert out["message_id"] == 42
    assert len(sess.calls) == 1
    assert sess.calls[0]["files"] and "photo" in sess.calls[0]["files"]
    assert "Brazil" in sess.calls[0]["data"]["caption"]
