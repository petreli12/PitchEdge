"""Tests for landing-page email capture."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from sqlalchemy import text

from pitchedge.dashboard.subscribers import (
    capture_subscriber_email,
    is_valid_email,
    normalize_email,
)

UTC = timezone.utc


def test_normalize_email():
    assert normalize_email("  Foo@Bar.COM ") == "foo@bar.com"


def test_is_valid_email():
    assert is_valid_email("a@b.co")
    assert not is_valid_email("not-an-email")


def test_capture_subscriber_idempotent(conn):
    ok1, _ = capture_subscriber_email("test@pitchedge.example", captured_utc=datetime(2026, 6, 1, tzinfo=UTC))
    ok2, msg2 = capture_subscriber_email("test@pitchedge.example", captured_utc=datetime(2026, 6, 2, tzinfo=UTC))
    assert ok1
    assert ok2
    assert "already" in msg2.lower()
    count = conn.execute(text("SELECT count(*) FROM subscribers WHERE email = :e"), {"e": "test@pitchedge.example"}).scalar_one()
    assert count == 1


def test_capture_rejects_invalid():
    ok, msg = capture_subscriber_email("bad")
    assert not ok
    assert "valid" in msg.lower()
