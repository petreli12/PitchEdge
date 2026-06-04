"""Landing-page email capture into ``subscribers``."""

from __future__ import annotations

import re
from datetime import datetime, timezone

from sqlalchemy import text

from pitchedge import db

_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")

INSERT_SUBSCRIBER_SQL = """
INSERT INTO subscribers (email, captured_utc)
VALUES (:email, :captured_utc)
ON CONFLICT (email) DO NOTHING
"""


def normalize_email(raw: str) -> str:
    """Strip and lower-case an email address for storage."""
    return raw.strip().lower()


def is_valid_email(email: str) -> bool:
    """Lightweight format check (not deliverability)."""
    return bool(_EMAIL_RE.match(email))


def capture_subscriber_email(
    raw_email: str,
    *,
    db_url: str | None = None,
    captured_utc: datetime | None = None,
) -> tuple[bool, str]:
    """Insert email if new. Returns ``(inserted, message)``."""
    email = normalize_email(raw_email)
    if not email:
        return False, "Enter an email address."
    if not is_valid_email(email):
        return False, "Please enter a valid email address (e.g. you@example.com)."

    captured = captured_utc or datetime.now(timezone.utc)
    with db.connect(db_url) as conn:
        result = conn.execute(
            text(INSERT_SUBSCRIBER_SQL),
            {"email": email, "captured_utc": captured},
        )
        inserted = result.rowcount > 0
    if inserted:
        return True, "Thanks — you are on the list."
    return True, "You are already subscribed."


def post_subscriber_email(
    raw_email: str,
    *,
    post_url: str,
    field: str = "email",
    timeout: float = 10.0,
) -> tuple[bool, str]:
    """Submit an email to an external capture endpoint over HTTP.

    Used on the public (DB-free) deploy so the landing keeps the in-app form and
    its success state without a writable database. ``post_url`` is a Formspree /
    Tally / Google Form action; ``field`` is the form field name the endpoint
    expects. Returns ``(ok, message)``; never raises on network errors.
    """
    email = normalize_email(raw_email)
    if not email:
        return False, "Enter an email address."
    if not is_valid_email(email):
        return False, "Please enter a valid email address (e.g. you@example.com)."
    if not post_url:
        return False, "Signup is not configured yet."

    import requests

    try:
        resp = requests.post(
            post_url,
            data={field: email},
            headers={"Accept": "application/json"},
            timeout=timeout,
        )
    except requests.RequestException:
        return False, "Could not reach the signup service. Please try again later."

    if resp.status_code in (200, 201, 202, 204):
        return True, "Thanks — you're on the list."
    return False, "Something went wrong signing you up. Please try again later."
