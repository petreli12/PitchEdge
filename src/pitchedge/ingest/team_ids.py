"""Stable integer team identifiers for historical nations.

``raw_results.home_id`` / ``away_id`` are not foreign keys to ``teams`` (the WC
48-team table). Historical ingests map each distinct team *name* to a stable
positive integer so the natural key ``(date, home_id, away_id, competition)``
is reproducible across runs.
"""

from __future__ import annotations

import hashlib


def team_name_to_id(name: str) -> int:
    """Map a team name string to a stable positive 31-bit integer.

    The same normalized name always yields the same id. Names are stripped and
    lowercased before hashing.
    """
    normalized = name.strip().lower()
    digest = hashlib.sha256(normalized.encode("utf-8")).digest()[:4]
    return int.from_bytes(digest, byteorder="big") & 0x7FFFFFFF


def history_label_for_model(name: str, history_name: str | None = None) -> str:
    """Kaggle ``results.csv`` label for ``team_name_to_id`` (WC display ``name`` otherwise)."""
    if history_name is not None:
        text = str(history_name).strip()
        if text:
            return text
    return name.strip()
