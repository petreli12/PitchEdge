"""Adapter for the-odds-api.com (v4) ``/scores`` -> ``raw_results`` + fixture status.

During the tournament finished games arrive after kickoff. This module pulls
recently completed matches from the-odds-api ``/scores`` endpoint, upserts each
final score into ``raw_results``, and marks the matching WC fixture
``status='final'`` so ``score.py`` can write ``prediction_scores`` (the public
calibration receipts). It mirrors ``ingest/odds.py``: HTTP is isolated in
``fetch_scores()`` so tests inject JSON fixtures with no live call.

Probabilities in ``match_predictions`` are never mutated; only ground-truth rows
and fixture status are written, via ``ingest.results.record_result``.

See https://the-odds-api.com/liveapi/guides/v4/#get-scores
  * ``daysFrom`` (1-3) returns completed games from that many days in the past.
  * Each event carries ``completed`` and a per-team ``scores`` list of strings.
  * Team strings must align with ``teams.name`` / ``teams.odds_name`` (matching is
    accent- and case-insensitive and order-independent, so a home/away swap on the
    API side is handled).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any, Mapping, Protocol

import requests

from pitchedge import config, db
from pitchedge.ingest.odds import _normalize_team, _parse_utc_timestamp
from pitchedge.ingest.results import record_result

log = logging.getLogger(__name__)


class ScoresHttpClient(Protocol):
    """Minimal HTTP client surface for ``fetch_scores`` (mockable in tests)."""

    def get(self, url: str, *, params: Mapping[str, Any], timeout: float) -> Any:
        ...


@dataclass(frozen=True)
class CompletedMatch:
    """One finished match parsed from the scores API (goals oriented to the API)."""

    home_team: str
    away_team: str
    commence_utc: datetime
    home_goals: int
    away_goals: int


@dataclass(frozen=True)
class FixtureEntry:
    """A WC fixture's matching keys (normalized names) and write-back labels."""

    fixture_id: int
    match_date: date
    home_history_name: str
    away_history_name: str
    home_names: frozenset[str]
    away_names: frozenset[str]


def fetch_scores(
    *,
    sport_key: str | None = None,
    api_key: str | None = None,
    days_from: int | None = None,
    date_format: str = "iso",
    base_url: str | None = None,
    client: ScoresHttpClient | None = None,
    timeout: float = 30.0,
) -> list[dict[str, Any]]:
    """GET /v4/sports/{sport}/scores and return the parsed JSON list of events.

    ``days_from`` defaults to ``config.ODDS_API_SCORES_DAYS_FROM``; values <= 0 omit
    the parameter (the API then returns only live/upcoming, no completed games).
    Raises on HTTP errors.
    """
    key = api_key or config.ODDS_API_KEY
    if not key:
        raise ValueError("ODDS_API_KEY is not set")

    sport = sport_key or config.ODDS_API_SPORT_KEY
    url = f"{(base_url or config.ODDS_API_BASE_URL).rstrip('/')}/v4/sports/{sport}/scores"
    days = config.ODDS_API_SCORES_DAYS_FROM if days_from is None else days_from
    params: dict[str, Any] = {"apiKey": key, "dateFormat": date_format}
    if days and days > 0:
        params["daysFrom"] = days

    http = client or requests
    response = http.get(url, params=params, timeout=timeout)
    if hasattr(response, "raise_for_status"):
        response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, list):
        raise ValueError(f"expected list of events from scores API, got {type(payload)}")
    return payload


def _parse_score_value(value: Any) -> int | None:
    """Coerce an API score (a string like ``"2"``) to int; ``None`` if not numeric."""
    if value is None:
        return None
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return None


def parse_completed_scores(events: list[dict[str, Any]]) -> list[CompletedMatch]:
    """Extract finished matches with both team goals from raw scores-API events.

    Skips events that are not ``completed``, lack a ``scores`` array, or whose
    home/away score cannot be resolved to an integer (logged at INFO).
    """
    out: list[CompletedMatch] = []
    for event in events:
        if not event.get("completed"):
            continue
        scores = event.get("scores")
        if not scores:
            continue
        home_team = event.get("home_team")
        away_team = event.get("away_team")
        if not home_team or not away_team:
            continue

        by_name: dict[str, int | None] = {}
        for entry in scores:
            name = entry.get("name")
            if name is None:
                continue
            by_name[_normalize_team(str(name))] = _parse_score_value(entry.get("score"))

        home_goals = by_name.get(_normalize_team(str(home_team)))
        away_goals = by_name.get(_normalize_team(str(away_team)))
        if home_goals is None or away_goals is None:
            log.info(
                "scores: incomplete score for %s vs %s; skipping",
                home_team,
                away_team,
            )
            continue

        out.append(
            CompletedMatch(
                home_team=str(home_team),
                away_team=str(away_team),
                commence_utc=_parse_utc_timestamp(str(event["commence_time"])),
                home_goals=home_goals,
                away_goals=away_goals,
            )
        )
    return out


_FIXTURE_INDEX_SQL = """
SELECT
    f.fixture_id,
    (f.kickoff_utc AT TIME ZONE 'UTC')::date AS match_date,
    th.name AS home_name,
    ta.name AS away_name,
    th.odds_name AS home_odds_name,
    ta.odds_name AS away_odds_name,
    COALESCE(NULLIF(TRIM(th.history_name), ''), th.name) AS home_history_name,
    COALESCE(NULLIF(TRIM(ta.history_name), ''), ta.name) AS away_history_name
FROM fixtures f
JOIN teams th ON th.team_id = f.home_id
JOIN teams ta ON ta.team_id = f.away_id
WHERE f.home_id IS NOT NULL
  AND f.away_id IS NOT NULL
"""


def _name_keys(*names: Any) -> frozenset[str]:
    """Normalized match keys for a team (display name plus optional odds alias)."""
    keys = set()
    for name in names:
        if name is None:
            continue
        text = str(name).strip()
        if text:
            keys.add(_normalize_team(text))
    return frozenset(keys)


def build_fixture_index_from_db(*, db_url: str | None = None) -> list[FixtureEntry]:
    """Build the in-memory fixture match index from ``fixtures`` joined to ``teams``."""
    rows = db.fetch_all(_FIXTURE_INDEX_SQL, db_url=db_url)
    index: list[FixtureEntry] = []
    for row in rows:
        match_date = row["match_date"]
        if isinstance(match_date, datetime):
            match_date = match_date.date()
        index.append(
            FixtureEntry(
                fixture_id=int(row["fixture_id"]),
                match_date=match_date,
                home_history_name=str(row["home_history_name"]),
                away_history_name=str(row["away_history_name"]),
                home_names=_name_keys(row["home_name"], row["home_odds_name"]),
                away_names=_name_keys(row["away_name"], row["away_odds_name"]),
            )
        )
    return index


def match_fixture(
    index: list[FixtureEntry],
    home_norm: str,
    away_norm: str,
    match_date: date,
) -> tuple[FixtureEntry, bool] | None:
    """Find the fixture for a (home, away) pair regardless of API home/away order.

    Returns ``(entry, swapped)`` where ``swapped`` is True when the API's home team
    is the fixture's away team. When several fixtures share the same pairing (e.g. a
    group match and a later knockout), the one with the closest kickoff date wins.
    """
    candidates: list[tuple[FixtureEntry, bool]] = []
    for entry in index:
        if home_norm in entry.home_names and away_norm in entry.away_names:
            candidates.append((entry, False))
        elif home_norm in entry.away_names and away_norm in entry.home_names:
            candidates.append((entry, True))
    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]
    candidates.sort(key=lambda c: abs((c[0].match_date - match_date).days))
    return candidates[0]


def sync_from_scores_api(
    events: list[dict[str, Any]] | None = None,
    *,
    fixture_index: list[FixtureEntry] | None = None,
    db_url: str | None = None,
    fetch_live: bool = False,
    days_from: int | None = None,
) -> tuple[int, int]:
    """Sync finished games into ``raw_results`` and mark fixtures final.

    Pass ``events`` (or set ``fetch_live=True`` to call the API). Goals are oriented
    to each matched fixture's home/away before writing, so a swapped API ordering is
    corrected. Idempotent: re-running upserts the same scores and re-marks nothing.
    Returns ``(matched, marked_final)``.
    """
    if events is None:
        if not fetch_live:
            raise ValueError("provide events=... or fetch_live=True")
        events = fetch_scores(days_from=days_from)

    completed = parse_completed_scores(events)
    if fixture_index is None:
        fixture_index = build_fixture_index_from_db(db_url=db_url)

    matched = 0
    marked_final = 0
    for game in completed:
        found = match_fixture(
            fixture_index,
            _normalize_team(game.home_team),
            _normalize_team(game.away_team),
            game.commence_utc.date(),
        )
        if found is None:
            log.info(
                "scores: no fixture for %s vs %s on %s; skipping",
                game.home_team,
                game.away_team,
                game.commence_utc.date(),
            )
            continue
        entry, swapped = found
        matched += 1
        home_goals = game.away_goals if swapped else game.home_goals
        away_goals = game.home_goals if swapped else game.away_goals
        detail = record_result(
            home_name=entry.home_history_name,
            away_name=entry.away_history_name,
            match_date=entry.match_date,
            home_goals=home_goals,
            away_goals=away_goals,
            fixture_id=entry.fixture_id,
            db_url=db_url,
        )
        marked_final += int(detail.get("fixture_marked_final") or 0)

    log.info(
        "scores sync: completed=%d matched=%d marked_final=%d",
        len(completed),
        matched,
        marked_final,
    )
    return matched, marked_final


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    matched, marked = sync_from_scores_api(fetch_live=True)
    print(f"scores: matched={matched} marked_final={marked}")


if __name__ == "__main__":
    main()
