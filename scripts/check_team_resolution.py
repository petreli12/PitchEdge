#!/usr/bin/env python3
"""Verify every WC ``teams`` row resolves across history, fixtures, and odds API.

Three sources:
  1. **Historical results** (Kaggle ``results.csv``): labels must exist so
     ``team_name_to_id(teams.name)`` matches ingested ``raw_results`` rows, or a
     documented history alias is required (e.g. Czechia → Czech Republic).
  2. **Fixtures** (``fixtures.csv`` / DB): each team appears on a group-stage row.
  3. **Odds API** (the-odds-api.com): team strings must map via ``odds_name`` or
     ``name`` + accent folding to ``build_fixture_lookup_from_db()`` keys; live
     events should list every team at least once.

Exit code 1 if any team fails a required check or any group fixture lacks odds.

Usage:
    uv run python scripts/check_team_resolution.py
    uv run python scripts/check_team_resolution.py --no-live-odds
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pitchedge import config
from pitchedge.ingest.odds import (
    _normalize_team,
    build_fixture_lookup_from_db,
    fetch_odds,
)
from pitchedge.ingest.team_ids import history_label_for_model, team_name_to_id

# Extra Kaggle labels to probe when ``teams.name`` is not an exact CSV string.
# Values are alternate ``home_team`` / ``away_team`` spellings in results.csv.
HISTORY_LABEL_ALIASES: dict[str, list[str]] = {
    "Czechia": ["Czech Republic"],
    "Turkiye": ["Turkey"],
    "Curacao": ["Curaçao", "Netherlands Antilles"],
    "South Korea": ["Korea Republic", "Korea, Republic of", "Republic of Korea"],
    "Ivory Coast": ["Côte d'Ivoire", "Cote d'Ivoire"],
    "United States": ["USA"],
    "DR Congo": ["Congo DR", "Democratic Republic of Congo", "Congo (Kinshasa)"],
    "Bosnia and Herzegovina": ["Bosnia-Herzegovina", "Bosnia & Herzegovina"],
    "Cape Verde": ["Cabo Verde"],
}


@dataclass
class TeamReport:
    team_id: int
    name: str
    odds_name: str | None
    group_label: str
    history_exact: bool = False
    history_alias_hits: list[tuple[str, int]] = field(default_factory=list)
    history_model_rows: int = 0
    history_alias_model_rows: int = 0
    in_group_fixtures: bool = False
    group_fixture_count: int = 0
    odds_lookup_keys: set[tuple[str, str]] = field(default_factory=set)
    in_live_api: bool = False
    live_api_labels: set[str] = field(default_factory=set)
    fixtures_missing_odds: list[int] = field(default_factory=list)

    @property
    def history_catalog_ok(self) -> bool:
        return self.history_exact or bool(self.history_alias_hits)

    @property
    def history_model_ok(self) -> bool:
        """Training rows exist for ``team_name_to_id(teams.name)``."""
        return self.history_model_rows > 0

    @property
    def history_resolved(self) -> bool:
        """Alias exists in CSV and can supply rows if we map the history label."""
        if self.history_model_ok:
            return True
        return self.history_alias_model_rows > 0

    @property
    def fixtures_ok(self) -> bool:
        return self.in_group_fixtures

    @property
    def odds_lookup_ok(self) -> bool:
        return self.group_fixture_count == 0 or len(self.odds_lookup_keys) >= self.group_fixture_count

    @property
    def odds_api_ok(self) -> bool:
        return self.in_live_api

    def history_warnings(self) -> list[str]:
        """Non-fatal: Kaggle uses a different label than ``teams.name``."""
        if self.history_model_ok:
            return []
        if not self.history_catalog_ok:
            return [
                f"no exact or alias label in results.csv for {self.name!r}"
            ]
        if not self.history_resolved:
            return [
                f"zero rows in results.csv for any known alias of {self.name!r}"
            ]
        labels = ", ".join(f"{lbl} ({n})" for lbl, n in self.history_alias_hits)
        return [
            f"Kaggle label differs from teams.name: set teams.history_name "
            f"(currently scoring id uses {self.name!r}); CSV has {labels}"
        ]

    def hard_failures(self, *, live_odds_checked: bool) -> list[str]:
        """Fixtures or odds broken (not history label spelling alone)."""
        out: list[str] = []
        if not self.history_catalog_ok:
            out.append("history: no exact or alias label in results.csv")
        if not self.history_resolved:
            out.append(
                "history: zero rows in results.csv for model or known aliases"
            )
        if not self.fixtures_ok:
            out.append("fixtures: not on any group-stage fixture")
        if not self.odds_lookup_ok:
            out.append(
                "odds DB: fixture pair not in build_fixture_lookup_from_db() "
                f"({len(self.odds_lookup_keys)}/{self.group_fixture_count} fixtures)"
            )
        if live_odds_checked and not self.odds_api_ok:
            out.append(
                "odds API: team string not seen in live events "
                f"(labels tried: {self._lookup_labels()})"
            )
        if self.fixtures_missing_odds:
            out.append(
                "odds DB: missing snapshots for fixture_id(s) "
                + ", ".join(str(x) for x in self.fixtures_missing_odds)
            )
        return out

    def _lookup_labels(self) -> list[str]:
        labels = [self.name]
        if self.odds_name and self.odds_name != self.name:
            labels.append(self.odds_name)
        return labels


def _load_history_names() -> set[str]:
    path = Path(config.HISTORY_CSV_PATH)
    frame = pd.read_csv(path, usecols=["home_team", "away_team"])
    return set(frame["home_team"].astype(str)) | set(frame["away_team"].astype(str))


def _history_row_counts(name: str, frame: pd.DataFrame) -> int:
    return int(((frame["home_team"] == name) | (frame["away_team"] == name)).sum())


def _model_row_count(team_name: str, frame: pd.DataFrame) -> int:
    tid = team_name_to_id(team_name)
    home_ids = frame["home_team"].map(team_name_to_id)
    away_ids = frame["away_team"].map(team_name_to_id)
    return int(((home_ids == tid) | (away_ids == tid)).sum())


def _load_teams() -> pd.DataFrame:
    return pd.read_csv(config.TEAMS_CSV_PATH)


def _load_group_fixtures() -> pd.DataFrame:
    df = pd.read_csv(config.FIXTURES_CSV_PATH)
    return df[df["stage"].astype(str).str.startswith("Group")].copy()


def _fixture_ids_missing_odds(db_url: str | None) -> set[int]:
    from sqlalchemy import text

    from pitchedge import db

    sql = """
    SELECT f.fixture_id
    FROM fixtures f
    WHERE f.status = 'scheduled'
      AND f.home_id IS NOT NULL
      AND f.away_id IS NOT NULL
      AND NOT EXISTS (
          SELECT 1 FROM odds_snapshots os WHERE os.fixture_id = f.fixture_id
      )
    """
    with db.connect(db_url) as conn:
        return {int(r[0]) for r in conn.execute(text(sql)).all()}


def _team_odds_labels(row: pd.Series) -> list[str]:
    name = str(row["name"]).strip()
    labels = [name]
    odds_raw = row.get("odds_name")
    if odds_raw is not None and str(odds_raw).strip() and not pd.isna(odds_raw):
        labels.append(str(odds_raw).strip())
    return labels


def _fixture_pair_in_lookup(
    home_id: int,
    away_id: int,
    teams_df: pd.DataFrame,
    lookup_keys: set[tuple[str, str]],
) -> bool:
    home_row = teams_df.loc[teams_df["team_id"] == home_id].iloc[0]
    away_row = teams_df.loc[teams_df["team_id"] == away_id].iloc[0]
    for h in _team_odds_labels(home_row):
        for a in _team_odds_labels(away_row):
            if (_normalize_team(h), _normalize_team(a)) in lookup_keys:
                return True
    return False


def build_reports(
    *,
    db_url: str | None,
    live_odds: bool,
) -> tuple[list[TeamReport], dict[str, Any]]:
    teams_df = _load_teams()
    hist_df = pd.read_csv(
        config.HISTORY_CSV_PATH,
        usecols=["home_team", "away_team"],
    )
    hist_names = _load_history_names()
    group_fix = _load_group_fixtures()
    missing_odds = _fixture_ids_missing_odds(db_url)

    fixture_lookup = build_fixture_lookup_from_db()
    lookup_keys = set(fixture_lookup.keys())

    api_normalized: set[str] = set()
    api_raw_by_norm: dict[str, set[str]] = {}
    live_checked = False
    if live_odds and config.ODDS_API_KEY:
        try:
            events = fetch_odds()
            live_checked = True
            for event in events:
                for label in (str(event["home_team"]), str(event["away_team"])):
                    norm = _normalize_team(label)
                    api_normalized.add(norm)
                    api_raw_by_norm.setdefault(norm, set()).add(label)
        except Exception as exc:
            print(f"WARNING: live odds fetch failed: {exc}", file=sys.stderr)

    reports: list[TeamReport] = []
    for row in teams_df.itertuples(index=False):
        name = str(row.name).strip()
        odds_raw = getattr(row, "odds_name", None)
        odds_name = (
            str(odds_raw).strip()
            if odds_raw is not None and str(odds_raw).strip() and not pd.isna(odds_raw)
            else None
        )
        hist_raw = getattr(row, "history_name", None)
        history_name = (
            str(hist_raw).strip()
            if hist_raw is not None and str(hist_raw).strip() and not pd.isna(hist_raw)
            else None
        )
        model_label = history_label_for_model(name, history_name)
        rep = TeamReport(
            team_id=int(row.team_id),
            name=name,
            odds_name=odds_name,
            group_label=str(row.group_label).strip(),
        )

        rep.history_exact = model_label in hist_names
        rep.history_model_rows = _model_row_count(model_label, hist_df)

        for alias in HISTORY_LABEL_ALIASES.get(name, []):
            n = _history_row_counts(alias, hist_df)
            if n > 0:
                rep.history_alias_hits.append((alias, n))
                rep.history_alias_model_rows += _model_row_count(alias, hist_df)

        gf = group_fix[
            (group_fix["home_id"] == rep.team_id)
            | (group_fix["away_id"] == rep.team_id)
        ]
        rep.in_group_fixtures = len(gf) > 0
        rep.group_fixture_count = len(gf)
        for _, fix in gf.iterrows():
            fid = int(fix["fixture_id"])
            h_id = int(fix["home_id"])
            a_id = int(fix["away_id"])
            if fid in missing_odds:
                rep.fixtures_missing_odds.append(fid)
            if _fixture_pair_in_lookup(h_id, a_id, teams_df, lookup_keys):
                rep.odds_lookup_keys.add(
                    (_normalize_team(str(teams_df.loc[teams_df["team_id"] == h_id, "name"].iloc[0])),
                     _normalize_team(str(teams_df.loc[teams_df["team_id"] == a_id, "name"].iloc[0]))),
                )

        labels = _team_odds_labels(teams_df.loc[teams_df["team_id"] == rep.team_id].iloc[0])
        if live_checked:
            for label in labels:
                norm = _normalize_team(label)
                if norm in api_normalized:
                    rep.in_live_api = True
                    rep.live_api_labels.update(api_raw_by_norm.get(norm, {label}))

        reports.append(rep)

    meta = {
        "live_odds_checked": live_checked,
        "api_team_count": len(api_normalized),
        "fixture_lookup_pairs": len(lookup_keys),
        "missing_odds_fixtures": len(missing_odds),
    }
    return reports, meta


def print_report(reports: list[TeamReport], meta: dict) -> int:
    """Print human-readable report; return exit code."""
    live = meta["live_odds_checked"]
    hard_failed: list[TeamReport] = []
    hist_warn: list[TeamReport] = []

    print("PitchEdge team resolution check (48 WC teams)")
    print(f"  history CSV: {config.HISTORY_CSV_PATH}")
    print(f"  fixtures CSV: {config.FIXTURES_CSV_PATH}")
    print(f"  live odds API: {'yes' if live else 'skipped'}")
    if live:
        print(f"  API distinct teams (normalized): {meta['api_team_count']}")
    print(f"  fixture lookup pairs in DB: {meta['fixture_lookup_pairs']}")
    print(f"  group fixtures missing odds_snapshots: {meta['missing_odds_fixtures']}")
    print()

    for rep in sorted(reports, key=lambda r: r.team_id):
        if rep.hard_failures(live_odds_checked=live):
            hard_failed.append(rep)
        for msg in rep.history_warnings():
            if rep not in hist_warn:
                hist_warn.append(rep)

    if hist_warn:
        print("=== History label mismatch (fixtures + odds OK) ===")
        for rep in hist_warn:
            for msg in rep.history_warnings():
                print(f"  [{rep.group_label}] {rep.name} (id={rep.team_id}): {msg}")
        print(
            "  Set teams.history_name in teams.csv (see HISTORY_NAME_OVERRIDES in "
            "scripts/wc2026_teams_data.py).\n"
        )

    if hard_failed:
        print("=== FAILURES (fixtures / odds / missing catalog) ===")
        for rep in hard_failed:
            print(f"  [{rep.group_label}] {rep.name} (id={rep.team_id})")
            for msg in rep.hard_failures(live_odds_checked=live):
                print(f"    - {msg}")
        print()

    ok = [
        r
        for r in reports
        if r not in hard_failed and not r.history_warnings()
    ]
    print("=== Summary ===")
    print(f"  Fully aligned (history id + fixtures + odds): {len(ok)}")
    print(f"  History label mismatch only: {len(hist_warn)}")
    print(f"  Hard failures: {len(hard_failed)}")

    if live and meta["api_team_count"] == 48:
        print("  South Korea / Korea Republic: API uses 'South Korea' — keep odds_name empty.")
    if not hard_failed and not hist_warn:
        print("\nAll 48 teams resolve across history, fixtures, and odds API.")
        return 0
    if not hard_failed and hist_warn:
        print(
            "\nNo fixture or odds API gaps; address history_name warnings above."
        )
        return 0 if not meta["missing_odds_fixtures"] else 1
    return 1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--no-live-odds",
        action="store_true",
        help="Skip the-odds-api.com live fetch (DB lookup + history only)",
    )
    args = parser.parse_args()

    reports, meta = build_reports(
        db_url=config.DB_URL,
        live_odds=not args.no_live_odds,
    )
    raise SystemExit(print_report(reports, meta))


if __name__ == "__main__":
    main()
