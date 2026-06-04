"""World Cup 2026 squad layout: groups, fixtures, model id mapping."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from pitchedge import config
from pitchedge.ingest.team_ids import history_label_for_model, team_name_to_id

GROUP_LABELS: tuple[str, ...] = tuple(chr(ord("A") + i) for i in range(12))


@dataclass(frozen=True)
class WcTeam:
    """One row from ``teams.csv`` (``team_id`` is the WC table id, 1–48)."""

    team_id: int
    name: str
    group_label: str
    model_team_id: int
    history_name: str | None = None


@dataclass(frozen=True)
class GroupFixture:
    """One group-stage match from ``fixtures.csv``."""

    fixture_id: int
    home_id: int
    away_id: int
    group_label: str


def load_wc_teams(path: str | Path | None = None) -> list[WcTeam]:
    """Load WC teams; ``model_team_id`` uses ``history_name`` when set (Kaggle label)."""
    p = Path(path or config.TEAMS_CSV_PATH)
    df = pd.read_csv(p)
    teams: list[WcTeam] = []
    for row in df.itertuples(index=False):
        name = str(row.name).strip()
        hist_raw = getattr(row, "history_name", None)
        history_name: str | None = None
        if hist_raw is not None and not (isinstance(hist_raw, float) and pd.isna(hist_raw)):
            text = str(hist_raw).strip()
            history_name = text or None
        label = history_label_for_model(name, history_name)
        teams.append(
            WcTeam(
                team_id=int(row.team_id),
                name=name,
                group_label=str(row.group_label).strip(),
                model_team_id=team_name_to_id(label),
                history_name=history_name,
            )
        )
    return teams


def teams_by_group(teams: list[WcTeam]) -> dict[str, list[WcTeam]]:
    """``group_label`` → four ``WcTeam`` rows."""
    out: dict[str, list[WcTeam]] = {g: [] for g in GROUP_LABELS}
    for t in teams:
        out[t.group_label].append(t)
    for g in GROUP_LABELS:
        if len(out[g]) != 4:
            raise ValueError(f"group {g} must have 4 teams, got {len(out[g])}")
    return out


def load_group_fixtures(path: str | Path | None = None) -> list[GroupFixture]:
    """Load the 72 group-stage fixtures (fixture_id 1–72)."""
    p = Path(path or config.FIXTURES_CSV_PATH)
    df = pd.read_csv(p)
    fixtures: list[GroupFixture] = []
    for row in df.itertuples(index=False):
        stage = str(row.stage).strip()
        if not stage.startswith("Group"):
            continue
        fixtures.append(
            GroupFixture(
                fixture_id=int(row.fixture_id),
                home_id=int(row.home_id),
                away_id=int(row.away_id),
                group_label=str(row.group_label).strip(),
            )
        )
    if len(fixtures) != 72:
        raise ValueError(f"expected 72 group fixtures, found {len(fixtures)}")
    return fixtures


def wc_id_to_model_id(teams: list[WcTeam] | None = None) -> dict[int, int]:
    """WC ``team_id`` → Dixon-Coles historical hash id.

    This is the **single** WC→model id resolver. Both ``predict.py`` and the
    tournament sim must obtain their mapping here so the two code paths can never
    diverge (the disagreement-board bug came from ``predict.py`` skipping this
    translation and feeding raw WC ids to a model keyed by historical hash ids,
    which silently collapsed every match to a coin flip). Pass an already-loaded
    ``teams`` list to avoid re-reading the CSV; otherwise it loads them.
    """
    resolved = teams if teams is not None else load_wc_teams()
    return {t.team_id: t.model_team_id for t in resolved}


def assert_wc_teams_in_model(
    wc_to_model: dict[int, int],
    model_team_ids: Iterable[int],
) -> None:
    """Raise if any resolved model id is absent from the fitted model.

    A WC team whose model id is not a key in the fitted Dixon-Coles parameters
    would silently fall through to the ``attack=0/defense=0`` default (equal
    lambdas, the ~34/32/34 coin flip). This guard turns that into a hard error.
    """
    known = set(model_team_ids)
    missing = {wc: mid for wc, mid in wc_to_model.items() if mid not in known}
    if missing:
        raise KeyError(
            "WC teams resolve to model ids absent from the fitted model "
            f"(would collapse to a coin flip): {missing}"
        )


def wc_id_to_group(teams: list[WcTeam]) -> dict[int, str]:
    return {t.team_id: t.group_label for t in teams}
