"""Vectorized group-stage simulation across Monte Carlo replicates."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pitchedge.sim.group_stage import (
    GroupStanding,
    rank_best_thirds,
    rank_group,
)
from pitchedge.sim.sampling import MatchupPair, build_matchup_catalog, sample_from_catalog
from pitchedge.sim.wc_teams import GroupFixture

if True:  # typing
    from pitchedge.model.dixon_coles import DixonColesModel


@dataclass(frozen=True)
class GroupBatchLayout:
    """Index layout for vectorized group-stage accumulation."""

    n_teams: int
    team_id_to_idx: dict[int, int]
    home_idx: np.ndarray
    away_idx: np.ndarray
    group_labels: tuple[str, ...]
    group_team_idx: dict[str, np.ndarray]
    group_match_cols: dict[str, np.ndarray]


def build_group_batch_layout(
    team_ids_by_group: dict[str, list[int]],
    fixtures: list[GroupFixture],
) -> GroupBatchLayout:
    """Map WC ``team_id``s to 0..47 and record which fixture columns belong to each group."""
    all_ids = [tid for ids in team_ids_by_group.values() for tid in ids]
    team_id_to_idx = {tid: tid - 1 for tid in all_ids}
    home_idx = np.array([team_id_to_idx[f.home_id] for f in fixtures], dtype=np.int16)
    away_idx = np.array([team_id_to_idx[f.away_id] for f in fixtures], dtype=np.int16)

    group_labels = tuple(sorted(team_ids_by_group.keys()))
    group_team_idx = {
        g: np.array([team_id_to_idx[t] for t in team_ids_by_group[g]], dtype=np.int16)
        for g in group_labels
    }
    group_match_cols: dict[str, list[int]] = {g: [] for g in group_labels}
    for col, fix in enumerate(fixtures):
        group_match_cols[fix.group_label].append(col)
    group_match_cols_arr = {
        g: np.array(cols, dtype=np.int16) for g, cols in group_match_cols.items()
    }
    return GroupBatchLayout(
        n_teams=len(all_ids),
        team_id_to_idx=team_id_to_idx,
        home_idx=home_idx,
        away_idx=away_idx,
        group_labels=group_labels,
        group_team_idx=group_team_idx,
        group_match_cols=group_match_cols_arr,
    )


def decode_group_scorelines(
    catalog: dict[MatchupPair, object],
    fixtures: list[GroupFixture],
    u: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Decode pre-drawn uniforms ``u`` (shape ``(n_sims, n_fixtures)``) to goal grids."""
    u = np.asarray(u, dtype=np.float64)
    if u.ndim != 2 or u.shape[1] != len(fixtures):
        raise ValueError("u must have shape (n_sims, n_fixtures)")
    n_sims, n_fix = u.shape
    home_goals = np.zeros((n_sims, n_fix), dtype=np.uint8)
    away_goals = np.zeros((n_sims, n_fix), dtype=np.uint8)
    for col, fix in enumerate(fixtures):
        hg, ag = sample_from_catalog(
            catalog, fix.home_id, fix.away_id, u[:, col]
        )
        home_goals[:, col] = hg
        away_goals[:, col] = ag
    return home_goals, away_goals


def sample_all_group_scorelines(
    catalog: dict[MatchupPair, object],
    fixtures: list[GroupFixture],
    n_sims: int,
    rng: np.random.Generator,
    *,
    group_uniforms: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample every group fixture for all ``n_sims`` replicates.

    Returns ``(home_goals, away_goals)`` with shape ``(n_sims, n_fixtures)``.
    Draw order is C-contiguous ``(sim, fixture)`` — same sequence as nested Python loops.

    Pass ``group_uniforms`` to fix the underlying draws (for aligned fast/legacy tests).
    """
    n_fix = len(fixtures)
    if group_uniforms is None:
        u = rng.random((n_sims, n_fix))
    else:
        u = np.asarray(group_uniforms, dtype=np.float64)
        if u.shape != (n_sims, n_fix):
            raise ValueError(f"group_uniforms must be ({n_sims}, {n_fix})")
    return decode_group_scorelines(catalog, fixtures, u)


def accumulate_standings(
    layout: GroupBatchLayout,
    home_goals: np.ndarray,
    away_goals: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Update points / goals-for / goals-against for all sims.

    Returns arrays with shape ``(n_sims, n_teams)``.
    """
    n_sims = home_goals.shape[0]
    n_teams = layout.n_teams
    points = np.zeros((n_sims, n_teams), dtype=np.int16)
    gf = np.zeros((n_sims, n_teams), dtype=np.int16)
    ga = np.zeros((n_sims, n_teams), dtype=np.int16)

    for col in range(home_goals.shape[1]):
        h = int(layout.home_idx[col])
        a = int(layout.away_idx[col])
        hg = home_goals[:, col].astype(np.int16)
        ag = away_goals[:, col].astype(np.int16)
        win_h = hg > ag
        win_a = ag > hg
        draw = hg == ag
        points[:, h] += np.where(win_h, 3, np.where(draw, 1, 0))
        points[:, a] += np.where(win_a, 3, np.where(draw, 1, 0))
        gf[:, h] += hg
        ga[:, h] += ag
        gf[:, a] += ag
        ga[:, a] += hg

    return points, gf, ga


@dataclass
class SingleSimGroupOutcome:
    """Group-stage qualifiers for one replicate (knockout input)."""

    ranked_by_group: dict[str, list[int]]
    best_eight_third_group_letters: list[str]
    third_by_group: dict[str, int]


def resolve_one_group_outcome(
    sim_index: int,
    layout: GroupBatchLayout,
    team_ids_by_group: dict[str, list[int]],
    idx_to_team_id: dict[int, int],
    points: np.ndarray,
    gf: np.ndarray,
    ga: np.ndarray,
    home_goals: np.ndarray,
    away_goals: np.ndarray,
    rng: np.random.Generator,
) -> SingleSimGroupOutcome:
    """Rank one replicate (same tiebreak RNG order as ``simulate_group_stage``)."""
    ranked_by_group: dict[str, list[int]] = {}
    third_by_group: dict[str, int] = {}
    thirds: list[GroupStanding] = []

    for g in layout.group_labels:
        tids = team_ids_by_group[g]
        standings = {
            tid: GroupStanding(
                team_id=tid,
                group_label=g,
                played=3,
                points=int(points[sim_index, layout.team_id_to_idx[tid]]),
                goals_for=int(gf[sim_index, layout.team_id_to_idx[tid]]),
                goals_against=int(ga[sim_index, layout.team_id_to_idx[tid]]),
            )
            for tid in tids
        }
        cols = layout.group_match_cols[g]
        results: list[tuple[int, int, int, int]] = []
        for col in cols:
            col_i = int(col)
            results.append(
                (
                    idx_to_team_id[int(layout.home_idx[col_i])],
                    idx_to_team_id[int(layout.away_idx[col_i])],
                    int(home_goals[sim_index, col_i]),
                    int(away_goals[sim_index, col_i]),
                )
            )
        ranked = rank_group(standings, results, rng)
        ranked_by_group[g] = ranked
        third_tid = ranked[2]
        third_by_group[g] = third_tid
        thirds.append(standings[third_tid])

    best = rank_best_thirds(thirds, rng)
    return SingleSimGroupOutcome(
        ranked_by_group=ranked_by_group,
        best_eight_third_group_letters=[t.group_label for t in best],
        third_by_group=third_by_group,
    )


def resolve_group_outcomes(
    layout: GroupBatchLayout,
    team_ids_by_group: dict[str, list[int]],
    idx_to_team_id: dict[int, int],
    points: np.ndarray,
    gf: np.ndarray,
    ga: np.ndarray,
    home_goals: np.ndarray,
    away_goals: np.ndarray,
    rng: np.random.Generator,
) -> list[SingleSimGroupOutcome]:
    """Rank groups per sim (tiebreakers unchanged; uses ``rank_group`` / ``rank_best_thirds``)."""
    n_sims = points.shape[0]
    return [
        resolve_one_group_outcome(
            s,
            layout,
            team_ids_by_group,
            idx_to_team_id,
            points,
            gf,
            ga,
            home_goals,
            away_goals,
            rng,
        )
        for s in range(n_sims)
    ]


def idx_to_team_id_map(layout: GroupBatchLayout) -> dict[int, int]:
    return {idx: tid for tid, idx in layout.team_id_to_idx.items()}


def prepare_group_batch(
    model: DixonColesModel,
    fixtures: list[GroupFixture],
    team_ids_by_group: dict[str, list[int]],
    wc_team_ids: list[int],
    wc_to_model_id: dict[int, int],
) -> tuple[dict[MatchupPair, object], GroupBatchLayout]:
    """Build CDF catalog and group layout once per Monte Carlo batch."""
    catalog = build_matchup_catalog(model, wc_team_ids, wc_to_model_id)
    layout = build_group_batch_layout(team_ids_by_group, fixtures)
    return catalog, layout
