"""Group-stage simulation and FIFA 2026 tiebreakers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from pitchedge.sim.sampling import MatchupPair, sample_from_catalog, sample_one
from pitchedge.sim.scoreline import sample_scoreline

if TYPE_CHECKING:
    from pitchedge.model.dixon_coles import DixonColesModel


@dataclass
class GroupStanding:
    """One team's group-stage record (goals are integer counts)."""

    team_id: int
    group_label: str
    played: int = 0
    points: int = 0
    goals_for: int = 0
    goals_against: int = 0
    fair_play: float = 0.0

    @property
    def goal_difference(self) -> int:
        return self.goals_for - self.goals_against


def _apply_result(
    standings: dict[int, GroupStanding],
    home_id: int,
    away_id: int,
    hg: int,
    ag: int,
) -> None:
    home = standings[home_id]
    away = standings[away_id]
    home.played += 1
    away.played += 1
    home.goals_for += hg
    home.goals_against += ag
    away.goals_for += ag
    away.goals_against += hg
    if hg > ag:
        home.points += 3
    elif hg < ag:
        away.points += 3
    else:
        home.points += 1
        away.points += 1


def _h2h_stats(
    team_ids: list[int],
    results: list[tuple[int, int, int, int]],
) -> dict[int, GroupStanding]:
    """Mini-league stats among ``team_ids`` from played pairs."""
    stats = {
        tid: GroupStanding(team_id=tid, group_label="")
        for tid in team_ids
    }
    id_set = set(team_ids)
    for home_id, away_id, hg, ag in results:
        if home_id in id_set and away_id in id_set:
            _apply_result(stats, home_id, away_id, hg, ag)
    return stats


def _compare_standings(a: GroupStanding, b: GroupStanding) -> int:
    """FIFA group tiebreaker (no H2H): points, GD, GF, fair-play (higher better)."""
    for av, bv in (
        (a.points, b.points),
        (a.goal_difference, b.goal_difference),
        (a.goals_for, b.goals_for),
        (a.fair_play, b.fair_play),
    ):
        if av != bv:
            return -1 if av > bv else 1
    return 0


def _rank_subset(
    team_ids: list[int],
    standings: dict[int, GroupStanding],
    results: list[tuple[int, int, int, int]],
    rng: np.random.Generator,
) -> list[int]:
    """Rank ``team_ids`` using FIFA order including head-to-head among ties."""
    if len(team_ids) == 1:
        return team_ids

    base = [standings[tid] for tid in team_ids]
    if len(set((s.points, s.goal_difference, s.goals_for) for s in base)) == 1:
        h2h = _h2h_stats(team_ids, results)
        h2h_list = [h2h[tid] for tid in team_ids]
        if len(set((s.points, s.goal_difference, s.goals_for) for s in h2h_list)) > 1:
            return sorted(
                team_ids,
                key=lambda tid: (
                    -h2h[tid].points,
                    -h2h[tid].goal_difference,
                    -h2h[tid].goals_for,
                    -h2h[tid].fair_play,
                    rng.random(),
                ),
            )

    return sorted(
        team_ids,
        key=lambda tid: (
            -standings[tid].points,
            -standings[tid].goal_difference,
            -standings[tid].goals_for,
            -standings[tid].fair_play,
            rng.random(),
        ),
    )


def rank_group(
    standings: dict[int, GroupStanding],
    results: list[tuple[int, int, int, int]],
    rng: np.random.Generator,
) -> list[int]:
    """Return ``team_id`` list best -> worst (length 4).

    Orders teams by the FIFA overall criteria (points, goal difference, goals
    for) in descending order, then resolves any block of teams that is exactly
    tied on all three via head-to-head / fair-play / drawing of lots
    (``_rank_subset``). The ordering must depend only on results, never on the
    insertion order of ``standings``.
    """

    def overall_key(tid: int) -> tuple[int, int, int]:
        s = standings[tid]
        return (s.points, s.goal_difference, s.goals_for)

    sorted_ids = sorted(standings.keys(), key=overall_key, reverse=True)

    ordered: list[int] = []
    i = 0
    n = len(sorted_ids)
    while i < n:
        j = i + 1
        while j < n and overall_key(sorted_ids[j]) == overall_key(sorted_ids[i]):
            j += 1
        block = sorted_ids[i:j]
        if len(block) == 1:
            ordered.append(block[0])
        else:
            ordered.extend(_rank_subset(block, standings, results, rng))
        i = j
    return ordered


def rank_best_thirds(
    third_place: list[GroupStanding],
    rng: np.random.Generator,
) -> list[GroupStanding]:
    """Rank 12 third-placed teams; return the top eight (FIFA cross-group criteria)."""
    return sorted(
        third_place,
        key=lambda s: (
            -s.points,
            -s.goal_difference,
            -s.goals_for,
            -s.fair_play,
            rng.random(),
        ),
    )[:8]


@dataclass
class GroupStageOutcome:
    """Qualifiers after one simulated group stage."""

    ranked_by_group: dict[str, list[int]]
    best_eight_third_group_letters: list[str]
    third_by_group: dict[str, int]


def simulate_group_stage(
    model: DixonColesModel,
    fixtures: list,
    team_ids_by_group: dict[str, list[int]],
    wc_to_model_id: dict[int, int],
    rng: np.random.Generator,
    *,
    catalog: dict[MatchupPair, object] | None = None,
    group_uniforms_row: np.ndarray | None = None,
) -> GroupStageOutcome:
    """Play all group matches; return rankings and best-eight third groups.

    When ``group_uniforms_row`` is set (shape ``(n_fixtures,)``), decode those uniforms
    instead of drawing new ones — matches vectorized ``decode_group_scorelines``.
    """
    standings: dict[int, GroupStanding] = {}
    for group, ids in team_ids_by_group.items():
        for tid in ids:
            standings[tid] = GroupStanding(team_id=tid, group_label=group)

    results: list[tuple[int, int, int, int]] = []
    for col, fix in enumerate(fixtures):
        if group_uniforms_row is not None and catalog is not None:
            u = float(group_uniforms_row[col])
            hg, ag = sample_from_catalog(
                catalog,
                fix.home_id,
                fix.away_id,
                np.array([u], dtype=np.float64),
            )
            hg, ag = int(hg[0]), int(ag[0])
        elif catalog is not None:
            hg, ag = sample_one(catalog, fix.home_id, fix.away_id, rng)
        else:
            hg, ag = sample_scoreline(
                model, fix.home_id, fix.away_id, wc_to_model_id, rng
            )
        results.append((fix.home_id, fix.away_id, hg, ag))
        _apply_result(standings, fix.home_id, fix.away_id, hg, ag)

    ranked_by_group: dict[str, list[int]] = {}
    third_by_group: dict[str, int] = {}
    thirds: list[GroupStanding] = []
    for group, ids in team_ids_by_group.items():
        group_results = [
            r
            for r in results
            if r[0] in ids or r[1] in ids
        ]
        ranked = rank_group(
            {tid: standings[tid] for tid in ids},
            group_results,
            rng,
        )
        ranked_by_group[group] = ranked
        third_by_group[group] = ranked[2]
        thirds.append(standings[ranked[2]])

    best = rank_best_thirds(thirds, rng)
    return GroupStageOutcome(
        ranked_by_group=ranked_by_group,
        best_eight_third_group_letters=[s.group_label for s in best],
        third_by_group=third_by_group,
    )
