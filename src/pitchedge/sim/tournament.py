"""Monte Carlo World Cup 2026 tournament simulation.

Scorelines are sampled from the standalone Dixon-Coles score matrix at
``MODEL_TEMPERATURE=1.0`` (matrix sampling; equivalent to T=1 on H/D/A). Venue
policy: ``neutral=True`` except co-host home fixtures (USA/Mexico/Canada as
``home_id``). Not blend, not market.

See docs/ARCHITECTURE.md §3.5 and ``data/wc2026/annex_c.json`` (FIFA Annex C).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np

from pitchedge import config, db
from pitchedge.model.dixon_coles import DixonColesModel, fit_dixon_coles_from_db
from pitchedge.sim.bracket import build_qualifiers, simulate_knockout
from pitchedge.sim.group_batch import (
    accumulate_standings,
    idx_to_team_id_map,
    prepare_group_batch,
    resolve_one_group_outcome,
    sample_all_group_scorelines,
)
from pitchedge.sim.group_stage import simulate_group_stage
from pitchedge.sim.scoreline import sample_scoreline as sample_scoreline  # re-export
from pitchedge.sim.wc_teams import (
    load_group_fixtures,
    load_wc_teams,
    teams_by_group,
    wc_id_to_model_id,
)

log = logging.getLogger(__name__)

INSERT_SIM_SQL = """
INSERT INTO sim_results (
    run_batch_utc, team_id,
    p_advance_group, p_r16, p_qf, p_sf, p_final, p_win,
    n_sims
) VALUES (
    :run_batch_utc, :team_id,
    :p_advance_group, :p_r16, :p_qf, :p_sf, :p_final, :p_win,
    :n_sims
)
ON CONFLICT (run_batch_utc, team_id) DO UPDATE SET
    p_advance_group = EXCLUDED.p_advance_group,
    p_r16 = EXCLUDED.p_r16,
    p_qf = EXCLUDED.p_qf,
    p_sf = EXCLUDED.p_sf,
    p_final = EXCLUDED.p_final,
    p_win = EXCLUDED.p_win,
    n_sims = EXCLUDED.n_sims
"""


@dataclass
class TeamSimCounters:
    """Per-team outcome counts across Monte Carlo replicates."""

    top_two: int = 0
    r16: int = 0
    qf: int = 0
    sf: int = 0
    final: int = 0
    win: int = 0


@dataclass
class SimAggregate:
    """Per-team probabilities from one batch."""

    run_batch_utc: datetime
    n_sims: int
    by_team: dict[int, dict[str, float]] = field(default_factory=dict)


def _r16_reached(match_winners: dict[str, int], team_id: int) -> bool:
    for n in range(73, 89):
        if match_winners.get(f"W{n}") == team_id:
            return True
    return False


def _qf_reached(match_winners: dict[str, int], team_id: int) -> bool:
    for n in range(89, 97):
        if match_winners.get(f"W{n}") == team_id:
            return True
    return False


def _sf_reached(match_winners: dict[str, int], team_id: int) -> bool:
    for n in (97, 98, 99, 100):
        if match_winners.get(f"W{n}") == team_id:
            return True
    return False


def _final_reached(match_winners: dict[str, int], team_id: int) -> bool:
    for n in (101, 102):
        if match_winners.get(f"W{n}") == team_id:
            return True
    return False


def _apply_knockout_counters(
    counters: dict[int, TeamSimCounters],
    champion: int,
    winners: dict[str, int],
) -> None:
    for tid in counters:
        if _r16_reached(winners, tid):
            counters[tid].r16 += 1
        if _qf_reached(winners, tid):
            counters[tid].qf += 1
        if _sf_reached(winners, tid):
            counters[tid].sf += 1
        if _final_reached(winners, tid):
            counters[tid].final += 1
        if tid == champion:
            counters[tid].win += 1


def run_one_tournament(
    model: DixonColesModel,
    fixtures: list,
    team_ids_by_group: dict[str, list[int]],
    wc_to_model_id: dict[int, int],
    rng: np.random.Generator,
    *,
    catalog: dict | None = None,
    group_uniforms_row: np.ndarray | None = None,
) -> dict[int, TeamSimCounters]:
    """Simulate one full tournament (legacy per-match loop)."""
    gs = simulate_group_stage(
        model,
        fixtures,
        team_ids_by_group,
        wc_to_model_id,
        rng,
        catalog=catalog,
        group_uniforms_row=group_uniforms_row,
    )
    counters: dict[int, TeamSimCounters] = {
        tid: TeamSimCounters()
        for ids in team_ids_by_group.values()
        for tid in ids
    }
    for group, ranked in gs.ranked_by_group.items():
        for tid in ranked[:2]:
            counters[tid].top_two += 1

    q = build_qualifiers(
        gs.ranked_by_group,
        gs.third_by_group,
        gs.best_eight_third_group_letters,
    )
    champion, winners = simulate_knockout(
        q, model, wc_to_model_id, rng, catalog=catalog
    )
    _apply_knockout_counters(counters, champion, winners)
    return counters


def run_monte_carlo_legacy(
    model: DixonColesModel,
    *,
    n_sims: int,
    seed: int | None = None,
    fixtures: list,
    team_ids_by_group: dict[str, list[int]],
    wc_to_model: dict[int, int],
    wc_team_ids: list[int],
    rng: np.random.Generator | None = None,
    group_uniforms: np.ndarray | None = None,
    catalog: dict | None = None,
) -> dict[int, TeamSimCounters]:
    """Original Python loop (for regression / distribution checks)."""
    from pitchedge.sim.sampling import build_matchup_catalog

    if rng is None:
        rng = np.random.default_rng(seed)
    if catalog is None:
        catalog = build_matchup_catalog(model, wc_team_ids, wc_to_model)
    totals: dict[int, TeamSimCounters] = {
        tid: TeamSimCounters()
        for ids in team_ids_by_group.values()
        for tid in ids
    }
    for s in range(n_sims):
        row = group_uniforms[s] if group_uniforms is not None else None
        one = run_one_tournament(
            model,
            fixtures,
            team_ids_by_group,
            wc_to_model,
            rng,
            catalog=catalog,
            group_uniforms_row=row,
        )
        for tid, c in one.items():
            t = totals[tid]
            t.top_two += c.top_two
            t.r16 += c.r16
            t.qf += c.qf
            t.sf += c.sf
            t.final += c.final
            t.win += c.win
    return totals


def run_monte_carlo_fast(
    model: DixonColesModel,
    *,
    n_sims: int,
    seed: int | None = None,
    fixtures: list,
    team_ids_by_group: dict[str, list[int]],
    wc_to_model: dict[int, int],
    wc_team_ids: list[int],
    rng: np.random.Generator | None = None,
    group_uniforms: np.ndarray | None = None,
    catalog: dict | None = None,
    layout=None,
) -> dict[int, TeamSimCounters]:
    """Vectorized group stage + CDF catalog; knockouts in a light per-sim loop."""
    if rng is None:
        rng = np.random.default_rng(seed)
    if catalog is None or layout is None:
        catalog, layout = prepare_group_batch(
            model, fixtures, team_ids_by_group, wc_team_ids, wc_to_model
        )

    home_goals, away_goals = sample_all_group_scorelines(
        catalog, fixtures, n_sims, rng, group_uniforms=group_uniforms
    )
    points, gf, ga = accumulate_standings(layout, home_goals, away_goals)
    idx_map = idx_to_team_id_map(layout)
    totals: dict[int, TeamSimCounters] = {
        tid: TeamSimCounters()
        for ids in team_ids_by_group.values()
        for tid in ids
    }
    for s in range(n_sims):
        outcome = resolve_one_group_outcome(
            s,
            layout,
            team_ids_by_group,
            idx_map,
            points,
            gf,
            ga,
            home_goals,
            away_goals,
            rng,
        )
        for group, ranked in outcome.ranked_by_group.items():
            for tid in ranked[:2]:
                totals[tid].top_two += 1
        q = build_qualifiers(
            outcome.ranked_by_group,
            outcome.third_by_group,
            outcome.best_eight_third_group_letters,
        )
        champion, winners = simulate_knockout(
            q, model, wc_to_model, rng, catalog=catalog
        )
        _apply_knockout_counters(totals, champion, winners)
    return totals


def _totals_to_aggregate(
    totals: dict[int, TeamSimCounters], n: int
) -> SimAggregate:
    batch_utc = datetime.now(timezone.utc)
    aggregate = SimAggregate(run_batch_utc=batch_utc, n_sims=n)
    for tid, c in totals.items():
        aggregate.by_team[tid] = {
            "p_advance_group": c.top_two / n,
            "p_r16": c.r16 / n,
            "p_qf": c.qf / n,
            "p_sf": c.sf / n,
            "p_final": c.final / n,
            "p_win": c.win / n,
        }
    return aggregate


def run_monte_carlo(
    model: DixonColesModel,
    *,
    n_sims: int | None = None,
    seed: int | None = None,
    fixtures_path: str | None = None,
    teams_path: str | None = None,
    use_fast: bool = True,
) -> SimAggregate:
    """Run ``n_sims`` tournament replicates and return per-team probabilities."""
    n = n_sims if n_sims is not None else config.N_SIMS
    if n < 1:
        raise ValueError("n_sims must be positive")
    seed_val = seed if seed is not None else config.RANDOM_SEED

    teams = load_wc_teams(teams_path)
    by_group = teams_by_group(teams)
    team_ids_by_group = {g: [t.team_id for t in ts] for g, ts in by_group.items()}
    wc_to_model = wc_id_to_model_id(teams)
    wc_team_ids = [t.team_id for t in teams]
    fixtures = load_group_fixtures(fixtures_path)

    if use_fast:
        totals = run_monte_carlo_fast(
            model,
            n_sims=n,
            seed=seed_val,
            fixtures=fixtures,
            team_ids_by_group=team_ids_by_group,
            wc_to_model=wc_to_model,
            wc_team_ids=wc_team_ids,
        )
    else:
        totals = run_monte_carlo_legacy(
            model,
            n_sims=n,
            seed=seed_val,
            fixtures=fixtures,
            team_ids_by_group=team_ids_by_group,
            wc_to_model=wc_to_model,
            wc_team_ids=wc_team_ids,
        )

    return _totals_to_aggregate(totals, n)


def write_sim_results(
    aggregate: SimAggregate,
    *,
    db_url: str | None = None,
) -> int:
    """Persist ``aggregate`` to ``sim_results``; return rows written."""
    rows = [
        {
            "run_batch_utc": aggregate.run_batch_utc,
            "team_id": tid,
            **probs,
            "n_sims": aggregate.n_sims,
        }
        for tid, probs in aggregate.by_team.items()
    ]
    count = db.execute(INSERT_SIM_SQL, rows, db_url=db_url)
    log.info(
        "sim_results: batch=%s n_sims=%d teams=%d",
        aggregate.run_batch_utc.isoformat(),
        aggregate.n_sims,
        len(rows),
    )
    return max(count, 0)


def run_and_persist(
    model: DixonColesModel | None = None,
    *,
    n_sims: int | None = None,
    seed: int | None = None,
    db_url: str | None = None,
) -> SimAggregate:
    """Fit or use provided model, run Monte Carlo, write ``sim_results``."""
    dc_model = model if model is not None else fit_dixon_coles_from_db(db_url=db_url)
    if abs(config.MODEL_TEMPERATURE - 1.0) > 1e-9:
        log.warning(
            "MODEL_TEMPERATURE=%.4f but sim samples raw score matrix (T=1 path); "
            "set MODEL_TEMPERATURE=1.0 for published receipts",
            config.MODEL_TEMPERATURE,
        )
    agg = run_monte_carlo(dc_model, n_sims=n_sims, seed=seed)
    write_sim_results(agg, db_url=db_url)
    return agg


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    agg = run_and_persist()
    top = sorted(agg.by_team.items(), key=lambda x: x[1]["p_win"], reverse=True)[:5]
    print(f"sim complete: n_sims={agg.n_sims} batch={agg.run_batch_utc.isoformat()}")
    print("top P(win):")
    for tid, probs in top:
        print(f"  team_id={tid}: p_win={probs['p_win']:.4f}")


if __name__ == "__main__":
    main()
