#!/usr/bin/env python3
"""Benchmark Phase 5 Monte Carlo: legacy loop vs vectorized fast path."""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pitchedge import config
from pitchedge.model.dixon_coles import DixonColesModel, fit_dixon_coles_from_db
from pitchedge.sim.tournament import run_monte_carlo_fast, run_monte_carlo_legacy
from pitchedge.sim.wc_teams import (
    load_group_fixtures,
    load_wc_teams,
    teams_by_group,
    wc_id_to_model_id,
)


def _load_layout():
    teams = load_wc_teams()
    by_group = teams_by_group(teams)
    return (
        load_group_fixtures(),
        {g: [t.team_id for t in ts] for g, ts in by_group.items()},
        wc_id_to_model_id(teams),
        [t.team_id for t in teams],
    )


def main() -> None:
    n_full = config.N_SIMS
    n_legacy = int(sys.argv[1]) if len(sys.argv) > 1 else min(200, n_full)
    seed = config.RANDOM_SEED

    try:
        model = fit_dixon_coles_from_db()
    except Exception:
        model = DixonColesModel(attack={}, defense={}, home_adv=0.1, rho=-0.05, xi=0.005)

    fixtures, team_ids_by_group, wc_to_model, wc_team_ids = _load_layout()
    common = dict(
        model=model,
        seed=seed,
        fixtures=fixtures,
        team_ids_by_group=team_ids_by_group,
        wc_to_model=wc_to_model,
        wc_team_ids=wc_team_ids,
    )

    print(f"Catalog build + fast path ({n_full} sims)...")
    t0 = time.perf_counter()
    run_monte_carlo_fast(n_sims=n_full, **common)
    fast_sec = time.perf_counter() - t0

    print(f"Legacy loop ({n_legacy} sims)...")
    t1 = time.perf_counter()
    run_monte_carlo_legacy(n_sims=n_legacy, **common)
    legacy_sample_sec = time.perf_counter() - t1
    legacy_est = legacy_sample_sec * (n_full / n_legacy)

    print()
    print(f"Fast ({n_full} sims):     {fast_sec:.1f}s  ({n_full / fast_sec:.1f} sims/s)")
    print(f"Legacy ({n_legacy} sims): {legacy_sample_sec:.1f}s  ({n_legacy / legacy_sample_sec:.1f} sims/s)")
    print(f"Legacy est. ({n_full}):   {legacy_est:.1f}s  (linear extrapolation)")
    print(f"Speedup (est.):           {legacy_est / fast_sec:.1f}x")


if __name__ == "__main__":
    main()
