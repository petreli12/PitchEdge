#!/usr/bin/env python3
"""A/B the tournament sim under correct vs broken WC->model id mapping.

The disagreement-board bug was an id-space mismatch: WC team ids (1..48) were fed
to a model keyed by historical hash ids, so every lookup missed and collapsed to
equal lambdas (the 34/32/34 coin-flip). The sim translates ids via
``wc_id_to_model_id`` and so was never affected; this script proves that by
running the SAME fitted model and SAME seed two ways:

  * correct  : wc_id -> historical model id (production path)
  * broken   : wc_id -> wc_id (identity; ids miss the model => coin-flips)

If the sim were stale/cached, both would be identical. A divergence (favorites
concentrate under the correct map, minnows thin) proves the sim is live and uses
the corrected model. Reproducibility: correct map run twice at one seed must be
byte-identical.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pitchedge import config
from pitchedge.model.dixon_coles import fit_dixon_coles_from_db
from pitchedge.sim.tournament import (
    _totals_to_aggregate,
    run_monte_carlo_fast,
)
from pitchedge.sim.wc_teams import (
    load_group_fixtures,
    load_wc_teams,
    teams_by_group,
    wc_id_to_model_id,
)


def _aggregate(model, *, wc_to_model, fixtures, team_ids_by_group, wc_team_ids, n, seed):
    totals = run_monte_carlo_fast(
        model,
        n_sims=n,
        seed=seed,
        fixtures=fixtures,
        team_ids_by_group=team_ids_by_group,
        wc_team_ids=wc_team_ids,
        wc_to_model=wc_to_model,
    )
    return _totals_to_aggregate(totals, n)


def _sum_pwin(agg) -> float:
    return sum(v["p_win"] for v in agg.by_team.values())


def _per_group_advance(agg, by_group) -> dict[str, float]:
    out: dict[str, float] = {}
    for g, ts in by_group.items():
        out[g] = sum(agg.by_team[t.team_id]["p_advance_group"] for t in ts)
    return out


def main() -> None:
    n = config.N_SIMS
    seed = config.RANDOM_SEED
    print(f"n_sims={n} seed={seed}")

    print("fitting Dixon-Coles (Elo shrinkage) ...")
    model = fit_dixon_coles_from_db()

    teams = load_wc_teams()
    name = {t.team_id: t.name for t in teams}
    by_group = teams_by_group(teams)
    team_ids_by_group = {g: [t.team_id for t in ts] for g, ts in by_group.items()}
    wc_team_ids = [t.team_id for t in teams]
    fixtures = load_group_fixtures()

    correct_map = wc_id_to_model_id(teams)               # production
    broken_map = {t.team_id: t.team_id for t in teams}   # identity => all miss model

    kw = dict(
        fixtures=fixtures,
        team_ids_by_group=team_ids_by_group,
        wc_team_ids=wc_team_ids,
        n=n,
    )

    agg_correct = _aggregate(model, wc_to_model=correct_map, seed=seed, **kw)
    agg_correct2 = _aggregate(model, wc_to_model=correct_map, seed=seed, **kw)
    agg_broken = _aggregate(model, wc_to_model=broken_map, seed=seed, **kw)

    # Reproducibility: same seed + correct map => identical.
    repro = all(
        abs(agg_correct.by_team[t]["p_win"] - agg_correct2.by_team[t]["p_win"]) < 1e-12
        for t in agg_correct.by_team
    )

    order = sorted(
        agg_correct.by_team, key=lambda t: -agg_correct.by_team[t]["p_win"]
    )[:15]

    print("\n=== Title odds: BROKEN map (buggy-equivalent) vs CORRECT map (fixed) ===")
    print(f"{'team':14} {'broken P(win)':>14} {'correct P(win)':>15} {'delta':>9}")
    print("-" * 56)
    for t in order:
        b = agg_broken.by_team[t]["p_win"]
        c = agg_correct.by_team[t]["p_win"]
        print(f"{name[t][:14]:14} {b:>14.4f} {c:>15.4f} {c - b:>+9.4f}")

    print("\n=== Property checks ===")
    print(f"sum P(win)  correct = {_sum_pwin(agg_correct):.6f}  (expect ~1.0)")
    print(f"sum P(win)  broken  = {_sum_pwin(agg_broken):.6f}  (expect ~1.0)")
    print(f"same-seed reproducible (correct map, run x2): {repro}")

    adv_c = _per_group_advance(agg_correct, by_group)
    adv_b = _per_group_advance(agg_broken, by_group)
    print("\nper-group sum P(advance to top-2)  (expect ~2.0 each):")
    print(f"{'group':6} {'correct':>9} {'broken':>9}")
    for g in sorted(adv_c):
        print(f"{g:6} {adv_c[g]:>9.4f} {adv_b[g]:>9.4f}")

    # Directional summary: how much mass favorites gained from the long tail.
    top6_c = sum(agg_correct.by_team[t]["p_win"] for t in order[:6])
    top6_b = sum(agg_broken.by_team[t]["p_win"] for t in order[:6])
    print(
        f"\ntop-6 favorites P(win) mass: broken={top6_b:.3f} -> correct={top6_c:.3f} "
        f"(+{top6_c - top6_b:.3f})"
    )


if __name__ == "__main__":
    main()
