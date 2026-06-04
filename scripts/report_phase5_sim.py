#!/usr/bin/env python3
"""Phase 5 title-odds report: full P(win) table + property checks (BUILD_PLAN)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pitchedge import config
from pitchedge.model.dixon_coles import fit_dixon_coles_from_db
from pitchedge.sim.tournament import run_monte_carlo
from pitchedge.sim.wc_teams import load_wc_teams, teams_by_group


def main() -> None:
    n = config.N_SIMS
    seed = config.RANDOM_SEED
    print(f"Monte Carlo: n_sims={n} seed={seed} MODEL_TEMPERATURE={config.MODEL_TEMPERATURE}")
    model = fit_dixon_coles_from_db()
    agg = run_monte_carlo(model, n_sims=n, seed=seed)
    agg2 = run_monte_carlo(model, n_sims=n, seed=seed)

    teams = load_wc_teams()
    names = {t.team_id: t.name for t in teams}
    by_group = teams_by_group(teams)

    rows = sorted(agg.by_team.items(), key=lambda x: x[1]["p_win"], reverse=True)
    print("\nP(win) | P(reach SF) | P(reach final) | team")
    print("-" * 72)
    for tid, p in rows:
        print(
            f"{p['p_win']:7.4f} | {p['p_sf']:11.4f} | {p['p_final']:13.4f} | "
            f"{names.get(tid, tid)} (id={tid})"
        )

    total_win = sum(p["p_win"] for p in agg.by_team.values())
    print(f"\nSum P(win) = {total_win:.6f} (gate: 1.0 +/- 0.01)")
    print("\nPer-group sum P(advance_group) (expect ~2.0):")
    for g, ts in by_group.items():
        s = sum(agg.by_team[t.team_id]["p_advance_group"] for t in ts)
        flag = " OK" if abs(s - 2.0) <= 0.12 else " WARN"
        print(f"  Group {g}: {s:.4f}{flag}")

    repro_ok = all(
        agg.by_team[tid]["p_win"] == agg2.by_team[tid]["p_win"]
        for tid in agg.by_team
    )
    print(f"\nSame-seed re-run identical: {repro_ok}")

    favorites = {37, 33, 29, 9, 16, 2}
    print("\nFavorite cluster (Argentina/France/Spain/Brazil + peers):")
    for tid in sorted(favorites, key=lambda t: agg.by_team[t]["p_win"], reverse=True):
        if tid in agg.by_team:
            print(f"  {names[tid]:18s} P(win)={agg.by_team[tid]['p_win']:.4f}")

    longshots = [48, 12, 7, 47]
    print("\nLongshot sanity (should trail favorites):")
    for tid in longshots:
        if tid in agg.by_team:
            print(f"  {names[tid]:18s} P(win)={agg.by_team[tid]['p_win']:.4f}")

    failures: list[str] = []
    if abs(total_win - 1.0) > 0.01:
        failures.append(f"sum P(win)={total_win:.4f}")
    for g, ts in by_group.items():
        s = sum(agg.by_team[t.team_id]["p_advance_group"] for t in ts)
        if abs(s - 2.0) > 0.12:
            failures.append(f"group {g} advance sum={s:.3f}")
    if not repro_ok:
        failures.append("reproducibility")
    if failures:
        print("\nPROPERTY CHECK FAILURES:")
        for f in failures:
            print(f"  - {f}")
        sys.exit(1)
    print("\nPhase 5 property checks passed.")


if __name__ == "__main__":
    main()
