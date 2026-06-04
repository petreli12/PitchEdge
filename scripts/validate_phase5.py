#!/usr/bin/env python3
"""Phase 5 validation per docs/BUILD_PLAN.md (property checks on sim output)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pitchedge import config
from pitchedge.model.dixon_coles import fit_dixon_coles_from_db
from pitchedge.sim.annex_c import load_annex_c
from pitchedge.sim.tournament import run_monte_carlo
from pitchedge.sim.wc_teams import load_wc_teams, teams_by_group


def main() -> None:
    failures: list[str] = []

    if len(load_annex_c()) != 495:
        failures.append("annex_c.json must have 495 combinations")

    print(f"Running {config.N_SIMS} sims (seed={config.RANDOM_SEED})...")
    model = fit_dixon_coles_from_db()
    a = run_monte_carlo(model, n_sims=config.N_SIMS, seed=config.RANDOM_SEED)
    b = run_monte_carlo(model, n_sims=min(200, config.N_SIMS), seed=config.RANDOM_SEED)

    total_win = sum(p["p_win"] for p in a.by_team.values())
    if abs(total_win - 1.0) > 0.02:
        failures.append(f"sum P(win)={total_win:.4f} (expected ~1.0)")

    teams = load_wc_teams()
    by_group = teams_by_group(teams)
    for g, ts in by_group.items():
        s = sum(a.by_team[t.team_id]["p_advance_group"] for t in ts)
        if abs(s - 2.0) > 0.12:
            failures.append(f"group {g} sum P(advance_group)={s:.3f} (expected ~2.0)")

    tid = 37
    if a.by_team[tid]["p_win"] != b.by_team[tid]["p_win"]:
        failures.append("reproducibility failed for fixed seed subsample")

    top = sorted(a.by_team.items(), key=lambda x: x[1]["p_win"], reverse=True)[:8]
    print("Top P(win):")
    names = {t.team_id: t.name for t in teams}
    for team_id, probs in top:
        print(f"  {names[team_id]:20s} {probs['p_win']:.4f}")

    if failures:
        print("FAIL:")
        for f in failures:
            print(f"  - {f}")
        sys.exit(1)
    print("Phase 5 checks passed.")


if __name__ == "__main__":
    main()
