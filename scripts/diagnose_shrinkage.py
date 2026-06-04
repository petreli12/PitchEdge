#!/usr/bin/env python3
"""Diagnose favorite-vs-minnow flattening from Dixon-Coles shrinkage.

For each fixture, print both teams' qualifying-match counts in the recency
window, fitted DC attack/defense (pre- and post-shrinkage), Elo, Elo-implied
win prob, and the DC neutral-venue 1X2. Confirms whether shrinkage pulls minnow
params so far toward the prior that the favorite's edge collapses.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pitchedge import config
from pitchedge.model.dixon_coles import (
    DixonColesConfig,
    apply_elo_shrinkage,
    fit_dixon_coles,
    load_latest_elo,
    load_training_matches,
    match_probs,
    filter_recency_window,
    _count_team_matches,
)
from pitchedge.model.elo import EloConfig, elo_win_prob
from pitchedge.sim.wc_teams import load_wc_teams

FIXTURES = [
    ("Brazil", "Haiti"),
    ("Germany", "Curacao"),
    ("Spain", "Cape Verde"),
]


def main() -> None:
    teams = {t.name: t for t in load_wc_teams()}
    matches = load_training_matches()
    elo = load_latest_elo()

    cfg = DixonColesConfig(
        xi=config.DC_XI,
        recency_years=config.DC_RECENCY_YEARS,
        min_matches=config.DC_MIN_MATCHES,
        max_goals=config.DC_MAX_GOALS,
        rho_max_iter=config.DC_RHO_MAX_ITER,
        tau_floor=config.DC_TAU_FLOOR,
    )

    # Fit without shrinkage, then apply shrinkage, so we can compare params.
    raw_model = fit_dixon_coles(matches, elo_by_team=None, dc_config=cfg)
    shrunk_model = apply_elo_shrinkage(raw_model, elo_by_team=elo, dc_config=cfg)

    work = filter_recency_window(matches, recency_years=cfg.recency_years)
    counts = _count_team_matches(work)

    elo_cfg = EloConfig(
        initial_rating=config.ELO_INITIAL,
        rating_divisor=config.ELO_RATING_DIVISOR,
    )

    print(f"min_matches={cfg.min_matches} recency_years={cfg.recency_years} "
          f"shrink_elo_scale={cfg.shrink_elo_scale}")
    print(f"raw_model teams={len(raw_model.attack)} window_matches={len(work)}\n")

    for home_name, away_name in FIXTURES:
        ht = teams[home_name]
        at = teams[away_name]
        hid = ht.model_team_id
        aid = at.model_team_id

        print("=" * 72)
        print(f"{home_name} (id={hid}) vs {away_name} (id={aid})  [neutral venue]")
        for label, t in ((home_name, ht), (away_name, at)):
            tid = t.model_team_id
            n = counts.get(tid, 0)
            raw_a = raw_model.attack.get(tid, float("nan"))
            raw_d = raw_model.defense.get(tid, float("nan"))
            sh_a = shrunk_model.attack.get(tid, float("nan"))
            sh_d = shrunk_model.defense.get(tid, float("nan"))
            e = elo.get(tid, elo_cfg.initial_rating)
            print(
                f"  {label:12} n={n:4d}  Elo={e:7.1f}  "
                f"raw(att={raw_a:+.3f} def={raw_d:+.3f})  "
                f"shrunk(att={sh_a:+.3f} def={sh_d:+.3f})"
            )

        e_home = elo.get(hid, elo_cfg.initial_rating)
        e_away = elo.get(aid, elo_cfg.initial_rating)
        p_elo = elo_win_prob(
            e_home, e_away, home_advantage=0.0,
            rating_divisor=elo_cfg.rating_divisor,
        )

        raw_probs = match_probs(raw_model, hid, aid, neutral=True)
        sh_probs = match_probs(shrunk_model, hid, aid, neutral=True)
        lh_raw, la_raw = raw_model.lambdas(hid, aid, neutral=True)
        lh_sh, la_sh = shrunk_model.lambdas(hid, aid, neutral=True)

        print(f"  Elo-implied P({home_name} win, no draw) = {p_elo:.3f}")
        print(f"  RAW    DC: P(home)={raw_probs.p_home:.3f} draw={raw_probs.p_draw:.3f} "
              f"away={raw_probs.p_away:.3f}  lambda=({lh_raw:.2f},{la_raw:.2f})")
        print(f"  SHRUNK DC: P(home)={sh_probs.p_home:.3f} draw={sh_probs.p_draw:.3f} "
              f"away={sh_probs.p_away:.3f}  lambda=({lh_sh:.2f},{la_sh:.2f})")
        print()


if __name__ == "__main__":
    main()
