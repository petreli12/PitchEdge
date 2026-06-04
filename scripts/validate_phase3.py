#!/usr/bin/env python3
"""Phase 3 validation per docs/BUILD_PLAN.md."""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pitchedge.ingest.team_ids import team_name_to_id
from pitchedge.model.blend import blend
from pitchedge.model.dixon_coles import (
    DixonColesConfig,
    DixonColesModel,
    apply_elo_shrinkage,
    fit_dixon_coles,
    fit_dixon_coles_from_db,
    load_training_matches,
    load_latest_elo,
    match_probs,
    wc_match_probs,
    wc_score_matrix,
)

logging.basicConfig(level=logging.WARNING)


def _check(name: str, ok: bool, detail: str, failures: list[str]) -> None:
    mark = "PASS" if ok else "FAIL"
    print(f"[{mark}] {name}: {detail}")
    if not ok:
        failures.append(name)


def main() -> None:
    failures: list[str] = []

    # Blend endpoints
    model_p = (0.55, 0.25, 0.20)
    market_p = (0.50, 0.27, 0.23)
    _check("blend w=0 == market", blend(model_p, market_p, w=0.0) == market_p, str(blend(model_p, market_p, w=0.0)), failures)
    _check("blend w=1 == model", blend(model_p, market_p, w=1.0) == model_p, str(blend(model_p, market_p, w=1.0)), failures)
    b = blend(model_p, market_p, w=0.3)
    _check("blend renormalizes", abs(sum(b) - 1.0) < 1e-9, f"sum={sum(b)}", failures)

    # Shrinkage warning
    cap: list[str] = []

    class H(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            cap.append(record.getMessage())

    h = H()
    lg = logging.getLogger("pitchedge.model.dixon_coles")
    lg.addHandler(h)
    lg.setLevel(logging.WARNING)
    low = DixonColesModel(
        attack={99: 0.4},
        defense={99: -0.1},
        home_adv=0.2,
        rho=-0.03,
        xi=0.005,
        match_counts={99: 5},
    )
    apply_elo_shrinkage(low, elo_by_team={99: 1500.0}, dc_config=DixonColesConfig(min_matches=30))
    _check(
        "low-data shrinkage warning",
        any("team_id=99" in m for m in cap),
        cap[-1] if cap else "none",
        failures,
    )

    # DB fit with recency window (fast path)
    print("\nFitting on DB (6-year recency window)...")
    t0 = time.perf_counter()
    try:
        model = fit_dixon_coles_from_db()
        elapsed = time.perf_counter() - t0
    except Exception as exc:
        _check("DB fit completes", False, str(exc), failures)
        print(f"\nFAILED {len(failures)} checks")
        sys.exit(1)

    mean_alpha = float(np.mean(list(model.attack.values())))
    _check(
        "optimizer finite NLL",
        model.fit_neg_log_likelihood is not None and model.fit_neg_log_likelihood < 1e8,
        f"nll={model.fit_neg_log_likelihood:.2f}",
        failures,
    )
    _check(
        "mean(alpha) approx 0",
        abs(mean_alpha) < 1e-4,
        f"mean_alpha={mean_alpha:.8f}",
        failures,
    )
    _check(
        "fit wall-clock reasonable",
        elapsed < 120.0,
        f"elapsed={elapsed:.2f}s (reported fit_seconds={model.fit_seconds})",
        failures,
    )

    pairs = [
        ("Argentina", "Brazil"),
        ("France", "Germany"),
        ("Spain", "Italy"),
        ("Mexico", "United States"),
    ]
    details: list[str] = []
    matrix_ok = True
    for hn, an in pairs:
        mat = wc_score_matrix(model, team_name_to_id(hn), team_name_to_id(an))
        s = float(mat.sum())
        details.append(f"{hn} v {an}={s:.8f}")
        if abs(s - 1.0) > 1e-6:
            matrix_ok = False
    _check("score_matrix sums to ~1", matrix_ok, "; ".join(details), failures)

    arg = team_name_to_id("Argentina")
    rsa = team_name_to_id("South Africa")
    probs_wc = wc_match_probs(model, arg, rsa)
    probs_gamma_demo = match_probs(model, arg, rsa, neutral=False)
    _check(
        "strong favorite > 0.5 (wc neutral venue)",
        probs_wc.p_home > 0.5,
        f"p_home={probs_wc.p_home:.3f} gamma={model.home_adv:.4f}",
        failures,
    )
    print(
        f"Argentina vs South Africa: wc_match_probs p_home={probs_wc.p_home:.3f} "
        f"(neutral=True); diagnostic gamma-on p_home={probs_gamma_demo.p_home:.3f}"
    )

    print("\n=== Phase 3 validation summary ===")
    print(f"DB fit wall_clock: {elapsed:.2f}s")
    print(f"Reported fit_seconds: {model.fit_seconds}")
    print(f"final_nll: {model.fit_neg_log_likelihood:.2f}")
    if failures:
        print(f"FAILED: {', '.join(failures)}")
        sys.exit(1)
    print("All Phase 3 checks passed.")


if __name__ == "__main__":
    main()
