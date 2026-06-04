"""Dixon-Coles bivariate Poisson model for international matches.

Fitting is two-stage for speed:
  1. Poisson GLM (statsmodels) for attack, defense, and home advantage ``γ``.
  2. Scalar ``ρ`` low-score correction via vectorized time-decayed DC likelihood.

Training uses a configurable recency window (default 6 years) plus ``exp(-ξ·Δt)``
weights. Probabilities are in [0, 1]; expected goals are non-negative counts.

See docs/ARCHITECTURE.md section 3.3.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any, Mapping, Sequence

import numpy as np
import statsmodels.api as sm
from scipy.optimize import minimize
from scipy.special import gammaln
from scipy.stats import poisson
from statsmodels.genmod.families import Poisson

from pitchedge import config, db
from pitchedge.model.elo import EloConfig

log = logging.getLogger(__name__)

_SUM_TOL = 1e-6


@dataclass
class DixonColesConfig:
    """Hyperparameters for fitting and prediction."""

    xi: float = 0.005  # time-decay per year; tune on validation (Phase 4)
    recency_years: float = 6.0  # training window length before reference date
    min_matches: int = 30  # shrink params when team has fewer qualifying matches
    max_goals: int = 10
    rho_max_iter: int = 100  # stage-2 optimizer iterations (ρ only)
    tau_floor: float = 1e-10  # floor on τ·Poisson mass before log()
    elo_config: EloConfig | None = None
    shrink_elo_scale: float = 0.15  # maps Elo delta to attack/defense prior magnitude


@dataclass
class MatchProbs:
    """Three-way and derived market probabilities (all in [0, 1] except xG)."""

    p_home: float
    p_draw: float
    p_away: float
    exp_home_goals: float
    exp_away_goals: float
    p_over_25: float
    p_btts: float


@dataclass
class DixonColesModel:
    """Fitted parameters keyed by historical ``team_id`` (same ids as raw_results)."""

    attack: dict[int, float]
    defense: dict[int, float]
    home_adv: float
    rho: float
    xi: float
    fit_neg_log_likelihood: float | None = None
    match_counts: dict[int, int] | None = None
    fit_seconds: float | None = None

    def lambdas(
        self, home_id: int, away_id: int, *, neutral: bool = False
    ) -> tuple[float, float]:
        """Expected goals ``(λ_home, λ_away)`` for the listed home/away slots.

        When ``neutral`` is True, home advantage ``γ`` is omitted (``ha = 0``).
        """
        ha = 0.0 if neutral else self.home_adv
        alpha_h = self.attack.get(home_id, 0.0)
        beta_h = self.defense.get(home_id, 0.0)
        alpha_a = self.attack.get(away_id, 0.0)
        beta_a = self.defense.get(away_id, 0.0)
        log_lh = alpha_h + beta_a + ha
        log_la = alpha_a + beta_h
        return (math.exp(log_lh), math.exp(log_la))


@dataclass
class MatchArrays:
    """Vectorized match data for likelihood evaluation (indices into ``team_ids``)."""

    home_idx: np.ndarray
    away_idx: np.ndarray
    home_goals: np.ndarray
    away_goals: np.ndarray
    neutral: np.ndarray
    weights: np.ndarray
    n_matches: int


def tau(
    home_goals: int,
    away_goals: int,
    lambda_home: float,
    lambda_away: float,
    rho: float,
) -> float:
    """Dixon-Coles adjustment factor τ for low-score cells."""
    if home_goals == 0 and away_goals == 0:
        return 1.0 - lambda_home * lambda_away * rho
    if home_goals == 0 and away_goals == 1:
        return 1.0 + lambda_home * rho
    if home_goals == 1 and away_goals == 0:
        return 1.0 + lambda_away * rho
    if home_goals == 1 and away_goals == 1:
        return 1.0 - rho
    return 1.0


def dc_log_prob(
    home_goals: int,
    away_goals: int,
    lambda_home: float,
    lambda_away: float,
    rho: float,
) -> float:
    """Log probability of a single scoreline under Dixon-Coles."""
    if home_goals < 0 or away_goals < 0:
        return -math.inf
    t = tau(home_goals, away_goals, lambda_home, lambda_away, rho)
    if t <= 0.0:
        return -math.inf
    return (
        math.log(max(t, config.DC_TAU_FLOOR))
        + poisson.logpmf(home_goals, lambda_home)
        + poisson.logpmf(away_goals, lambda_away)
    )


def score_matrix(
    model: DixonColesModel,
    home_id: int,
    away_id: int,
    *,
    max_goals: int | None = None,
    neutral: bool = False,
) -> np.ndarray:
    """Return score-probability matrix ``P[home_goals, away_goals]`` (sums to ~1).

    Dimensions ``(max_goals+1, max_goals+1)``. Row index = goals for ``home_id``,
    column index = goals for ``away_id``. When ``neutral`` is True, ``γ`` is not
    applied to expected goals (neutral-venue / World Cup group knockout sites).
    """
    mg = max_goals if max_goals is not None else config.DC_MAX_GOALS
    lambda_h, lambda_a = model.lambdas(home_id, away_id, neutral=neutral)
    mat = np.zeros((mg + 1, mg + 1), dtype=float)
    for i in range(mg + 1):
        for j in range(mg + 1):
            t = tau(i, j, lambda_h, lambda_a, model.rho)
            mat[i, j] = t * poisson.pmf(i, lambda_h) * poisson.pmf(j, lambda_a)
    total = mat.sum()
    if total <= 0.0:
        raise ValueError("score matrix collapsed to zero mass")
    return mat / total


def match_probs(
    model: DixonColesModel,
    home_id: int,
    away_id: int,
    *,
    max_goals: int | None = None,
    neutral: bool = False,
) -> MatchProbs:
    """Aggregate the score matrix into 1X2, xG, over 2.5, and BTTS probabilities.

    ``p_home`` / ``p_away`` refer to ``home_id`` / ``away_id``. When ``neutral`` is
    True, ``γ`` is zeroed; swapping home/away should mirror 1X2 (see tests).
    """
    mat = score_matrix(
        model, home_id, away_id, max_goals=max_goals, neutral=neutral
    )
    p_home = 0.0
    p_draw = 0.0
    p_away = 0.0
    exp_h = 0.0
    exp_a = 0.0
    p_over = 0.0
    p_btts = 0.0
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            p = mat[i, j]
            exp_h += i * p
            exp_a += j * p
            if i > j:
                p_home += p
            elif i == j:
                p_draw += p
            else:
                p_away += p
            if i + j >= 3:
                p_over += p
            if i >= 1 and j >= 1:
                p_btts += p

    triple = (p_home, p_draw, p_away)
    total = sum(triple)
    if not math.isclose(total, 1.0, rel_tol=0, abs_tol=_SUM_TOL):
        raise ValueError(f"match_probs must sum to 1, got {total}")

    return MatchProbs(
        p_home=p_home,
        p_draw=p_draw,
        p_away=p_away,
        exp_home_goals=exp_h,
        exp_away_goals=exp_a,
        p_over_25=p_over,
        p_btts=p_btts,
    )


def wc_score_matrix(
    model: DixonColesModel,
    home_id: int,
    away_id: int,
    *,
    max_goals: int | None = None,
    host_home: bool | None = None,
) -> np.ndarray:
    """Score matrix for WC / international-tournament fixtures (``γ`` usually off).

    Defaults to neutral venue. Pass ``host_home=True`` or use a co-host as
    ``home_id`` (see ``pitchedge.model.venues``) to apply home advantage.
    """
    from pitchedge.model.venues import dixon_coles_neutral_for_wc_fixture

    neutral = dixon_coles_neutral_for_wc_fixture(home_id, host_home=host_home)
    return score_matrix(
        model, home_id, away_id, max_goals=max_goals, neutral=neutral
    )


def wc_match_probs(
    model: DixonColesModel,
    home_id: int,
    away_id: int,
    *,
    max_goals: int | None = None,
    host_home: bool | None = None,
) -> MatchProbs:
    """Match probabilities for WC / international-tournament fixtures.

    Same venue policy as ``wc_score_matrix``. Tournament sim and ``predict.py``
    should call this (or ``fixture_model_probs``) rather than bare ``match_probs``.
    """
    from pitchedge.model.venues import dixon_coles_neutral_for_wc_fixture

    neutral = dixon_coles_neutral_for_wc_fixture(home_id, host_home=host_home)
    return match_probs(
        model, home_id, away_id, max_goals=max_goals, neutral=neutral
    )


def tournament_score_matrix(
    model: DixonColesModel,
    home_id: int,
    away_id: int,
    *,
    max_goals: int | None = None,
    host_home: bool = False,
) -> np.ndarray:
    """Score matrix for historical cup backtests (Phase 4); neutral unless ``host_home``."""
    from pitchedge.model.venues import dixon_coles_neutral_for_tournament_fixture

    neutral = dixon_coles_neutral_for_tournament_fixture(host_home=host_home)
    return score_matrix(
        model, home_id, away_id, max_goals=max_goals, neutral=neutral
    )


def tournament_match_probs(
    model: DixonColesModel,
    home_id: int,
    away_id: int,
    *,
    max_goals: int | None = None,
    host_home: bool = False,
) -> MatchProbs:
    """Match probabilities for historical cup backtests (Phase 4)."""
    from pitchedge.model.venues import dixon_coles_neutral_for_tournament_fixture

    neutral = dixon_coles_neutral_for_tournament_fixture(host_home=host_home)
    return match_probs(
        model, home_id, away_id, max_goals=max_goals, neutral=neutral
    )


def _years_before(reference: date, match_date: date) -> float:
    return max(0.0, (reference - match_date).days / 365.25)


def filter_recency_window(
    matches: Sequence[Mapping[str, Any]],
    *,
    recency_years: float,
) -> list[dict[str, Any]]:
    """Keep matches within ``recency_years`` of the latest date in ``matches``."""
    if not matches:
        return []
    ref = max(r["date"] for r in matches)
    cutoff = ref - timedelta(days=int(recency_years * 365.25))
    return [dict(r) for r in matches if r["date"] >= cutoff]


def _count_team_matches(matches: Sequence[Mapping[str, Any]]) -> dict[int, int]:
    counts: dict[int, int] = {}
    for row in matches:
        h = int(row["home_id"])
        a = int(row["away_id"])
        counts[h] = counts.get(h, 0) + 1
        counts[a] = counts.get(a, 0) + 1
    return counts


def _build_match_arrays(
    matches: Sequence[Mapping[str, Any]],
    team_ids: list[int],
    *,
    reference_date: date,
    xi: float,
) -> MatchArrays:
    """Build numpy index arrays for vectorized likelihood evaluation."""
    tid_to_idx = {tid: i for i, tid in enumerate(team_ids)}
    n = len(matches)
    home_idx = np.empty(n, dtype=np.intp)
    away_idx = np.empty(n, dtype=np.intp)
    home_goals = np.empty(n, dtype=np.int64)
    away_goals = np.empty(n, dtype=np.int64)
    neutral = np.empty(n, dtype=bool)
    weights = np.empty(n, dtype=float)

    for i, row in enumerate(matches):
        home_idx[i] = tid_to_idx[int(row["home_id"])]
        away_idx[i] = tid_to_idx[int(row["away_id"])]
        home_goals[i] = int(row["home_goals"])
        away_goals[i] = int(row["away_goals"])
        neutral[i] = bool(row.get("neutral", False))
        weights[i] = math.exp(-xi * _years_before(reference_date, row["date"]))

    return MatchArrays(
        home_idx=home_idx,
        away_idx=away_idx,
        home_goals=home_goals,
        away_goals=away_goals,
        neutral=neutral,
        weights=weights,
        n_matches=n,
    )


def vectorized_neg_log_likelihood(
    rho: float,
    attack: np.ndarray,
    defense: np.ndarray,
    gamma: float,
    arrays: MatchArrays,
    *,
    tau_floor: float,
) -> float:
    """Time-decayed Dixon-Coles negative log-likelihood (single vectorized pass)."""
    hi = arrays.home_idx
    ai = arrays.away_idx
    x = arrays.home_goals.astype(float)
    y = arrays.away_goals.astype(float)
    ha = np.where(arrays.neutral, 0.0, gamma)

    log_lh = attack[hi] + defense[ai] + ha
    log_la = attack[ai] + defense[hi]
    lh = np.exp(log_lh)
    la = np.exp(log_la)

    log_p_home = x * log_lh - lh - gammaln(x + 1.0)
    log_p_away = y * log_la - la - gammaln(y + 1.0)

    tau_vec = np.ones_like(lh)
    m00 = (x == 0) & (y == 0)
    m01 = (x == 0) & (y == 1)
    m10 = (x == 1) & (y == 0)
    m11 = (x == 1) & (y == 1)
    tau_vec[m00] = 1.0 - lh[m00] * la[m00] * rho
    tau_vec[m01] = 1.0 + lh[m01] * rho
    tau_vec[m10] = 1.0 + la[m10] * rho
    tau_vec[m11] = 1.0 - rho

    log_mass = np.log(np.maximum(tau_vec, tau_floor)) + log_p_home + log_p_away
    log_mass = np.where(np.isfinite(log_mass), log_mass, -1e12)
    return float(-np.sum(arrays.weights * log_mass))


def _center_attack(attack: np.ndarray) -> np.ndarray:
    """Identifiability: mean(α) = 0."""
    return attack - np.mean(attack)


def _sum_zero_team_columns(
    x_mat: np.ndarray,
    row: int,
    team_idx: int,
    col_offset: int,
    n_teams: int,
) -> None:
    """Add sum-to-zero team effect: cols ``col_offset .. col_offset+n_teams-2``."""
    if team_idx < n_teams - 1:
        x_mat[row, col_offset + team_idx] += 1.0
    else:
        x_mat[row, col_offset : col_offset + n_teams - 1] -= 1.0


def _expand_sum_zero_params(beta: np.ndarray, n_teams: int) -> np.ndarray:
    """Recover full team vector from ``n_teams-1`` sum-to-zero coefficients."""
    full = np.zeros(n_teams, dtype=float)
    full[: n_teams - 1] = beta
    full[n_teams - 1] = -float(np.sum(beta))
    return full


def _fit_poisson_glm_stage1(
    team_ids: list[int],
    arrays: MatchArrays,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Stage 1: Poisson GLM for attack, defense, and home advantage (γ).

    Uses sum-to-zero contrasts on attack and defense so the design is identifiable
    without a giant joint optimizer. Falls back to light ridge if IRLS fails.
    """
    n_teams = len(team_ids)
    n = arrays.n_matches
    n_rows = 2 * n
    n_free = n_teams - 1
    n_cols = 2 * n_free + 1
    def_off = n_free
    gamma_col = 2 * n_free

    y = np.empty(n_rows, dtype=float)
    w = np.empty(n_rows, dtype=float)
    x_mat = np.zeros((n_rows, n_cols), dtype=float)

    for m in range(n):
        hi = int(arrays.home_idx[m])
        ai = int(arrays.away_idx[m])
        ha = 0.0 if arrays.neutral[m] else 1.0
        hg = float(arrays.home_goals[m])
        ag = float(arrays.away_goals[m])
        wt = float(arrays.weights[m])

        row_h = 2 * m
        row_a = 2 * m + 1
        y[row_h] = hg
        y[row_a] = ag
        w[row_h] = wt
        w[row_a] = wt

        _sum_zero_team_columns(x_mat, row_h, hi, 0, n_teams)
        _sum_zero_team_columns(x_mat, row_h, ai, def_off, n_teams)
        x_mat[row_h, gamma_col] = ha

        _sum_zero_team_columns(x_mat, row_a, ai, 0, n_teams)
        _sum_zero_team_columns(x_mat, row_a, hi, def_off, n_teams)

    glm = sm.GLM(y, x_mat, family=Poisson(), freq_weights=w)
    try:
        result = glm.fit(maxiter=300, disp=0)
    except Exception:
        log.warning("Poisson GLM IRLS failed; retrying with ridge (alpha=1e-3)")
        result = glm.fit_regularized(alpha=1e-3, L1_wt=0.0, maxiter=300)

    attack = _expand_sum_zero_params(
        np.asarray(result.params[:n_free], dtype=float), n_teams
    )
    defense = _expand_sum_zero_params(
        np.asarray(result.params[n_free:gamma_col], dtype=float), n_teams
    )
    attack = _center_attack(attack)
    gamma = float(result.params[gamma_col])
    return attack, defense, gamma


def _fit_rho_stage2(
    attack: np.ndarray,
    defense: np.ndarray,
    gamma: float,
    arrays: MatchArrays,
    *,
    dc_config: DixonColesConfig,
) -> tuple[float, float]:
    """Stage 2: optimize ρ only with vectorized DC likelihood."""

    def objective(theta: np.ndarray) -> float:
        return vectorized_neg_log_likelihood(
            float(theta[0]),
            attack,
            defense,
            gamma,
            arrays,
            tau_floor=dc_config.tau_floor,
        )

    result = minimize(
        objective,
        x0=np.array([-0.03]),
        method="L-BFGS-B",
        bounds=[(-0.2, 0.2)],
        options={"maxiter": dc_config.rho_max_iter},
    )
    if not result.success:
        log.warning("Dixon-Coles rho optimizer: %s", result.message)
    rho = float(result.x[0])
    nll = float(result.fun)
    return rho, nll


def _elo_attack_defense_priors(
    team_ids: list[int],
    elo_by_team: Mapping[int, float],
    *,
    dc_config: DixonColesConfig,
) -> tuple[dict[int, float], dict[int, float]]:
    """Map Elo ratings to weak attack/defense priors for shrinkage."""
    cfg = dc_config.elo_config or EloConfig(
        initial_rating=config.ELO_INITIAL,
        rating_divisor=config.ELO_RATING_DIVISOR,
    )
    scale = dc_config.shrink_elo_scale
    baseline = cfg.initial_rating
    attack: dict[int, float] = {}
    defense: dict[int, float] = {}
    for tid in team_ids:
        elo = elo_by_team.get(tid, baseline)
        delta = (elo - baseline) / cfg.rating_divisor
        attack[tid] = scale * delta
        defense[tid] = -scale * delta
    return attack, defense


def _rebalance_attack_defense(
    attack: dict[int, float], defense: dict[int, float]
) -> tuple[dict[int, float], dict[int, float]]:
    """Shift attack/defense so mean(α)=0 without changing any match λ."""
    mean_alpha = float(np.mean(list(attack.values())))
    if abs(mean_alpha) < 1e-12:
        return attack, defense
    atk = {tid: v - mean_alpha for tid, v in attack.items()}
    dfn = {tid: defense[tid] + mean_alpha for tid in defense}
    return atk, dfn


def apply_elo_shrinkage(
    model: DixonColesModel,
    *,
    elo_by_team: Mapping[int, float],
    dc_config: DixonColesConfig | None = None,
) -> DixonColesModel:
    """Shrink low-data team params toward Elo-implied priors; log each flagged team."""
    cfg = dc_config or DixonColesConfig(
        min_matches=config.DC_MIN_MATCHES,
        xi=model.xi,
        recency_years=config.DC_RECENCY_YEARS,
        tau_floor=config.DC_TAU_FLOOR,
    )
    counts = model.match_counts or {}
    atk_prior, def_prior = _elo_attack_defense_priors(
        list(model.attack.keys()), elo_by_team, dc_config=cfg
    )

    attack = dict(model.attack)
    defense = dict(model.defense)
    for tid in attack:
        n = counts.get(tid, 0)
        if n >= cfg.min_matches:
            continue
        lam = 1.0 - (n / cfg.min_matches) if cfg.min_matches > 0 else 1.0
        attack[tid] = (1.0 - lam) * attack[tid] + lam * atk_prior.get(tid, 0.0)
        defense[tid] = (1.0 - lam) * defense[tid] + lam * def_prior.get(tid, 0.0)
        log.warning(
            "Dixon-Coles shrinkage: team_id=%s has %d qualifying matches (< %d); "
            "pulled toward Elo prior",
            tid,
            n,
            cfg.min_matches,
        )

    attack, defense = _rebalance_attack_defense(attack, defense)
    return DixonColesModel(
        attack=attack,
        defense=defense,
        home_adv=model.home_adv,
        rho=model.rho,
        xi=model.xi,
        fit_neg_log_likelihood=model.fit_neg_log_likelihood,
        match_counts=model.match_counts,
        fit_seconds=model.fit_seconds,
    )


def fit_dixon_coles(
    matches: Sequence[Mapping[str, Any]],
    *,
    elo_by_team: Mapping[int, float] | None = None,
    dc_config: DixonColesConfig | None = None,
    apply_recency: bool = True,
) -> DixonColesModel:
    """Fit DC parameters on historical matches (pure function; no DB I/O).

    Applies the recency window by default. Returns a model with Elo shrinkage
    when ``elo_by_team`` is provided.
    """
    if not matches:
        raise ValueError("cannot fit Dixon-Coles without matches")

    cfg = dc_config or DixonColesConfig(
        xi=config.DC_XI,
        recency_years=config.DC_RECENCY_YEARS,
        min_matches=config.DC_MIN_MATCHES,
        max_goals=config.DC_MAX_GOALS,
        rho_max_iter=config.DC_RHO_MAX_ITER,
        tau_floor=config.DC_TAU_FLOOR,
    )

    t0 = time.perf_counter()
    work = (
        filter_recency_window(matches, recency_years=cfg.recency_years)
        if apply_recency
        else list(matches)
    )
    if not work:
        raise ValueError("no matches remain after recency window filter")

    team_ids = sorted(
        {int(r["home_id"]) for r in work} | {int(r["away_id"]) for r in work}
    )
    reference_date = max(r["date"] for r in work)
    arrays = _build_match_arrays(work, team_ids, reference_date=reference_date, xi=cfg.xi)

    attack, defense, gamma = _fit_poisson_glm_stage1(team_ids, arrays)
    rho, nll = _fit_rho_stage2(attack, defense, gamma, arrays, dc_config=cfg)

    attack_dict = {tid: float(attack[i]) for i, tid in enumerate(team_ids)}
    defense_dict = {tid: float(defense[i]) for i, tid in enumerate(team_ids)}
    counts = _count_team_matches(work)
    elapsed = time.perf_counter() - t0

    model = DixonColesModel(
        attack=attack_dict,
        defense=defense_dict,
        home_adv=gamma,
        rho=rho,
        xi=cfg.xi,
        fit_neg_log_likelihood=nll,
        match_counts=counts,
        fit_seconds=elapsed,
    )

    log.info(
        "Dixon-Coles fit: teams=%d matches=%d window_years=%.1f final_nll=%.2f "
        "mean_alpha=%.6f fit_seconds=%.2f",
        len(team_ids),
        len(work),
        cfg.recency_years,
        nll,
        float(np.mean(list(attack_dict.values()))),
        elapsed,
    )
    print(
        f"Dixon-Coles fit: final_nll={nll:.2f} mean_alpha={np.mean(list(attack_dict.values())):.6f} "
        f"wall_clock_seconds={elapsed:.2f}"
    )

    if elo_by_team is not None:
        model = apply_elo_shrinkage(model, elo_by_team=elo_by_team, dc_config=cfg)
    return model


def load_training_matches(
    *,
    since: date | None = None,
    db_url: str | None = None,
) -> list[dict[str, Any]]:
    """Load scored ``raw_results`` for fitting."""
    sql = """
        SELECT date, home_id, away_id, home_goals, away_goals, neutral
        FROM raw_results
        WHERE home_goals IS NOT NULL AND away_goals IS NOT NULL
    """
    params: dict[str, Any] = {}
    if since is not None:
        sql += " AND date >= :since"
        params["since"] = since
    sql += " ORDER BY date ASC, match_id ASC"
    return db.fetch_all(sql, params, db_url=db_url)


def load_latest_elo(*, db_url: str | None = None) -> dict[int, float]:
    """Most recent Elo snapshot per team_id."""
    sql = """
        SELECT DISTINCT ON (team_id) team_id, elo
        FROM team_ratings
        WHERE elo IS NOT NULL
        ORDER BY team_id, as_of_date DESC
    """
    rows = db.fetch_all(sql, db_url=db_url)
    return {int(r["team_id"]): float(r["elo"]) for r in rows}


def fit_dixon_coles_from_db(
    *,
    since: date | None = None,
    db_url: str | None = None,
) -> DixonColesModel:
    """Fit on ``raw_results`` (recency window applied) with Elo shrinkage."""
    matches = load_training_matches(since=since, db_url=db_url)
    elo = load_latest_elo(db_url=db_url)
    return fit_dixon_coles(matches, elo_by_team=elo)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    fit_dixon_coles_from_db()


if __name__ == "__main__":
    main()
