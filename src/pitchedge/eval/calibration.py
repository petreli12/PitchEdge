"""Reliability diagrams with Wilson confidence bands."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from pitchedge.eval.metrics import OUTCOME_HOME


def wilson_interval(
    successes: int,
    trials: int,
    z: float = 1.96,
) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion (in [0, 1])."""
    if trials <= 0:
        return (0.0, 0.0)
    phat = successes / trials
    denom = 1.0 + z**2 / trials
    center = (phat + z**2 / (2 * trials)) / denom
    margin = (
        z
        * np.sqrt((phat * (1 - phat) + z**2 / (4 * trials)) / trials)
        / denom
    )
    return (float(max(0.0, center - margin)), float(min(1.0, center + margin)))


def reliability_curve(
    predicted: np.ndarray,
    observed: np.ndarray,
    *,
    n_bins: int = 10,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Bin ``predicted`` probabilities and return centers, rates, lo, hi."""
    predicted = np.asarray(predicted, dtype=float)
    observed = np.asarray(observed, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    centers: list[float] = []
    rates: list[float] = []
    lo: list[float] = []
    hi: list[float] = []

    for i in range(n_bins):
        lo_edge = edges[i]
        hi_edge = edges[i + 1]
        if i < n_bins - 1:
            mask = (predicted >= lo_edge) & (predicted < hi_edge)
        else:
            mask = (predicted >= lo_edge) & (predicted <= hi_edge)
        n = int(mask.sum())
        if n == 0:
            continue
        centers.append(float(predicted[mask].mean()))
        successes = int(observed[mask].sum())
        rates.append(successes / n)
        lo_b, hi_b = wilson_interval(successes, n)
        lo.append(lo_b)
        hi.append(hi_b)

    return (
        np.array(centers),
        np.array(rates),
        np.array(lo),
        np.array(hi),
    )


def plot_reliability_diagram(
    *,
    model_pred_home: np.ndarray,
    market_pred_home: np.ndarray | None,
    outcomes_home: np.ndarray,
    output_path: str | Path,
    title: str = "Home win reliability (held-out tournaments)",
) -> Path:
    """Save reliability diagram PNG for model vs market on the same axes."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Perfect calibration")

    mc, mr, mlo, mhi = reliability_curve(model_pred_home, outcomes_home)
    ax.plot(mc, mr, "o-", color="#2563eb", label="Dixon-Coles model")
    ax.fill_between(mc, mlo, mhi, color="#2563eb", alpha=0.2)

    if market_pred_home is not None and len(market_pred_home) > 0:
        kc, kr, klo, khi = reliability_curve(market_pred_home, outcomes_home)
        ax.plot(kc, kr, "s-", color="#dc2626", label="De-vigged market")
        ax.fill_between(kc, klo, khi, color="#dc2626", alpha=0.2)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Predicted P(home win)")
    ax.set_ylabel("Observed home-win frequency")
    ax.set_title(title)
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def outcomes_to_home_indicator(outcomes: np.ndarray) -> np.ndarray:
    """Binary indicator: 1 if home win."""
    return (np.asarray(outcomes) == OUTCOME_HOME).astype(float)


def describe_reliability_bias(
    predicted_home: np.ndarray,
    outcomes_home: np.ndarray,
    *,
    n_bins: int = 10,
) -> str:
    """Summarize whether the curve sits above/below the diagonal (home-win ECE).

    Observed rate above predicted in a bin → underconfident; below → overconfident.
    """
    centers, rates, _, _ = reliability_curve(predicted_home, outcomes_home, n_bins=n_bins)
    if len(centers) == 0:
        return "insufficient data to assess calibration bias"

    gaps = rates - centers
    mean_gap = float(np.mean(gaps))
    if mean_gap > 0.02:
        return (
            "systematically underconfident on home-win probability "
            "(observed frequency runs above the diagonal in most bins)"
        )
    if mean_gap < -0.02:
        return (
            "systematically overconfident on home-win probability "
            "(observed frequency runs below the diagonal in most bins)"
        )
    return (
        "reasonably aligned with the diagonal on home-win probability "
        "(no large systematic over/under-confidence)"
    )
