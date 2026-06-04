"""Held-out tournament backtest and blend-weight tuning (Phase 4)."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pitchedge import config
from pitchedge.eval.calibration import (
    describe_reliability_bias,
    outcomes_to_home_indicator,
    plot_reliability_diagram,
)
from pitchedge.eval.metrics import (
    expected_calibration_error,
    home_win_ece,
    mean_brier,
    mean_log_loss,
    outcome_from_goals,
    paired_log_loss_difference,
)
from pitchedge.eval.temperature_scaling import (
    apply_temperature,
    fit_temperature,
    scale_prob_list,
)
from pitchedge.eval.tournaments import HELD_OUT_TOURNAMENTS, HeldOutTournament
from pitchedge.ingest.backtest_odds import (
    build_backtest_odds_table,
    lookup_market_probs,
)
from pitchedge.ingest.history import load_history_frame
from pitchedge.ingest.team_ids import team_name_to_id
from pitchedge.model.blend import blend
from pitchedge.model.dixon_coles import fit_dixon_coles, tournament_match_probs
from pitchedge.model.elo import EloConfig, fit_elo

log = logging.getLogger(__name__)

BLEND_WEIGHT_GRID = np.linspace(0.0, 1.0, 21)


def _dataframe_to_markdown(frame: pd.DataFrame) -> str:
    """Simple markdown table without optional ``tabulate`` dependency."""
    headers = list(frame.columns)
    lines = [
        "| " + " | ".join(str(h) for h in headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in frame.itertuples(index=False):
        lines.append("| " + " | ".join(f"{v:.4f}" if isinstance(v, float) else str(v) for v in row) + " |")
    return "\n".join(lines)


@dataclass
class PredictionRow:
    """One scored match in the backtest."""

    tournament_slug: str
    tournament_label: str
    date: date
    home_team: str
    away_team: str
    home_id: int
    away_id: int
    outcome: int
    p_model: tuple[float, float, float]
    p_market: tuple[float, float, float] | None


def _frame_to_match_rows(frame: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for rec in frame.to_dict(orient="records"):
        d = rec["date"]
        if hasattr(d, "date"):
            d = d.date()
        hg = rec.get("home_score")
        ag = rec.get("away_score")
        if hg is None or ag is None or pd.isna(hg) or pd.isna(ag):
            continue
        rows.append(
            {
                "date": d,
                "home_team": str(rec["home_team"]).strip(),
                "away_team": str(rec["away_team"]).strip(),
                "home_id": team_name_to_id(str(rec["home_team"])),
                "away_id": team_name_to_id(str(rec["away_team"])),
                "home_goals": int(hg),
                "away_goals": int(ag),
                "competition": str(rec["tournament"]).strip(),
                "neutral": bool(rec.get("neutral", False)),
            }
        )
    return rows


def training_matches(
    all_matches: list[dict[str, Any]],
    tournament: HeldOutTournament,
) -> list[dict[str, Any]]:
    """Matches used to fit Elo/DC before ``tournament`` (exclude that edition)."""
    return [
        m
        for m in all_matches
        if m["date"] < tournament.start
        and not (
            m["competition"] == tournament.competition
            and tournament.start <= m["date"] <= tournament.end
        )
    ]


def eval_matches(
    all_matches: list[dict[str, Any]],
    tournament: HeldOutTournament,
) -> list[dict[str, Any]]:
    """Finished matches in the held-out tournament window."""
    return [
        m
        for m in all_matches
        if m["competition"] == tournament.competition
        and tournament.start <= m["date"] <= tournament.end
    ]


def generate_predictions(
    all_matches: list[dict[str, Any]],
    odds_df: pd.DataFrame,
) -> list[PredictionRow]:
    """Walk-forward fits and pre-match probabilities (``neutral=True``)."""
    predictions: list[PredictionRow] = []

    for tournament in HELD_OUT_TOURNAMENTS:
        train = training_matches(all_matches, tournament)
        if not train:
            log.warning("No training data before %s", tournament.label)
            continue

        elo_ratings, _ = fit_elo(train)
        dc_model = fit_dixon_coles(
            train,
            elo_by_team=elo_ratings,
            apply_recency=True,
        )

        for m in eval_matches(all_matches, tournament):
            probs = tournament_match_probs(
                dc_model,
                m["home_id"],
                m["away_id"],
                host_home=False,
            )
            p_model = (probs.p_home, probs.p_draw, probs.p_away)
            p_market = lookup_market_probs(
                odds_df,
                home_team=m["home_team"],
                away_team=m["away_team"],
                match_date=m["date"],
            )
            predictions.append(
                PredictionRow(
                    tournament_slug=tournament.slug,
                    tournament_label=tournament.label,
                    date=m["date"],
                    home_team=m["home_team"],
                    away_team=m["away_team"],
                    home_id=m["home_id"],
                    away_id=m["away_id"],
                    outcome=outcome_from_goals(m["home_goals"], m["away_goals"]),
                    p_model=p_model,
                    p_market=p_market,
                )
            )
    return predictions


def model_probs_for_row(
    row: PredictionRow,
    *,
    temperature: float = 1.0,
) -> tuple[float, float, float]:
    """Standalone Dixon-Coles probs, optionally temperature-scaled."""
    if abs(temperature - 1.0) < 1e-12:
        return row.p_model
    return apply_temperature(row.p_model, temperature)


def sweep_blend_weight(
    rows: list[PredictionRow],
    *,
    model_temperature: float = 1.0,
) -> tuple[float, pd.DataFrame]:
    """Return ``(best_w, sweep_table)`` minimizing log loss on rows with market odds."""
    scored = [r for r in rows if r.p_market is not None]
    if not scored:
        raise ValueError("cannot sweep blend weight without market odds")

    outcomes = [r.outcome for r in scored]
    records: list[dict[str, Any]] = []
    best_w = 0.0
    best_ll = float("inf")

    for w in BLEND_WEIGHT_GRID:
        blended = [
            blend(
                model_probs_for_row(r, temperature=model_temperature),
                r.p_market,
                w=w,
            )
            for r in scored  # type: ignore[arg-type]
        ]
        ll = mean_log_loss(blended, outcomes)
        br = mean_brier(blended, outcomes)
        records.append({"w": float(w), "log_loss": ll, "brier": br})
        if ll < best_ll:
            best_ll = ll
            best_w = float(w)

    return best_w, pd.DataFrame(records)


def calibration_metrics(
    rows: list[PredictionRow],
    *,
    model_temperature: float = 1.0,
) -> dict[str, Any]:
    """ECE and reliability bias for model vs market (matches with odds only)."""
    market_rows = [r for r in rows if r.p_market is not None]
    if not market_rows:
        return {}

    outcomes = [r.outcome for r in market_rows]
    model_probs = [
        model_probs_for_row(r, temperature=model_temperature) for r in market_rows
    ]
    market_probs = [r.p_market for r in market_rows]  # type: ignore[misc]
    outcomes_home = outcomes_to_home_indicator(np.array(outcomes))

    model_home = np.array([p[0] for p in model_probs])
    market_home = np.array([p[0] for p in market_probs])  # type: ignore[index]

    return {
        "model_ece": expected_calibration_error(model_probs, outcomes),
        "market_ece": expected_calibration_error(market_probs, outcomes),
        "model_home_ece": home_win_ece(model_probs, outcomes),
        "market_home_ece": home_win_ece(market_probs, outcomes),
        "model_reliability_note": describe_reliability_bias(model_home, outcomes_home),
        "market_reliability_note": describe_reliability_bias(market_home, outcomes_home),
        "paired_log_loss": paired_log_loss_difference(model_probs, market_probs, outcomes),
    }


def temperature_scaling_report(
    rows: list[PredictionRow],
) -> dict[str, Any]:
    """Fit T on held-out rows with market odds; before/after proper scores."""
    market_rows = [r for r in rows if r.p_market is not None]
    if not market_rows:
        return {"applied": False, "temperature": 1.0}

    outcomes = [r.outcome for r in market_rows]
    raw_probs = [r.p_model for r in market_rows]
    temperature = fit_temperature(raw_probs, outcomes)
    scaled_probs = scale_prob_list(raw_probs, temperature)

    before = {
        "log_loss": mean_log_loss(raw_probs, outcomes),
        "brier": mean_brier(raw_probs, outcomes),
        "ece": expected_calibration_error(raw_probs, outcomes),
    }
    after = {
        "log_loss": mean_log_loss(scaled_probs, outcomes),
        "brier": mean_brier(scaled_probs, outcomes),
        "ece": expected_calibration_error(scaled_probs, outcomes),
    }
    return {
        "applied": abs(temperature - 1.0) > 1e-4,
        "temperature": temperature,
        "before": before,
        "after": after,
    }


def aggregate_metrics(
    rows: list[PredictionRow],
    *,
    best_w: float,
    model_temperature: float = 1.0,
) -> dict[str, Any]:
    """Pool and per-tournament Brier / log loss."""
    outcomes = [r.outcome for r in rows]
    model_probs = [model_probs_for_row(r, temperature=model_temperature) for r in rows]
    market_rows = [r for r in rows if r.p_market is not None]
    market_probs = [r.p_market for r in market_rows]  # type: ignore[misc]
    market_outcomes = [r.outcome for r in market_rows]
    blended_probs = [
        blend(
            model_probs_for_row(r, temperature=model_temperature),
            r.p_market,
            w=best_w,
        )
        for r in market_rows  # type: ignore[arg-type]
    ]

    by_tournament: dict[str, dict[str, Any]] = {}
    for t in HELD_OUT_TOURNAMENTS:
        sub = [r for r in rows if r.tournament_slug == t.slug]
        if not sub:
            continue
        sub_out = [r.outcome for r in sub]
        sub_model = [
            model_probs_for_row(r, temperature=model_temperature) for r in sub
        ]
        sub_mkt = [r for r in sub if r.p_market is not None]
        entry: dict[str, Any] = {
            "n_matches": len(sub),
            "model_brier": mean_brier(sub_model, sub_out),
            "model_log_loss": mean_log_loss(sub_model, sub_out),
            "n_with_market": len(sub_mkt),
        }
        if sub_mkt:
            sm = [r.p_market for r in sub_mkt]  # type: ignore[misc]
            so = [r.outcome for r in sub_mkt]
            entry["market_brier"] = mean_brier(sm, so)
            entry["market_log_loss"] = mean_log_loss(sm, so)
            sb = [
                blend(
                    model_probs_for_row(r, temperature=model_temperature),
                    r.p_market,
                    w=best_w,
                )
                for r in sub_mkt  # type: ignore[arg-type]
            ]
            entry["blend_brier"] = mean_brier(sb, so)
            entry["blend_log_loss"] = mean_log_loss(sb, so)
        by_tournament[t.slug] = entry

    model_probs_mkt = [
        model_probs_for_row(r, temperature=model_temperature) for r in market_rows
    ]

    return {
        "n_total": len(rows),
        "n_with_market": len(market_rows),
        "model_brier": mean_brier(model_probs, outcomes),
        "model_log_loss": mean_log_loss(model_probs, outcomes),
        "model_brier_pooled": mean_brier(model_probs_mkt, market_outcomes)
        if market_rows
        else None,
        "model_log_loss_pooled": mean_log_loss(model_probs_mkt, market_outcomes)
        if market_rows
        else None,
        "market_brier": mean_brier(market_probs, market_outcomes) if market_rows else None,
        "market_log_loss": mean_log_loss(market_probs, market_outcomes)
        if market_rows
        else None,
        "blend_brier": mean_brier(blended_probs, market_outcomes) if market_rows else None,
        "blend_log_loss": mean_log_loss(blended_probs, market_outcomes)
        if market_rows
        else None,
        "by_tournament": by_tournament,
        "best_w": best_w,
    }


def write_backtest_report(
    metrics: dict[str, Any],
    sweep_df: pd.DataFrame,
    *,
    calibration: dict[str, Any],
    temp_report: dict[str, Any],
    scaled_sweep_df: pd.DataFrame | None = None,
    odds_meta: dict[str, Any] | None = None,
    output_path: str | Path,
) -> Path:
    """Write ``backtest_report.md`` with explicit model vs market comparison."""
    path = Path(output_path)
    m_brier = metrics.get("model_brier_pooled") or metrics["model_brier"]
    k_brier = metrics["market_brier"]
    m_ll = metrics.get("model_log_loss_pooled") or metrics["model_log_loss"]
    k_ll = metrics["market_log_loss"]
    m_ll_all = metrics["model_log_loss"]
    b_brier = metrics["blend_brier"]
    b_ll = metrics["blend_log_loss"]
    w = metrics["best_w"]

    model_ll_lower = k_ll is not None and m_ll < k_ll
    model_brier_lower = k_brier is not None and m_brier < k_brier

    lines = [
        "# PitchEdge Phase 4 backtest report",
        "",
        "Held-out tournaments: FIFA World Cup 2018 & 2022, UEFA Euro 2024, "
        "Copa América 2024. Dixon-Coles predictions use **`neutral=True`** "
        "(no home-advantage γ) on all tournament matches.",
        "",
        "## Data sources",
        "",
        "- **Results:** Kaggle international results (`raw_results` / HISTORY_CSV).",
        "- **Market odds:** football-data.co.uk average H/D/A (`H-Avg`, `D-Avg`, `A-Avg`) "
        "for World Cup 2018 & 2022; Euro 2024 & Copa 2024 via the-odds-api historical "
        "endpoint (paid plan) or `data/backtest/euro_copa_odds_cache.csv`.",
        "",
    ]
    if odds_meta:
        blocked = odds_meta.get("odds_api_historical_blocked")
        lines.extend(
            [
                "### Market odds coverage (this run)",
                "",
                f"- Euro/Copa matches needing odds: **{odds_meta.get('euro_copa_matches_requested', 0)}**",
                f"- Rows from cache file: **{odds_meta.get('euro_copa_cache_rows', 0)}**",
                f"- Rows from Odds API historical: **{odds_meta.get('odds_api_rows', 0)}**",
                f"- Pooled matches with market odds: **{metrics['n_with_market']}** / "
                f"{metrics['n_total']} total held-out matches",
            ]
        )
        if blocked:
            lines.append(
                "- **Odds API:** `ODDS_API_KEY` is valid for live odds but **historical "
                "odds returned 401** (`HISTORICAL_UNAVAILABLE_ON_FREE_USAGE_PLAN`). "
                "Upgrade at [the-odds-api.com](https://the-odds-api.com) or populate "
                f"`{config.BACKTEST_EURO_COPA_CACHE}`, then re-run `make backtest`."
            )
        lines.append("")
    lines.extend(
        [
        "## Blend weight sweep",
        "",
        f"Grid: w ∈ [0, 1] (step {BLEND_WEIGHT_GRID[1] - BLEND_WEIGHT_GRID[0]:.2f}). "
        f"**Chosen `BLEND_W` = {w:.2f}** (minimizes pooled log loss on matches with "
        "market odds).",
        "",
        "### Sweep (log loss vs w)",
        "",
        _dataframe_to_markdown(sweep_df),
        "",
        "## Pooled scores (matches with market odds)",
        "",
        "| Source | Brier ↓ | Log loss ↓ | n |",
        "|--------|---------|------------|---|",
        f"| Dixon-Coles model | {m_brier:.4f} | {m_ll:.4f} | {metrics['n_with_market']} |",
        ]
    )
    if k_brier is not None:
        lines.append(
            f"| De-vigged market | {k_brier:.4f} | {k_ll:.4f} | {metrics['n_with_market']} |"
        )
    if b_brier is not None:
        lines.append(
            f"| Blend (w={w:.2f}) | {b_brier:.4f} | {b_ll:.4f} | {metrics['n_with_market']} |"
        )

    lines.extend(
        [
            "",
            f"**All tournament matches (model only):** n={metrics['n_total']}, "
            f"Brier={metrics['model_brier']:.4f}, log loss={m_ll_all:.4f}. "
            f"Pooled comparison above uses the same n={metrics['n_with_market']} "
            f"matches that have market odds.",
            "",
            "## Model vs market on the held-out sample",
            "",
        ]
    )
    if k_ll is None:
        lines.append(
            "Market odds were not available for the pooled sample; cannot compare."
        )
    elif model_ll_lower:
        lines.append(
            f"On pooled log loss, the **model scores lower than the de-vigged "
            f"market** ({m_ll:.4f} vs {k_ll:.4f}) on this held-out sample. The "
            f"blend at w={w:.2f} scores {b_ll:.4f}. We report the metric; we do not "
            f"claim a forward-looking edge — the market remains the calibration anchor."
        )
    else:
        lines.append(
            f"On pooled log loss, the **model scores higher than the de-vigged "
            f"market** ({m_ll:.4f} vs {k_ll:.4f}). This is expected and acceptable — "
            f"the de-vigged market remains the calibration anchor. The blend at "
            f"w={w:.2f} scores {b_ll:.4f} (target: stay close to the market)."
        )

    if k_brier is not None:
        if model_brier_lower:
            lines.append(
                f"\nOn pooled Brier, the model scores lower than the market "
                f"({m_brier:.4f} vs {k_brier:.4f})."
            )
        else:
            lines.append(
                f"\nOn pooled Brier, the model is **not** lower than the market "
                f"({m_brier:.4f} vs {k_brier:.4f})."
            )

    lines.extend(["", "## Per tournament", ""])
    for t in HELD_OUT_TOURNAMENTS:
        entry = metrics["by_tournament"].get(t.slug)
        if not entry:
            continue
        lines.append(f"### {t.label}")
        lines.append(f"- Matches: {entry['n_matches']}")
        lines.append(
            f"- Model: Brier={entry['model_brier']:.4f}, "
            f"log loss={entry['model_log_loss']:.4f}"
        )
        if entry.get("n_with_market", 0) > 0:
            lines.append(
                f"- Market (n={entry['n_with_market']}): Brier={entry['market_brier']:.4f}, "
                f"log loss={entry['market_log_loss']:.4f}"
            )
            lines.append(
                f"- Blend w={w:.2f}: Brier={entry['blend_brier']:.4f}, "
                f"log loss={entry['blend_log_loss']:.4f}"
            )
        else:
            lines.append("- Market: no odds joined for this edition.")
        lines.append("")

    lines.extend(["", "## Calibration vs accuracy", ""])
    if calibration:
        pl = calibration["paired_log_loss"]
        lines.append(
            "Proper scores (log loss, Brier) measure **accuracy** (how often we pick "
            "the right outcome). **ECE** measures **calibration** (whether stated "
            "probabilities match observed frequencies). A model can be accurate but "
            "miscalibrated, or vice versa."
        )
        lines.append("")
        lines.append(
            f"| Source | ECE (max-conf) ↓ | Home-win ECE ↓ | n (with odds) |"
        )
        lines.append("|--------|------------------|----------------|---------------|")
        lines.append(
            f"| Dixon-Coles model | {calibration['model_ece']:.4f} | "
            f"{calibration['model_home_ece']:.4f} | {metrics['n_with_market']} |"
        )
        lines.append(
            f"| De-vigged market | {calibration['market_ece']:.4f} | "
            f"{calibration['market_home_ece']:.4f} | {metrics['n_with_market']} |"
        )
        lines.append("")
        lines.append(f"**Model reliability (home win):** {calibration['model_reliability_note']}.")
        lines.append(
            f"**Market reliability (home win):** {calibration['market_reliability_note']}."
        )
        lines.append("")
        lines.append(
            f"### Log-loss difference (paired, n={metrics['n_with_market']} with odds)"
        )
        lines.append(
            f"- Mean (model − market): **{pl['mean_diff']:.4f}** "
            f"(pooled means: model {m_ll:.4f}, market {k_ll:.4f})"
        )
        lines.append(f"- Std of paired differences: {pl['std_diff']:.4f}")
        lines.append(f"- Standard error: **{pl['se_diff']:.4f}**")
        within_one_se = abs(pl["mean_diff"]) <= pl["se_diff"] if pl["se_diff"] > 0 else False
        lines.append(
            f"- |mean| within one SE? **{'yes' if within_one_se else 'no'}** "
            f"(difference is {'not ' if within_one_se else ''}larger than sampling noise "
            f"at the 1-SE threshold; this does not claim statistical significance)."
        )

    lines.extend(["", "## Temperature scaling", ""])
    if temp_report.get("before"):
        t = temp_report["temperature"]
        before = temp_report["before"]
        after = temp_report["after"]
        lines.append(
            f"Single-parameter T fit on held-out matches with market odds "
            f"(minimize log loss). **Fitted T = {t:.3f}**."
        )
        lines.append("")
        lines.append("| Metric | Before | After T-scaling |")
        lines.append("|--------|--------|-----------------|")
        lines.append(
            f"| Log loss | {before['log_loss']:.4f} | {after['log_loss']:.4f} |"
        )
        lines.append(f"| Brier | {before['brier']:.4f} | {after['brier']:.4f} |")
        lines.append(f"| ECE | {before['ece']:.4f} | {after['ece']:.4f} |")
        if temp_report.get("applied") and scaled_sweep_df is not None:
            lines.append("")
            lines.append(
                f"Blend sweep re-run on **temperature-scaled** standalone model "
                f"(T={t:.3f}). Chosen `BLEND_W` = {metrics.get('scaled_best_w', w):.2f}."
            )
            lines.append("")
            lines.append("### Sweep after temperature scaling")
            lines.append("")
            lines.append(_dataframe_to_markdown(scaled_sweep_df))

    lines.extend(
        [
            "",
            "## Reliability diagrams",
            "",
            "See `reports/reliability_diagram.png` (model vs market, matches with odds) "
            "and `reports/reliability_model_all.png` (model, all held-out matches). "
            "Wilson bands on bin frequencies. A curve hugging the diagonal is well "
            "calibrated; systematic offset indicates over- or under-confidence.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def run_backtest(
    *,
    history_csv: str | Path | None = None,
    odds_xlsx: str | Path | None = None,
    report_path: str | Path | None = None,
    reliability_png: str | Path | None = None,
    use_odds_api: bool = True,
) -> dict[str, Any]:
    """Execute full Phase 4 backtest; returns metrics dict and writes artifacts."""
    csv_path = Path(history_csv or config.HISTORY_CSV_PATH)
    frame = load_history_frame(csv_path)
    all_matches = _frame_to_match_rows(frame)

    euro_copa_payload: list[dict[str, Any]] = []
    for tournament in HELD_OUT_TOURNAMENTS:
        if not tournament.odds_api_sport_key:
            continue
        for m in eval_matches(all_matches, tournament):
            euro_copa_payload.append(
                {
                    "tournament_slug": tournament.slug,
                    "date": m["date"],
                    "home_team": m["home_team"],
                    "away_team": m["away_team"],
                }
            )

    odds_df, odds_meta = build_backtest_odds_table(
        xlsx_path=odds_xlsx,
        use_odds_api=use_odds_api,
        euro_copa_matches=euro_copa_payload,
    )
    log.info(
        "Loaded %d odds rows for backtest joins (Euro/Copa cache=%d, odds-api=%d)",
        len(odds_df),
        odds_meta.get("euro_copa_cache_rows", 0),
        odds_meta.get("odds_api_rows", 0),
    )

    predictions = generate_predictions(all_matches, odds_df)
    best_w, sweep_df = sweep_blend_weight(predictions)
    metrics = aggregate_metrics(predictions, best_w=best_w)

    calibration = calibration_metrics(predictions)
    temp_report = temperature_scaling_report(predictions)
    model_temperature = float(temp_report.get("temperature", 1.0))
    scaled_sweep_df: pd.DataFrame | None = None
    if temp_report.get("applied"):
        scaled_best_w, scaled_sweep_df = sweep_blend_weight(
            predictions,
            model_temperature=model_temperature,
        )
        scaled_metrics = aggregate_metrics(
            predictions,
            best_w=scaled_best_w,
            model_temperature=model_temperature,
        )
        metrics["scaled_best_w"] = scaled_best_w
        metrics["scaled_model_log_loss"] = scaled_metrics["model_log_loss"]
        metrics["scaled_model_brier"] = scaled_metrics["model_brier"]
        metrics["scaled_blend_log_loss"] = scaled_metrics["blend_log_loss"]
        calibration_scaled = calibration_metrics(
            predictions,
            model_temperature=model_temperature,
        )
        metrics["calibration_scaled"] = calibration_scaled
    else:
        model_temperature = 1.0

    metrics["model_temperature"] = model_temperature
    metrics["calibration"] = calibration
    metrics["temperature_report"] = temp_report
    metrics["odds_meta"] = odds_meta

    report = Path(report_path or config.BACKTEST_REPORT_PATH)
    write_backtest_report(
        metrics,
        sweep_df,
        calibration=calibration,
        temp_report=temp_report,
        scaled_sweep_df=scaled_sweep_df,
        odds_meta=odds_meta,
        output_path=report,
    )

    rel_path = Path(reliability_png or config.BACKTEST_RELIABILITY_PNG)
    model_home = np.array(
        [
            model_probs_for_row(r, temperature=model_temperature)[0]
            for r in predictions
        ]
    )
    outcomes = np.array([r.outcome for r in predictions])
    market_rows = [r for r in predictions if r.p_market is not None]
    if market_rows:
        market_home = np.array([r.p_market[0] for r in market_rows])  # type: ignore[index]
        model_home_mkt = np.array(
            [
                model_probs_for_row(r, temperature=model_temperature)[0]
                for r in market_rows
            ]
        )
        market_outcomes = np.array([r.outcome for r in market_rows])
        plot_reliability_diagram(
            model_pred_home=model_home_mkt,
            market_pred_home=market_home,
            outcomes_home=outcomes_to_home_indicator(market_outcomes),
            output_path=rel_path,
            title="Home win reliability (matches with market odds)",
        )
    plot_reliability_diagram(
        model_pred_home=model_home,
        market_pred_home=None,
        outcomes_home=outcomes_to_home_indicator(outcomes),
        output_path=rel_path.with_name("reliability_model_all.png"),
        title="Home win reliability — model (all held-out matches)",
    )

    metrics["sweep_table"] = sweep_df
    metrics["predictions"] = predictions
    return metrics


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    result = run_backtest()
    print(f"Chosen BLEND_W={result['best_w']:.2f}")
    if result.get("model_log_loss_pooled") is not None:
        print(
            f"Pooled (n={result['n_with_market']}): model LL="
            f"{result['model_log_loss_pooled']:.4f}  market LL={result['market_log_loss']:.4f}"
        )
    print(f"Model log loss (all matches)={result['model_log_loss']:.4f}")
    cal = result.get("calibration", {})
    if cal:
        print(f"Model ECE={cal['model_ece']:.4f}  Market ECE={cal['market_ece']:.4f}")
        pl = cal["paired_log_loss"]
        print(
            f"Paired LL diff={pl['mean_diff']:.4f}  SE={pl['se_diff']:.4f}  "
            f"within_1SE={abs(pl['mean_diff']) <= pl['se_diff']}"
        )
    tr = result.get("temperature_report", {})
    if tr.get("before"):
        print(f"Temperature T={tr['temperature']:.3f}")
        print(
            f"After scaling: LL={tr['after']['log_loss']:.4f}  "
            f"Brier={tr['after']['brier']:.4f}  ECE={tr['after']['ece']:.4f}"
        )
    print(f"Report written to {config.BACKTEST_REPORT_PATH}")


if __name__ == "__main__":
    main()
