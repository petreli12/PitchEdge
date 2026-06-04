"""PitchEdge public dashboard (Streamlit).

Read-only display over persisted DB receipts — no probability recomputation.
Run: ``uv run streamlit run src/pitchedge/app.py`` or ``make dashboard``.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

# Repo-root `.env` (Streamlit cwd is often `src/pitchedge/`).
load_dotenv(Path(__file__).resolve().parents[2] / ".env", override=False)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from pitchedge import config
from pitchedge.content.calibration_tracker import model_vs_market_gap
from pitchedge.dashboard import data as ds
from pitchedge.dashboard.subscribers import (
    capture_subscriber_email,
    post_subscriber_email as capture_subscriber_email_http,
)
from pitchedge.eval.calibration import reliability_curve

UTC = timezone.utc

SMALL_SAMPLE_CAVEAT = (
    "Receipts update match-by-match as results land. Early in the tournament the "
    "sample is small: calibration bins are thin and Wilson bands are wide. We report "
    "proper scores (Brier, log loss) and reliability either way. Tournament title "
    "odds from the sim are not scored here (one outcome per team)."
)

MOBILE_CSS = """
<style>
.pe-nav { margin-bottom: 1rem; }
@media (max-width: 768px) {
  .block-container { padding-left: 0.75rem; padding-right: 0.75rem; }
  [data-testid="stDataFrame"] { font-size: 0.78rem; }
  [data-testid="stMetric"] label { font-size: 0.75rem; }
  h1 { font-size: 1.45rem; }
  h2 { font-size: 1.15rem; }
}
.pe-model { font-size: 1.35rem; font-weight: 700; color: #3b82f6; }
</style>
"""

def _pages() -> tuple[str, ...]:
    """Nav sections. The launch board renders public-safe copy in snapshot mode."""
    return ("Predictions", "Launch board", "About")


def _pct(x: float | None) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    return f"{100.0 * float(x):.1f}%"


def _kickoff_label(ko: Any) -> str:
    if hasattr(ko, "strftime"):
        return ko.strftime("%b %d %H:%M UTC")
    return str(ko)


def _reliability_figure(series: dict[str, Any]) -> plt.Figure:
    """Home-win reliability diagram (model vs market, same axes)."""
    model_h = series["model_home"]
    market_h = series["market_home"]
    outcomes_h = series["outcomes_home"]
    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Perfect calibration")
    if len(model_h) > 0:
        mc, mr, mlo, mhi = reliability_curve(model_h, outcomes_h)
        ax.plot(mc, mr, "o-", color="#2563eb", label="PitchEdge model", markersize=6)
        ax.fill_between(mc, mlo, mhi, color="#2563eb", alpha=0.2)
    if len(market_h) > 0:
        kc, kr, klo, khi = reliability_curve(market_h, outcomes_h)
        ax.plot(kc, kr, "s-", color="#dc2626", label="De-vigged market", markersize=6)
        ax.fill_between(kc, klo, khi, color="#dc2626", alpha=0.2)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Predicted P(home win)")
    ax.set_ylabel("Observed home-win rate")
    ax.set_title("Home-win reliability (scored match receipts)")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def _render_subscribe_form(submit_fn) -> None:
    """Render the email form + persistent success/error state.

    ``submit_fn(email) -> (ok, message)`` is the storage strategy (DB write in
    live mode, HTTP POST to an external endpoint on the public deploy).
    """
    st.subheader("Stay in the loop")
    with st.form("subscribe", clear_on_submit=False):
        email = st.text_input("Email", placeholder="you@example.com")
        submitted = st.form_submit_button("Get updates")
        if submitted:
            ok, msg = submit_fn(email)
            st.session_state["subscribe_ok"] = ok
            if ok:
                st.session_state["subscribe_msg"] = msg
                st.session_state.pop("subscribe_error", None)
            else:
                st.session_state.pop("subscribe_msg", None)
                st.session_state["subscribe_error"] = msg

    if st.session_state.get("subscribe_ok"):
        st.success(st.session_state.get("subscribe_msg", "Thanks — you're on the list."))
    elif st.session_state.get("subscribe_error"):
        st.error(st.session_state["subscribe_error"])


def render_landing(sim_rows: list[dict[str, Any]] | None) -> None:
    """Public About page: copy, title-odds proof, email capture, optional Telegram."""
    st.title("PitchEdge")
    st.markdown(
        """
PitchEdge is a **calibrated** World Cup prediction engine that **shows its receipts**.

We publish standalone Dixon-Coles probabilities, log them before kickoff, and score
every finished match with proper rules (Brier, log loss). The de-vigged market is our
benchmark — not something we claim to defeat. Calibration matters more than a vanity
win rate.

**Independent of the market:** our model and receipts track *our* numbers; market
rows are shown alongside for comparison only.
        """
    )

    st.subheader("Tournament title odds")
    st.caption(
        "From our latest Monte Carlo sim (logged probabilities, not bookmaker outrights)."
    )
    if sim_rows:
        top_six = sim_rows[:6]
        proof = pd.DataFrame(
            [
                {"Team": row["name"], "P(win title)": float(row["p_win"])}
                for row in top_six
            ]
        )
        st.dataframe(
            proof,
            use_container_width=True,
            hide_index=True,
            column_config={
                "P(win title)": st.column_config.NumberColumn(
                    format="percent",
                    help="Model sim probability to win the tournament",
                ),
            },
        )
    else:
        st.markdown(
            "Title odds will appear here once tournament simulations are published."
        )

    form_url = config.SUBSCRIBE_FORM_URL.strip()
    post_url = config.SUBSCRIBE_POST_URL.strip()
    join_url = config.TELEGRAM_JOIN_URL.strip()
    snapshot = ds.is_snapshot_mode()

    # "Stay in the loop" only renders when there is a real capture path. We never
    # show an in-app form that would silently drop submissions:
    #   - live DB mode      -> in-app form writing to `subscribers`
    #   - snapshot + POST URL-> in-app form submitting over HTTP (success state kept)
    #   - snapshot + form URL-> a button linking out to an external form
    #   - otherwise          -> nothing (Telegram CTA below, if configured)
    if not snapshot:
        _render_subscribe_form(lambda e: capture_subscriber_email(e))
    elif post_url:
        _render_subscribe_form(
            lambda e: capture_subscriber_email_http(
                e, post_url=post_url, field=config.SUBSCRIBE_EMAIL_FIELD
            )
        )
    elif form_url:
        st.subheader("Stay in the loop")
        st.link_button("Get updates", form_url, use_container_width=True)

    if join_url:
        st.subheader("Join on Telegram")
        st.link_button("Join on Telegram", join_url, use_container_width=True)


def render_upcoming_tab(rows: list[dict[str, Any]]) -> None:
    st.caption(
        "Probabilities are read from logged `match_predictions` (append-only receipts). "
        "**PitchEdge model** is the published, scored source — not the blend."
    )
    if not rows:
        st.info("No upcoming fixtures with predictions yet. Run `make predict` after ingest.")
        return

    for row in rows:
        home = row["home"]
        away = row["away"]
        stage = row.get("stage") or ""
        group = row.get("group_label") or ""
        label = f"{home} vs {away}"
        if group:
            label += f" (Group {group})"
        with st.expander(f"{label} — {_kickoff_label(row['kickoff_utc'])}"):
            st.markdown(
                f'<p class="pe-model">Model: H {_pct(row.get("model_p_home"))} · '
                f"D {_pct(row.get('model_p_draw'))} · "
                f"A {_pct(row.get('model_p_away'))}</p>",
                unsafe_allow_html=True,
            )
            if row.get("model_predicted_utc"):
                st.caption(f"Logged { _kickoff_label(row['model_predicted_utc']) }")
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**Market (de-vigged)**")
                if row.get("market_p_home") is not None:
                    st.write(
                        f"H {_pct(row['market_p_home'])} · "
                        f"D {_pct(row['market_p_draw'])} · "
                        f"A {_pct(row['market_p_away'])}"
                    )
                else:
                    st.write("No market row logged for this fixture.")
            with c2:
                st.markdown("**Blend** (config `BLEND_W`)")
                if row.get("blend_p_home") is not None:
                    st.write(
                        f"H {_pct(row['blend_p_home'])} · "
                        f"D {_pct(row['blend_p_draw'])} · "
                        f"A {_pct(row['blend_p_away'])}"
                    )
                else:
                    st.write("—")


def render_tournament_odds_tab(sim_rows: list[dict[str, Any]]) -> None:
    if not sim_rows:
        st.info("No `sim_results` yet. Run `make sim` after fitting the model.")
        return
    batch = sim_rows[0].get("run_batch_utc")
    n_sims = sim_rows[0].get("n_sims")
    st.caption(f"Latest sim batch: {_kickoff_label(batch)} · n_sims={n_sims}")

    st.warning(
        "Title and advancement percentages come from Monte Carlo sims of our model. "
        "The distribution is intentionally **flatter** than typical bookmaker outrights: "
        "we aim for calibration, not the sharpest single-number headline. "
        "The market futures book is often more concentrated on favorites."
    )

    df = pd.DataFrame(
        [
            {
                "Team": r["name"],
                "Group": r.get("group_label") or "",
                "P(win)": float(r["p_win"]),
                "P(final)": float(r["p_final"]),
                "P(SF)": float(r["p_sf"]),
                "P(QF)": float(r["p_qf"]),
                "P(R16)": float(r["p_r16"]),
                "P(advance)": float(r["p_advance_group"]),
            }
            for r in sim_rows
        ]
    )
    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "P(win)": st.column_config.NumberColumn(format="percent", help="Title"),
            "P(final)": st.column_config.NumberColumn(format="percent"),
            "P(SF)": st.column_config.NumberColumn(format="percent"),
            "P(QF)": st.column_config.NumberColumn(format="percent"),
            "P(R16)": st.column_config.NumberColumn(format="percent"),
            "P(advance)": st.column_config.NumberColumn(format="percent"),
        },
    )


def render_advancement_tab(sim_rows: list[dict[str, Any]]) -> None:
    if not sim_rows:
        st.info("No simulation results to display.")
        return
    st.caption(
        "From the latest `sim_results` batch. **P(top 2 in group)** is the chance of "
        "finishing 1st or 2nd in the group table only (the four teams in each group "
        "sum to 200%). **P(R16)** is the chance of reaching the round of 32, including "
        "via the best-third-place route — usually lower than P(top 2)."
    )
    by_group: dict[str, list[dict[str, Any]]] = {}
    for r in sim_rows:
        g = r.get("group_label") or "—"
        by_group.setdefault(str(g), []).append(r)

    for group in sorted(by_group.keys()):
        teams = sorted(by_group[group], key=lambda x: -float(x["p_advance_group"]))
        st.markdown(f"### Group {group}")
        gdf = pd.DataFrame(
            [
                {
                    "Team": t["name"],
                    "P(top 2 in group)": float(t["p_advance_group"]),
                    "P(R16)": float(t["p_r16"]),
                    "P(SF)": float(t["p_sf"]),
                    "P(final)": float(t["p_final"]),
                    "P(win)": float(t["p_win"]),
                }
                for t in teams
            ]
        )
        st.dataframe(
            gdf,
            use_container_width=True,
            hide_index=True,
            column_config={
                "P(top 2 in group)": st.column_config.NumberColumn(format="percent"),
                "P(R16)": st.column_config.NumberColumn(format="percent"),
                "P(SF)": st.column_config.NumberColumn(format="percent"),
                "P(final)": st.column_config.NumberColumn(format="percent"),
                "P(win)": st.column_config.NumberColumn(format="percent"),
            },
        )


def render_calibration_tab() -> None:
    st.markdown("### The receipts")
    st.info(SMALL_SAMPLE_CAVEAT)

    try:
        summary = ds.calibration_summary()
    except Exception as exc:
        st.error(f"Could not load calibration summary: {exc}")
        return

    model = summary.get("model")
    market = summary.get("market")
    if not model and not market:
        st.warning(
            "No scored match predictions yet. After the tournament starts, run "
            "`make score` on finished fixtures."
        )
    else:
        cols = st.columns(2)
        for col, src in zip(cols, (model, market)):
            if not src:
                col.metric("—", "No data")
                continue
            col.markdown(f"**{src.source.title()}** (n={src.n})")
            col.metric("Mean Brier", f"{src.mean_brier:.4f}" if src.n else "—")
            col.metric("Mean log loss", f"{src.mean_log_loss:.4f}" if src.n else "—")
            col.metric("ECE (max-conf)", f"{src.ece:.4f}" if src.n else "—")

        gaps = model_vs_market_gap(summary)
        if gaps.get("brier_gap") is not None:
            st.caption(
                "Gaps are model minus market on the same scored matches. "
                "Lower Brier/log loss is better accuracy; lower ECE is better calibration."
            )

    try:
        series = ds.scored_home_win_series()
    except Exception as exc:
        st.error(f"Could not load scored series for diagram: {exc}")
        return

    n = int(series.get("n_fixtures", 0))
    if n == 0:
        st.warning("Reliability diagram needs at least one scored fixture with model and market rows.")
        return

    st.caption(f"Diagram uses {n} finished fixtures with paired model and market receipts.")
    fig = _reliability_figure(series)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


def render_launch_board(
    sim_rows: list[dict[str, Any]],
    receipt: dict[str, Any],
) -> None:
    public = ds.is_snapshot_mode()
    st.title("Pre-tournament launch board")
    if public:
        st.caption(
            "Every team's title odds and our biggest model-vs-market disagreements. "
            "All match probabilities here were logged before kickoff (append-only "
            "receipts) — a locked baseline, not a live recompute. The market is shown "
            "for comparison; it is our benchmark, not something we claim to beat."
        )
    else:
        st.caption(
            "Internal preview. Every match probability below must already exist in "
            "`match_predictions` from `make predict` (pre-kickoff guard) before "
            "June 11. This board is the locked receipts baseline, not a live recompute."
        )

    # Internal QA only (prediction coverage + missing odds): hidden on the public deploy.
    if not public:
        upcoming = int(receipt.get("upcoming_fixtures") or 0)
        with_model = int(receipt.get("fixtures_with_model_pred") or 0)
        if upcoming == 0:
            st.error("No upcoming fixtures in DB.")
        elif with_model < upcoming:
            st.error(
                f"Only {with_model}/{upcoming} upcoming fixtures have a model prediction. "
                "Run `make predict` until all are logged before launch."
            )
        else:
            st.success(
                f"All {upcoming} upcoming fixtures have model predictions logged "
                f"(batch window {_kickoff_label(receipt.get('latest_model_utc'))})."
            )

        missing_odds = ds.fixtures_missing_odds()
        if missing_odds:
            st.warning(
                f"{len(missing_odds)} scheduled fixture(s) still have no `odds_snapshots` row."
            )
            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "Fixture": f"{r['home']} vs {r['away']}",
                            "Kickoff": _kickoff_label(r["kickoff_utc"]),
                            "fixture_id": r["fixture_id"],
                        }
                        for r in missing_odds
                    ]
                ),
                use_container_width=True,
                hide_index=True,
            )

    st.subheader("Title odds (latest sim)")
    if sim_rows:
        title_df = pd.DataFrame(
            [{"Team": r["name"], "P(win)": float(r["p_win"])} for r in sim_rows]
        )
        st.dataframe(
            title_df,
            use_container_width=True,
            hide_index=True,
            column_config={"P(win)": st.column_config.NumberColumn(format="percent")},
        )
    elif not public:
        st.info("Run `make sim` to populate title odds.")
    else:
        st.info("Title odds will appear once simulations are published.")

    st.subheader("Top model vs market disagreements")
    st.caption(
        "Fixtures where our standalone model differs most from the de-vigged market."
    )
    try:
        ranked = ds.ranked_disagreements(limit=8)
    except Exception as exc:
        st.error("Could not load model-vs-market disagreements.")
        if not public:
            with st.expander("Technical details"):
                st.code(str(exc))
        ranked = []

    if not ranked:
        if public:
            st.info("No model-vs-market disagreements to show yet.")
        else:
            st.info(
                "No paired model and market predictions found for upcoming fixtures. "
                "Run `make predict` after odds are ingested."
            )
    else:
        for d in ranked:
            c = d.candidate
            line = f"**{c.home} vs {c.away}** ({c.stage})"
            if not public:
                line += f" — TVD {d.tvd:.3f}"
            st.markdown(f"{line}  \n{d.note}")


def _load_dashboard_data() -> tuple[
    list[dict[str, Any]] | None,
    list[dict[str, Any]] | None,
    dict[str, Any] | None,
    str | None,
]:
    """Fetch views from the active source; return ``(upcoming, sim, receipt, error)``."""
    try:
        upcoming, sim_rows, receipt = ds.dashboard_views()
        return upcoming, sim_rows, receipt, None
    except Exception as exc:
        return None, None, None, str(exc)


def render_nav() -> str:
    """Top navigation (always visible — sidebar is easy to miss on mobile)."""
    st.markdown('<div class="pe-nav">', unsafe_allow_html=True)
    page = st.radio(
        "Section",
        list(_pages()),
        horizontal=True,
        index=0,
        label_visibility="collapsed",
    )
    st.markdown("</div>", unsafe_allow_html=True)
    return page


def main() -> None:
    st.set_page_config(
        page_title="PitchEdge",
        page_icon=None,
        layout="wide",
        initial_sidebar_state="auto",
    )
    st.markdown(MOBILE_CSS, unsafe_allow_html=True)

    page = render_nav()
    upcoming, sim_rows, receipt, db_err = _load_dashboard_data()

    if page == "About":
        render_landing(sim_rows)
        return

    if db_err:
        if ds.is_snapshot_mode():
            st.error(
                "Could not read the bundled snapshot. Re-generate it with "
                "`make export-snapshot` against the live DB, then redeploy.\n\n"
                f"Details: {db_err}"
            )
        else:
            st.error(
                "Could not reach Postgres. Start the DB and load data, then refresh:\n\n"
                "```\nmake db-up && make migrate && make reload-data\n"
                "make sim && make predict\n```\n\n"
                f"Details: {db_err}"
            )
        st.stop()

    assert upcoming is not None and sim_rows is not None and receipt is not None

    if page == "Launch board":
        render_launch_board(sim_rows, receipt)
        return

    st.title("PitchEdge predictions")
    st.caption("Calibrated probabilities and receipts — model vs market, never blend-as-model.")
    tab_up, tab_sim, tab_adv, tab_cal = st.tabs(
        ["Upcoming matches", "Tournament odds", "Advancement", "Calibration (receipts)"]
    )
    with tab_up:
        render_upcoming_tab(upcoming)
    with tab_sim:
        render_tournament_odds_tab(sim_rows)
    with tab_adv:
        render_advancement_tab(sim_rows)
    with tab_cal:
        render_calibration_tab()


if __name__ == "__main__":
    main()
