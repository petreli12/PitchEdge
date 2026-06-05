"""Nightly pipeline orchestrator for PitchEdge.

Runs the end-to-end nightly job in order:

    refresh data  -> refit ratings -> log predictions -> run sim
                  -> score finished matches -> push the day's content

Each stage is idempotent or append-only; re-running a night never corrupts data
(``ON CONFLICT DO NOTHING`` on ingests/scores; ``match_predictions`` and
``sim_results`` append a fresh batch and downstream readers take the latest).

Design notes:
  * ``--dry-run`` walks every stage with ZERO side effects: no DB writes, no
    Telegram post, no Anthropic call, no live odds fetch. Where it is cheap and
    read-only it reports what the real run would do (e.g. how many fixtures would
    be predicted, how many finished predictions are unscored).
  * Logging is structured (``key=value``) and parseable, one line per stage
    transition, so a cron/launchd log can be grepped after the fact.
  * The expensive Dixon-Coles fit happens once in ``refit_ratings`` and is reused
    by both ``predict`` and ``sim``.

CLI:
    python -m pitchedge.scheduler --dry-run
    python -m pitchedge.scheduler --log-file logs/nightly.log
    python -m pitchedge.scheduler --stages predict,sim,score
    python -m pitchedge.scheduler --continue-on-error
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from pitchedge import config

log = logging.getLogger("pitchedge.scheduler")


# --------------------------------------------------------------------------- #
# Pipeline context + stage results                                            #
# --------------------------------------------------------------------------- #
@dataclass
class PipelineContext:
    """Shared state threaded through every stage of one nightly run."""

    db_url: str | None
    dry_run: bool
    now: datetime
    seed: int
    n_sims: int
    # Look-ahead window for the daily disagreement post. 36h suits an in-tournament
    # daily cadence (post about imminent fixtures); widen it for a pre-tournament
    # splash where the next match is days away.
    within_hours: int = 36
    # Populated by the refit stage and reused by predict + sim so the costly
    # Dixon-Coles fit only happens once per night.
    model: Any = None


@dataclass
class StageResult:
    name: str
    status: str  # "ok" | "skipped" | "failed"
    duration_s: float
    detail: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


def _fmt_detail(detail: dict[str, Any]) -> str:
    if not detail:
        return ""
    return " " + " ".join(f"{k}={v}" for k, v in detail.items())


# --------------------------------------------------------------------------- #
# Stages                                                                       #
# Each stage returns ("ok"|"skipped", detail_dict). Raising marks it failed.   #
# --------------------------------------------------------------------------- #
def stage_ingest_history(ctx: PipelineContext) -> tuple[str, dict[str, Any]]:
    """Load the historical results CSV (idempotent). Tolerant of a missing file."""
    csv_path = Path(config.HISTORY_CSV_PATH)
    if not csv_path.exists():
        return "skipped", {"reason": "csv_missing", "path": str(csv_path)}
    if ctx.dry_run:
        return "skipped", {"reason": "dry_run", "path": str(csv_path)}

    from pitchedge.ingest.history import ingest_history

    attempted, inserted = ingest_history(db_url=ctx.db_url)
    return "ok", {"attempted": attempted, "inserted": inserted}


def stage_ingest_fixtures(ctx: PipelineContext) -> tuple[str, dict[str, Any]]:
    """Load WC teams + fixtures CSVs (idempotent). Tolerant of missing files."""
    teams_csv = Path(config.TEAMS_CSV_PATH)
    fixtures_csv = Path(config.FIXTURES_CSV_PATH)
    if not teams_csv.exists() or not fixtures_csv.exists():
        return "skipped", {"reason": "csv_missing"}
    if ctx.dry_run:
        return "skipped", {"reason": "dry_run"}

    from pitchedge.ingest.fixtures import ingest_fixtures, ingest_teams

    t_attempted, t_inserted = ingest_teams(db_url=ctx.db_url)
    f_attempted, f_inserted = ingest_fixtures(db_url=ctx.db_url)
    return "ok", {
        "teams_inserted": t_inserted,
        "fixtures_inserted": f_inserted,
        "teams_attempted": t_attempted,
        "fixtures_attempted": f_attempted,
    }


def stage_ingest_odds(ctx: PipelineContext) -> tuple[str, dict[str, Any]]:
    """Fetch live odds into ``odds_snapshots``. Skips if no API key is set."""
    if not config.ODDS_API_KEY.strip():
        return "skipped", {"reason": "no_odds_api_key"}
    if ctx.dry_run:
        return "skipped", {"reason": "dry_run", "would": "fetch_live_odds"}

    from pitchedge.ingest.odds import ingest_odds_snapshots

    attempted, inserted = ingest_odds_snapshots(fetch_live=True, db_url=ctx.db_url)
    return "ok", {"attempted": attempted, "inserted": inserted}


def stage_refit_ratings(ctx: PipelineContext) -> tuple[str, dict[str, Any]]:
    """Refit Elo (writes ``team_ratings``) then Dixon-Coles (held in ctx.model)."""
    if ctx.dry_run:
        return "skipped", {"reason": "dry_run", "would": "fit_elo+dixon_coles"}

    from pitchedge.model.dixon_coles import fit_dixon_coles_from_db
    from pitchedge.model.elo import fit_elo_from_db

    teams_written, as_of = fit_elo_from_db(db_url=ctx.db_url)
    ctx.model = fit_dixon_coles_from_db(db_url=ctx.db_url)
    nll = ctx.model.fit_neg_log_likelihood
    return "ok", {
        "elo_teams": teams_written,
        "elo_as_of": as_of,
        "dc_nll": f"{nll:.4f}" if nll is not None else "na",
    }


def stage_predict(ctx: PipelineContext) -> tuple[str, dict[str, Any]]:
    """Log model/market/blend predictions for upcoming fixtures (pre-kickoff)."""
    if ctx.dry_run:
        # Read-only: report how many fixtures WOULD be predicted.
        from pitchedge import db
        from pitchedge.predict import fetch_upcoming_fixtures

        with db.connect(ctx.db_url) as conn:
            fixtures = fetch_upcoming_fixtures(conn, now_utc=ctx.now)
        return "skipped", {"reason": "dry_run", "upcoming_fixtures": len(fixtures)}

    from pitchedge.predict import log_upcoming_predictions

    rows = log_upcoming_predictions(
        model=ctx.model, db_url=ctx.db_url, predicted_utc=ctx.now
    )
    return "ok", {"rows_logged": rows}


def stage_sim(ctx: PipelineContext) -> tuple[str, dict[str, Any]]:
    """Run the Monte Carlo tournament sim and persist ``sim_results``."""
    if ctx.dry_run:
        return "skipped", {
            "reason": "dry_run",
            "would": f"run_{ctx.n_sims}_sims_seed_{ctx.seed}",
        }

    from pitchedge.sim.tournament import run_and_persist

    agg = run_and_persist(
        model=ctx.model, n_sims=ctx.n_sims, seed=ctx.seed, db_url=ctx.db_url
    )
    return "ok", {"n_sims": agg.n_sims, "teams": len(agg.by_team)}


def stage_score(ctx: PipelineContext) -> tuple[str, dict[str, Any]]:
    """Score newly-final fixtures (idempotent; never mutates predictions)."""
    if ctx.dry_run:
        # Read-only count of predictions that would be scored.
        from sqlalchemy import text

        from pitchedge import db
        from pitchedge.score import UNSCORED_FINAL_SQL

        with db.connect(ctx.db_url) as conn:
            pending = conn.execute(text(UNSCORED_FINAL_SQL)).mappings().all()
        return "skipped", {"reason": "dry_run", "unscored_pending": len(pending)}

    from pitchedge.score import score_finished_matches

    attempted, inserted = score_finished_matches(db_url=ctx.db_url)
    return "ok", {"attempted": attempted, "inserted": inserted}


def stage_content(ctx: PipelineContext) -> tuple[str, dict[str, Any]]:
    """Post the day's top model-vs-market disagreement to Telegram."""
    if not config.TELEGRAM_BOT_TOKEN.strip() or not config.TELEGRAM_CHAT_ID.strip():
        return "skipped", {"reason": "telegram_not_configured"}

    from pitchedge import db
    from pitchedge.content.daily_disagreement import fetch_candidates, select_top

    with db.connect(ctx.db_url) as conn:
        candidates = fetch_candidates(conn, within_hours=ctx.within_hours)

    if ctx.dry_run:
        top = select_top(candidates)
        detail: dict[str, Any] = {
            "reason": "dry_run",
            "candidates": len(candidates),
        }
        if top is not None:
            detail["top"] = f"{top.candidate.home}_vs_{top.candidate.away}"
            detail["tvd"] = f"{top.tvd:.3f}"
        return "skipped", detail

    if not candidates:
        return "skipped", {"reason": "no_candidates"}
    if not config.ANTHROPIC_API_KEY.strip():
        # Narrative needs the LLM; do not post a half-built card.
        return "skipped", {"reason": "no_anthropic_key", "candidates": len(candidates)}

    from pitchedge.content.telegram import post_daily_disagreement

    result = post_daily_disagreement(candidates, dry_run=False)
    if result is None:
        return "skipped", {"reason": "no_fixture_cleared_threshold"}
    return "ok", {"message_id": result.get("message_id")}


# Ordered pipeline. Name -> callable. Order matters (dependencies flow downward).
STAGES: list[tuple[str, Callable[[PipelineContext], tuple[str, dict[str, Any]]]]] = [
    ("ingest_history", stage_ingest_history),
    ("ingest_fixtures", stage_ingest_fixtures),
    ("ingest_odds", stage_ingest_odds),
    ("refit_ratings", stage_refit_ratings),
    ("predict", stage_predict),
    ("sim", stage_sim),
    ("score", stage_score),
    ("content", stage_content),
]


# --------------------------------------------------------------------------- #
# Runner                                                                        #
# --------------------------------------------------------------------------- #
def run_pipeline(
    ctx: PipelineContext,
    *,
    stages: list[tuple[str, Callable[[PipelineContext], tuple[str, dict[str, Any]]]]]
    | None = None,
    continue_on_error: bool = False,
) -> list[StageResult]:
    """Run the pipeline stages in order; return per-stage results.

    On a stage failure the run aborts (fail-fast) unless ``continue_on_error`` is
    set, in which case remaining stages are still attempted and the failure is
    recorded. The caller decides the process exit code from the results.
    """
    selected = stages if stages is not None else STAGES
    results: list[StageResult] = []
    mode = "dry_run" if ctx.dry_run else "live"
    log.info(
        "pipeline start mode=%s db=%s seed=%d n_sims=%d stages=%d",
        mode,
        _db_label(ctx.db_url),
        ctx.seed,
        ctx.n_sims,
        len(selected),
    )

    for name, fn in selected:
        log.info("stage start name=%s", name)
        start = time.monotonic()
        try:
            status, detail = fn(ctx)
            duration = time.monotonic() - start
            results.append(StageResult(name, status, duration, detail))
            log.info(
                "stage end name=%s status=%s dur=%.2fs%s",
                name,
                status,
                duration,
                _fmt_detail(detail),
            )
        except Exception as exc:  # noqa: BLE001 - we want to record any failure
            duration = time.monotonic() - start
            results.append(
                StageResult(name, "failed", duration, error=str(exc))
            )
            log.error(
                "stage end name=%s status=failed dur=%.2fs error=%r",
                name,
                duration,
                str(exc),
            )
            if not continue_on_error:
                log.error("pipeline aborting after failed stage name=%s", name)
                break

    ok = sum(1 for r in results if r.status == "ok")
    skipped = sum(1 for r in results if r.status == "skipped")
    failed = sum(1 for r in results if r.status == "failed")
    log.info(
        "pipeline end mode=%s ok=%d skipped=%d failed=%d",
        mode,
        ok,
        skipped,
        failed,
    )
    return results


def _db_label(db_url: str | None) -> str:
    """Database name only (never the full connection string with credentials)."""
    url = db_url or config.DB_URL
    return url.rsplit("/", 1)[-1].split("?")[0] if url else "default"


def check_db_reachable(db_url: str | None) -> tuple[bool, str]:
    """Preflight: confirm Postgres answers a trivial ``SELECT 1``.

    Returns ``(ok, detail)`` and never raises. Its purpose is to turn a common
    failure mode -- the laptop woke for the 04:00 job but Docker/Postgres is not
    running -- into one clear log line instead of a stack trace deep inside the
    first DB stage. The bound connect timeout lives in ``db.CONNECT_TIMEOUT_S``.
    """
    from sqlalchemy import text

    from pitchedge import db

    label = _db_label(db_url)
    try:
        with db.connect(db_url) as conn:
            conn.execute(text("SELECT 1"))
        return True, f"db={label} reachable"
    except Exception as exc:  # noqa: BLE001 - report any failure as unreachable
        first_line = (str(exc).splitlines() or [""])[0]
        return False, f"db={label} unreachable: {type(exc).__name__}: {first_line}"


def _configure_logging(log_file: str | None, verbose: bool) -> None:
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stderr)]
    if log_file:
        path = Path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(path))
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        handlers=handlers,
        force=True,
    )


def _select_stages(
    names: str | None,
) -> list[tuple[str, Callable[[PipelineContext], tuple[str, dict[str, Any]]]]]:
    if not names:
        return STAGES
    wanted = [n.strip() for n in names.split(",") if n.strip()]
    by_name = dict(STAGES)
    unknown = [n for n in wanted if n not in by_name]
    if unknown:
        raise SystemExit(
            f"unknown stage(s): {', '.join(unknown)}; "
            f"valid: {', '.join(by_name)}"
        )
    return [(n, by_name[n]) for n in wanted]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="walk the pipeline with no DB writes, no posts, no external calls",
    )
    parser.add_argument(
        "--log-file",
        default=None,
        help="also append structured logs to this file (created if needed)",
    )
    parser.add_argument(
        "--stages",
        default=None,
        help="comma-separated subset to run, in order (default: all)",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="attempt all stages even if one fails (default: fail-fast)",
    )
    parser.add_argument(
        "--n-sims",
        type=int,
        default=None,
        help=f"override Monte Carlo sims (default: config N_SIMS={config.N_SIMS})",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help=f"override sim seed (default: config RANDOM_SEED={config.RANDOM_SEED})",
    )
    parser.add_argument(
        "--within-hours",
        type=int,
        default=36,
        help="content: look-ahead window for the daily disagreement (default: 36)",
    )
    parser.add_argument(
        "--skip-db-check",
        action="store_true",
        help="skip the Postgres reachability preflight",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    args = parser.parse_args(argv)

    _configure_logging(args.log_file, args.verbose)

    ctx = PipelineContext(
        db_url=None,
        dry_run=args.dry_run,
        now=datetime.now(timezone.utc),
        seed=args.seed if args.seed is not None else config.RANDOM_SEED,
        n_sims=args.n_sims if args.n_sims is not None else config.N_SIMS,
        within_hours=args.within_hours,
    )

    # Preflight DB check. Fatal for a live run (no point fitting/posting against a
    # dead DB); only a warning in dry-run so inspection/CI never depends on a
    # running Postgres.
    if not args.skip_db_check:
        ok, detail = check_db_reachable(ctx.db_url)
        if ok:
            log.info("preflight %s", detail)
        elif ctx.dry_run:
            log.warning("preflight %s (continuing: dry-run)", detail)
        else:
            log.error(
                "preflight %s -- aborting. Is Docker/Postgres up? "
                "Start it (docker compose up -d db) and re-run; see RUNBOOK.md.",
                detail,
            )
            return 2

    results = run_pipeline(
        ctx,
        stages=_select_stages(args.stages),
        continue_on_error=args.continue_on_error,
    )
    failed = [r for r in results if r.status == "failed"]
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
