"""Tests for the nightly pipeline orchestrator (no DB, no network)."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from pitchedge import scheduler
from pitchedge.scheduler import PipelineContext, run_pipeline


def _ctx(dry_run: bool = False) -> PipelineContext:
    return PipelineContext(
        db_url=None,
        dry_run=dry_run,
        now=datetime(2026, 6, 5, tzinfo=timezone.utc),
        seed=1,
        n_sims=10,
    )


def test_run_pipeline_runs_stages_in_order():
    calls: list[str] = []

    def make(name):
        def fn(ctx):
            calls.append(name)
            return "ok", {}
        return name, fn

    stages = [make("a"), make("b"), make("c")]
    results = run_pipeline(_ctx(), stages=stages)
    assert calls == ["a", "b", "c"]
    assert [r.name for r in results] == ["a", "b", "c"]
    assert all(r.status == "ok" for r in results)


def test_run_pipeline_fail_fast_aborts_remaining():
    calls: list[str] = []

    def ok(name):
        def fn(ctx):
            calls.append(name)
            return "ok", {}
        return name, fn

    def boom(ctx):
        calls.append("boom")
        raise RuntimeError("kaboom")

    stages = [ok("a"), ("boom", boom), ok("c")]
    results = run_pipeline(_ctx(), stages=stages)
    assert calls == ["a", "boom"]  # "c" never runs
    assert [r.status for r in results] == ["ok", "failed"]
    assert "kaboom" in results[1].error


def test_run_pipeline_continue_on_error_attempts_all():
    calls: list[str] = []

    def ok(name):
        def fn(ctx):
            calls.append(name)
            return "ok", {}
        return name, fn

    def boom(ctx):
        calls.append("boom")
        raise RuntimeError("kaboom")

    stages = [ok("a"), ("boom", boom), ok("c")]
    results = run_pipeline(_ctx(), stages=stages, continue_on_error=True)
    assert calls == ["a", "boom", "c"]
    assert [r.status for r in results] == ["ok", "failed", "ok"]


@pytest.mark.parametrize(
    "stage_fn",
    [
        scheduler.stage_ingest_history,
        scheduler.stage_ingest_fixtures,
        scheduler.stage_ingest_odds,
        scheduler.stage_refit_ratings,
        scheduler.stage_sim,
    ],
)
def test_dry_run_mutating_stages_skip_without_side_effects(monkeypatch, stage_fn):
    """In dry-run these stages must short-circuit before any write/refit/fetch."""
    # Poison the heavy entry points: if a dry-run stage reaches them, fail loudly.
    import pitchedge.ingest.history as history
    import pitchedge.ingest.fixtures as fixtures
    import pitchedge.ingest.odds as odds
    import pitchedge.model.elo as elo
    import pitchedge.sim.tournament as tournament

    def poison(*args, **kwargs):  # pragma: no cover - must never run
        raise AssertionError("mutating call made during dry-run")

    monkeypatch.setattr(history, "ingest_history", poison)
    monkeypatch.setattr(fixtures, "ingest_teams", poison)
    monkeypatch.setattr(fixtures, "ingest_fixtures", poison)
    monkeypatch.setattr(odds, "ingest_odds_snapshots", poison)
    monkeypatch.setattr(elo, "fit_elo_from_db", poison)
    monkeypatch.setattr(tournament, "run_and_persist", poison)

    status, detail = stage_fn(_ctx(dry_run=True))
    assert status == "skipped"
    assert detail.get("reason") in {"dry_run", "csv_missing", "no_odds_api_key"}


def test_content_stage_skips_when_telegram_unconfigured(monkeypatch):
    monkeypatch.setattr(scheduler.config, "TELEGRAM_BOT_TOKEN", "")
    monkeypatch.setattr(scheduler.config, "TELEGRAM_CHAT_ID", "")
    status, detail = scheduler.stage_content(_ctx(dry_run=True))
    assert status == "skipped"
    assert detail["reason"] == "telegram_not_configured"


def test_select_stages_rejects_unknown():
    with pytest.raises(SystemExit):
        scheduler._select_stages("predict,not_a_stage")


def test_select_stages_subset_in_order():
    selected = scheduler._select_stages("sim,predict")
    assert [name for name, _ in selected] == ["sim", "predict"]


def test_db_label_hides_credentials():
    label = scheduler._db_label(
        "postgresql+psycopg://user:secret@localhost:5432/pitchedge"
    )
    assert label == "pitchedge"
    assert "secret" not in label


def test_main_dry_run_non_db_stages_returns_zero():
    rc = scheduler.main(
        [
            "--dry-run",
            "--stages",
            "ingest_history,ingest_fixtures,ingest_odds,refit_ratings,sim",
        ]
    )
    assert rc == 0
