"""Idempotent schema migration for PitchEdge.

Implements the full data model from docs/ARCHITECTURE.md section 2. Running this
module is safe to repeat: every object uses ``CREATE ... IF NOT EXISTS`` (or
``CREATE OR REPLACE`` for functions, ``DROP ... IF EXISTS`` then ``CREATE`` for
triggers), so ``make migrate`` can run any number of times without error.

Integrity rules enforced at the database layer (defense in depth, in addition to
the app-level guards):
  * ``match_predictions`` is APPEND-ONLY and pre-kickoff. A BEFORE INSERT trigger
    rejects any row whose ``predicted_utc >= fixtures.kickoff_utc``. A
    BEFORE UPDATE/DELETE trigger rejects all mutations of logged predictions.
  * ``prediction_scores`` is likewise immutable once written.

Run with:  uv run python -m pitchedge.migrations
"""

from __future__ import annotations

import logging

from pitchedge import db

log = logging.getLogger(__name__)

# Tables defined by ARCHITECTURE section 2, in dependency order. Used by the
# migration and by tests to assert the schema is complete.
EXPECTED_TABLES: tuple[str, ...] = (
    "teams",
    "raw_results",
    "fixtures",
    "odds_snapshots",
    "team_ratings",
    "match_predictions",
    "prediction_scores",
    "sim_results",
    "subscribers",
)

SCHEMA_SQL = """
-- ---------------------------------------------------------------------------
-- teams: the 48 World Cup teams.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS teams (
    team_id       INTEGER PRIMARY KEY,
    name          TEXT NOT NULL,
    fifa_code     TEXT,
    confederation TEXT,
    group_label   TEXT
);

-- ---------------------------------------------------------------------------
-- raw_results: historical & tournament matches (training data + ground truth).
-- home_id/away_id are NOT FKs to teams: the historical corpus spans far more
-- nations than the 48-team WC set. The natural key dedupes idempotent ingests.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS raw_results (
    match_id    BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    date        DATE NOT NULL,
    home_id     INTEGER NOT NULL,
    away_id     INTEGER NOT NULL,
    home_goals  INTEGER,
    away_goals  INTEGER,
    competition TEXT,
    neutral     BOOLEAN NOT NULL DEFAULT FALSE,
    CONSTRAINT uq_raw_results_natural
        UNIQUE (date, home_id, away_id, competition)
);

-- ---------------------------------------------------------------------------
-- fixtures: scheduled World Cup matches.
-- home_id/away_id are nullable (knockout slots are unknown pre-draw).
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS fixtures (
    fixture_id  INTEGER PRIMARY KEY,
    kickoff_utc TIMESTAMPTZ NOT NULL,
    home_id     INTEGER REFERENCES teams (team_id),
    away_id     INTEGER REFERENCES teams (team_id),
    stage       TEXT,
    group_label TEXT,
    status      TEXT NOT NULL DEFAULT 'scheduled'
        CHECK (status IN ('scheduled', 'final'))
);

-- ---------------------------------------------------------------------------
-- odds_snapshots: multiple pre-match odds snapshots per fixture (decimal).
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS odds_snapshots (
    id           BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    fixture_id   INTEGER NOT NULL REFERENCES fixtures (fixture_id),
    book         TEXT NOT NULL,
    captured_utc TIMESTAMPTZ NOT NULL,
    home_odds    NUMERIC(8, 3),
    draw_odds    NUMERIC(8, 3),
    away_odds    NUMERIC(8, 3),
    CONSTRAINT uq_odds_snapshot
        UNIQUE (fixture_id, book, captured_utc)
);

-- ---------------------------------------------------------------------------
-- team_ratings: snapshotted ratings so model inputs are reproducible.
-- team_id uses the same integer ids as raw_results (historical nations), NOT
-- necessarily rows in teams (which is only the 48-team WC squad).
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS team_ratings (
    id               BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    team_id          INTEGER NOT NULL,
    as_of_date       DATE NOT NULL,
    elo              DOUBLE PRECISION,
    attack_strength  DOUBLE PRECISION,
    defense_strength DOUBLE PRECISION,
    CONSTRAINT uq_team_rating_snapshot
        UNIQUE (team_id, as_of_date)
);

-- ---------------------------------------------------------------------------
-- match_predictions: THE RECEIPTS. Append-only and immutable after kickoff.
-- Probabilities are in [0,1] and must sum to ~1. The pre-kickoff rule is
-- enforced by a trigger (a CHECK cannot reference fixtures.kickoff_utc).
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS match_predictions (
    id             BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    fixture_id     INTEGER NOT NULL REFERENCES fixtures (fixture_id),
    model_version  TEXT NOT NULL,
    predicted_utc  TIMESTAMPTZ NOT NULL DEFAULT now(),
    p_home         DOUBLE PRECISION NOT NULL CHECK (p_home BETWEEN 0 AND 1),
    p_draw         DOUBLE PRECISION NOT NULL CHECK (p_draw BETWEEN 0 AND 1),
    p_away         DOUBLE PRECISION NOT NULL CHECK (p_away BETWEEN 0 AND 1),
    exp_home_goals DOUBLE PRECISION,
    exp_away_goals DOUBLE PRECISION,
    source         TEXT NOT NULL CHECK (source IN ('model', 'market', 'blend')),
    CONSTRAINT ck_prediction_probs_sum
        CHECK (abs((p_home + p_draw + p_away) - 1.0) < 0.01)
);

-- ---------------------------------------------------------------------------
-- prediction_scores: written only after a match is final; never mutated.
-- One score row per prediction (prediction_id is the PK -> idempotent scoring).
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS prediction_scores (
    prediction_id BIGINT PRIMARY KEY REFERENCES match_predictions (id),
    brier         DOUBLE PRECISION,
    log_loss      DOUBLE PRECISION,
    outcome       CHAR(1) CHECK (outcome IN ('H', 'D', 'A')),
    scored_utc    TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- ---------------------------------------------------------------------------
-- sim_results: per-team Monte Carlo advancement/title probabilities per batch.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS sim_results (
    id              BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    run_batch_utc   TIMESTAMPTZ NOT NULL,
    team_id         INTEGER NOT NULL REFERENCES teams (team_id),
    p_advance_group DOUBLE PRECISION,
    p_r16           DOUBLE PRECISION,
    p_qf            DOUBLE PRECISION,
    p_sf            DOUBLE PRECISION,
    p_final         DOUBLE PRECISION,
    p_win           DOUBLE PRECISION,
    n_sims          INTEGER NOT NULL,
    CONSTRAINT uq_sim_result UNIQUE (run_batch_utc, team_id)
);

-- ---------------------------------------------------------------------------
-- Helpful indexes for the read paths.
-- ---------------------------------------------------------------------------
CREATE INDEX IF NOT EXISTS ix_raw_results_date ON raw_results (date);
CREATE INDEX IF NOT EXISTS ix_fixtures_kickoff ON fixtures (kickoff_utc);
CREATE INDEX IF NOT EXISTS ix_odds_fixture ON odds_snapshots (fixture_id);
CREATE INDEX IF NOT EXISTS ix_team_ratings_team ON team_ratings (team_id, as_of_date);
CREATE INDEX IF NOT EXISTS ix_predictions_fixture ON match_predictions (fixture_id);
CREATE INDEX IF NOT EXISTS ix_sim_results_batch ON sim_results (run_batch_utc);

-- ---------------------------------------------------------------------------
-- subscribers: landing-page email capture (unique email, append-only).
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS subscribers (
    email         TEXT PRIMARY KEY,
    captured_utc  TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Optional alias for the-odds-api team strings (when they differ from ``name``).
ALTER TABLE teams ADD COLUMN IF NOT EXISTS odds_name TEXT;

-- Kaggle results.csv label when it differs from display ``name`` (e.g. Czechia).
ALTER TABLE teams ADD COLUMN IF NOT EXISTS history_name TEXT;

-- Elo ids are historical nation ids (raw_results), not WC teams FK.
ALTER TABLE team_ratings DROP CONSTRAINT IF EXISTS team_ratings_team_id_fkey;

-- ---------------------------------------------------------------------------
-- Integrity triggers (append-only receipts + pre-kickoff guard).
-- ---------------------------------------------------------------------------

-- Reject a prediction logged at or after its fixture's kickoff.
CREATE OR REPLACE FUNCTION pitchedge_enforce_pre_kickoff()
RETURNS trigger AS $fn$
DECLARE
    v_kickoff timestamptz;
BEGIN
    SELECT kickoff_utc INTO v_kickoff
    FROM fixtures
    WHERE fixture_id = NEW.fixture_id;

    IF v_kickoff IS NULL THEN
        RAISE EXCEPTION
            'match_predictions: fixture % does not exist or has no kickoff_utc',
            NEW.fixture_id;
    END IF;

    IF NEW.predicted_utc >= v_kickoff THEN
        RAISE EXCEPTION
            'match_predictions is pre-kickoff only: predicted_utc (%) must be '
            'strictly before kickoff_utc (%)', NEW.predicted_utc, v_kickoff;
    END IF;

    RETURN NEW;
END;
$fn$ LANGUAGE plpgsql;

-- Reject any UPDATE/DELETE on an immutable, append-only table.
CREATE OR REPLACE FUNCTION pitchedge_reject_mutation()
RETURNS trigger AS $fn$
BEGIN
    RAISE EXCEPTION
        '% is append-only and immutable; % on logged rows is not permitted',
        TG_TABLE_NAME, TG_OP;
END;
$fn$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_predictions_pre_kickoff ON match_predictions;
CREATE TRIGGER trg_predictions_pre_kickoff
    BEFORE INSERT ON match_predictions
    FOR EACH ROW EXECUTE FUNCTION pitchedge_enforce_pre_kickoff();

DROP TRIGGER IF EXISTS trg_predictions_immutable ON match_predictions;
CREATE TRIGGER trg_predictions_immutable
    BEFORE UPDATE OR DELETE ON match_predictions
    FOR EACH ROW EXECUTE FUNCTION pitchedge_reject_mutation();

DROP TRIGGER IF EXISTS trg_scores_immutable ON prediction_scores;
CREATE TRIGGER trg_scores_immutable
    BEFORE UPDATE OR DELETE ON prediction_scores
    FOR EACH ROW EXECUTE FUNCTION pitchedge_reject_mutation();
"""


def migrate(db_url: str | None = None) -> None:
    """Apply the full schema. Idempotent: safe to run repeatedly."""
    log.info("applying pitchedge schema migration")
    db.execute_script(SCHEMA_SQL, db_url=db_url)
    log.info("migration complete: %d tables ensured", len(EXPECTED_TABLES))


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    migrate()
    print("PitchEdge migration applied. Tables: " + ", ".join(EXPECTED_TABLES))


if __name__ == "__main__":
    main()
