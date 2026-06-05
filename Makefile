# PitchEdge developer tasks.
# Requires: uv (https://docs.astral.sh/uv/) and Docker Compose.

# Postgres URL for local Docker Compose (port 5432). Overrides a stale DB_URL in
# `.env` for targets that talk to the containerized database.
DOCKER_DB_URL := postgresql+psycopg://pitchedge:pitchedge@localhost:5432/pitchedge

# Throwaway database for the test suite. DB-backed tests truncate ingest tables,
# so the suite MUST NOT run against the live `pitchedge` database. The truncating
# fixture additionally refuses to run unless the connected db name ends in
# `_test` (see tests/conftest.py), so this is belt-and-suspenders.
TEST_DB_URL := postgresql+psycopg://pitchedge:pitchedge@localhost:5432/pitchedge_test

.PHONY: help install db-up db-down db-logs migrate test fmt \
	ingest-history ingest-fixtures ingest-odds fit-elo fit-dc backtest sim \
	verify-annex report-sim predict score db-status reload-data dashboard check-teams \
	telegram-test telegram-post telegram-dry-run nightly nightly-dry-run

help:
	@echo "Targets:"
	@echo "  install   - sync the uv-managed virtualenv (deps + dev deps)"
	@echo "  db-up     - start Postgres 16 via Docker Compose (waits for healthy)"
	@echo "  db-down   - stop the Postgres container"
	@echo "  db-logs   - tail Postgres logs"
	@echo "  migrate   - apply the idempotent schema migration"
	@echo "  ingest-history  - load Kaggle/GitHub results.csv into raw_results"
	@echo "  ingest-fixtures - load WC teams + fixtures CSVs"
	@echo "  ingest-odds     - fetch live odds (requires ODDS_API_KEY)"
	@echo "  fit-elo         - fit Elo on raw_results -> team_ratings"
	@echo "  fit-dc          - fit Dixon-Coles on raw_results (in-memory; logs NLL)"
	@echo "  backtest        - Phase 4 held-out tournaments + blend w sweep"
	@echo "  sim             - Phase 5 Monte Carlo tournament (N_SIMS, RANDOM_SEED)"
	@echo "  verify-annex    - compare annex_c.json to official FIFA PDF"
	@echo "  report-sim      - Phase 5 title-odds table + property checks"
	@echo "  predict         - log model/market/blend predictions (pre-kickoff)"
	@echo "  score           - Brier/log loss on final fixtures (idempotent)"
	@echo "  dashboard       - Streamlit public UI (reads Docker DB on 5432)"
	@echo "  telegram-test   - live sample card+caption to TELEGRAM_CHAT_ID"
	@echo "  telegram-post   - live daily_disagreement from DB (Anthropic + predict)"
	@echo "  telegram-dry-run - disagreement pipeline without sending"
	@echo "  check-teams     - WC team resolution across history, fixtures, odds API"
	@echo "  nightly         - run the full nightly pipeline (scheduler)"
	@echo "  nightly-dry-run - walk the nightly pipeline with no writes/posts"
	@echo "  db-status       - show DB_URL and table row counts (app vs docker check)"
	@echo "  reload-data     - re-ingest history, fixtures, odds, then fit Elo"
	@echo "  test      - run pytest against \$$TEST_DB_URL (pitchedge_test; DB tests truncate it)"

install:
	uv sync --extra dev

db-up:
	docker compose up -d db
	@echo "Waiting for Postgres to become healthy..."
	@until [ "$$(docker inspect -f '{{.State.Health.Status}}' pitchedge-db 2>/dev/null)" = "healthy" ]; do \
		sleep 1; \
	done
	@echo "Postgres is healthy."

db-down:
	docker compose down

db-logs:
	docker compose logs -f db

migrate:
	DB_URL=$(DOCKER_DB_URL) uv run python -m pitchedge.migrations

ingest-history:
	DB_URL=$(DOCKER_DB_URL) uv run python -m pitchedge.ingest.history

ingest-fixtures:
	DB_URL=$(DOCKER_DB_URL) uv run python -m pitchedge.ingest.fixtures

ingest-odds:
	DB_URL=$(DOCKER_DB_URL) uv run python -m pitchedge.ingest.odds

fit-elo:
	DB_URL=$(DOCKER_DB_URL) uv run python -m pitchedge.model.elo

fit-dc:
	DB_URL=$(DOCKER_DB_URL) uv run python -m pitchedge.model.dixon_coles

fetch-euro-copa-odds:
	uv run python scripts/fetch_euro_copa_odds.py

backtest:
	MPLBACKEND=Agg uv run python -m pitchedge.eval.backtest

sim:
	DB_URL=$(DOCKER_DB_URL) uv run python -m pitchedge.sim.tournament

verify-annex:
	uv run python scripts/verify_annex_c.py

report-sim:
	DB_URL=$(DOCKER_DB_URL) uv run python scripts/report_phase5_sim.py

predict:
	DB_URL=$(DOCKER_DB_URL) uv run python -m pitchedge.predict

score:
	DB_URL=$(DOCKER_DB_URL) uv run python -m pitchedge.score

db-status:
	DB_URL=$(DOCKER_DB_URL) uv run python scripts/db_status.py

dashboard:
	DB_URL=$(DOCKER_DB_URL) uv run streamlit run src/pitchedge/app.py

# Freeze the dashboard's datasets into data/snapshot/ for the public deploy.
# Reads the live DB (locked predictions + latest sim); recomputes nothing.
export-snapshot:
	DB_URL=$(DOCKER_DB_URL) uv run python scripts/export_snapshot.py

# Regenerate the snapshot from the live DB and commit+push it (redeploys the
# public Streamlit site). Same code path the nightly uses with --publish-snapshot.
publish-snapshot:
	DB_URL=$(DOCKER_DB_URL) uv run python -m pitchedge.scheduler --publish-snapshot --stages snapshot

# Run the dashboard locally in snapshot mode (no DB), exactly as the public deploy.
dashboard-snapshot:
	DASHBOARD_SNAPSHOT_DIR=data/snapshot uv run streamlit run src/pitchedge/app.py

telegram-test:
	DB_URL=$(DOCKER_DB_URL) uv run python scripts/telegram_post.py sample

telegram-post:
	DB_URL=$(DOCKER_DB_URL) uv run python scripts/telegram_post.py disagreement

telegram-dry-run:
	DB_URL=$(DOCKER_DB_URL) uv run python scripts/telegram_post.py disagreement --dry-run

check-teams:
	DB_URL=$(DOCKER_DB_URL) uv run python scripts/check_team_resolution.py

# Full nightly pipeline: refresh -> refit -> predict -> sim -> score -> content.
nightly:
	DB_URL=$(DOCKER_DB_URL) uv run python -m pitchedge.scheduler

# Same ordering, zero side effects (no DB writes, no posts, no external calls).
nightly-dry-run:
	DB_URL=$(DOCKER_DB_URL) uv run python -m pitchedge.scheduler --dry-run

reload-data: ingest-history ingest-fixtures ingest-odds fit-elo
	@DB_URL=$(DOCKER_DB_URL) $(MAKE) db-status

test:
	DB_URL=$(TEST_DB_URL) uv run pytest
