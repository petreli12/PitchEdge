# PitchEdge Runbook — nightly pipeline operations

Operational guide for the automated nightly job (`src/pitchedge/scheduler.py`):
how to run it by hand, how it's scheduled, where logs go, and how to recover
from a failed night.

## What the nightly does

The scheduler runs these stages **in order**. Each is idempotent or append-only,
so re-running a night never corrupts data.

| # | Stage            | Action                                                        | Writes |
|---|------------------|---------------------------------------------------------------|--------|
| 1 | `ingest_history` | Load historical results CSV (skips if file absent)            | `raw_results` (ON CONFLICT DO NOTHING) |
| 2 | `ingest_fixtures`| Load WC teams + fixtures CSVs (skips if files absent)          | `teams`, `fixtures` (idempotent) |
| 3 | `ingest_odds`    | Fetch live odds (skips if `ODDS_API_KEY` unset)               | `odds_snapshots` (ON CONFLICT DO NOTHING) |
| 4 | `refit_ratings`  | Fit Elo, then Dixon-Coles (held in memory, reused below)      | `team_ratings` |
| 5 | `predict`        | Log model/market/blend probs for upcoming fixtures            | `match_predictions` (append-only, pre-kickoff guard) |
| 6 | `sim`            | Monte Carlo tournament sim                                     | `sim_results` (new batch per run) |
| 7 | `score`          | Brier/log loss for newly-final fixtures                       | `prediction_scores` (ON CONFLICT DO NOTHING) |
| 8 | `content`        | Post the day's top model-vs-market disagreement to Telegram   | Telegram (no DB write) |

Notes:
- The expensive Dixon-Coles fit happens **once** (stage 4) and is reused by
  `predict` and `sim`.
- `predict` is **append-only**: each run writes a fresh batch (immutable
  receipts). Downstream readers (`score`, dashboard, disagreement) use the
  **latest** batch per fixture, so re-running is safe — it never rewrites or
  scores stale rows.
- `sim` writes a new `run_batch_utc`; the dashboard reads `MAX(run_batch_utc)`.
- `content` only posts a fixture whose model-vs-market gap clears the threshold
  and falls within `--within-hours` (default 36h). Pre-tournament, when no match
  is imminent, it simply skips.

## Prerequisites

- **Docker Postgres running**: `make db-up` (Docker Desktop must be up).
- **`.env` populated** (gitignored): `DB_URL`, and for full function
  `ODDS_API_KEY`, `ANTHROPIC_API_KEY`, `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID`.
  Missing optional keys make the relevant stage **skip**, not fail.

## Run it manually

```bash
# Dry-run: walk every stage with ZERO side effects (no writes, no posts,
# no live odds fetch, no Anthropic call). Always safe.
make nightly-dry-run
# or with options:
.venv/bin/python -m pitchedge.scheduler --dry-run --within-hours 36

# Full live run:
make nightly
# or directly, with a dated structured log file:
.venv/bin/python -m pitchedge.scheduler --log-file logs/nightly-$(date +%F).log
```

Useful flags:
- `--dry-run` — no side effects.
- `--stages a,b,c` — run only a subset, in order (e.g. `--stages score,content`).
- `--continue-on-error` — attempt every stage even if one fails (default is
  fail-fast: stop after the first failure).
- `--within-hours N` — content look-ahead window (default 36).
- `--n-sims N`, `--seed N` — override sim size / seed.
- `--log-file PATH`, `-v` — file logging / debug verbosity.

Exit code is non-zero if any stage failed (useful for cron alerting).

## Scheduling

### macOS (launchd) — mirrors the OptionsEdge setup

```bash
# 1. Edit the absolute paths in the plist (replace /Users/CHANGE_ME/...).
# 2. Install and load:
cp deploy/com.pitchedge.nightly.plist ~/Library/LaunchAgents/
launchctl load ~/Library/LaunchAgents/com.pitchedge.nightly.plist
# 3. (optional) trigger once now:
launchctl start com.pitchedge.nightly
# Inspect / remove:
launchctl list | grep pitchedge
launchctl unload ~/Library/LaunchAgents/com.pitchedge.nightly.plist
```

Runs daily at 04:00 local. Adjust `StartCalendarInterval` in the plist.

#### Running overnight while the laptop sleeps

macOS does not run scheduled jobs while asleep. `launchd` will run a *missed*
`StartCalendarInterval` job once when the Mac next wakes, but for a predictable
overnight run schedule a power-management wake just before the job:

```bash
# Wake (or power on, if plugged in) every day at 03:55, 5 min before the 04:00 job.
sudo pmset repeat wakeorpoweron MTWRFSU 03:55:00
pmset -g sched          # verify the repeating schedule
sudo pmset repeat cancel # remove it later
```

Requirements / caveats:
- The laptop must be **plugged in** (battery wake from deep sleep is unreliable;
  `poweron` only works on AC).
- **Docker Desktop must already be running** before sleep so the Postgres
  container is up on wake (set Docker to "Start when you log in").
- `deploy/run_nightly.sh` runs the pipeline under `caffeinate -i`, so once it
  starts the Mac will not idle-sleep until the run finishes.

### Linux / cron alternative

See `deploy/crontab.example` — one line, `crontab -e`, edit the path.

Both call `deploy/run_nightly.sh`, which resolves the repo root, ensures the log
dir, picks the in-repo `.venv`, and forwards args to the scheduler.

## Logs

- Structured, parseable, one line per stage transition:
  `... pitchedge.scheduler stage end name=predict status=ok dur=12.30s rows_logged=216`
- Scheduler log: `logs/nightly-YYYY-MM-DD.log` (via `--log-file`).
- launchd stdout/stderr: `logs/launchd.out`, `logs/launchd.err`.
- cron: `logs/cron.log`.
- `logs/` is gitignored.

Quick health check after a run:
```bash
grep "pipeline end" logs/nightly-$(date +%F).log   # ok/skipped/failed counts
grep "status=failed" logs/nightly-*.log            # any failures
```

## Recovering from a failed night

The pipeline is fail-fast by default, so a failure leaves earlier stages
applied and later ones not run. Recovery is just **re-running** — every stage is
idempotent or append-only.

1. **Find the failed stage**: `grep "status=failed" logs/nightly-<date>.log`.
2. **Diagnose** common causes:
   - `could not connect` / DB errors → Docker not running. `make db-up`, retry.
   - `ingest_odds` failed → odds API down or quota hit. Safe to skip; rerun later.
   - `content` skipped `no_anthropic_key` / `telegram_not_configured` → set the
     key(s) in `.env`; until then the post is simply skipped (not an error).
   - `refit_ratings` failed → inspect the Dixon-Coles fit; rerun `make fit-elo`
     then `make fit-dc` to reproduce locally.
3. **Re-run from the failed stage onward**, e.g. if `sim` failed:
   ```bash
   .venv/bin/python -m pitchedge.scheduler --stages sim,score,content \
       --log-file logs/recover-$(date +%F).log
   ```
   Or just re-run the whole pipeline (`make nightly`) — idempotent stages no-op,
   `predict`/`sim` append a fresh (correct) batch that supersedes the partial one.
4. To push through transient single-stage failures in one shot, use
   `--continue-on-error` and review the summary line.

## Refreshing the public dashboard (separate, deliberate)

The nightly does **not** push to the public Streamlit deploy — publishing stays
a manual decision so a bad night can't auto-publish. To refresh the live site
after a good run:

```bash
make export-snapshot          # freeze DB -> data/snapshot/
git add data/snapshot && git commit -m "Refresh dashboard snapshot" && git push
```

(Streamlit Cloud redeploys on push.) Verify locally first with
`make dashboard-snapshot`.

## Integrity invariants (do not break)

- Every `match_predictions` row has `predicted_utc < kickoff_utc` (DB guard +
  application guard in `predict.py`).
- No stage updates or deletes a logged prediction or score.
- Sims are reproducible under the fixed seed (`RANDOM_SEED`).
