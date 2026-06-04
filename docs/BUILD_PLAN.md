# PitchEdge — Build Plan (Cursor)

How to use: work **one phase at a time**. Paste the phase's prompt into Cursor
(agent/composer mode), let it build, then run the **Validation** block. Do not
advance until every checkbox passes. `@`-reference `docs/PRD.md`,
`docs/ARCHITECTURE.md`, and `.cursorrules` in each prompt so Cursor keeps context.

Sprint maps to a 7-day runway before the June 11 kickoff. Phases 0–6 are the
critical path; Phase 7 is launch-day; Phase 8+ are post-launch.

---

## Phase 0 — Scaffold & infra  (Day 1, morning)

**Objective:** repo skeleton, Docker Postgres, config, DB layer, migrations.

**Cursor prompt:**
> Read `.cursorrules`, `@docs/PRD.md`, and `@docs/ARCHITECTURE.md`. Scaffold the
> PitchEdge project: a `uv`-managed `pyproject.toml` (Python 3.12) with deps
> pandas, numpy, scipy, sqlalchemy, psycopg[binary], streamlit, pillow,
> matplotlib, python-dotenv, requests, anthropic, pytest. Create the
> `src/pitchedge/` package and `tests/` layout. Add a `docker-compose.yml` with
> Postgres 16. Create `src/pitchedge/config.py` that loads env vars (DB_URL,
> ANTHROPIC_API_KEY, TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, RANDOM_SEED, BLEND_W)
> with sensible defaults, plus `.env.example`. Create `src/pitchedge/db.py` with
> an engine factory and a parameterized-query helper. Implement the full schema
> from ARCHITECTURE §2 as a migration script `src/pitchedge/migrations.py` that
> is idempotent (CREATE TABLE IF NOT EXISTS). Enforce the append-only/pre-kickoff
> rule on `match_predictions` with a CHECK or trigger. Add a `Makefile` with
> `db-up`, `migrate`, `test` targets.

**Validation:**
- [ ] `make db-up && make migrate` runs clean; all tables in ARCHITECTURE §2 exist.
- [ ] Inserting a `match_predictions` row with `predicted_utc >= kickoff_utc`
      is rejected.
- [ ] `.env.example` lists every required var; no secret is hardcoded anywhere.
- [ ] `make test` collects and runs (even if near-empty).

---

## Phase 1 — Data ingestion  (Day 1, afternoon)

**Objective:** load historical results, WC fixtures/teams, and odds snapshots —
idempotently.

> ⚠️ Before this phase, decide your sources and check their *current* free-tier
> limits yourself: historical results (Kaggle "International football results
> 1872→now" CSV or openfootball), fixtures/results (football-data.org or
> API-Football), odds (the-odds-api.com). Confirm endpoints — do not let Cursor
> invent them.

**Cursor prompt:**
> Build `src/pitchedge/ingest/`. (1) `history.py`: load a historical
> international-results CSV (I will provide the path/columns: date, home_team,
> away_team, home_score, away_score, tournament, neutral) into `raw_results`,
> deduping on a natural key. (2) `fixtures.py`: load the 48 WC teams (with
> confederation + group) and the 104 fixtures from a CSV I provide, into `teams`
> and `fixtures`. (3) `odds.py`: an adapter for the odds API that writes
> `odds_snapshots`; isolate the HTTP call behind a function so it's mockable, and
> write a fixture-based test using a saved sample JSON response. All ingests must
> be idempotent (re-running inserts nothing new). Add `pytest` tests for dedup
> and for the odds adapter against the sample fixture. Mark any uncertain API
> field with a TODO and tell me what to confirm.

**Validation:**
- [ ] Re-running each ingest twice leaves row counts unchanged (idempotent).
- [ ] `teams` has 48 rows across 12 groups; `fixtures` has 104 rows.
- [ ] Odds adapter test passes against the saved sample JSON (no live call in tests).
- [ ] Spot-check 3 historical matches against a known source — scores match.

---

## Phase 2 — De-vig + Elo  (Day 1 eve / Day 2 morning)

**Objective:** the market baseline and the robust rating prior.

**Cursor prompt:**
> Build `src/pitchedge/model/devig.py` and `src/pitchedge/model/elo.py`.
> `devig.py`: convert decimal 3-way odds to probabilities via proportional
> normalization (default) with a flagged option for Shin's method; return a
> normalized (p_home, p_draw, p_away). `elo.py`: a World-Football-style Elo that
> ingests `raw_results` chronologically with goal-difference weighting and a home
> advantage term (skip HA for neutral matches), writing snapshots to
> `team_ratings`. Expose `elo_win_prob(elo_a, elo_b, home_adv)`. Add pytest:
> de-vigged probs sum to 1 and overround is removed; Elo updates are
> zero-sum; a stronger team gets >0.5 win prob.

**Validation:**
- [ ] De-vig output sums to 1.0 (±1e-9) and is < the raw implied sum.
- [ ] After processing history, plausible teams top the Elo table (sanity: the
      usual elites are high; minnows low).
- [ ] Elo total rating change per match ≈ 0 (zero-sum check passes).

---

## Phase 3 — Dixon-Coles + blend  (Day 2)

**Objective:** the differentiating model and the market blend.

**Cursor prompt:**
> Build `src/pitchedge/model/dixon_coles.py` per ARCHITECTURE §3.3. Fit attack
> `α_i` / defense `β_i` per team, global home advantage `γ`, and low-score
> correction `ρ` by maximizing the time-decayed (`exp(-ξ·Δt)`) Dixon-Coles
> log-likelihood with `scipy.optimize.minimize` (L-BFGS-B), identifiability
> constraint `mean(α)=0`. Provide `score_matrix(home_id, away_id, max_goals=10)`
> returning the corrected Poisson score-prob matrix, and `match_probs(...)`
> summing it to (p_home, p_draw, p_away) plus expected goals, over/under 2.5,
> BTTS. For teams with fewer than N qualifying matches, shrink params toward the
> Elo-implied prior and log a warning. Then build
> `src/pitchedge/model/blend.py`: `blend(p_model, p_market, w)` → renormalized
> mix, with `w` from config. Tests: score matrix sums to 1; match_probs sum to
> 1; a strong-vs-weak fixture yields sensible favorite prob; blend with w=0
> equals market, w=1 equals model.

**Validation:**
- [ ] Optimizer converges (report final NLL); `mean(α)≈0`.
- [ ] `score_matrix` sums to ~1 (±1e-6) for several fixtures.
- [ ] Low-data teams trigger the shrinkage warning.
- [ ] Blend endpoints (w=0, w=1) behave exactly as expected.

---

## Phase 4 — Backtest & blend tuning  (Day 2 eve / Day 3 morning)

**Objective:** prove calibration and pick `w` honestly. This is the credibility
phase — do not skip or rush it.

**Cursor prompt:**
> Build `src/pitchedge/eval/backtest.py`. Using held-out historical tournaments
> (e.g. WC 2018, WC 2022, Euro 2024, Copa America 2024 — I'll specify), generate
> model, market (where odds exist), and blended pre-match probabilities, then
> compute multi-class Brier and log loss for each. Sweep blend weight `w` over
> [0,1] and report the `w` minimizing out-of-sample log loss. Build
> `src/pitchedge/eval/calibration.py`: bin predictions and produce a reliability
> diagram (predicted vs observed frequency) with Wilson confidence bands, for our
> model and the market on the same axes; save as PNG. Write the chosen `w` and
> the scores to a `backtest_report.md`. Be explicit in the report if the model
> does NOT beat the market — that is an acceptable and expected outcome.

**Validation:**
- [ ] `backtest_report.md` shows Brier + log loss for model / market / blend.
- [ ] Chosen `w` is justified by the log-loss sweep (expect it to lean toward
      market — that's fine and on-brand).
- [ ] Reliability PNG renders; our curve is reasonably near the diagonal.
- [ ] Set `BLEND_W` in config to the chosen value.

---

## Phase 5 — Monte Carlo tournament sim  (Day 3)

**Objective:** per-team advancement and title probabilities under the real 2026
format. The bracket logic is the trap — give it the validation it deserves.

**Cursor prompt:**
> Build `src/pitchedge/sim/tournament.py` per ARCHITECTURE §3.5. Simulate the
> 12-group stage (single round-robin) by sampling scorelines from the blended
> model's score matrix; rank with the official FIFA 2026 tiebreakers in order
> (points, GD, GF, head-to-head, fair-play, random). Select top 2 per group plus
> the 8 best third-placed teams, then assign them into the Round-of-32 bracket
> using the official 2026 mapping (implement the mapping table explicitly —
> I will provide/confirm it). Simulate knockouts with ET and 50/50 penalties on
> draws. Run N=50000 sims with the config seed; aggregate per-team
> P(advance/R16/QF/SF/final/win) into `sim_results`. Tests: each group's
> P(advance) sums to ~2 over top-2 logic; global P(win) sums to ~1; results are
> reproducible under a fixed seed.

**Validation:**
- [ ] Sum of all teams' P(win title) ≈ 1.0 (±0.01).
- [ ] Within each group, the four teams' P(advance) sum to ~2.0 (top-2) before
      adding best-thirds.
- [ ] Re-running with the same seed gives identical numbers.
- [ ] Eyeball: favorites' title probs are in a sane ballpark vs. public market
      futures (not off by an order of magnitude).

---

## Phase 6 — Receipts + scoring + content engine  (Day 4)

**Objective:** log predictions immutably, score finished matches, and generate
+ post content.

**Cursor prompt:**
> Build three things. (1) `src/pitchedge/predict.py`: for each upcoming fixture,
> compute model/market/blend probs and INSERT into `match_predictions` (source
> tagged), enforcing predicted_utc < kickoff_utc. (2) `src/pitchedge/score.py`:
> for fixtures now final, compute Brier + log loss per logged prediction and
> write `prediction_scores` (never mutate predictions). (3)
> `src/pitchedge/content/`: `narrative.py` calls the Anthropic API
> (model claude-sonnet, max_tokens ~700) with a structured JSON of the fixture's
> probabilities and asks for a tight, non-hype preview — pass the numbers in, do
> not let the LLM invent stats; `card.py` renders a fixed-template PNG with
> Pillow/matplotlib showing the three probabilities + expected score; `telegram.py`
> posts text+image via the Bot API. Add a `daily_disagreement` function that
> finds the fixture where |model − market| is largest and drafts that post.
> Tests: prediction insert is rejected post-kickoff; scoring is idempotent;
> narrative function is tested with a mocked API response (no live call).

**Validation:**
- [ ] A prediction logged for an upcoming fixture appears in `match_predictions`;
      attempting to log one after kickoff is rejected.
- [ ] After feeding a finished result, `prediction_scores` populates; re-running
      does not double-score.
- [ ] `card.py` produces a clean, on-brand PNG for a sample fixture.
- [ ] A test post reaches your Telegram channel (text + card).
- [ ] Narrative test passes with the mocked Anthropic response.

---

## Phase 7 — Dashboard, landing page, LAUNCH  (Day 5–6)

**Objective:** public surface + the pre-tournament splash.

**Cursor prompt:**
> Build `src/pitchedge/app.py` (Streamlit): tabs for (a) upcoming match
> probabilities, (b) tournament-odds table from `sim_results` sorted by P(win),
> (c) bracket/advancement view, (d) the live calibration tracker reading
> `prediction_scores` (reliability diagram + running Brier/log loss for model vs
> market, with the small-sample caveat shown in the UI). Add a single-page
> landing section with an email capture and a Telegram-join CTA. Read the
> `.cursorrules` and follow the calibration-over-accuracy framing in all copy;
> never state we beat the market.

**Validation:**
- [ ] Dashboard runs (`streamlit run`) and all four tabs render from real DB data.
- [ ] Calibration tab shows the caveat about small samples.
- [ ] **Day 6 launch:** publish the full pre-tournament board (every team's title
      %, your top model-vs-market disagreements). This is your splash AND your
      locked-in receipts baseline — every prob logged before June 11.

---

## Phase 8 — Scheduler & hardening  (Day 7, buffer)

**Cursor prompt:**
> Build `src/pitchedge/scheduler.py` orchestrating the nightly pipeline: refresh
> data → refit ratings → log predictions for upcoming fixtures (pre-kickoff
> guard) → run sim → score finished matches → push the day's content. Provide a
> cron entry and a macOS launchd plist (mirror the OptionsEdge setup). Add
> structured logging and a dry-run flag. Write a short `RUNBOOK.md`: how to run
> manually, how to recover from a failed night, where logs go.

**Validation:**
- [ ] `scheduler.py --dry-run` walks the full pipeline without writing/posting.
- [ ] A real run logs each stage and completes idempotently.
- [ ] `RUNBOOK.md` exists and is accurate.

---

## Cross-cutting validation (run before launch)

- [ ] Every prediction in `match_predictions` has `predicted_utc < kickoff_utc`.
      (Run a SQL audit — this is the integrity of your entire brand.)
- [ ] No code path updates or deletes a logged prediction or score.
- [ ] No secret committed; `.env` gitignored.
- [ ] Sims reproducible under the fixed seed.
- [ ] No UI/log/comment string claims the model beats the market.
- [ ] You can explain, in one sentence each, where `w` came from and how the
      track record is scored. If you can't, neither can your audience — fix it.
