# PitchEdge — Architecture & Methodology

## 1. System overview

```
                 ┌─────────────────────────────────────────────┐
                 │                Postgres 16                   │
                 │  raw_results · fixtures · odds_snapshots ·   │
                 │  team_ratings · match_predictions (receipts) │
                 │  prediction_scores · sim_results             │
                 └───────────────▲───────────────▲─────────────┘
                                 │               │
   ┌──────────────┐   ingest     │               │   read
   │ data sources │──────────────┘               │
   │ odds / fixt. │                              │
   │ results/hist │              ┌───────────────┴───────────────┐
   └──────────────┘              │           model layer          │
                                 │  elo · dixon_coles · devig ·   │
   ┌──────────────┐  schedule    │  blend · monte_carlo_sim       │
   │ cron/launchd │──────────────┤                                │
   └──────────────┘              └───────────────┬───────────────┘
                                                 │
        ┌────────────────────────────────────────┼──────────────────────┐
        │                                         │                      │
 ┌──────▼───────┐                        ┌────────▼────────┐    ┌────────▼────────┐
 │ scoring job  │                        │ content engine  │    │ Streamlit + LP  │
 │ Brier/logloss│                        │ LLM + Pillow →  │    │ dashboard +     │
 │ reliability  │                        │ Telegram / X    │    │ calibration UI  │
 └──────────────┘                        └─────────────────┘    └─────────────────┘
```

Single Python package `pitchedge`, one Postgres instance, all orchestrated by a
nightly scheduler. Same topology as OptionsEdge: pipeline → model → notify →
dashboard.

## 2. Data model (Postgres)

Tables (Cursor will write the migrations; this is the contract):

- **teams**: `team_id PK`, `name`, `fifa_code`, `confederation`, `group_label`.
- **raw_results**: historical & tournament matches — `match_id PK`, `date`,
  `home_id`, `away_id`, `home_goals`, `away_goals`, `competition`, `neutral bool`.
  This is both training data and (post-kickoff) ground truth.
- **fixtures**: scheduled WC matches — `fixture_id PK`, `kickoff_utc`,
  `home_id`, `away_id`, `stage`, `group_label`, `status` (scheduled/final).
- **odds_snapshots**: `id PK`, `fixture_id FK`, `book`, `captured_utc`,
  `home_odds`, `draw_odds`, `away_odds` (decimal). Multiple snapshots per fixture.
- **team_ratings**: `id PK`, `team_id FK`, `as_of_date`, `elo`,
  `attack_strength`, `defense_strength`. Snapshotted so ratings are reproducible.
- **match_predictions** (THE RECEIPTS — append-only): `id PK`, `fixture_id FK`,
  `model_version`, `predicted_utc`, `p_home`, `p_draw`, `p_away`, `exp_home_goals`,
  `exp_away_goals`, `source` ('model'|'market'|'blend'). A CHECK constraint or
  app-level guard must reject inserts where `predicted_utc >= kickoff_utc`.
- **prediction_scores**: `prediction_id FK`, `brier`, `log_loss`, `outcome`
  (H/D/A), `scored_utc`. Written only after the match is final.
- **sim_results**: `id PK`, `run_batch_utc`, `team_id FK`, `p_advance_group`,
  `p_r16`, `p_qf`, `p_sf`, `p_final`, `p_win`, `n_sims`.

Immutability rule: `match_predictions` and `prediction_scores` are never UPDATEd
for an existing prediction. New model version → new rows. This is what makes the
track record credible.

## 3. Model methodology

### 3.1 De-vig (market baseline) — the calibration floor
For a 3-way market with decimal odds `(d_H, d_D, d_A)`:
- Raw implied: `r_i = 1/d_i`. Overround = `Σ r_i` (≈1.05–1.08).
- **MVP:** proportional normalization `p_i = r_i / Σ r_j`.
- **Upgrade:** Shin's method or the power/`Wisdom-of-Crowd` method to correct
  favorite–longshot bias. Implement proportional first; gate Shin behind a flag.

This de-vigged vector is the single most calibrated input we have. Treat it as
the benchmark to beat *and* the anchor we blend toward.

### 3.2 Elo (fast, robust baseline)
World-Football-style Elo: update after each historical match with goal-difference
weighting and home advantage. Cheap, hard to break, good cold-start prior for
low-data teams. Use Elo-implied win prob as a feature and a sanity check.

### 3.3 Dixon-Coles bivariate Poisson (the differentiator)
Per team: attack strength `α_i`, defense strength `β_i`; global home advantage
`γ`; low-score dependence correction `τ(·)` governed by `ρ`. Expected goals:
- `λ_home = exp(α_home + β_away + γ)` (omit `γ` when `neutral=True`)
- `λ_away = exp(α_away + β_home)`
**Prediction venue policy:** WC 2026 and historical cup backtests call
`wc_match_probs` / `tournament_match_probs` (wrappers defaulting to `neutral=True`).
Only co-host nations listed as `fixtures.home_id` (USA/Mexico/Canada `team_id`
13/1/5) or an explicit `host_home=True` use `neutral=False`. See
`src/pitchedge/model/venues.py`.
Score matrix from the Poisson PMFs with the Dixon-Coles `τ` correction on the
(0,0),(0,1),(1,0),(1,1) cells. **Time-decay**: weight each historical match by
`exp(-ξ · Δt)` so recent form dominates — fit `ξ` on a validation tournament.
Sum the score matrix into P(home/draw/away); it also yields correct-score,
over/under, BTTS for free (content gold).

For internationals, **shrink** `α_i, β_i` toward confederation/Elo-implied priors
for teams with few matches. Do not trust raw params for low-data teams.

### 3.4 Blend
`p_blend = w · p_model + (1 - w) · p_market`, renormalized. Choose `w` by
minimizing out-of-sample log loss on backtests — expect `w` to land low
(market is strong). Honest framing: the blend is to avoid embarrassment, not to
claim alpha.

### 3.5 Monte Carlo tournament sim
- **Group stage:** 12 groups of 4, double round-robin? No — single round-robin
  (each team plays the other 3). Simulate each match by sampling a scoreline from
  the model's score matrix. Apply **real FIFA 2026 tiebreakers** in order: points
  → goal difference → goals scored → head-to-head (points, GD, goals among tied)
  → fair-play → drawing of lots (random). 
- **Qualification:** top 2 of each group + the **8 best third-placed teams**
  (ranked by pts, GD, GF) advance to a 32-team knockout.
- **Bracket:** the 2026 R32 bracket assignment of third-placed teams is
  combinatorially fiddly (depends on *which* groups' thirds qualify). Implement
  the official mapping table; this gets its own validation phase.
- **Knockouts:** single-elimination; draws after 90' → simulate ET, then
  penalties as ~50/50 (slight nudge toward higher-Elo team is optional, document
  the choice). 
- Run ≥50k sims with a fixed seed; aggregate to per-team advancement/title probs.

## 4. Scoring & calibration

- **Per prediction:** multi-class Brier and log loss vs. realized H/D/A.
- **Reliability diagram:** bin predicted probs, plot predicted vs. observed
  frequency, with Wilson confidence bands (small-sample honesty).
- Report our model **and** the market baseline on the same axes. The story is
  "are we calibrated," not "did we win."

## 5. Content pipeline

Structured model output (JSON) → Anthropic API prompt → narrative text →
Pillow/matplotlib renders a fixed-template card → post to Telegram Bot API.
Cards use a brand template (NOT AI image generation): free, instant, consistent.

## 6. Scheduling

One nightly job (cron on Linux / launchd on macOS, mirroring OptionsEdge):
refresh data → refit ratings → regenerate predictions for upcoming fixtures
(guarded to pre-kickoff) → run sim → score finished matches → push content.

## 7. Tech choices

Python 3.12, deps via `uv`. `scipy.stats.poisson` + `scipy.optimize.minimize`
(L-BFGS-B) for the Dixon-Coles fit. `pandas`/`numpy` throughout. `psycopg`/
`SQLAlchemy` for DB. `streamlit` for dashboard. `pytest` for tests. Docker
Compose for Postgres + app.
