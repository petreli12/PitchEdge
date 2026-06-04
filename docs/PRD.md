# PitchEdge — Product Requirements Document (PRD)

## 1. Problem & positioning

Sports prediction content is saturated with overconfident, unscored "hot
takes." There is almost no widely-followed source that (a) produces its own
defensible probabilistic model and (b) publishes a transparent, scored track
record. The 2026 World Cup (kickoff June 11, final July 19) is a ~5-week window
of maximal attention and a natural acquisition event.

**Positioning:** "The calibrated World Cup model that shows its receipts."
We compete on honesty and presentation, not on alpha.

## 2. Goals (this cycle)

| # | Goal | Metric |
|---|------|--------|
| G1 | Ship a working model before kickoff | Pre-tournament board published by June 10 |
| G2 | Be calibrated, not lucky | Out-of-sample log loss within ~2% of de-vigged market baseline on backtests |
| G3 | Build an audience | Telegram subs + email captures; track daily |
| G4 | Establish the "receipts" brand | Public, auto-updating calibration tracker live by Round of 32 |

**Explicit non-goal:** beating the closing line. If we beat it, great, but the
product does not depend on it and we never market it.

## 3. Users

- **Primary:** soccer fans who want smarter-than-pundit analysis and like data.
- **Secondary (later):** bettors who want a calibrated reference (analytics
  only), and small media/tipster accounts who'd license the probabilities (B2B).

## 4. Functional requirements

### 4.1 Data
- Ingest fixtures, results, and pre-match odds for all 48 teams / 104 matches.
- Ingest historical international results (training data, ~1990→present).
- Idempotent nightly refresh; never double-insert.

### 4.2 Model
- **Baseline:** de-vigged market probabilities per match (the calibration floor).
- **Own model:** Dixon-Coles bivariate Poisson with time-decay + an Elo feature.
- **Blend:** shrink own-model probs toward market with a tunable weight `w`.
- **Match outputs:** P(home win / draw / away win), expected goals, correct-score
  matrix, over/under 2.5, BTTS.
- **Tournament sim:** Monte Carlo (≥50k runs) producing, per team: P(advance from
  group), P(reach R16/QF/SF/final), P(win title). Must implement the real 2026
  format (12 groups of 4 → top 2 + 8 best third-placed → Round of 32).

### 4.3 Track record ("receipts")
- Log every match probability with a server timestamp **strictly before
  kickoff**. Immutable once kickoff passes.
- After results post, score each prediction: Brier, log loss, and a hit flag.
- Maintain a running reliability diagram (predicted vs. observed frequency) for
  both our model and the market baseline.

### 4.4 Content engine
- For each upcoming match: generate a narrative preview from structured model
  output via the Anthropic API.
- Render a branded card image (fixed template) with the key probabilities.
- Auto-distribute to Telegram channel. X posting optional (see constraints).
- "Model vs. market disagreement" daily post — highest-engagement format.

### 4.5 Public surface
- Streamlit dashboard: live match probabilities, tournament-odds table, bracket
  sim, and the live calibration tracker.
- One-page landing site with email + Telegram capture.

## 5. Non-functional requirements

- **Reproducible:** fixed seeds for sims; deterministic given inputs.
- **Auditable:** predictions are append-only; scoring never mutates a logged prob.
- **Cheap:** free-tier data sources; Telegram free; LLM + image gen the only real
  variable cost (pennies/match).
- **Calibrated > accurate:** we optimize and report proper scoring rules, not
  win-rate vanity metrics.

## 6. Constraints & honest risks

- **Small-sample calibration.** 104 matches is statistically noisy. Present the
  track record with humility; one cold streak should not read as "the model is
  broken." Show confidence bands on the reliability curve.
- **International data is thin.** Many teams play few comparable matches → heavy
  reliance on shrinkage and the market blend. Do not over-trust the raw
  Dixon-Coles params for low-data teams.
- **X (Twitter) API is no longer free-ish.** Write access is rate-limited and
  paid tiers are pricey. MVP = Telegram auto-post + manual X posting. Do not
  block the build on X automation.
- **Distribution is the real bottleneck**, not the code. Budget real effort for
  it (subreddits, soccer X, your own channels).
- **Post-tournament churn.** Audience built on a one-month event decays fast.
  A "what happens after July 19" plan is a P1 follow-up, out of scope here.

## 7. Out of scope (this cycle)
- Real-money betting features of any kind.
- In-play / live model updates.
- Mobile app (the dashboard is web-only for now).
- Paid subscription plumbing (audience first, monetize later).
