# PitchEdge

A **publicly calibrated** World Cup prediction engine and content flywheel.

> Working name — rename freely. It deliberately echoes the OptionsEdge brand family.

## The one-sentence pitch

The only World Cup model that posts its probabilities *before* kickoff and
publishes its own calibration scorecard — model vs. market, with receipts.

## What this is (and is not)

- **It is** an analytics + content product. We forecast match and tournament
  outcomes, generate narrative + visual content, and publish a transparent,
  scored track record.
- **It is not** a betting service, tipster account, or anything that places,
  brokers, or facilitates real-money wagers. We sit strictly on the
  information/analytics side. (Not legal advice — confirm your own
  jurisdiction before adding anything that touches real money.)

## The strategic wedge

We do **not** claim to beat the market. The closing line from sharp books is
better-calibrated than anything we can train on thin international data in a
week. Our edge is **intellectual honesty and presentation**:

1. A model that is *defensibly our own* (Dixon-Coles + Elo), blended toward the
   de-vigged market so it never embarrasses itself.
2. Every pre-kickoff probability logged immutably, then scored (Brier, log loss)
   against results.
3. A weekly published reliability curve. Almost nobody in sports content does
   this. That is the moat.

## Repo layout

```
pitchedge/
├── README.md            # this file
├── .cursorrules         # Cursor project rules — read first
├── docs/
│   ├── PRD.md           # product requirements
│   ├── ARCHITECTURE.md  # system design, data model, model methodology
│   └── BUILD_PLAN.md    # phased build w/ Cursor prompts + validations
├── src/pitchedge/       # (built by Cursor)
├── tests/               # (built by Cursor)
├── docker-compose.yml   # Postgres + app (built by Cursor)
└── pyproject.toml       # deps via uv (built by Cursor)
```

## How to use these docs with Cursor

1. Drop this whole folder into a new repo and open it in Cursor.
2. Cursor auto-loads `.cursorrules` as context. Keep `docs/` in the project so
   you can `@`-reference them in prompts.
3. Work through `docs/BUILD_PLAN.md` **one phase at a time**. Paste the phase
   prompt, let Cursor build, then run the phase's validation block **before**
   moving on. Do not batch phases — that is how silent bugs compound.

## Stack

Python 3.12 · uv · Postgres 16 (Docker) · pandas/numpy/scipy · Streamlit ·
Anthropic API (narrative) · Pillow + matplotlib (cards) · Telegram Bot API ·
cron / launchd (scheduler).

Deliberately the same shape as OptionsEdge so ~60% of that infra muscle ports.
