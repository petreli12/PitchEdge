# World Cup 2026 — teams & fixtures

## Quick start (recommended)

We ship a generator from the **confirmed draw** and a published **104-match schedule**:

```bash
# Regenerate CSVs (safe to re-run)
uv run python scripts/build_wc2026_csvs.py

# Load into Postgres (Docker must be running)
make db-up && make migrate && make ingest-fixtures
```

That produces:

- `teams.csv` — **48 teams**, 12 groups (A–L), with optional `odds_name` for the-odds-api spelling
- `fixtures.csv` — **104 fixtures** (72 group + 32 knockout placeholders with empty `home_id`/`away_id`)
- `annex_c.json` — **495** FIFA Annex C third-place → R32 slot mappings (confirm against official PDF before production)

Kickoffs are converted from **US/Eastern → UTC**. Spot-check a few matches on [FIFA](https://www.fifa.com) before you lock predictions.

## Odds API — step by step

PitchEdge uses [the-odds-api.com](https://the-odds-api.com/) (v4), **not** a sportsbook. You only **read** prices into `odds_snapshots`.

### 1. Get a key

1. Create a free account at https://the-odds-api.com/
2. Copy your API key from the dashboard
3. Add to `.env`:

```env
ODDS_API_KEY=your_key_here
ODDS_API_SPORT_KEY=soccer_fifa_world_cup
ODDS_API_REGIONS=us,uk,eu
```

**Sport key:** `soccer_fifa_world_cup` is the standard key for WC match odds. Outrights (tournament winner) use `soccer_fifa_world_cup_winner` — different endpoint, not wired into `ingest/odds.py` yet.

**Quota:** Each request costs `(number of markets) × (number of regions)`. Default ingest uses one market (`h2h`) and your regions string — e.g. `us,uk,eu` = 3× per call. Check your plan limits on the dashboard.

### 2. Probe without guessing

```bash
uv run python scripts/probe_odds_api.py
```

This prints:

- World Cup–related sport keys on **your** account (and whether they’re `active`)
- Sample `home_team` / `away_team` strings from live odds
- A sample `h2h` outcomes block (verify `"Draw"` and team spelling)

### 3. Align team names

`ingest/odds.py` matches API teams to fixtures via `(home_name, away_name)`:

- Uses `teams.odds_name` when set, otherwise `teams.name`
- `teams.csv` from the generator already sets `odds_name` where we expect differences, e.g.:
  - `South Korea` → leave `odds_name` empty (API uses `South Korea`, not `Korea Republic`)
  - `Ivory Coast` → `Côte d'Ivoire`
  - `DR Congo` → `Congo DR`
  - `Turkiye` → `Turkey`
  - `Cape Verde` → `Cabo Verde`

If `probe_odds_api.py` shows different spellings, edit `odds_name` in `teams.csv` and re-run `make ingest-fixtures` (idempotent).

### 4. Ingest odds

```bash
make ingest-fixtures   # must be loaded first (needs team IDs + kickoffs)
make ingest-odds       # live API call; requires ODDS_API_KEY
```

Events with no matching fixture are **skipped** (logged). Knockout rows without `home_id`/`away_id` are skipped until you fill them in.

**When to run:** The API returns **upcoming** matches. Empty results before books post WC lines are normal; retry closer to kickoff.

## Manual CSVs (if you prefer)

### `teams.csv`

| Column | Required | Description |
|--------|----------|-------------|
| team_id | yes | Primary key (integer) |
| name | yes | Display / default match name |
| fifa_code | no | FIFA code |
| confederation | yes | UEFA, CONMEBOL, CAF, AFC, CONCACAF, OFC |
| group_label | yes | A–L |
| odds_name | no | the-odds-api string if different from `name` |
| history_name | no | Kaggle `results.csv` label if different from `name` (e.g. Czechia → Czech Republic) |

### `fixtures.csv`

| Column | Required | Description |
|--------|----------|-------------|
| fixture_id | yes | Primary key (1–104) |
| kickoff_utc | yes | ISO-8601 UTC |
| home_id | yes* | FK to `teams` (*empty for TBD knockout slots) |
| away_id | yes* | FK to `teams` |
| stage | yes | Group A, Round of 32, Final, … |
| group_label | yes | Group letter or empty for knockouts |
| status | no | `scheduled` (default) or `final` |

## Validate

```bash
make test
# After ingest:
#   SELECT count(*) FROM teams;      -- 48
#   SELECT count(*) FROM fixtures;   -- 104
```
