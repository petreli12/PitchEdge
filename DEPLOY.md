# Deploying the PitchEdge public dashboard

The public dashboard is a **read-only** Streamlit app that serves a frozen
snapshot of the locked pre-tournament board. It does **not** connect to a
database in production — it reads `data/snapshot/*.json`, which is generated
from the live DB before launch. This matches the "locked receipts baseline, not
a live recompute" design.

## 1. Generate / refresh the snapshot (local, against the live DB)

```bash
make export-snapshot          # writes data/snapshot/*.json from Postgres
make dashboard-snapshot       # optional: preview locally in snapshot mode (no DB)
```

`export-snapshot` recomputes nothing; it freezes exactly what the live queries
return (upcoming model/market/blend probs, sim title odds, advancement,
disagreements, and any scored calibration receipts).

## 2. Commit and push to GitHub

```bash
git add -A
git commit -m "Publish pre-tournament board snapshot"
git push                      # to your GitHub remote
```

Committed for deploy: source (`src/`), `streamlit_app.py`, `requirements.txt`,
`data/snapshot/`, `data/wc2026/`. Excluded: `.env`, `.streamlit/secrets.toml`,
and the large/licensed `data/kaggle/` and `data/backtest/` inputs.

## 3. Deploy on Streamlit Community Cloud

1. Go to https://share.streamlit.io and "Create app" from your GitHub repo.
2. **Main file path:** `streamlit_app.py`
3. **Python version:** 3.12
4. (Optional) **Secrets** — none are required. To enable extras, paste from
   `.streamlit/secrets.toml.example`:
   - `TELEGRAM_JOIN_URL` — renders the "Join on Telegram" button.
   - `SUBSCRIBE_FORM_URL` — external email-capture link (Google Form / Tally /
     Formspree), since the public deploy has no writable subscribers DB.

The app auto-detects the bundled `data/snapshot/` and runs read-only.

## Refreshing after launch

When matches finish and you re-score, re-run step 1 and push again; Streamlit
Cloud redeploys on push. The snapshot's `meta.json` records `generated_utc` and
the source DB so you can confirm freshness.

## Notes

- The internal "Launch board" preview is hidden automatically in snapshot mode;
  the public surface is **Predictions** (4 tabs) + **About**.
- Live post-launch scoring with a hosted Postgres is a later option: set `DB_URL`
  in Streamlit secrets to a publicly-reachable database and the app switches to
  live mode (no snapshot needed).
