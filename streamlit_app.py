"""Streamlit Community Cloud entrypoint for the PitchEdge public dashboard.

Set this file as the app's "Main file path" on Streamlit Cloud. It:
  1. Puts the ``src/`` package layout on the import path.
  2. Bridges Streamlit secrets into environment variables so ``pitchedge.config``
     (which reads ``os.environ``) picks them up — must happen before any
     ``pitchedge`` import.
  3. Executes the real app (``src/pitchedge/app.py``) on every rerun.

No secrets are required for a working deploy: the app auto-detects the bundled
``data/snapshot/`` and serves it read-only (no Postgres). Optional secrets:
``TELEGRAM_JOIN_URL`` (renders a Join button) and ``SUBSCRIBE_FORM_URL``
(external email-capture link). ``DASHBOARD_SNAPSHOT_DIR`` may override the
snapshot location.
"""

from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

# Bridge Streamlit secrets -> env BEFORE importing pitchedge.config.
_BRIDGED_KEYS = (
    "DASHBOARD_SNAPSHOT_DIR",
    "SUBSCRIBE_FORM_URL",
    "TELEGRAM_JOIN_URL",
    "DB_URL",
)
try:
    import streamlit as st

    for _key in _BRIDGED_KEYS:
        try:
            _val = st.secrets[_key]  # type: ignore[index]
        except Exception:
            continue
        if _val is not None and not os.getenv(_key):
            os.environ[_key] = str(_val)
except Exception:
    # No secrets configured / streamlit not yet initialized: defaults apply.
    pass

# This entrypoint is the public snapshot deploy: default to the bundled snapshot
# unless a secret/env explicitly overrides (e.g. DB_URL for a live deploy).
os.environ.setdefault("DASHBOARD_SNAPSHOT_DIR", str(_SRC.parent / "data" / "snapshot"))

runpy.run_path(str(_SRC / "pitchedge" / "app.py"), run_name="__main__")
