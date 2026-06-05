#!/usr/bin/env bash
#
# PitchEdge nightly pipeline wrapper for cron / launchd.
#
# Resolves the repo root from this script's location, ensures the log directory
# exists, and runs the scheduler with the project's virtualenv. Any extra args
# are forwarded to the scheduler (e.g. --dry-run, --within-hours 36).
#
# Prerequisites: the Docker Postgres must be running (docker compose up -d db)
# and `.env` must be populated. See RUNBOOK.md.

set -euo pipefail

# Repo root = parent of this script's directory.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT}"

LOG_DIR="${PITCHEDGE_LOG_DIR:-${ROOT}/logs}"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/nightly-$(date +%F).log"

# Prefer the in-repo virtualenv; fall back to `uv run` if it is not present.
if [[ -x "${ROOT}/.venv/bin/python" ]]; then
  PY=("${ROOT}/.venv/bin/python")
else
  PY=(uv run python)
fi

# Prevent idle sleep for the duration of the run (macOS). Critical when a pmset
# scheduled wake fires this job overnight: caffeinate holds the system awake
# until the scheduler process exits, so the Mac can't doze back off mid-pipeline.
CAFFEINATE=()
if command -v caffeinate >/dev/null 2>&1; then
  CAFFEINATE=(caffeinate -i)
fi

echo "=== pitchedge nightly $(date -u +%FT%TZ) ===" >>"${LOG_FILE}"
exec "${CAFFEINATE[@]}" "${PY[@]}" -m pitchedge.scheduler --log-file "${LOG_FILE}" "$@"
