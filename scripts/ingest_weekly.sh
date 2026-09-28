#!/usr/bin/env bash
# Weekly nflverse snapshot; safe to run from cron/launchd (append-only, never overwrites).
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs
exec .venv/bin/lineseek-ingest weekly >> "logs/ingest_weekly_$(date -u +%Y%m%d).log" 2>&1
