#!/bin/zsh
# Runs `finscope check` for launchd (see scripts/com.finscope.check.plist and the README).
# Steps: go to the project folder, where .env and data/ live; append to data/logs/check.log;
# write the date; wait for the network; run check. uv is called by its full path: launchd
# doesn't load your PATH.
cd "${0:A:h}/.." || exit 1
mkdir -p data/logs
exec >>data/logs/check.log 2>&1
echo "=== $(date '+%Y-%m-%d %H:%M %Z') ==="
# A run missed while the Mac slept starts the moment it wakes, often before the network is up
# (2026-11-22: every request failed with "nodename nor servname provided"). Wait up to two
# minutes for SEC to answer; if it never does, run anyway: check reports the failure.
for attempt in {1..12}; do
  /usr/bin/curl --silent --output /dev/null --max-time 5 https://www.sec.gov && break
  echo "network not ready, waiting (attempt $attempt of 12)"
  sleep 10
done
# Unbuffered, so errors land in the log next to the section they belong to.
export PYTHONUNBUFFERED=1
exec "${FINSCOPE_UV:-$HOME/.local/bin/uv}" run --quiet python -m finscope check
