#!/bin/zsh
# Runs `finscope check` for launchd (see scripts/com.finscope.check.plist and the README).
# Steps: go to the project folder, where .env and data/ live; append to data/logs/check.log;
# write the date; run check. uv is called by its full path: launchd doesn't load your PATH.
cd "${0:A:h}/.." || exit 1
mkdir -p data/logs
exec >>data/logs/check.log 2>&1
echo "=== $(date '+%Y-%m-%d %H:%M %Z') ==="
exec "${FINSCOPE_UV:-$HOME/.local/bin/uv}" run --quiet python -m finscope check
