#!/usr/bin/env bash
# Cron launches a job in an owned session; the Python lock serializes all ticks.
set -euo pipefail
daily_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
daily_log="$daily_root/refs/scans/paper-daily-cron.log"
mkdir -p "$daily_root/refs/scans"
# Deadline monitoring runs even while the scheduler is processing a long paper.
daily_watch_session="exec-paper-daily-watch"
if ! tmux has-session -t "=$daily_watch_session" 2>/dev/null; then
  printf -v daily_watch_cmd 'cd %q && /usr/bin/python3 scripts/daily_pipeline.py watch >> %q 2>&1' "$daily_root" "$daily_log"
  tmux new-session -d -s "$daily_watch_session" "$daily_watch_cmd"
fi
daily_session="exec-paper-daily"
if tmux has-session -t "=$daily_session" 2>/dev/null; then
  exit 0
fi
printf -v daily_cmd 'cd %q && /usr/bin/python3 scripts/daily_pipeline.py tick >> %q 2>&1' "$daily_root" "$daily_log"
tmux new-session -d -s "$daily_session" "$daily_cmd"
tmux set-option -t "$daily_session" remain-on-exit off 2>/dev/null || true
