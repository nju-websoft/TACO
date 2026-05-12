#!/usr/bin/env bash
# stop_seed_sim.sh — Kill all running seed_similarity_filter.py processes.

cd "$(dirname "$0")"

pids=$(pgrep -f "seed_similarity_filter.py")
if [ -z "$pids" ]; then
  echo "No running seed_similarity_filter.py processes."
  exit 0
fi

echo "Killing PIDs: $pids"
echo "$pids" | xargs -r kill
sleep 2

remaining=$(pgrep -f "seed_similarity_filter.py")
if [ -n "$remaining" ]; then
  echo "Force-killing remaining: $remaining"
  echo "$remaining" | xargs -r kill -9
fi
echo "Stopped."
