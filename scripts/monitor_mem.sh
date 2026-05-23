#!/usr/bin/env bash
# Background memory monitor for training runs.
#
# Records to two CSVs while $PID is alive:
#   - monitor_gpu.csv  — nvidia-smi --query-gpu=memory.used,memory.free,util,timestamp
#   - monitor_ps.csv   — ps rss,etime,%cpu,%mem for the training process
#
# Usage:
#   scripts/monitor_mem.sh <pid> [output_dir] [interval_sec]
#
# Both files truncate on start. Loop exits when the PID dies, so launch it from
# the same tmux pane as the trainer (or wrap it after the trainer call).

set -euo pipefail

PID="${1:?Usage: $0 <pid> [output_dir] [interval_sec]}"
OUTDIR="${2:-.}"
INTERVAL="${3:-2}"

mkdir -p "$OUTDIR"
GPU_CSV="$OUTDIR/monitor_gpu.csv"
PS_CSV="$OUTDIR/monitor_ps.csv"

if ! kill -0 "$PID" 2>/dev/null; then
  echo "ERROR: PID $PID not running" >&2
  exit 1
fi

# Header rows.
echo "timestamp,gpu_mem_used_mib,gpu_mem_free_mib,gpu_util_pct" > "$GPU_CSV"
echo "timestamp,pid,rss_kb,etime,cpu_pct,mem_pct" > "$PS_CSV"

echo "monitor_mem.sh: tracking PID=$PID, interval=${INTERVAL}s, out=$OUTDIR"

while kill -0 "$PID" 2>/dev/null; do
  TS=$(date -u +"%Y-%m-%dT%H:%M:%SZ")

  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi \
      --query-gpu=memory.used,memory.free,utilization.gpu \
      --format=csv,noheader,nounits \
      | head -1 \
      | awk -v ts="$TS" -F', *' '{printf "%s,%s,%s,%s\n", ts, $1, $2, $3}' \
      >> "$GPU_CSV"
  fi

  ps -p "$PID" -o pid=,rss=,etime=,%cpu=,%mem= 2>/dev/null \
    | awk -v ts="$TS" '{printf "%s,%s,%s,%s,%s,%s\n", ts, $1, $2, $3, $4, $5}' \
    >> "$PS_CSV"

  sleep "$INTERVAL"
done

echo "monitor_mem.sh: PID $PID exited; logs written to $OUTDIR"
