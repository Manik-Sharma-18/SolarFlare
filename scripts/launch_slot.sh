#!/usr/bin/env bash
# Multi-user-safe training launcher for SolarFlare.
#
# Session naming: sf-<user>-<slot>-<script_name>
# Occupancy determined from tmux pane + OS process state.
#
# Usage:
#   scripts/launch_slot.sh <slot> <script_path> [extra_args...]
#
# Slots: mini_mps  mini_cpu  studio_mps  studio_cpu  5060ti_cuda
# Exit:  0=launched  1=error  2=slot occupied  3=host unreachable

set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <slot> <script_path> [extra_args...]"
  echo "Slots: mini_mps mini_cpu studio_mps studio_cpu 5060ti_cuda"
  exit 1
fi

SLOT="$1"; SCRIPT="$2"; shift 2

UNSAFE_CUDA=0
SKIP_SYNC=0
SYNC_FIX=0
SKIP_VRAM=0
NEW_ARGS=()
for arg in "$@"; do
  case "$arg" in
    --unsafe-cuda-launch) UNSAFE_CUDA=1 ;;
    --skip-sync-check)    SKIP_SYNC=1 ;;
    --sync-fix)           SYNC_FIX=1 ;;
    --skip-vram-check)    SKIP_VRAM=1 ;;
    *) NEW_ARGS+=("$arg") ;;
  esac
done
set -- "${NEW_ARGS[@]+"${NEW_ARGS[@]}"}"

# VRAM buffer rule (CUDA only): launch blocked if used/total > (100 - BUFFER)%.
# Default buffer 40% → max used 60%. Override via env SF_VRAM_BUFFER_PCT.
VRAM_BUFFER_PCT="${SF_VRAM_BUFFER_PCT:-40}"

USER_PREFIX="${SF_USER:-$(whoami)}"
REPO_LOCAL="/Volumes/T9/IndraAstra/manik/SolarFlare"

# shellcheck source=_sf_audit.sh
source "$REPO_LOCAL/scripts/_sf_audit.sh"

if [[ "$SLOT" == "5060ti_cuda" ]]; then
  if [[ "$UNSAFE_CUDA" == "1" ]]; then
    echo "WARNING: --unsafe-cuda-launch; skipping CUDA audit for $SCRIPT" >&2
  else
    audit_cuda "$SCRIPT" || exit 1
  fi
  [[ "$SKIP_VRAM" == "0" ]] && { audit_vram "$@" || exit 1; }
fi

# --- Slot config ------------------------------------------------------------
case "$SLOT" in
  mini_mps)    DEVICE=mps;  HOST=local;      DIR="$REPO_LOCAL";                    TMUX="tmux";                    PY="python3" ;;
  mini_cpu)    DEVICE=cpu;  HOST=local;      DIR="$REPO_LOCAL";                    TMUX="tmux";                    PY="python3" ;;
  studio_mps)  DEVICE=mps;  HOST=mac-studio; DIR="/Users/admin/ml/manik/SolarFlare"; TMUX="/opt/homebrew/bin/tmux"; PY="python3" ;;
  studio_cpu)  DEVICE=cpu;  HOST=mac-studio; DIR="/Users/admin/ml/manik/SolarFlare"; TMUX="/opt/homebrew/bin/tmux"; PY="python3" ;;
  5060ti_cuda) DEVICE=cuda; HOST=5060ti;     DIR="/home/indra/solarflare";          TMUX="/usr/bin/tmux";           PY="venv/bin/python3" ;;
  *) echo "ERROR: unknown slot '$SLOT'"; exit 1 ;;
esac

STEP=$(basename "$SCRIPT" .py)
SESSION="sf-${USER_PREFIX}-${SLOT}-${STEP}"

# --- sync verification (remote slots only) ----------------------------------
if [[ "$HOST" != "local" && "$SKIP_SYNC" == "0" ]]; then
  SYNC_ARGS=(--slot "$SLOT" --level both)
  [[ "$SYNC_FIX" == "1" ]] && SYNC_ARGS+=(--fix)
  if ! bash "$REPO_LOCAL/scripts/sync_verify.sh" "${SYNC_ARGS[@]}"; then
    echo "ERROR: sync verification failed for $SLOT (host=$HOST)" >&2
    echo "  Re-run with --sync-fix to auto-rsync, or --skip-sync-check to bypass." >&2
    exit 1
  fi
fi

# --- helpers ----------------------------------------------------------------
run() {
  if [[ "$HOST" == "local" ]]; then TERM=xterm-256color bash -c "$1"
  else ssh -o ConnectTimeout=5 "$HOST" "$1"; fi
}

count_children() {
  run "pgrep -P $1 2>/dev/null | wc -l | tr -d ' '" 2>/dev/null || echo "0"
}

# --- check occupancy --------------------------------------------------------
EXISTING=$(run "$TMUX list-sessions -F '#{session_name}' 2>/dev/null | grep -E '^sf-[^-]+-${SLOT}-'" 2>/dev/null || true)

if [[ -n "$EXISTING" ]]; then
  while IFS= read -r s; do
    info=$(run "$TMUX list-panes -t '$s' -F '#{pane_dead} #{pane_pid}' 2>/dev/null | head -1" 2>/dev/null || echo "1 0")
    dead=$(awk '{print $1}' <<< "$info"); pid=$(awk '{print $2}' <<< "$info")
    if [[ "$dead" == "1" ]]; then
      run "$TMUX kill-session -t '$s' 2>/dev/null" || true; continue
    fi
    n=$(count_children "$pid")
    if [[ "${n:-0}" -gt 0 ]]; then
      echo "SLOT OCCUPIED: $SLOT has running session '$s'"; exit 2
    else
      run "$TMUX kill-session -t '$s' 2>/dev/null" || true
    fi
  done <<< "$EXISTING"
fi

# --- launch -----------------------------------------------------------------
LOG="$DIR/logs/${STEP}__${SLOT}.log"
run "mkdir -p $DIR/logs"

CMD="cd $DIR && SF_SLOT=$SLOT $PY -u $SCRIPT --device $DEVICE $* 2>&1 | tee $LOG"
LAUNCH="/tmp/sf_launch_${SESSION}.sh"

if [[ "$HOST" == "local" ]]; then
  echo "$CMD" > "$LAUNCH" && chmod +x "$LAUNCH"
  # Unset $TMUX env var: controller daemon runs inside sf-controller tmux session,
  # subprocess inherits $TMUX, then nested `tmux new-session` dies ~35s. Strip it here.
  env -u TMUX TERM=xterm-256color $TMUX new-session -d -s "$SESSION" bash "$LAUNCH"
else
  ssh -o ConnectTimeout=5 "$HOST" "cat > $LAUNCH && chmod +x $LAUNCH" <<< "$CMD"
  ssh -o ConnectTimeout=5 "$HOST" "$TMUX new-session -d -s $SESSION bash $LAUNCH"
fi

sleep 1
INFO=$(run "$TMUX list-panes -t $SESSION -F '#{pane_dead} #{pane_pid}' 2>/dev/null | head -1" 2>/dev/null || echo "1 0")
[[ "$(awk '{print $1}' <<< "$INFO")" == "1" ]] && { echo "ERROR: pane died. Check $LOG on $HOST"; exit 3; }

echo "LAUNCHED: $SESSION on $SLOT (host=$HOST)"
echo "  Log: $LOG"
echo "  Monitor: bash scripts/slot_status.sh"
