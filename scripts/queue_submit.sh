#!/usr/bin/env bash
# queue_submit.sh — Submit a training run to the SolarFlare controller queue.
#
# Identity (in order): --user flag > SF_USER env > .sf_user file > interactive prompt
#
# Usage:
#   queue_submit.sh <script_path> \
#       [--user <name>] [--step-name <name>] \
#       [--device-pref any|cuda|mps|cpu] \
#       [--slot-pref mini_mps|mini_cpu|studio_mps|studio_cpu|5060ti_cuda] \
#       [--args "<extra args>"] [--priority <int>]
#
# Env: SF_CONTROLLER_URL (default http://mac-mini.local:7434)

set -euo pipefail

CONTROLLER_URL="${SF_CONTROLLER_URL:-http://mac-mini.local:7434}"
USER_FILE="$(pwd)/.sf_user"

[[ $# -lt 1 ]] && { echo "Usage: $0 <script_path> [options]" >&2; exit 1; }

SCRIPT_PATH="$1"; shift
STEP_NAME="" DEVICE_PREF="any" SLOT_PREF="" EXTRA_ARGS="" PRIORITY=0 USER_FLAG=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --user)        USER_FLAG="$2";   shift 2 ;;
    --step-name)   STEP_NAME="$2";   shift 2 ;;
    --device-pref) DEVICE_PREF="$2"; shift 2 ;;
    --slot-pref)   SLOT_PREF="$2";   shift 2 ;;
    --args)        EXTRA_ARGS="$2";  shift 2 ;;
    --priority)    PRIORITY="$2";    shift 2 ;;
    *) echo "Unknown option: $1" >&2; exit 1 ;;
  esac
done

# Identity resolution
if [[ -n "$USER_FLAG" ]]; then
  SF_USER="$USER_FLAG"
elif [[ -n "${SF_USER:-}" ]]; then
  :
elif [[ -f "$USER_FILE" ]]; then
  SF_USER=$(head -1 "$USER_FILE" | tr -d '[:space:]')
fi

if [[ -z "${SF_USER:-}" ]]; then
  if [[ ! -t 0 && ! -r /dev/tty ]]; then
    echo "ERROR: no identity. Create $USER_FILE or pass --user <name>." >&2; exit 1
  fi
  DEFAULT=$(basename "$(dirname "$(pwd)")")
  echo "" >&2
  echo "── First-time setup: SolarFlare identity not configured ──" >&2
  printf "  Enter your user id [%s]: " "$DEFAULT" >&2
  read -r input < /dev/tty
  SF_USER="${input:-$DEFAULT}"
  SF_USER=$(echo "$SF_USER" | tr -d '[:space:]')
  [[ -z "$SF_USER" ]] && { echo "ERROR: empty user id." >&2; exit 1; }
  echo "$SF_USER" > "$USER_FILE"
  echo "  Saved → $USER_FILE" >&2
fi

[[ -z "$STEP_NAME" ]] && STEP_NAME=$(basename "$SCRIPT_PATH" .py)

case "$DEVICE_PREF" in
  any|cuda|mps|cpu) ;;
  *) echo "ERROR: --device-pref must be: any cuda mps cpu" >&2; exit 1 ;;
esac

if [[ -n "$SLOT_PREF" ]]; then
  case "$SLOT_PREF" in
    mini_mps|mini_cpu|studio_mps|studio_cpu|5060ti_cuda) ;;
    *) echo "ERROR: --slot-pref must be: mini_mps mini_cpu studio_mps studio_cpu 5060ti_cuda" >&2; exit 1 ;;
  esac
fi

PAYLOAD=$(python3 -c "
import json, sys
d = {'user': sys.argv[1], 'step_name': sys.argv[2], 'script_path': sys.argv[3],
     'args': sys.argv[4], 'device_pref': sys.argv[5], 'priority': int(sys.argv[6])}
if sys.argv[7]: d['slot_pref'] = sys.argv[7]
print(json.dumps(d))
" "$SF_USER" "$STEP_NAME" "$SCRIPT_PATH" "$EXTRA_ARGS" "$DEVICE_PREF" "$PRIORITY" "$SLOT_PREF")

RESPONSE=$(curl -sf -X POST -H "Content-Type: application/json" -d "$PAYLOAD" \
  "${CONTROLLER_URL}/submit") || {
  echo "ERROR: Could not reach controller at ${CONTROLLER_URL}" >&2
  echo "  Start it: python3 scripts/experiment_controller.py" >&2
  exit 1
}

QUEUE_ID=$(python3 -c "import json,sys; print(json.loads(sys.argv[1])['id'])" "$RESPONSE" 2>/dev/null || echo "?")
echo "Queued: id=${QUEUE_ID}  user=${SF_USER}  step=${STEP_NAME}  device_pref=${DEVICE_PREF}${SLOT_PREF:+  slot_pref=}${SLOT_PREF}"
