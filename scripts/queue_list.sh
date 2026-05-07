#!/usr/bin/env bash
# queue_list.sh — Pretty-print the SolarFlare experiment queue.
#
# Usage: queue_list.sh [--user <name>] [--all-statuses]
# Env:   SF_CONTROLLER_URL (default http://mac-mini.local:7434)

set -euo pipefail

CONTROLLER_URL="${SF_CONTROLLER_URL:-http://mac-mini.local:7434}"
USER_FILTER="" ALL_STATUSES=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --user)         USER_FILTER="$2"; shift 2 ;;
    --all-statuses) ALL_STATUSES=1;   shift ;;
    -h|--help)      echo "Usage: $0 [--user <name>] [--all-statuses]"; exit 0 ;;
    *) echo "Unknown option: $1" >&2; exit 1 ;;
  esac
done

URL="${CONTROLLER_URL}/queue"
[[ -n "$USER_FILTER" ]] && URL="${URL}?user=${USER_FILTER}"

RESPONSE=$(curl -sf "$URL") || {
  echo "ERROR: Could not reach controller at ${CONTROLLER_URL}" >&2
  echo "  Start it: python3 scripts/experiment_controller.py" >&2
  exit 1
}

python3 - "$RESPONSE" "$ALL_STATUSES" <<'PYEOF'
import json, sys

data = json.loads(sys.argv[1])
all_statuses = sys.argv[2] == "1"
ACTIVE = {"queued", "running"}
COLOR = {"queued": "\033[33m", "running": "\033[32m",
         "done": "\033[90m", "failed": "\033[31m", "cancelled": "\033[90m"}
RESET = "\033[0m"

entries = [e for e in data if all_statuses or e["status"] in ACTIVE]
if not entries:
    msg = "Queue empty." if all_statuses else "No queued/running entries. Use --all-statuses for history."
    print(msg); sys.exit(0)

hdr = f"{'ID':>5}  {'USER':<12} {'STATUS':<11} {'STEP_NAME':<30} {'DEVICE':>6} {'SLOT_PREF':<14} SUBMITTED"
print(hdr); print("-" * len(hdr))
for e in entries:
    c = COLOR.get(e["status"], "")
    slot = e["slot_pref"] or ""
    ts = (e["submitted_at"] or "")[:19].replace("T", " ")
    print(f"{e['id']:>5}  {e['user']:<12} {c}{e['status']:<11}{RESET} {e['step_name'][:29]:<30} {e['device_pref']:>6} {slot:<14} {ts}")
print(f"\nTotal: {len(entries)} entries")
PYEOF
