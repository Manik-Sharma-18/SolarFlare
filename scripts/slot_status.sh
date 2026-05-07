#!/usr/bin/env bash
# Show state of all 5 SolarFlare training slots across 3 machines.
# State: RUNNING (pane alive + active children) | DONE (idle) | FREE (no session)
# Session convention: sf-<user>-<slot>-<step>
#
# Usage: scripts/slot_status.sh

set -uo pipefail

PREFIX="sf-"

remote_cmd() {
  local host="$1" cmd="$2"
  if [[ "$host" == "local" ]]; then TERM=xterm-256color bash -c "$cmd"
  else ssh -o ConnectTimeout=4 "$host" "$cmd"; fi
}

log_dir_for_host() {
  case "$1" in
    local)      echo "/Volumes/T9/IndraAstra/manik/SolarFlare/logs" ;;
    mac-studio) echo "/Users/admin/ml/manik/SolarFlare/logs" ;;
    5060ti)     echo "/home/indra/solarflare/logs" ;;
    *)          echo "logs" ;;
  esac
}

check_crash() {
  local host="$1" log_dir="$2" slot="$3"
  local last_log crash_line
  last_log=$(remote_cmd "$host" \
    "ls -t ${log_dir}/*__${slot}.log 2>/dev/null | head -1" 2>/dev/null || true)
  [[ -z "$last_log" ]] && return
  crash_line=$(remote_cmd "$host" \
    "grep -m1 'Traceback\|RuntimeError\|CUDA error\|AssertionError\|KeyboardInterrupt' '${last_log}' 2>/dev/null | tail -1" \
    2>/dev/null || true)
  [[ -n "$crash_line" ]] && printf "  *** CRASHED: %s ***\n" "$crash_line"
}

check_slot() {
  local slot="$1" host="$2" tmux_cmd="$3"
  printf "%-13s " "$slot"

  local sessions
  sessions=$(remote_cmd "$host" \
    "$tmux_cmd list-sessions -F '#{session_name}' 2>/dev/null | grep -E '^sf-[^-]+-${slot}-'" \
    2>/dev/null || true)

  local log_dir; log_dir=$(log_dir_for_host "$host")

  if [[ -z "$sessions" ]]; then
    printf "[FREE]\n"; check_crash "$host" "$log_dir" "$slot"; return
  fi

  local s any_done="" any_dead=""
  while IFS= read -r s; do
    local info dead pid
    info=$(remote_cmd "$host" \
      "$tmux_cmd list-panes -t '$s' -F '#{pane_dead} #{pane_pid}' 2>/dev/null | head -1" \
      2>/dev/null || echo "1 0")
    dead=$(awk '{print $1}' <<< "$info"); pid=$(awk '{print $2}' <<< "$info")

    if [[ "$dead" == "1" ]]; then any_dead="$s"; continue; fi

    local n
    n=$(remote_cmd "$host" "pgrep -P $pid 2>/dev/null | wc -l | tr -d ' '" 2>/dev/null || echo "0")
    if [[ "${n:-0}" -gt 0 ]]; then
      printf "[RUNNING]    session=%s\n" "$s"; return
    else
      any_done="$s"
    fi
  done <<< "$sessions"

  if [[ -n "$any_done" ]]; then
    printf "[DONE]       session=%s  (script finished)\n" "$any_done"
    check_crash "$host" "$log_dir" "$slot"
  elif [[ -n "$any_dead" ]]; then
    printf "[FREE]       (dead pane from session=%s)\n" "$any_dead"
    check_crash "$host" "$log_dir" "$slot"
  else
    printf "[FREE]\n"
  fi
}

list_sessions() {
  local host="$1" tmux_cmd="$2"
  local sessions
  sessions=$(remote_cmd "$host" "$tmux_cmd list-sessions 2>/dev/null" || true)
  if [[ -z "$sessions" ]]; then echo "  (no tmux sessions)"; return; fi

  local sf_sessions other
  sf_sessions=$(grep "^${PREFIX}" <<< "$sessions" || true)
  other=$(grep -v "^${PREFIX}" <<< "$sessions" || true)

  if [[ -n "$sf_sessions" ]]; then
    echo "  [SF training]"
    while IFS= read -r line; do
      local s="${line%%:*}"
      local info dead pid n state
      info=$(remote_cmd "$host" \
        "$tmux_cmd list-panes -t '$s' -F '#{pane_dead} #{pane_pid}' 2>/dev/null | head -1" \
        2>/dev/null || echo "1 0")
      dead=$(awk '{print $1}' <<< "$info"); pid=$(awk '{print $2}' <<< "$info")
      if [[ "$dead" == "1" ]]; then state="DEAD"
      else
        n=$(remote_cmd "$host" "pgrep -P $pid 2>/dev/null | wc -l | tr -d ' '" 2>/dev/null || echo "0")
        [[ "${n:-0}" -gt 0 ]] && state="RUNNING" || state="DONE"
      fi
      printf "    [%-7s] %s\n" "$state" "$line"
    done <<< "$sf_sessions"
  fi

  if [[ -n "$other" ]]; then
    echo "  [NON-SF — DO NOT TOUCH]"
    grep -v "^${PREFIX}" <<< "$sessions" | sed 's/^/    /'
  fi
}

echo "===== SolarFlare Slot Status ====="
check_slot "mini_mps"    "local"      "tmux"
check_slot "mini_cpu"    "local"      "tmux"
check_slot "studio_mps"  "mac-studio" "/opt/homebrew/bin/tmux"
check_slot "studio_cpu"  "mac-studio" "/opt/homebrew/bin/tmux"
check_slot "5060ti_cuda" "5060ti"     "/usr/bin/tmux"

echo
echo "===== Tmux sessions by machine ====="
echo "--- Mac Mini (local) ---"; list_sessions "local" "tmux"
echo "--- Mac Studio ---";       list_sessions "mac-studio" "/opt/homebrew/bin/tmux"
echo "--- 5060ti ---";           list_sessions "5060ti" "/usr/bin/tmux"
