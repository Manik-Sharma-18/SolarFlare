#!/usr/bin/env bash
# Show state of all 5 SolarFlare training slots across 3 machines.
# State: RUNNING (pane alive + active children) | DONE (idle) | FREE (no session) | UNKNOWN (probe failed)
# Session convention: sf-<user>-<slot>-<step>
#
# Usage: scripts/slot_status.sh

set -uo pipefail

PREFIX="sf-"
SSH_ERR_SENTINEL="__SSH_ERR__"

remote_cmd() {
  # Echo stdout on success; echo SSH_ERR_SENTINEL only when ssh itself fails (exit 255 = conn/auth).
  # Remote cmd non-zero exits (grep no-match etc) pass through as empty stdout — same as before.
  # Never silently treat a host-unreachable as empty output (that's what triggered the reconcile race).
  local host="$1" cmd="$2" out rc
  if [[ "$host" == "local" ]]; then
    TERM=xterm-256color bash -c "$cmd" 2>/dev/null
    return 0
  fi
  out=$(ssh -o ConnectTimeout=4 -o BatchMode=yes "$host" "$cmd" 2>/dev/null); rc=$?
  if [[ $rc -eq 255 ]]; then echo "$SSH_ERR_SENTINEL"; return 1; fi
  printf '%s\n' "$out"
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
    "$tmux_cmd list-sessions -F '#{session_name}' 2>/dev/null | grep -E '^sf-[^-]+-${slot}-'")

  if [[ "$sessions" == "$SSH_ERR_SENTINEL" ]]; then
    printf "[UNKNOWN]    (ssh probe failed)\n"; return
  fi

  local log_dir; log_dir=$(log_dir_for_host "$host")

  if [[ -z "$sessions" ]]; then
    printf "[FREE]\n"; check_crash "$host" "$log_dir" "$slot"; return
  fi

  local s any_done="" any_dead=""
  while IFS= read -r s; do
    local info dead pid
    info=$(remote_cmd "$host" \
      "$tmux_cmd list-panes -t '$s' -F '#{pane_dead} #{pane_pid}' 2>/dev/null | head -1")
    if [[ "$info" == "$SSH_ERR_SENTINEL" ]]; then
      printf "[UNKNOWN]    session=%s (pane probe failed)\n" "$s"; return
    fi
    dead=$(awk '{print $1}' <<< "$info"); pid=$(awk '{print $2}' <<< "$info")

    if [[ "$dead" == "1" ]]; then any_dead="$s"; continue; fi

    local n_out
    n_out=$(remote_cmd "$host" "pgrep -P $pid 2>/dev/null | wc -l | tr -d ' '")
    if [[ "$n_out" == "$SSH_ERR_SENTINEL" ]]; then
      printf "[UNKNOWN]    session=%s (pgrep probe failed)\n" "$s"; return
    fi
    local n="${n_out:-0}"
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
  sessions=$(remote_cmd "$host" "$tmux_cmd list-sessions 2>/dev/null")
  if [[ "$sessions" == "$SSH_ERR_SENTINEL" ]]; then
    echo "  (ssh probe failed — host unreachable)"; return
  fi
  if [[ -z "$sessions" ]]; then echo "  (no tmux sessions)"; return; fi

  local sf_sessions other
  sf_sessions=$(grep "^${PREFIX}" <<< "$sessions" || true)
  other=$(grep -v "^${PREFIX}" <<< "$sessions" || true)

  if [[ -n "$sf_sessions" ]]; then
    echo "  [SF training]"
    while IFS= read -r line; do
      local s="${line%%:*}"
      local info dead pid n_out n state
      info=$(remote_cmd "$host" \
        "$tmux_cmd list-panes -t '$s' -F '#{pane_dead} #{pane_pid}' 2>/dev/null | head -1")
      if [[ "$info" == "$SSH_ERR_SENTINEL" ]]; then state="UNKNOWN"
      else
        dead=$(awk '{print $1}' <<< "$info"); pid=$(awk '{print $2}' <<< "$info")
        if [[ "$dead" == "1" ]]; then state="DEAD"
        else
          n_out=$(remote_cmd "$host" "pgrep -P $pid 2>/dev/null | wc -l | tr -d ' '")
          if [[ "$n_out" == "$SSH_ERR_SENTINEL" ]]; then state="UNKNOWN"
          else n="${n_out:-0}"; [[ "${n:-0}" -gt 0 ]] && state="RUNNING" || state="DONE"
          fi
        fi
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
