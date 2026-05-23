#!/usr/bin/env bash
# Audit helpers sourced by launch_slot.sh.
# Provides: audit_cuda, audit_vram, get_config_path, get_vram_budget_mib.
# Expects: $REPO_LOCAL, $VRAM_BUFFER_PCT in caller scope.

audit_cuda() {
  local script="$1"
  [[ -f "$script" ]] || { echo "ERROR: cannot read script: $script" >&2; return 1; }
  local c; c=$(cat "$script")
  grep -q "# CUDA-5060ti-validated" <<< "$c" && return 0
  local fails=()
  grep -q "pin_memory=True" <<< "$c"   || fails+=("  - Missing pin_memory=True on DataLoader")
  grep -q "non_blocking=True" <<< "$c" || fails+=("  - Missing non_blocking=True on .to(device)")
  if grep -q "GradScaler" <<< "$c" && ! grep -qE "GradScaler\([^)]*enabled=False|use_amp=False" <<< "$c"; then
    fails+=("  - GradScaler without enabled=False — fp32 is faster on Blackwell")
  fi
  if [[ ${#fails[@]} -gt 0 ]]; then
    echo "" >&2
    echo "┌── CUDA LAUNCH BLOCKED: $script" >&2
    echo "│   See: .claude/skills/solarflare-training/SKILL.md" >&2
    for f in "${fails[@]}"; do echo "│ $f" >&2; done
    echo "│   Override: append --unsafe-cuda-launch" >&2
    echo "└──" >&2
    return 1
  fi
}

get_config_path() {
  local prev=""
  for arg in "$@"; do
    [[ "$prev" == "--config" ]] && { echo "$arg"; return; }
    [[ "$arg" == --config=* ]] && { echo "${arg#--config=}"; return; }
    prev="$arg"
  done
}

get_vram_budget_mib() {
  local key budget table="$REPO_LOCAL/configs/_vram_budget.yaml"
  [[ -f "$table" ]] || return
  key=$(basename "$1" .yaml)
  budget=$(awk -F': *' -v k="$key" '$1==k {print $2; exit}' "$table")
  [[ -z "$budget" ]] && budget=$(awk -F': *' '$1=="default" {print $2; exit}' "$table")
  echo "$budget"
}

audit_vram() {
  local out used total free
  out=$(ssh -o ConnectTimeout=5 5060ti \
    'nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits' 2>/dev/null) || {
    echo "ERROR: nvidia-smi unreachable on 5060ti" >&2; return 1; }
  used=$(awk -F',' '{gsub(/ /,""); print $1}' <<< "$out")
  total=$(awk -F',' '{gsub(/ /,""); print $2}' <<< "$out")
  [[ -z "$used" || -z "$total" || "$total" -eq 0 ]] && {
    echo "ERROR: bad nvidia-smi output: '$out'" >&2; return 1; }
  free=$((total - used))

  local cfg budget required
  cfg=$(get_config_path "$@")
  [[ -n "$cfg" ]] && budget=$(get_vram_budget_mib "$cfg")

  if [[ -n "$budget" ]]; then
    required=$((budget * (100 + VRAM_BUFFER_PCT) / 100))
    if (( free < required )); then
      echo "" >&2
      echo "┌── VRAM BLOCKED: free=${free} MiB < required=${required} MiB" >&2
      echo "│   config=$(basename "$cfg") budget=${budget} MiB buffer=${VRAM_BUFFER_PCT}%" >&2
      echo "│   used=${used}/${total} MiB. Override: --skip-vram-check" >&2
      echo "└──" >&2
      return 1
    fi
    echo "VRAM OK: free=${free} MiB ≥ ${required} MiB (budget ${budget} + ${VRAM_BUFFER_PCT}%)" >&2
  else
    local max_pct=$((100 - VRAM_BUFFER_PCT))
    local used_pct=$((used * 100 / total))
    if (( used_pct > max_pct )); then
      echo "" >&2
      echo "┌── VRAM BLOCKED (no budget entry): used=${used_pct}% > max=${max_pct}%" >&2
      echo "│   Add entry to configs/_vram_budget.yaml or pass --skip-vram-check" >&2
      echo "└──" >&2
      return 1
    fi
    echo "VRAM OK (% fallback): ${used}/${total} MiB (${used_pct}%)" >&2
  fi
}
