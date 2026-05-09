#!/usr/bin/env bash
# Two-level sync verification for SolarFlare remote slots.
#
# Level 1 (code): SHA256 hash manifest of *.py *.sh *.yaml *.yml *.md
#   (excludes outputs/ data/ logs/ .git/ .venv/ venv/ __pycache__/ .controller/)
# Level 2 (data): list of data/*.zarr names + sizes (du -sk per cube)
#
# Usage:
#   scripts/sync_verify.sh --slot <slot> [--level code|data|both] [--fix] [--quiet]
#
# Exit: 0=in sync   1=error   2=out of sync (no --fix)

set -euo pipefail

SLOT=""; LEVEL="both"; FIX=0; QUIET=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --slot)  SLOT="$2"; shift 2 ;;
    --level) LEVEL="$2"; shift 2 ;;
    --fix)   FIX=1; shift ;;
    --quiet) QUIET=1; shift ;;
    -h|--help)
      sed -n '2,12p' "$0"; exit 0 ;;
    *) echo "ERROR: unknown arg '$1'" >&2; exit 1 ;;
  esac
done

[[ -z "$SLOT" ]] && { echo "ERROR: --slot required" >&2; exit 1; }
[[ "$LEVEL" =~ ^(code|data|both)$ ]] || { echo "ERROR: --level must be code|data|both" >&2; exit 1; }

REPO_LOCAL="/Volumes/T9/IndraAstra/manik/SolarFlare"
case "$SLOT" in
  mini_mps|mini_cpu)       HOST=local;      DIR="$REPO_LOCAL" ;;
  studio_mps|studio_cpu)   HOST=mac-studio; DIR="/Users/admin/ml/manik/SolarFlare" ;;
  5060ti_cuda)             HOST=5060ti;     DIR="/home/indra/solarflare" ;;
  *) echo "ERROR: unknown slot '$SLOT'" >&2; exit 1 ;;
esac

if [[ "$HOST" == "local" ]]; then
  [[ "$QUIET" == "0" ]] && echo "sync_verify: slot '$SLOT' is local — skipping"
  exit 0
fi

log() { [[ "$QUIET" == "0" ]] && echo "$@" >&2 || true; }

# --- code-side hashing ------------------------------------------------------
code_manifest_cmd() {
  cat <<'EOF'
find . -type f \
  \( -name '*.py' -o -name '*.sh' -o -name '*.yaml' -o -name '*.yml' -o -name '*.md' \) \
  -not -path './outputs/*' -not -path './outputs_*/*' \
  -not -path './data/*' -not -path './logs/*' \
  -not -path './.git/*' -not -path './.venv/*' -not -path './venv/*' \
  -not -path './__pycache__/*' -not -path './*/__pycache__/*' \
  -not -path './.controller/*' -not -path './.claude/projects/*' \
  | LC_ALL=C sort
EOF
}

hash_files_local() {
  cd "$REPO_LOCAL"
  bash -c "$(code_manifest_cmd)" | xargs -I {} shasum -a 256 {} \
    | awk '{print $2"  "$1}' | LC_ALL=C sort
}

hash_files_remote() {
  ssh -o ConnectTimeout=5 "$HOST" "cd $DIR && $(code_manifest_cmd) | xargs -I {} sha256sum {}" \
    | awk '{print $2"  "$1}' | LC_ALL=C sort
}

# --- data-side listing ------------------------------------------------------
# Uses file count + apparent byte total per cube — block-size agnostic
# (APFS vs ext4 disagree on `du -sk`, but content bytes match).
data_list_cmd() {
  cat <<'EOF'
[ -d data ] || exit 0
for z in data/*.zarr; do
  [ -d "$z" ] || continue
  n=$(find "$z" -type f | wc -l | tr -d ' ')
  b=$(find "$z" -type f -exec wc -c {} + 2>/dev/null | awk 'END{print $1+0}')
  echo "$z $n $b"
done | LC_ALL=C sort
EOF
}

data_list_local() {
  cd "$REPO_LOCAL"
  bash -c "$(data_list_cmd)"
}

data_list_remote() {
  ssh -o ConnectTimeout=5 "$HOST" "cd $DIR && $(data_list_cmd)" || echo ""
}

# --- diff helpers -----------------------------------------------------------
check_code() {
  log "sync_verify[code]: hashing local + remote ($HOST:$DIR)…"
  local L R
  L=$(hash_files_local) || return 1
  R=$(hash_files_remote) || { echo "ERROR: remote hash failed" >&2; return 1; }
  if [[ "$L" == "$R" ]]; then log "  OK code in sync"; return 0; fi
  echo "OUT OF SYNC: code differs between local and $HOST" >&2
  diff <(echo "$L") <(echo "$R") | head -40 >&2
  return 2
}

check_data() {
  log "sync_verify[data]: listing data/*.zarr local + remote…"
  local L R
  L=$(data_list_local)
  R=$(data_list_remote)
  if [[ "$L" == "$R" ]]; then log "  OK data in sync"; return 0; fi
  echo "OUT OF SYNC: data/*.zarr differs between local and $HOST" >&2
  diff <(echo "$L") <(echo "$R") | head -40 >&2
  return 2
}

# --- rsync fix --------------------------------------------------------------
fix_code() {
  log "sync_verify[code]: rsync local -> ${HOST}:${DIR}..."
  rsync -az --delete \
    --include='*/' \
    --include='*.py' --include='*.sh' --include='*.yaml' --include='*.yml' --include='*.md' \
    --exclude='outputs/' --exclude='outputs_*/' --exclude='data/' --exclude='logs/' \
    --exclude='.git/' --exclude='.venv/' --exclude='venv/' \
    --exclude='__pycache__/' --exclude='.controller/' --exclude='.claude/projects/' \
    --exclude='*' \
    "$REPO_LOCAL/" "$HOST:$DIR/"
}

fix_data() {
  log "sync_verify[data]: rsync data/ local -> ${HOST}:${DIR}/data/..."
  rsync -az --delete "$REPO_LOCAL/data/" "$HOST:$DIR/data/"
}

# --- run --------------------------------------------------------------------
RC=0
if [[ "$LEVEL" == "code" || "$LEVEL" == "both" ]]; then
  if ! check_code; then
    if [[ "$FIX" == "1" ]]; then fix_code && check_code || RC=2; else RC=2; fi
  fi
fi
if [[ "$LEVEL" == "data" || "$LEVEL" == "both" ]]; then
  if ! check_data; then
    if [[ "$FIX" == "1" ]]; then fix_data && check_data || RC=2; else RC=2; fi
  fi
fi

if [[ "$RC" == "2" ]]; then
  echo "" >&2
  echo "Resolve via:" >&2
  echo "  scripts/sync_verify.sh --slot $SLOT --level $LEVEL --fix" >&2
  echo "Or override launch with: scripts/launch_slot.sh ... --skip-sync-check" >&2
fi
exit $RC
