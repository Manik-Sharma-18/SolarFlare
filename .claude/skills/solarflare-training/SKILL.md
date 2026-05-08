---
name: solarflare-training
description: Multi-machine training slot coordination for SolarFlare — queue experiments across Mac Mini (MPS/CPU), Mac Studio (MPS/CPU), and RTX 5060 Ti (CUDA). Invoke when launching training runs, checking slot status, or onboarding to the queue workflow.
---

# SolarFlare Training Workflow

Multi-machine training queue for SolarFlare. Five slots across three machines.
Prevents slot collisions. Dispatches via `scripts/launch_slot.sh`.

---

## 0. Machines & Slots

| Slot ID        | Machine             | Device | Working dir                                      | Python          |
|----------------|---------------------|--------|--------------------------------------------------|-----------------|
| `mini_mps`     | Mac Mini (local)    | MPS    | `/Volumes/T9/IndraAstra/manik/SolarFlare`        | `python3`       |
| `mini_cpu`     | Mac Mini (local)    | CPU    | same                                             | `python3`       |
| `studio_mps`   | Mac Studio (`mac-studio`) | MPS | `/Users/admin/ml/manik/SolarFlare`          | `python3`       |
| `studio_cpu`   | Mac Studio          | CPU    | same                                             | `python3`       |
| `5060ti_cuda`  | RTX 5060 Ti (`5060ti`) | CUDA | `/home/indra/solarflare`                        | `venv/bin/python3` |

**RAM check before launch:** Mac Studio ≥50 GB free+inactive; Mac Mini ≥20 GB; 5060ti ≥16 GB VRAM.

---

## 1. Quick Commands

```bash
# Submit (auto-picks free slot)
scripts/queue_submit.sh main.py --args "--config config.yaml"

# CUDA-only (RTX 5060 Ti)
scripts/queue_submit.sh main_v5.py --device-pref cuda --args "--config configs/v5_path_a.yaml"

# Hard slot reservation
scripts/queue_submit.sh main_v5.py --slot-pref 5060ti_cuda

# Check all slots
scripts/slot_status.sh

# View queue
scripts/queue_list.sh
scripts/queue_list.sh --user manik

# Cancel entry
curl -X POST http://mac-mini.local:7434/cancel/<id>
```

---

## 2. Launch Protocol (MANDATORY)

Every training run goes through `scripts/queue_submit.sh` or `scripts/launch_slot.sh`.
Never launch training directly — slot collisions cause silent OOM or corrupted runs.

```bash
# One-time identity setup (auto-prompted on first submit)
echo "manik" > .sf_user

# Submit to queue
scripts/queue_submit.sh <script> [--device-pref cuda|mps|any] [--slot-pref <slot>] [--args "..."]

# Direct launch (only when you know the slot is yours)
scripts/launch_slot.sh <slot> <script> [extra_args]
```

---

## 3. Slot Coordination

`scripts/slot_status.sh` shows all 5 slots as `[FREE]`, `[RUNNING]`, or `[DONE]`.

Session naming: `sf-<user>-<slot>-<script_name>` — encodes slot identity.

Cleanup happens automatically when training exits. If ssh disconnects, the tmux
session survives; the next launcher detects the stale state and recovers.

---

## 4. Controller Daemon

Runs on Mac Mini, port 7434. Dispatches every 30s. SQLite at `.controller/queue.db`.

**Preflight before any submit:** `curl -sf http://localhost:7434/status >/dev/null || nohup python3 scripts/experiment_controller.py > logs/experiment_controller.log 2>&1 & disown`

**On the Mac Mini itself, `mac-mini.local` does not resolve.** Export `SF_CONTROLLER_URL=http://localhost:7434` (or prefix submit/cancel commands) — default URL only works from other machines on the LAN.

```bash
# Start (one-time, Mac Mini only)
nohup python3 scripts/experiment_controller.py > logs/experiment_controller.log 2>&1 &

# Status check (off-Mac-Mini)
curl http://mac-mini.local:7434/status | python3 -m json.tool
# Status check (on Mac Mini)
curl http://localhost:7434/status | python3 -m json.tool
```

Round-robins across users. `--device-pref cuda` entries only run on `5060ti_cuda`.
`--slot-pref <slot>` reserves that slot exclusively for the entry.

---

## 5. CUDA Launch Checklist (5060ti_cuda)

Scripts launched on `5060ti_cuda` must pass this audit (enforced by `launch_slot.sh`):

| Marker            | Why                                        |
|-------------------|--------------------------------------------|
| `pin_memory=True` | DMA overlap, async H→D transfer            |
| `non_blocking=True` | Concurrent transfer + compute            |
| No fp16 GradScaler | fp32 is faster on Blackwell (step801 evidence from dhiraj/sgnnet) |

Add `# CUDA-5060ti-validated` anywhere in script to bypass audit (use sparingly).

Override: `scripts/launch_slot.sh 5060ti_cuda script.py --unsafe-cuda-launch`

---

## 6. Sync Verification (MANDATORY for remote slots)

`launch_slot.sh` runs `scripts/sync_verify.sh` as preflight whenever
`HOST != local` (i.e. `studio_*` and `5060ti_cuda`). Two levels:

| Level | What it checks | How |
|-------|---------------|-----|
| **code** | All `*.py *.sh *.yaml *.yml *.md` outside `outputs/`, `data/`, `logs/`, `.git/`, `venv/`, `__pycache__/`, `.controller/` | SHA256 hash manifest local vs remote |
| **data** | `data/*.zarr` set + sizes (kB) | `du -sk data/*.zarr` local vs remote |

Exit codes: `0`=in sync, `1`=error, `2`=out of sync.

If out of sync, the launch aborts. Resolve by:

```bash
# Auto-rsync (code + data)
scripts/sync_verify.sh --slot 5060ti_cuda --level both --fix

# Or rsync as part of launch
scripts/launch_slot.sh 5060ti_cuda main_v5.py --sync-fix --config configs/v5_path_a.yaml

# Bypass (only when remote is intentionally diverged — e.g. branch experiment)
scripts/launch_slot.sh 5060ti_cuda main_v5.py --skip-sync-check ...
```

Why this exists: silent staleness on `5060ti` (local code newer than remote)
caused multiple "fixes that did nothing" — remote was running old code.
Verification + auto-rsync makes that scenario impossible to hit by accident.

Direct standalone use:

```bash
# Check only (no transfer)
scripts/sync_verify.sh --slot 5060ti_cuda --level code
scripts/sync_verify.sh --slot 5060ti_cuda --level data
scripts/sync_verify.sh --slot 5060ti_cuda --level both

# Push local → remote
scripts/sync_verify.sh --slot studio_mps --level code --fix
```

---

## 7. Monitor Live Training

```bash
# Attach to local session
tmux attach -t sf-manik-mini_mps-main_v5

# Mac Studio
ssh mac-studio -t /opt/homebrew/bin/tmux attach -t sf-manik-studio_mps-main_v5

# RTX 5060 Ti
ssh 5060ti -t /usr/bin/tmux attach -t sf-manik-5060ti_cuda-main_v5
```

---

## 8. Smoke Test Before Launch

```bash
python3 main.py --help
python3 main_v5.py --help
```

Never launch a script you haven't sanity-checked — catches import errors and config typos.
