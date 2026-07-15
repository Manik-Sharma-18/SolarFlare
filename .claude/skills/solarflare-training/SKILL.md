---
name: solarflare-training
description: Multi-machine training slot coordination for SolarFlare — queue experiments across Mac Mini (MPS/CPU), Mac Studio (MPS/CPU), and RTX 5060 Ti (CUDA). Invoke when launching training runs, checking slot status, or onboarding to the queue workflow.
---

# SolarFlare Training Workflow

Multi-machine training queue. Five slots across three machines. Dispatches via
`scripts/launch_slot.sh`. Prevents slot collisions.

See `CLAUDE.md` § "Training queue gotchas" for recurring failure modes.

---

## 0. Machines & Slots

| Slot ID        | Machine             | Device | Working dir                                      | Python          |
|----------------|---------------------|--------|--------------------------------------------------|-----------------|
| `mini_mps`     | Mac Mini (local)    | MPS    | `/Volumes/T9/IndraAstra/manik/SolarFlare`        | `python3`       |
| `mini_cpu`     | Mac Mini (local)    | CPU    | same                                             | `python3`       |
| `studio_mps`   | Mac Studio (`mac-studio`) | MPS | `/Users/admin/ml/manik/SolarFlare`          | `python3`       |
| `studio_cpu`   | Mac Studio          | CPU    | same                                             | `python3`       |
| `5060ti_cuda`  | RTX 5060 Ti (`5060ti`) | CUDA | `/home/indra/solarflare`                        | `venv/bin/python3` |

**Free-resource gate:** Mac Studio ≥50 GB RAM; Mac Mini ≥20 GB RAM; 5060ti ≥16 GB VRAM.

Session naming: `sf-<user>-<slot>-<script_name>`. Tmux state survives ssh disconnects; the launcher recovers stale state on next dispatch.

---

## 1. Quick Commands

```bash
# Submit (auto-picks free slot; --device-pref filters; --slot-pref reserves)
scripts/queue_submit.sh main.py --args "--config configs/finetune_winding_flux.yaml"
scripts/queue_submit.sh main.py --slot-pref 5060ti_cuda --args "..."
scripts/queue_submit.sh main.py --device-pref cuda      --args "..."

# Inspect
scripts/slot_status.sh             # FREE / RUNNING / DONE per slot
scripts/queue_list.sh [--user manik]
sqlite3 -header -column .controller/queue.db \
  "SELECT id, step_name, status, launched_slot FROM queue_entries \
   WHERE status IN ('running','queued') ORDER BY id DESC"

# Cancel
curl -X POST http://mac-mini.local:7434/cancel/<id>
```

**Protocol:** every training run goes through `queue_submit.sh` or `launch_slot.sh`. Direct `python3 main.py` collides with active slots and corrupts shared GPU state.

---

## 2. Controller Daemon

Runs on Mac Mini, port 7434. Dispatches every 30s. SQLite at `.controller/queue.db`. Round-robins across users; `--device-pref cuda` confines to `5060ti_cuda`; `--slot-pref` is exclusive.

**Start (Mac Mini only, from venv-activated shell):**

```bash
source /Volumes/T9/IndraAstra/.venv/bin/activate   # MUST resolve `python3` to a torch-having interpreter
nohup python3 scripts/experiment_controller.py >> logs/experiment_controller.log 2>&1 &
disown
```

Without venv activation, `launch_slot.sh` (which uses bare `python3` for mini/studio slots) spawns torch-less children → silent `ModuleNotFoundError: torch` in ~35 s. See `[[controller_torch_path]]`.

**Status:**

```bash
curl -s "${SF_CONTROLLER_URL:-http://mac-mini.local:7434}/status" | python3 -m json.tool
```

On the Mac Mini itself, `mac-mini.local` doesn't resolve — export `SF_CONTROLLER_URL=http://localhost:7434` first.

**Preflight before any submit:**

```bash
curl -sf "${SF_CONTROLLER_URL:-http://localhost:7434}/status" >/dev/null \
  || echo "Controller down — start it (see above) before submitting"
```

---

## 3. CUDA Launch Checklist (5060ti_cuda)

`launch_slot.sh` audits scripts before dispatching to `5060ti_cuda`:

| Marker              | Why                                                |
|---------------------|----------------------------------------------------|
| `pin_memory=True`   | DMA overlap, async H→D transfer                    |
| `non_blocking=True` | Concurrent transfer + compute                      |
| No fp16 GradScaler  | fp32 faster on Blackwell (step801 / dhiraj/sgnnet) |

Add `# CUDA-5060ti-validated` anywhere in the script to bypass, or pass `--unsafe-cuda-launch`. Use sparingly.

**VRAM preflight (MANDATORY):** every 5060ti_cuda launch reads remote `nvidia-smi` against `configs/_vram_budget.yaml`. Blocked unless `free ≥ budget × (1 + buffer)`. Default buffer 40%; tune via `SF_VRAM_BUFFER_PCT`; bypass via `--skip-vram-check`. Tuning notes in `USAGE.md` § CUDA Best Practices.

---

## 4. Sync Verification (remote slots)

`launch_slot.sh` runs `scripts/sync_verify.sh` automatically whenever `HOST != local` (`studio_*`, `5060ti_cuda`). Two levels:

| Level | What it checks                                              | How                                       |
|-------|-------------------------------------------------------------|-------------------------------------------|
| code  | `*.py *.sh *.yaml *.yml *.md` outside outputs/data/logs/etc | SHA256 manifest local vs remote           |
| data  | `data/*.zarr` set + sizes                                   | `du -sk data/*.zarr` local vs remote      |

Exit: `0`=in sync, `1`=error, `2`=out of sync. Out-of-sync aborts the launch.

```bash
# Inspect or repair
scripts/sync_verify.sh --slot <slot> --level {code|data|both} [--fix]

# Launch with auto-rsync
scripts/launch_slot.sh 5060ti_cuda main.py --sync-fix --config <cfg>

# Bypass (only when remote is intentionally diverged — branch experiment, etc.)
scripts/launch_slot.sh 5060ti_cuda main.py --skip-sync-check --config <cfg>
```

Why this exists: silent staleness on 5060ti (local code newer than remote) caused multiple "fixes that did nothing" — remote was running old code.

---

## 5. Monitor Live Training

For **read-only** status (preferred — no risk of accidental Ctrl-C):

```bash
# Local mini
tmux capture-pane -t sf-manik-mini_mps-main -p | tail -60

# Remote (replace <host>/<slot>/<tmux>: mac-studio uses /opt/homebrew/bin/tmux,
# 5060ti uses /usr/bin/tmux)
ssh <host> "<tmux> capture-pane -t sf-manik-<slot>-main -p | tail -60"

# Collapse tqdm spam to per-epoch summary
ssh <host> "grep -E 'Epoch [0-9]+/|val_loss|Saved best|CLS-CSI' \
  <repo>/logs/main__<slot>.log | tail -40"
```

For interactive debug (attach — be careful with Ctrl-C):

```bash
tmux attach -t sf-manik-<slot>-main                     # local
ssh <host> -t <tmux> attach -t sf-manik-<slot>-main     # remote
```

Output dirs live under each slot's working dir (see §0), NOT the local repo — `find outputs/` locally won't see remote runs.

---

## 6. Smoke Test Before Launch

```bash
python3 main.py --help                                                          # import + config sanity
scripts/launch_slot.sh 5060ti_cuda main.py --config configs/smoke_5060ti.yaml   # 1-epoch full-pipeline dry run
```

Never launch a script you haven't sanity-checked — catches import errors and config typos before a slot reservation is wasted.
