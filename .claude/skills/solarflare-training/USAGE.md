# SolarFlare Training Queue — Usage Guide

## Setup (one-time)

### 1. Start the controller on Mac Mini

```bash
cd /Volumes/T9/IndraAstra/manik/SolarFlare
nohup python3 scripts/experiment_controller.py \
    > logs/experiment_controller.log 2>&1 &
echo $! > .controller/controller.pid
```

Controller runs on port **7434** (7433 is taken by dhiraj's SGNNET controller).

### 2. Set your identity

```bash
echo "manik" > .sf_user
# or per-session: export SF_USER=manik
```

### 3. Verify slots are visible

```bash
scripts/slot_status.sh
```

Expected output:
```
===== SolarFlare Slot Status =====
mini_mps      [FREE]
mini_cpu      [FREE]
studio_mps    [FREE]  (or [UNREACHABLE] if mac-studio not configured)
studio_cpu    [FREE]
5060ti_cuda   [FREE]  (or [UNREACHABLE] if 5060ti not configured)
```

---

## Submitting a Training Run

### Basic (auto-device, round-robin slot)

```bash
scripts/queue_submit.sh main.py --args "--config config.yaml"
```

### Force CUDA (RTX 5060 Ti)

```bash
scripts/queue_submit.sh main_v5.py \
    --device-pref cuda \
    --args "--config configs/v5_path_a.yaml"
```

### Force MPS (local Mac Mini)

```bash
scripts/queue_submit.sh main_v5.py \
    --slot-pref mini_mps \
    --args "--config configs/v5_path_a.yaml"
```

### High-priority run

```bash
scripts/queue_submit.sh main_v5.py --priority 10 --args "--config configs/v5_path_a.yaml"
```

---

## Checking Status

```bash
# All slots
scripts/slot_status.sh

# Queue (active entries only)
scripts/queue_list.sh

# Queue (all history)
scripts/queue_list.sh --all-statuses

# Controller internals
curl http://mac-mini.local:7434/status | python3 -m json.tool
```

---

## Cancelling an Entry

```bash
# Get the entry ID from queue_list.sh, then:
curl -X POST http://mac-mini.local:7434/cancel/42
```

---

## Watching Live Training

```bash
# Local Mac Mini
tmux attach -t sf-manik-mini_mps-main_v5

# Mac Studio (ssh)
ssh mac-studio -t /opt/homebrew/bin/tmux attach -t sf-manik-studio_mps-main_v5

# RTX 5060 Ti (ssh)
ssh 5060ti -t /usr/bin/tmux attach -t sf-manik-5060ti_cuda-main_v5
```

---

## CUDA Best Practices (5060ti_cuda)

SolarFlare's `utils/device.py` already handles device resolution correctly.
For DataLoaders targeting `5060ti_cuda`, add to your training script:

```python
# CUDA-5060ti-validated
train_loader = DataLoader(dataset, batch_size=BATCH,
                          shuffle=True, num_workers=0,
                          pin_memory=True)  # required

# In training loop:
x = x.to(device, non_blocking=True)   # required
y = y.to(device, non_blocking=True)   # required
```

Do NOT use fp16 GradScaler on Blackwell (RTX 5060 Ti). fp32 is faster.
Use `utils/device.py::get_grad_scaler(use_amp=False, device)` — returns DummyGradScaler.

---

## Remote Machine Setup (TODO)

Configure SSH hosts in `~/.ssh/config`:

```
Host mac-studio
    HostName <ip-or-hostname>
    User admin

Host 5060ti
    HostName <ip-or-hostname>
    User indra
```

Update paths in `scripts/launch_slot.sh` if remote dirs differ:
- `studio_mps` / `studio_cpu` → currently `/Users/admin/ml/manik/SolarFlare`
- `5060ti_cuda` → currently `/home/indra/solarflare`

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `Could not reach controller` | Start controller: `python3 scripts/experiment_controller.py` |
| Slot shows `[RUNNING]` but no active job | `tmux kill-session -t <session>` on the host machine |
| CUDA launch blocked | Add `pin_memory=True`, `non_blocking=True`, or `--unsafe-cuda-launch` |
| SSH unreachable | Entry stays queued; fix SSH config and restart controller |
| Controller crash | `python3 scripts/experiment_controller.py` — SQLite state persists |
