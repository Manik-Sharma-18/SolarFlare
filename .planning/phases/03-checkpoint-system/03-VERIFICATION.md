---
phase: 03-checkpoint-system
verified: 2026-02-03T20:26:02Z
status: passed
score: 7/7 must-haves verified
re_verification: false
---

# Phase 3: Checkpoint System Verification Report

**Phase Goal:** Training crashes lose at most one epoch of progress, and checkpoints are portable across devices

**Verified:** 2026-02-03T20:26:02Z

**Status:** passed

**Re-verification:** No — initial verification

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | Killing training mid-epoch and restarting with `resume_from: latest_checkpoint.pt` continues from the last completed epoch with correct LR, optimizer state, and patience counter | ✓ VERIFIED | trainer.py:310-324 loads checkpoint via `load_checkpoint_for_resume()`, restores `start_epoch`, `best_val_loss`, `patience_counter`, `history`, optimizer/scheduler/scaler state. Training loop uses `range(start_epoch, epochs + 1)` at line 374. |
| 2 | Checkpoint files are never corrupted by a crash (atomic write-to-temp-then-rename) | ✓ VERIFIED | checkpoint.py:38-62 implements `_atomic_save()` using `tempfile.NamedTemporaryFile()`, `torch.save()`, `os.fsync()`, then `os.replace()` for atomic rename (line 54). All saves use this pattern. |
| 3 | A checkpoint saved on CUDA loads and resumes correctly on MPS or CPU (optimizer state tensors remapped to target device) | ✓ VERIFIED | checkpoint.py:224 loads with `map_location='cpu'` (device-neutral). checkpoint.py:192-197 `_optimizer_to_device()` moves optimizer state tensors to target device. checkpoint.py:276 calls this after loading optimizer state. |
| 4 | Normalization parameters are embedded in the checkpoint file -- deleting metadata.json does not break inference from a checkpoint | ✓ VERIFIED | checkpoint.py:116 embeds `'normalization_params': normalization_params` in checkpoint dict. inference.py:70 extracts `checkpoint.get('normalization_params')` and returns it from `load_model()`. inference.py:205-210 prefers checkpoint normalization over metadata.json fallback. |
| 5 | Training saves two checkpoints: best_model.pt (best val loss) and checkpoint_epoch_NNN_valloss_X.XXXX.pt (latest) in checkpoints/ subfolder | ✓ VERIFIED | trainer.py:327-328 creates `checkpoints_dir = save_dir / 'checkpoints'`. Best saved at line 420 (`checkpoints_dir / 'best_model.pt'`). Latest saved at line 436 (`checkpoints_dir / new_filename` where `new_filename = f"checkpoint_epoch_{epoch:03d}_valloss_{val_loss:.4f}.pt"`). |
| 6 | Old latest checkpoint is deleted when new one is saved; old best is never deleted by rolling logic | ✓ VERIFIED | trainer.py:443-447 deletes old `latest_checkpoint_path` if it exists AND is not `best_model.pt`, then updates `latest_checkpoint_path = new_path`. Best model deletion is prevented by explicit check at line 445. |
| 7 | resume_from pointing to a missing file fails at startup with clear error | ✓ VERIFIED | config_validator.py:239-243 checks `resume_from` exists: `if not resume_path.exists(): errors.append(f"resume_from: file not found: {resume_from}")`. Validation runs before data loading (main.py:53). |

**Score:** 7/7 truths verified

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `utils/checkpoint.py` | Centralized checkpoint I/O module | ✓ VERIFIED | 341 lines, 9 functions. Exports: `save_checkpoint`, `load_checkpoint`, `load_checkpoint_for_resume`, `load_checkpoint_for_inference`, `CHECKPOINT_VERSION=1`. Substantive implementation with atomic writes, version validation, architecture checks, config diffing, optimizer device remapping. |
| `training/trainer.py` | Updated training loop with resume support | ✓ VERIFIED | 479 lines. Imports centralized checkpoint functions (line 19). Uses `save_checkpoint()` 3 times (best, latest, emergency). Resume logic at lines 304-324. No direct `torch.save` calls remaining. |
| `inference.py` | Self-contained inference with normalization from checkpoint | ✓ VERIFIED | 244 lines. Uses `load_checkpoint_for_inference()` (line 24, 41). `load_model()` returns `(model, normalization_params)` tuple (line 76). Handles both nested and flat config formats (lines 48-63). |
| `config.yaml` | resume_from field | ✓ VERIFIED | Line 8: `resume_from: null` (defaults to start fresh). Field exists and documented. |
| `utils/config_validator.py` | resume_from validation | ✓ VERIFIED | Lines 239-245 validate `resume_from`: checks file exists if set, warns if not `.pt` extension. Runs at startup via main.py:53. |
| `main.py` | Normalization params threading and resume_from handling | ✓ VERIFIED | Line 143 extracts `normalization_params = metadata.get('normalization', {})`. Line 155 adds `'resume_from': config.get('resume_from')` to train_config. Line 164 passes `normalization_params=normalization_params` to `train_model()`. Line 175 loads from `checkpoints/best_model.pt`. |

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|----|--------|---------|
| `training/trainer.py` | `utils/checkpoint.py` | save_checkpoint() and load_checkpoint_for_resume() | ✓ WIRED | Import at line 19. Called 3 times for save (lines 364, 419, 436), once for load (line 311). |
| `inference.py` | `utils/checkpoint.py` | load_checkpoint_for_inference() | ✓ WIRED | Import at line 24, called at line 41. Returns `(checkpoint, device)` tuple. |
| `main.py` | `training/trainer.py` | passes normalization_params and resume_from in train_config | ✓ WIRED | normalization_params extracted line 143, added to train_config, passed to train_model at line 164. resume_from added to train_config at line 155. |
| `utils/checkpoint.py` | torch.save/torch.load | _atomic_save wraps torch.save with temp file + os.replace | ✓ WIRED | _atomic_save() lines 38-62 uses tempfile, torch.save (line 50), os.fsync (line 53), os.replace (line 54). load_checkpoint() uses torch.load with map_location='cpu', weights_only=False (line 224). |
| `utils/checkpoint.py` | optimizer state | _move_optimizer_state_to_cpu recursively converts tensors, _optimizer_to_device restores to target | ✓ WIRED | _move_optimizer_state_to_cpu() lines 16-35 recursively walks dict/list and calls `.cpu()` on tensors. Called at line 124 before save. _optimizer_to_device() lines 192-197 moves to target device. Called at line 276 after load. |

### Requirements Coverage

Requirements mapped to Phase 3: CHK-01, CHK-02, CHK-03, CHK-04

| Requirement | Status | Blocking Issue |
|-------------|--------|----------------|
| CHK-01: Full checkpoint resume -- restore model, optimizer, scheduler, scaler, epoch, best_val_loss, patience_counter, history | ✓ SATISFIED | All state restored via load_checkpoint_for_resume() |
| CHK-02: Atomic checkpoint writes -- write to temp file then rename | ✓ SATISFIED | _atomic_save() implements temp + os.replace pattern |
| CHK-03: Embed normalization parameters in checkpoint file | ✓ SATISFIED | normalization_params in checkpoint dict, used by inference |
| CHK-04: Cross-device resume -- load CUDA checkpoint on MPS/CPU | ✓ SATISFIED | map_location='cpu' + _optimizer_to_device() enables cross-device |

### Anti-Patterns Found

**No anti-patterns detected.**

- No TODO/FIXME/placeholder comments
- No stub implementations
- No console.log-only handlers
- No empty return statements
- No direct torch.save/torch.load outside utils/checkpoint.py

### Human Verification Required

The following items require human testing to fully validate goal achievement:

#### 1. End-to-end resume test (cross-epoch)

**Test:** Start training for 5 epochs, kill process mid-epoch 3, set `resume_from: outputs/checkpoints/checkpoint_epoch_002_valloss_X.XXXX.pt`, restart training.

**Expected:** Training resumes from epoch 3 (not epoch 1). Validation loss continues from previous best. Patience counter preserved. LR matches epoch 3's scheduler state. Training history includes epochs 1-2 from checkpoint.

**Why human:** Requires actually running training, killing the process, and observing resume behavior across multiple epochs.

#### 2. Cross-device checkpoint portability

**Test:** Train on CUDA (or MPS) for 3 epochs. Copy `best_model.pt` to a different machine with only CPU. Set `device: cpu` and `resume_from: best_model.pt`. Resume training.

**Expected:** Training resumes successfully. Optimizer state loads without device mismatch errors. Model forward pass works on CPU. Loss values are reasonable (not starting from scratch).

**Why human:** Requires access to multiple devices (CUDA/MPS and CPU) and physically moving checkpoint files between machines.

#### 3. Self-contained inference without metadata.json

**Test:** Train a model, then delete `outputs/metadata.json`. Run `python inference.py` with `CHECKPOINT_PATH = 'outputs/checkpoints/best_model.pt'`.

**Expected:** Inference loads successfully. Normalization parameters loaded from checkpoint (log message: "Loaded normalization from checkpoint (self-contained)"). Predictions are unnormalized correctly using checkpoint normalization.

**Why human:** Requires running inference.py and verifying output matches expected behavior without metadata.json present.

#### 4. Atomic write crash resilience

**Test:** Start training. While checkpoint is being saved (watch disk I/O), forcefully kill the process (kill -9 or power loss simulation). Attempt to load the checkpoint file.

**Expected:** Either (a) the checkpoint file does not exist yet (temp file was being written), or (b) the checkpoint file is complete and valid (atomic rename completed). Never a corrupted partial checkpoint.

**Why human:** Requires precise timing to interrupt during save, and external process monitoring. Difficult to automate reliably.

#### 5. Emergency checkpoint on Ctrl+C

**Test:** Start training. After 2 epochs complete, press Ctrl+C during epoch 3 training loop.

**Expected:** Message: "SIGINT received. Saving emergency checkpoint after current epoch...". Epoch 3 completes. File `outputs/checkpoints/EMERGENCY_checkpoint_epoch_003.pt` created. Process exits cleanly. Emergency checkpoint contains `emergency=True` and `emergency_reason='user_interrupt'`.

**Why human:** Requires manual interrupt timing and verification of emergency checkpoint contents.

---

## Summary

Phase 3 goal **ACHIEVED**. All 7 observable truths verified. All 6 required artifacts substantive and wired correctly. All 4 requirements satisfied. No stub patterns or anti-patterns detected.

**Automated verification:** All structural checks passed. Checkpoint system is fully integrated:
- Centralized I/O module with atomic writes (os.replace)
- Full resume with all training state (model, optimizer, scheduler, scaler, patience, history)
- Cross-device portability (CPU-normalized tensors, optimizer device remapping)
- Self-contained inference (normalization embedded in checkpoint)
- Two-checkpoint management (best + rolling latest)
- Config validation for resume_from

**Human verification needed:** 5 items require manual testing to validate runtime behavior (end-to-end resume, cross-device portability, inference without metadata, crash resilience, emergency checkpoint). These are functional tests that cannot be verified structurally.

**Ready for:** Phase 4 (Data Pipeline) can proceed. Checkpoint system provides the foundation for long training runs with resume capability.

---

_Verified: 2026-02-03T20:26:02Z_
_Verifier: Claude (gsd-verifier)_
