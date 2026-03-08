---
phase: 9
slug: training-policy
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-03-08
---

# Phase 9 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest (existing) |
| **Config file** | tests/conftest.py |
| **Quick run command** | `python -m pytest tests/test_data_pipeline.py tests/test_config.py -x -q` |
| **Full suite command** | `python -m pytest tests/ -x -q` |
| **Estimated runtime** | ~30 seconds |

---

## Sampling Rate

- **After every task commit:** Run `python -m pytest tests/test_data_pipeline.py tests/test_config.py -x -q`
- **After every plan wave:** Run `python -m pytest tests/ -x -q`
- **Before `/gsd:verify-work`:** Full suite must be green
- **Max feedback latency:** 30 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 09-01-01 | 01 | 0 | TRAIN-04 | unit | `python -m pytest tests/test_data_pipeline.py::TestBuildIndex::test_build_index_returns_flare_flags -x -q` | ❌ W0 | ⬜ pending |
| 09-01-02 | 01 | 0 | TRAIN-04 | unit | `python -m pytest tests/test_data_pipeline.py::TestBuildIndex::test_flare_flags_detect_extreme_values -x -q` | ❌ W0 | ⬜ pending |
| 09-01-03 | 01 | 0 | TRAIN-04 | unit | `python -m pytest tests/test_data_pipeline.py::TestBuildIndex::test_flare_flags_only_check_output_frames -x -q` | ❌ W0 | ⬜ pending |
| 09-01-04 | 01 | 0 | TRAIN-04 | unit | `python -m pytest tests/test_data_pipeline.py::TestWeightedSampler::test_sampler_replaces_shuffle -x -q` | ❌ W0 | ⬜ pending |
| 09-01-05 | 01 | 0 | TRAIN-04 | unit | `python -m pytest tests/test_data_pipeline.py::TestWeightedSampler::test_sampler_weights_match_flare_flags -x -q` | ❌ W0 | ⬜ pending |
| 09-01-06 | 01 | 0 | TRAIN-04 | unit | `python -m pytest tests/test_config.py::test_flare_oversample_with_no_augmentation_warning -x -q` | ❌ W0 | ⬜ pending |
| 09-01-07 | 01 | 0 | TRAIN-04 | unit | `python -m pytest tests/test_config.py::test_flare_oversample_weight_valid -x -q` | ❌ W0 | ⬜ pending |
| 09-02-01 | 01 | 1 | TRAIN-01 | unit | `python -m pytest tests/test_config.py -x -q` | ✅ | ⬜ pending |
| 09-02-02 | 01 | 1 | TRAIN-02 | unit | `python -m pytest tests/test_data_pipeline.py::TestBuildIndex -x -q` | ✅ | ⬜ pending |
| 09-02-03 | 01 | 1 | TRAIN-03 | unit | `python -m pytest tests/test_config.py -x -q` | ✅ | ⬜ pending |
| 09-02-04 | 01 | 1 | TRAIN-05 | unit | `python -m pytest tests/test_config.py -x -q` | ✅ | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `tests/test_data_pipeline.py::TestBuildIndex::test_build_index_returns_flare_flags` — stubs for TRAIN-04
- [ ] `tests/test_data_pipeline.py::TestBuildIndex::test_flare_flags_detect_extreme_values` — stubs for TRAIN-04
- [ ] `tests/test_data_pipeline.py::TestBuildIndex::test_flare_flags_only_check_output_frames` — stubs for TRAIN-04
- [ ] `tests/test_data_pipeline.py::TestWeightedSampler::test_sampler_replaces_shuffle` — stubs for TRAIN-04
- [ ] `tests/test_data_pipeline.py::TestWeightedSampler::test_sampler_weights_match_flare_flags` — stubs for TRAIN-04
- [ ] `tests/test_config.py::test_flare_oversample_with_no_augmentation_warning` — stubs for TRAIN-04
- [ ] `tests/test_config.py::test_flare_oversample_weight_valid` — stubs for TRAIN-04

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| LR curve visible in training logs | TRAIN-01 | Requires visual inspection of training_history.png | Run training for 5+ epochs, check LR curve in output |
| Model converges over 50 epochs | TRAIN-05 | Requires full training run | Run 50-epoch training, verify val_loss stabilizes/decreases |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 30s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
