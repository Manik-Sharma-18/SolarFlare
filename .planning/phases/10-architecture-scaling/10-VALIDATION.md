---
phase: 10
slug: architecture-scaling
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-03-08
---

# Phase 10 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest (already configured) |
| **Config file** | tests/conftest.py (shared fixtures including base_config, device, tiny_model_config) |
| **Quick run command** | `python -m pytest tests/test_model.py tests/test_sa_convlstm.py tests/test_attention.py -x -q` |
| **Full suite command** | `python -m pytest tests/ -x -q` |
| **Estimated runtime** | ~5 seconds |

---

## Sampling Rate

- **After every task commit:** Run `python -m pytest tests/test_model.py tests/test_sa_convlstm.py tests/test_attention.py -x -q`
- **After every plan wave:** Run `python -m pytest tests/ -x -q`
- **Before `/gsd:verify-work`:** Full suite must be green
- **Max feedback latency:** 5 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 10-01-01 | 01 | 0 | ARCH-01 | unit | `python -m pytest tests/test_sa_convlstm.py -x -q` | ❌ W0 | ⬜ pending |
| 10-01-02 | 01 | 0 | ARCH-03, ARCH-07 | unit | `python -m pytest tests/test_attention.py -x -q` | ❌ W0 | ⬜ pending |
| 10-01-03 | 01 | 0 | ARCH-02, ARCH-04, ARCH-05 | unit | `python -m pytest tests/test_model.py -x -q` | ❌ W0 (update) | ⬜ pending |
| 10-02-01 | 02 | 1 | ARCH-01 | unit | `python -m pytest tests/test_sa_convlstm.py -x -q` | ❌ W0 | ⬜ pending |
| 10-02-02 | 02 | 1 | ARCH-07 | unit | `python -m pytest tests/test_attention.py::TestTemporalAttention -x -q` | ❌ W0 | ⬜ pending |
| 10-02-03 | 02 | 1 | ARCH-03 | unit | `python -m pytest tests/test_attention.py::TestAttentionGate -x -q` | ❌ W0 | ⬜ pending |
| 10-02-04 | 02 | 1 | ARCH-02 | unit | `python -m pytest tests/test_model.py::TestDeltaScale -x -q` | ❌ W0 | ⬜ pending |
| 10-03-01 | 03 | 2 | ALL | integration | `python -m pytest tests/test_model.py -x -q` | ❌ W0 (update) | ⬜ pending |
| 10-03-02 | 03 | 2 | ARCH-06 | unit | `python -m pytest tests/test_model.py::TestForwardOutput::test_forward_with_dropout -x -q` | ✅ (update) | ⬜ pending |
| 10-03-03 | 03 | 2 | ALL | smoke | `python -m pytest tests/test_model.py -m mps -x -q` | ❌ W0 (update) | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `tests/test_sa_convlstm.py` — stubs for ARCH-01 (SAConvLSTMCell shape, SAM attention weights, parameter count)
- [ ] `tests/test_attention.py` — stubs for ARCH-03, ARCH-07 (AttentionGate, TemporalAttention shapes and properties)
- [ ] Update `tests/test_model.py` — add tests for ARCH-02 (delta_scale), ARCH-04 (wider channels), ARCH-05 (kernel 5), full architecture integration test, MPS smoke test
- [ ] Update `tests/conftest.py` — add fixture for SA model config with all ARCH features enabled

*Existing infrastructure covers basic model testing; new files needed for new modules.*

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Attention weights not collapsed to uniform/zero during training | ARCH-03 | Requires actual training run to verify learning dynamics | Train for 5 epochs, check attention entropy logs are not at max/zero |
| delta_scale value changes during training | ARCH-02 | Requires actual training run | Train for 5 epochs, verify delta_scale != 100.0 after training |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 5s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
