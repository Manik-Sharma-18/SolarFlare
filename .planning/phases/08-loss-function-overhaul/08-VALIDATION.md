---
phase: 8
slug: loss-function-overhaul
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-03-08
---

# Phase 8 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest (installed) |
| **Config file** | tests/conftest.py |
| **Quick run command** | `python -m pytest tests/test_losses.py -x -q` |
| **Full suite command** | `python -m pytest tests/ -x -q` |
| **Estimated runtime** | ~15 seconds |

---

## Sampling Rate

- **After every task commit:** Run `python -m pytest tests/test_losses.py -x -q`
- **After every plan wave:** Run `python -m pytest tests/ -x -q`
- **Before `/gsd:verify-work`:** Full suite must be green
- **Max feedback latency:** 15 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 08-01-01 | 01 | 1 | LOSS-04 | unit | `python -m pytest tests/test_losses.py::TestWeightedMAE -x` | Existing (update) | ⬜ pending |
| 08-01-02 | 01 | 1 | LOSS-05 | unit | `python -m pytest tests/test_losses.py::TestAsymmetricExtremeLoss -x` | ❌ W0 | ⬜ pending |
| 08-01-03 | 01 | 1 | LOSS-01 | unit | `python -m pytest tests/test_losses.py::TestTemporalDiffLoss -x` | ❌ W0 | ⬜ pending |
| 08-01-04 | 01 | 1 | LOSS-03 | unit | `python -m pytest tests/test_losses.py::TestTemporalVarPenalty -x` | ❌ W0 | ⬜ pending |
| 08-01-05 | 01 | 1 | LOSS-02 | unit | `python -m pytest tests/test_losses.py::TestTemporalWeighting -x` | ❌ W0 | ⬜ pending |
| 08-01-06 | 01 | 1 | LOSS-06, LOSS-07 | unit | `python -m pytest tests/test_losses.py::TestCompositeLoss -x` | Existing (update) | ⬜ pending |
| 08-02-01 | 02 | 2 | LOSS-07 | unit | `python -m pytest tests/test_losses.py -x -q` | Existing (update) | ⬜ pending |
| 08-02-02 | 02 | 2 | LOSS-07 | integration | `python -m pytest tests/ -x -q` | Existing (update) | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `tests/test_losses.py::TestTemporalDiffLoss` — stubs for LOSS-01 (temporal diff non-zero for different dynamics, zero for identical)
- [ ] `tests/test_losses.py::TestTemporalWeighting` — stubs for LOSS-02 (later timesteps weighted more)
- [ ] `tests/test_losses.py::TestTemporalVarPenalty` — stubs for LOSS-03 (negative value, capped at target variation)
- [ ] `tests/test_losses.py::TestAsymmetricExtremeLoss` — stubs for LOSS-05 (asymmetric above threshold, symmetric below)
- [ ] Update `tests/test_losses.py::TestWeightedMAE` — covers LOSS-04 (absolute threshold, consistent across batches)
- [ ] Update `tests/test_losses.py::TestCompositeLoss` — covers LOSS-07 (all 6 component keys in return dict)
- [ ] Update `tests/test_losses.py::TestGetLossFunction` — covers LOSS-06 (new config keys)

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Compact console summary readable | LOSS-07 | Visual formatting | Run 1 epoch, verify `Loss: X | TDiff: X | TVar: X | Extreme: X` appears |
| Loss breakdown plot correct | LOSS-07 | Visual rendering | Run training, check `outputs/training_history.png` for new subplot row |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 15s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
