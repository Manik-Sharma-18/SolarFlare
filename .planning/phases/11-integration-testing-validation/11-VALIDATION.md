---
phase: 11
slug: integration-testing-validation
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-03-09
---

# Phase 11 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest (conftest.py with custom markers: mps, cuda) |
| **Config file** | tests/conftest.py |
| **Quick run command** | `python -m pytest tests/ -x --timeout=60` |
| **Full suite command** | `python -m pytest tests/ -v` |
| **Estimated runtime** | ~30 seconds (unit tests only) |

---

## Sampling Rate

- **After every task commit:** Run `python -m pytest tests/ -x --timeout=60`
- **After every plan wave:** Run `python -m pytest tests/ -v`
- **Before `/gsd:verify-work`:** Full suite must be green
- **Max feedback latency:** 30 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| SC-01 | 01 | 1 | End-to-end | integration (manual) | `python main.py` (smoke: 3-5 epochs) | Yes | ⬜ pending |
| SC-02 | 01 | 1 | Temporal variation | integration (manual) | Check `outputs/test_results.json` | N/A | ⬜ pending |
| SC-03 | 01 | 1 | CSI improvement | integration (manual) | Check `outputs/test_results.json` | N/A | ⬜ pending |
| SC-04 | 02 | 2 | Comparison report | smoke | `python generate_comparison.py && test -f COMPARISON.md` | ❌ W0 | ⬜ pending |
| EXISTING | - | 0 | All prior reqs | unit | `python -m pytest tests/ -x` | ✅ | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `generate_comparison.py` — standalone comparison report generator (new file, Wave 0 or Plan 02)
- [ ] Verify existing test suite passes before any training: `python -m pytest tests/ -x`

*Existing infrastructure covers unit/integration testing. Only the comparison script is new.*

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| 50-epoch training completes without errors/NaN | SC-01 | Long-running training process (2-8 hours) | Run `python main.py`, monitor for errors, NaN losses, crashes |
| Temporal variation ratio > 0.060 | SC-02 | Requires full training to produce results | Inspect `outputs/test_results.json` after training |
| CSI > 0.051 | SC-03 | Requires full training to produce results | Inspect `outputs/test_results.json` after training |
| Visual quality of comparison charts | SC-04 | Subjective visual assessment | Review PNGs referenced in COMPARISON.md |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 30s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
