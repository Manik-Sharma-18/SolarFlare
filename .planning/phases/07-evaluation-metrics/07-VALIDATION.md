---
phase: 7
slug: evaluation-metrics
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-03-07
---

# Phase 7 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest (already configured) |
| **Config file** | none — pytest defaults + conftest.py markers |
| **Quick run command** | `python -m pytest tests/test_metrics.py -x -q` |
| **Full suite command** | `python -m pytest tests/ -v` |
| **Estimated runtime** | ~5 seconds |

---

## Sampling Rate

- **After every task commit:** Run `python -m pytest tests/test_metrics.py -x -q`
- **After every plan wave:** Run `python -m pytest tests/ -v`
- **Before `/gsd:verify-work`:** Full suite must be green
- **Max feedback latency:** 10 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 07-01-F1 | 01 | 1 | EVAL-01..07 | TDD unit | `python -m pytest tests/test_metrics.py -v -x` | Created by Plan 01 | pending |
| 07-02-T1 | 02 | 2 | ALL | integration | `python -m pytest tests/test_metrics.py -v -x && python -m pytest tests/ -x -q` | Created by Plan 01 | pending |
| 07-02-T2 | 02 | 2 | ALL | config+viz | `python -m pytest tests/ -x -q` | Created by Plan 01 | pending |

*Status: pending / green / red / flaky*

---

## Wave 0 Requirements

- [ ] `tests/test_metrics.py` — stubs for all EVAL-* requirements (unit + integration)
- [ ] Fixture: `tiny_predictions` — small (2, 1, 4, 8, 8) tensor pair for fast metric tests
- [ ] No framework install needed (pytest already configured)
- [ ] No conftest changes needed (existing fixtures sufficient)

*Wave 0 creates test stubs; implementation waves fill them in.*

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Console summary format | ALL | Visual formatting check | Run 1-epoch training, verify epoch summary shows new metrics |
| Visualization plots | ALL | Visual output check | Run training, inspect new metric plots in outputs/ |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 10s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
