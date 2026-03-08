---
phase: 10-architecture-scaling
plan: 01
subsystem: model
tags: [sa-convlstm, channel-attention, temporal-attention, attention-gate, pytorch, bmm]

# Dependency graph
requires:
  - phase: existing
    provides: "ConvLSTMCell base class for composition"
provides:
  - "SelfAttentionMemory, SAConvLSTMCell, SAConvLSTM classes in models/sa_convlstm.py"
  - "TemporalAttention, AttentionGate classes in models/attention.py"
  - "Comprehensive test suites for both modules"
affects: [10-02-predictor-integration, 10-03-config-infrastructure]

# Tech tracking
tech-stack:
  added: []
  patterns: [channel-attention-via-bmm, composition-over-inheritance, 3-tuple-hidden-state, attention-unet-gate]

key-files:
  created:
    - models/sa_convlstm.py
    - models/attention.py
    - tests/test_sa_convlstm.py
    - tests/test_attention.py
  modified: []

key-decisions:
  - "Memory projection layer added to SelfAttentionMemory for hidden_dim shape consistency across timesteps"
  - "SAM parameter count ~4*C^2 (not 3.5*C^2) due to memory projection -- still modest overhead"
  - "TemporalAttention proj_dim defaults to channels (full projection) for maximum expressiveness"
  - "AttentionGate f_int defaults to max(encoder_channels//2, 8) for small channel counts"

patterns-established:
  - "Composition pattern: SAConvLSTMCell wraps ConvLSTMCell via self.convlstm_cell"
  - "3-tuple hidden state: (h, c, m) replaces (h, c) for SA-ConvLSTM modules"
  - "Manual attention: torch.bmm + torch.softmax everywhere for MPS compatibility"
  - "Channel attention: global avg pool -> outer product -> softmax -> bmm with values"

requirements-completed: [ARCH-01, ARCH-03, ARCH-07]

# Metrics
duration: 5min
completed: 2026-03-08
---

# Phase 10 Plan 01: Standalone Modules Summary

**SA-ConvLSTM cell with channel attention memory, temporal attention over encoder states, and attention U-Net gate -- all MPS-safe via manual bmm+softmax**

## Performance

- **Duration:** 5 min
- **Started:** 2026-03-08T13:21:32Z
- **Completed:** 2026-03-08T13:26:17Z
- **Tasks:** 2
- **Files modified:** 4

## Accomplishments
- SelfAttentionMemory module with channel attention (bmm+softmax), gated h/m fusion, and residual output
- SAConvLSTMCell wrapping ConvLSTMCell via composition, returning (h, c, m) 3-tuple
- SAConvLSTM multi-step wrapper with 3-tuple hidden state init and continuation
- TemporalAttention module with Q/K/V via 1x1 Conv2d, global avg pool queries, and attention weight output
- AttentionGate with Attention U-Net pattern (no BatchNorm), producing sigmoid-gated skip features
- 25 comprehensive tests covering shapes, finiteness, manual attention verification, composition, and edge cases

## Task Commits

Each task was committed atomically:

1. **Task 1: TDD SA-ConvLSTM module** (TDD)
   - `1ebb2be` (test: add failing tests for SA-ConvLSTM module)
   - `32672a2` (feat: implement SA-ConvLSTM module with channel attention memory)

2. **Task 2: TDD attention modules** (TDD)
   - `5238023` (test: add failing tests for attention modules)
   - `6ad8e67` (feat: implement TemporalAttention and AttentionGate modules)

## Files Created/Modified
- `models/sa_convlstm.py` - SelfAttentionMemory, SAConvLSTMCell, SAConvLSTM classes
- `models/attention.py` - TemporalAttention and AttentionGate modules
- `tests/test_sa_convlstm.py` - 13 unit tests for SA-ConvLSTM module
- `tests/test_attention.py` - 12 unit tests for attention modules

## Decisions Made

1. **Memory projection layer added** - SelfAttentionMemory's fused output (z_fused) has attn_dim channels, but memory state must maintain hidden_dim channels for consistency across timesteps. Added memory_proj Conv2d(attn_dim, hidden_dim, 1) to project memory back to hidden_dim. This increases SAM params from 3.5*C^2 to ~4*C^2 but is necessary for correctness.

2. **Parameter count adjustment** - The plan estimated 3.5*C^2 params per SAM. Actual count is ~4*C^2 (+ biases) due to the memory projection. For [32,64,128] channels, total SAM overhead across 6 cells will be approximately 90K params instead of 75K -- still modest relative to the full model.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Memory state shape mismatch**
- **Found during:** Task 1 (GREEN phase)
- **Issue:** z_fused has shape (B, attn_dim, H, W) but m_new must be (B, hidden_dim, H, W) to be consistent with hidden state initialization and continuation across timesteps
- **Fix:** Added memory_proj Conv2d(attn_dim, hidden_dim, 1) to project fused memory back to hidden_dim
- **Files modified:** models/sa_convlstm.py
- **Verification:** All 13 SA-ConvLSTM tests pass including shape tests and hidden state continuation
- **Committed in:** 32672a2

---

**Total deviations:** 1 auto-fixed (1 bug)
**Impact on plan:** Essential for correctness. Memory state must maintain consistent shape across timesteps. No scope creep.

## Issues Encountered
None.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- SA-ConvLSTM and attention modules ready for integration into SolarFluxPredictor (Plan 02)
- All modules independently tested with comprehensive test suites
- 3-tuple hidden state pattern established -- predictor.py must update all state unpacking sites
- No BatchNorm in attention gate -- consistent with batch_size=1 constraint
- All attention uses manual bmm+softmax -- MPS-safe for cross-device compatibility

## Self-Check: PASSED

All 4 created files verified on disk. All 4 commit hashes verified in git log. 25/25 tests pass.

---
*Phase: 10-architecture-scaling*
*Completed: 2026-03-08*
