---
phase: 10-architecture-scaling
plan: 02
subsystem: model
tags: [sa-convlstm, temporal-attention, attention-gate, delta-scale, predictor, pytorch]

# Dependency graph
requires:
  - phase: 10-01
    provides: "SA-ConvLSTM, TemporalAttention, AttentionGate standalone modules"
provides:
  - "SolarFluxPredictor with full v3.0 architecture support (SA-ConvLSTM, temporal attention, attention gate, delta scale)"
  - "Backward-compatible predictor: default params preserve original ConvLSTM behavior"
  - "Updated models/__init__.py with SA-ConvLSTM and attention module exports"
  - "sa_model_config fixture and comprehensive v3.0 integration tests"
affects: [10-03-config-infrastructure, 11-integration]

# Tech tracking
tech-stack:
  added: []
  patterns: [conditional-convlstm-class, additive-temporal-attention, gated-skip-connection, learnable-delta-scale, packed-encoder-states]

key-files:
  created: []
  modified:
    - models/predictor.py
    - models/__init__.py
    - main.py
    - tests/conftest.py
    - tests/test_model.py

key-decisions:
  - "Pack encoder h3 states as tensor (not list) for gradient checkpoint compatibility, unpack after"
  - "nn.Dropout replaces nn.Dropout2d for 5D ConvLSTM outputs to avoid PyTorch deprecation warning"
  - "Temporal attention queries decoder_state3[0][0] (hidden state h) not dec_h3 output for richer context"

patterns-established:
  - "Conditional module class: ConvLSTMClass = SAConvLSTM if use_sa_convlstm else ConvLSTM"
  - "Additive injection: dec_h3_frame = dec_h3[:,:,0] + context (graceful degradation)"
  - "Packed encoder states: stack to tensor for checkpoint, unpack to list for attention"

requirements-completed: [ARCH-01, ARCH-02, ARCH-03, ARCH-04, ARCH-05, ARCH-06, ARCH-07]

# Metrics
duration: 4min
completed: 2026-03-08
---

# Phase 10 Plan 02: Predictor Integration Summary

**SA-ConvLSTM, temporal attention, attention gate, and delta scaling wired into SolarFluxPredictor with full backward compatibility and 55 passing tests**

## Performance

- **Duration:** 4 min
- **Started:** 2026-03-08T13:29:36Z
- **Completed:** 2026-03-08T13:33:27Z
- **Tasks:** 2
- **Files modified:** 5

## Accomplishments
- Predictor accepts use_sa_convlstm, temporal_attention, attention_gate, and delta_scale_init params
- All 6 ConvLSTM modules conditionally replaced with SAConvLSTM via single ConvLSTMClass variable
- Encoder collects packed h3 states for temporal attention with gradient checkpoint compatibility
- Temporal attention additively injected into decoder loop output, attention gate applied to skip connection
- delta_scale nn.Parameter(100.0) multiplies raw output delta when enabled
- 3-tuple hidden state handling for SA-ConvLSTM decoder initialization
- nn.Dropout replaces nn.Dropout2d for 5D tensor deprecation fix
- main.py updated to pass new config params in both run_training() and run_inference()
- 13 new architecture tests added: delta scale, full arch, dual channel, downsample, SA module replacement, wider channels, kernel 5, MPS
- All 55 tests pass (15 original + 25 from plan 01 + 15 new)

## Task Commits

Each task was committed atomically:

1. **Task 1: Wire SA-ConvLSTM, temporal attention, attention gate, and delta_scale into predictor.py** - `eed8041` (feat)
2. **Task 2: Add SA model fixture and v3.0 architecture tests** - `ba75dc0` (feat)

## Files Created/Modified
- `models/predictor.py` - Updated SolarFluxPredictor with SA-ConvLSTM, temporal attention, attention gate, delta_scale, nn.Dropout fix
- `models/__init__.py` - Added SelfAttentionMemory, SAConvLSTMCell, SAConvLSTM, TemporalAttention, AttentionGate exports
- `main.py` - Updated run_training() and run_inference() to pass v3.0 config params
- `tests/conftest.py` - Added sa_model_config fixture with all ARCH features enabled
- `tests/test_model.py` - Added TestDeltaScale, TestFullArchitecture classes, SA-ConvLSTM shape tests, MPS full arch test

## Decisions Made
1. **Pack encoder h3 states as tensor** - Encoder h3 states are stacked into a (B, T, C, H, W) tensor before returning from _encoder_forward, then unpacked to a list after. This ensures compatibility with torch.utils.checkpoint which requires tensor returns, while temporal attention needs a list of states.
2. **nn.Dropout instead of nn.Dropout2d** - Changed all three dropout layers from nn.Dropout2d to nn.Dropout to fix PyTorch deprecation warning when applied to 5D ConvLSTM outputs (per RESEARCH.md pitfall 6).
3. **Query decoder hidden state (not output)** - Temporal attention queries decoder_state3[0][0] (the h hidden state) rather than dec_h3[:,:,0] (the ConvLSTM output tensor). The hidden state captures the full decoder context for more meaningful attention.

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
None.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- Full v3.0 architecture integration complete and tested
- All ARCH requirements (01-07) covered by tests
- Ready for Plan 03 (config infrastructure) to add v3.0 config presets and training setup
- Backward compatibility verified: default params (no SA features) produce identical behavior

## Self-Check: PASSED

All 5 modified files verified on disk. Both commit hashes verified in git log. 55/55 tests pass.

---
*Phase: 10-architecture-scaling*
*Completed: 2026-03-08*
