# Phase 6: Test Coverage - Context

**Gathered:** 2026-02-04
**Status:** Ready for planning

<domain>
## Phase Boundary

Automated tests validating all stabilization work from Phases 1-5: device management, config validation, error handling, checkpoints, data pipeline, and MPS compatibility. Every module touched in Phases 1-5 gets at least one test. No new features — purely verification.

</domain>

<decisions>
## Implementation Decisions

### Test organization
- Mirror source tree: tests/test_device.py for utils/device.py, tests/test_checkpoint.py for utils/checkpoint.py, etc.
- Shared conftest.py in tests/ with common fixtures (tmp dirs, sample configs, device detection)
- Test names describe behavior: test_nan_loss_skips_optimizer, test_checkpoint_roundtrip, test_mmap_loading_correct_shape
- pytest markers for device filtering: @pytest.mark.mps, @pytest.mark.cuda — CI can skip device-specific tests

### Device coverage
- Device-specific tests skip entirely (via marker) when target device unavailable — no CPU fallback
- Real device detection only — no mocking of torch.cuda.is_available() or torch.backends.mps.is_available()
- MPS-specific ops (SSIM tiling, Welford uncertainty) tested on available device: verify no crash + valid output, no cross-device comparison
- DummyGradScaler gets an explicit test: scale() returns input unchanged, step() calls optimizer, update() is no-op

### Fixture & data strategy
- Data pipeline tests use synthetic .npy files on disk (via tmp_path) — exercises real mmap loading path
- Shared base config fixture in conftest.py; individual tests override specific fields as needed
- Checkpoint roundtrip tests use real SolarFlare model with minimal dimensions (tiny but real)
- Data fixtures use small realistic dimensions (32x32) to exercise convolution/pooling code paths

### Coverage targets
- Unit tests only — no end-to-end training loop test
- Config validation: representative sample of known bad configs, not exhaustive per-rule coverage
- Error handling tests verify actual behavior: inject NaN loss and check optimizer step skipped; simulate signal and check emergency checkpoint exists
- Every module touched in Phases 1-5 gets at least one test — no acceptable gaps

### Claude's Discretion
- Exact test file count and grouping within the mirror structure
- Helper utilities in conftest.py vs test-local helpers
- Parametrize usage for testing multiple input shapes/configs
- Assertion tolerance values for floating-point comparisons

</decisions>

<specifics>
## Specific Ideas

No specific requirements — open to standard approaches

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 06-test-coverage*
*Context gathered: 2026-02-04*
