# Phase 1: Device Foundation & Tech Debt - Context

**Gathered:** 2026-02-02
**Status:** Ready for planning

<domain>
## Phase Boundary

Pipeline auto-detects the best available device and runs identically on CUDA, MPS, and CPU. Legacy dead code removed, reproducible seeding in place. This phase covers device detection, config interface, AMP handling per device, legacy cleanup, and seeding. Error handling, checkpoint portability, and MPS op alternatives are separate phases.

</domain>

<decisions>
## Implementation Decisions

### Config interface for device
- Config field: `device` with values `auto`, `cuda`, `mps`, `cpu`
- `auto` is the default — detects best available device
- If a forced device (e.g. `device: mps`) is unavailable, hard error and exit with a clear message — no silent fallback
- Single log line at startup: e.g. `Using device: mps (Apple M2 Pro)`
- Device resolved once at startup, then passed as a parameter through function arguments — no global `get_device()` singleton

### Seeding & reproducibility
- Statistically equivalent runs, not bit-for-bit identical — no need to enforce deterministic algorithms
- Seed configurable in config.yaml with a fixed default (e.g. `seed: 42`)
- Seed all three sources: `torch`, `numpy`, and Python `random`
- Same-device reproducibility only — different devices may diverge, that's acceptable

### Claude's Discretion
- Device detection priority order (CUDA > MPS > CPU is standard, but Claude can adjust)
- DummyGradScaler implementation details for MPS
- Legacy cleanup approach for ConvLSTM.py and inference.py
- Where exactly to place the seeding call in the training flow

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

*Phase: 01-device-foundation-tech-debt*
*Context gathered: 2026-02-02*
