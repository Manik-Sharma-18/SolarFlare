# Phase 2: Config Validation & Error Handling - Context

**Gathered:** 2026-02-03
**Status:** Ready for planning

<domain>
## Phase Boundary

Catch bad configurations and runtime anomalies with clear messages before they waste compute or corrupt training. Covers: config validation at startup, NaN/gradient anomaly handling during training, data file failure policies, and graceful shutdown on interruption signals.

</domain>

<decisions>
## Implementation Decisions

### Validation strictness
- All config validation runs at startup, before any data loading or model creation — fail fast
- Severity is tiered: warnings for unusual-but-valid values (e.g., very high LR), hard errors for invalid values (e.g., negative LR, unknown device)
- Report all validation errors at once — user fixes everything in one pass, not iterative fix-run cycles
- Cross-field validation enabled: detect mismatches like `dual_channel: true` with `input_channels: 1`, or AMP on CPU-only

### NaN/anomaly response
- NaN loss detected → skip that batch's optimizer step, log a warning, continue training
- Consecutive NaN threshold: after N consecutive NaN batches, abort training (something is fundamentally wrong)
- The consecutive NaN threshold is configurable in config.yaml
- Gradient norm monitoring enabled: log warnings when gradient norms exceed a threshold (early warning before NaN)

### Data failure policy
- Pre-flight scan: validate all data files before training starts — know dataset health upfront
- Individual bad files are skipped and logged, training continues with remaining data
- If >10% of files fail validation, abort training — something is systematically wrong
- Failure report lists each failed file with its specific error reason (full detail, not just counts)

### Graceful shutdown
- SIGINT (Ctrl+C) and SIGTERM both trigger graceful shutdown
- On first signal: save emergency checkpoint, then exit cleanly — never lose progress
- On second Ctrl+C: force quit immediately, skip checkpoint save
- Emergency checkpoints are clearly labeled (distinct naming or metadata flag) so users know it's a partial/interrupted save

### Claude's Discretion
- Exact gradient norm warning threshold
- Emergency checkpoint naming convention details
- Warning vs error classification for edge-case config values
- Pre-flight scan implementation approach (full read vs header check)

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

*Phase: 02-config-validation-error-handling*
*Context gathered: 2026-02-03*
