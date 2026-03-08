# Phase 11: Integration Testing & Validation - Research

**Researched:** 2026-03-09
**Domain:** End-to-end training validation, metric comparison, diagnostic reporting
**Confidence:** HIGH

## Summary

Phase 11 is a validation phase -- no new features are built. The objective is to run the complete v3.0 training pipeline with all features enabled simultaneously (SA-ConvLSTM, temporal attention, attention gates, temporal loss, cosine LR, flare oversampling, balanced augmentation) and produce a diagnostic comparison report against the v2.0 baseline. The codebase is already fully integrated: `main.py` runs the complete pipeline from data loading through testing with visualization, and `config.yaml` already has all v3.0 defaults configured.

The primary technical challenge is not feature integration (already done in Phases 7-10) but rather: (1) ensuring a full 50-epoch training run completes without crashes/NaN on MPS, (2) parsing both v2.0 and v3.0 results into a structured comparison, and (3) producing meaningful visualizations that expose whether the model has truly learned temporal dynamics vs. near-persistence behavior.

**Primary recommendation:** Run a 3-epoch smoke test first to catch integration issues early, then execute the full 50-epoch run, then build a standalone comparison script that reads `outputs/test_results.json` (v3.0) against hardcoded v2.0 baselines and generates `COMPARISON.md` with embedded PNG charts.

<user_constraints>

## User Constraints (from CONTEXT.md)

### Locked Decisions
- **v2.0 baseline strategy:** Use known v2.0 values from `outputs copy/diagnostic_results.json` (test split only). Hardcode v2.0 baseline numbers directly in the comparison script/report -- no separate v2.0 config file.
  - v2.0 test baselines: MAE [0.102, 0.109, 0.112, 0.114], RMSE [0.145, 0.153, 0.156, 0.157], correlation [0.565, 0.508, 0.483, 0.467], persistence skill [2.9%, 4.7%, 5.2%, 5.1%], CSI 0.051, HSS 0.092, var ratio 0.060
- **Comparison report format:** Markdown report at project root: `COMPARISON.md`. Metric comparison tables with delta columns (v3.0 value, v2.0 value, change, % change). Bar charts / line plots comparing per-timestep metrics saved as PNGs referenced from markdown. Include sample prediction visualizations. Summary verdict at top (PASS/MIXED/REGRESSION). Tradeoffs documented neutrally.
- **Success thresholds:** No fixed minimum targets -- "any improvement" over v2.0 on temporal variation ratio and CSI counts as success. MAE/RMSE regression acceptable if temporal metrics improve.
- **Run strategy:** Single training run, seed 42, 50 epochs with cosine schedule. Smoke test first (3-5 epochs). Full 50-epoch run after smoke test passes. Device: MPS (Mac GPU), use_amp: false. If issues arise: diagnose and fix, then re-run.

### Claude's Discretion
- Smoke test epoch count (3-5 range)
- Chart styling and layout for comparison visualizations
- Report prose structure and section ordering
- How to select representative prediction samples for qualitative comparison
- Whether to include attention entropy analysis in the report

### Deferred Ideas (OUT OF SCOPE)
None -- discussion stayed within phase scope

</user_constraints>

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| matplotlib | >=3.5.0 | Comparison bar charts, line plots, prediction grids | Already in requirements.txt, used by existing visualization code |
| json | stdlib | Read v3.0 test_results.json | Standard library |
| numpy | >=1.21.0 | Metric computation, array manipulation | Already in requirements.txt |
| torch | >=2.0.0 | Model training (via existing main.py) | Already in requirements.txt |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| pathlib | stdlib | File path handling | All file I/O |
| datetime | stdlib | Timestamp in report | Report generation |
| textwrap | stdlib | Markdown formatting | Optional for report prose |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| matplotlib | plotly/seaborn | Unnecessary dependency; matplotlib already available and sufficient |
| Manual markdown | jinja2 templates | Over-engineering for a single comparison report |
| Hardcoded baselines | v2.0 config + re-run | User explicitly chose hardcoded approach -- faster, reproducible |

**Installation:**
```bash
# No new packages needed -- all dependencies already in requirements.txt
pip install -r requirements.txt
```

## Architecture Patterns

### Phase Workflow Structure
```
Phase 11 Execution Flow:
1. Smoke test (3 epochs)          # Validate integration
   |-> Pass? Continue             # No crashes, no NaN, loss decreasing
   |-> Fail? Diagnose & fix       # LR, loss imbalance, attention collapse
2. Full training run (50 epochs)  # python main.py (config.yaml already correct)
   |-> outputs/test_results.json  # v3.0 results saved here
3. Comparison script              # generate_comparison.py
   |-> COMPARISON.md              # Final report at project root
   |-> comparison_*.png           # Chart images
```

### Recommended File Structure
```
project_root/
  generate_comparison.py    # Standalone comparison script
  COMPARISON.md             # Generated comparison report
  comparison_metrics.png    # Per-timestep metric bar charts
  comparison_temporal.png   # Temporal dynamics comparison
  comparison_samples.png    # Sample prediction visualizations
  outputs/
    test_results.json       # v3.0 results (generated by main.py)
    training_history.json   # v3.0 training curves
```

### Pattern 1: Hardcoded v2.0 Baseline
**What:** Embed v2.0 baseline values directly in the comparison script as a Python dict rather than reading from file.
**When to use:** User decision -- avoids path dependencies, makes script self-contained.
**Example:**
```python
V2_BASELINE = {
    "test_mae_per_timestep": [0.102, 0.109, 0.112, 0.114],
    "test_rmse_per_timestep": [0.145, 0.153, 0.156, 0.157],
    "test_correlation_per_timestep": [0.565, 0.508, 0.483, 0.467],
    "persistence_skill_per_timestep": [2.9, 4.7, 5.2, 5.1],
    "test_csi": 0.051,
    "test_hss": 0.092,
    "temporal_variation_ratio": 0.060,
    # Additional v2.0 diagnostics for completeness
    "pred_variation": 0.006,
    "target_variation": 0.105,
}
```

### Pattern 2: Verdict Classification
**What:** Automated verdict based on key metric comparison.
**When to use:** Summary header of COMPARISON.md.
**Example:**
```python
def compute_verdict(v3_results, v2_baseline):
    """PASS: temporal var ratio AND CSI both improved.
    MIXED: some key metrics improved, some regressed.
    REGRESSION: key temporal metrics worse than v2.0."""
    var_ratio_improved = v3_results['temporal_variation_ratio'] > v2_baseline['temporal_variation_ratio']
    csi_improved = v3_results['test_csi'] > v2_baseline['test_csi']

    if var_ratio_improved and csi_improved:
        return "PASS"
    elif var_ratio_improved or csi_improved:
        return "MIXED"
    else:
        return "REGRESSION"
```

### Pattern 3: Smoke Test via Config Override
**What:** Run a short training for integration validation before the full run.
**When to use:** Before committing to 50-epoch run.
**Example:**
```python
# Modify config.yaml temporarily or use a separate smoke config
# Key checks after smoke test:
# 1. No NaN/Inf losses in any epoch
# 2. Loss decreasing (not stuck or diverging)
# 3. Temporal variation ratio > 0 (model producing some variation)
# 4. No Python errors or crashes
# 5. Attention entropy not collapsed to near-zero
```

### Anti-Patterns to Avoid
- **Re-running v2.0:** User explicitly said hardcode baselines. Do not create a v2.0 config or retrain a v2.0 model.
- **Overwriting v2.0 outputs:** The `outputs copy/` directory contains the v2.0 checkpoint and results. Never modify this directory.
- **Fixed success thresholds:** User explicitly said "any improvement" counts. Do not impose arbitrary minimum improvement percentages.
- **Modifying existing model/training code:** Phase 11 is validation only. If bugs are found, fix them, but do not add new features.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Training pipeline | Custom training loop | `python main.py` (existing) | Complete pipeline already exists with all v3.0 features |
| Metric computation | Manual CSI/HSS/SSIM | Existing `utils/metrics.py` | All metrics already computed during test evaluation |
| Training visualization | Custom training curves | Existing `plot_training_history()` | 9-subplot comprehensive visualization already exists |
| Test evaluation | Custom test loop | Existing `validate()` in trainer.py | Full metric dict with 16+ metrics already returned |
| Config validation | Manual config checks | Existing `validate_config()` | Config validation already catches mismatches |

**Key insight:** The entire training/evaluation pipeline is already complete. Phase 11's only new code is the comparison report generator.

## Common Pitfalls

### Pitfall 1: MPS Memory Issues During Full Training
**What goes wrong:** 50-epoch training with SA-ConvLSTM (4x params vs v2.0) may OOM on MPS with batch_size=1.
**Why it happens:** SA-ConvLSTM has Self-Attention Memory (SAM) with additional projection layers. Model is ~400K+ params at [32,64,128] channels vs ~100K at [16,32,64].
**How to avoid:** Batch size is already 1. `use_checkpointing: false` could be toggled to `true` if OOM occurs. `clear_device_cache()` already runs between epochs in the trainer.
**Warning signs:** MPS process killed, training suddenly stops without error message.

### Pitfall 2: NaN Loss from Temporal Loss Components
**What goes wrong:** The composite loss has 6 components including temporal difference and temporal variation penalty. Interaction between temporal_var_penalty (negative loss) and other components can cause NaN if values diverge.
**Why it happens:** Temporal variation penalty subtracts from total loss. If pred_variation grows large due to SA-ConvLSTM + temporal attention, the negative term can destabilize training.
**How to avoid:** The trainer already has NaN detection with `max_consecutive_nan: 10` and emergency checkpointing. If NaN occurs, reduce `temporal_var_lambda` from 0.1 to 0.05 or reduce `temporal_diff_weight` from 1.0 to 0.5.
**Warning signs:** Loss suddenly becomes negative, then NaN. Watch for "High gradient norm" warnings in training output.

### Pitfall 3: Attention Entropy Collapse
**What goes wrong:** Temporal attention weights collapse to focus on a single encoder timestep (entropy near 0).
**Why it happens:** With small dataset (568 samples), temporal attention can overfit to attending to the last frame only, which degenerates to persistence behavior.
**How to avoid:** The trainer already logs `temporal_attn_entropy` during validation. Max entropy for T_in=10 is ln(10) = 2.303. If entropy drops below 0.5 during training, attention is collapsing. Consider increasing dropout or reducing channels as mitigation.
**Warning signs:** Temporal variation ratio near 0, attention entropy consistently declining.

### Pitfall 4: Test Results JSON Schema Mismatch
**What goes wrong:** The comparison script expects certain keys in test_results.json but the v3.0 output format differs from v2.0 diagnostic format.
**Why it happens:** v2.0's `diagnostic_results.json` uses keys like `mae_per_t`, `corr_per_t`, `var_ratio`, while v3.0's `test_results.json` uses `test_mae_per_timestep`, `test_correlation_per_timestep`, `temporal_variation_ratio`.
**How to avoid:** The comparison script must handle the v3.0 key naming from `main.py` lines 253-276. V2.0 baselines are hardcoded, so no key mapping needed for v2.0 side.
**Warning signs:** KeyError when loading test_results.json.

### Pitfall 5: Misleading Persistence Skill Numbers
**What goes wrong:** Persistence skill can appear low even when the model is good because the dataset has low temporal variation to begin with.
**Why it happens:** Skill = (1 - model_MAE/persistence_MAE) * 100. When persistence is already quite good (as in this solar flux data), even a much better model shows modest skill percentages.
**How to avoid:** Report persistence skill alongside absolute MAE values. A 3-5% persistence skill in this domain is genuinely significant. The comparison report should contextualize the numbers, not judge them by standards from other domains.
**Warning signs:** User interprets 5% skill as "poor" when it's actually meaningful improvement.

### Pitfall 6: Config Already Modified from Prior Phases
**What goes wrong:** The `config.yaml` may have been modified during Phase 10 development with non-default values.
**Why it happens:** Phase 10 implementation may have left experimental settings.
**How to avoid:** Verify config.yaml matches the expected v3.0 defaults before running. The current config.yaml has already been reviewed and looks correct (all v3.0 features enabled with proper values).
**Warning signs:** Unexpected training behavior that doesn't match expectations.

## Code Examples

### Reading v3.0 Test Results
```python
# Source: main.py lines 253-276 (verified from codebase)
import json
from pathlib import Path

def load_v3_results(output_dir="./outputs"):
    with open(Path(output_dir) / "test_results.json") as f:
        return json.load(f)

# Available keys in v3.0 test_results.json:
# test_loss, test_mae_per_timestep, test_rmse_per_timestep,
# test_correlation_per_timestep, test_csi, test_csi_per_timestep,
# test_hss, test_hss_per_timestep, test_ssim, test_ssim_per_timestep,
# persistence_mae_per_timestep, persistence_skill_per_timestep,
# persistence_csi, persistence_hss, peak_flux_error_per_timestep,
# temporal_variation_ratio
```

### Comparison Bar Chart Pattern
```python
# Source: matplotlib standard patterns
import matplotlib.pyplot as plt
import numpy as np

def plot_per_timestep_comparison(v2_values, v3_values, metric_name, save_path):
    """Side-by-side bar chart for per-timestep metrics."""
    timesteps = [f"t+{i+1}" for i in range(len(v2_values))]
    x = np.arange(len(timesteps))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    bars1 = ax.bar(x - width/2, v2_values, width, label='v2.0', color='#4C72B0')
    bars2 = ax.bar(x + width/2, v3_values, width, label='v3.0', color='#DD8452')

    ax.set_xlabel('Timestep')
    ax.set_ylabel(metric_name)
    ax.set_title(f'{metric_name} Comparison: v2.0 vs v3.0')
    ax.set_xticks(x)
    ax.set_xticklabels(timesteps)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=8)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
```

### Markdown Report Generation Pattern
```python
# Source: standard Python string formatting
def generate_comparison_table(v2, v3, metric_name, per_timestep=True):
    """Generate a markdown table comparing v2.0 and v3.0 metrics."""
    if per_timestep:
        header = f"| Timestep | v2.0 | v3.0 | Delta | % Change |\n"
        header += "|----------|------|------|-------|----------|\n"
        rows = []
        for i, (v2_val, v3_val) in enumerate(zip(v2, v3)):
            delta = v3_val - v2_val
            pct = (delta / abs(v2_val) * 100) if v2_val != 0 else 0
            direction = "+" if delta > 0 else ""
            rows.append(f"| t+{i+1} | {v2_val:.4f} | {v3_val:.4f} | "
                       f"{direction}{delta:.4f} | {direction}{pct:.1f}% |")
        return header + "\n".join(rows)
    else:
        delta = v3 - v2
        pct = (delta / abs(v2) * 100) if v2 != 0 else 0
        direction = "+" if delta > 0 else ""
        return f"| {metric_name} | {v2:.4f} | {v3:.4f} | {direction}{delta:.4f} | {direction}{pct:.1f}% |"
```

### Smoke Test Validation Checks
```python
# Source: derived from training/trainer.py behavior
def validate_smoke_test(history, n_epochs=3):
    """Check smoke test passed basic integration criteria."""
    checks = {
        "no_nan_loss": not any(np.isnan(l) for l in history['train_loss']),
        "loss_decreasing": history['val_loss'][-1] < history['val_loss'][0],
        "var_ratio_positive": history['temporal_variation_ratio'][-1] > 0,
        "epochs_completed": len(history['train_loss']) >= n_epochs,
    }
    # Optional: attention entropy check
    if history.get('temporal_attn_entropy'):
        checks["entropy_not_collapsed"] = history['temporal_attn_entropy'][-1] > 0.5

    all_passed = all(checks.values())
    return all_passed, checks
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| v2.0 simple ConvLSTM [16,32,64] | SA-ConvLSTM [32,64,128] with attention | Phase 10 | ~4x params, temporal attention, spatial attention gates |
| L1 + SSIM loss only | 6-component composite loss | Phase 8 | Temporal diff, temporal var penalty, asymmetric extreme loss |
| Fixed LR, no augmentation | Cosine LR, balanced augmentation, flare oversampling | Phase 9 | Better training policy for 568-sample dataset |
| 4 basic metrics | 16+ metrics including CSI, HSS, SSIM per timestep | Phase 7 | Comprehensive evaluation framework |
| var_ratio = 0.060 (near-persistence) | Expected significant improvement | Phase 11 | Primary success signal |

**Key v2.0 weakness (the problem we are solving):** v2.0 produces only 6% of target's frame-to-frame variation (pred_variation: 0.006 vs target: 0.105). The model essentially learned persistence with minimal temporal dynamics. All v3.0 features are designed to fix this.

## Open Questions

1. **Expected training time for 50 epochs on MPS?**
   - What we know: v2.0 trained in ~1 hour on RTX 3050 (25 epochs, smaller model). v3.0 is ~4x params with SA-ConvLSTM + attention overhead, 50 epochs.
   - What's unclear: MPS performance vs CUDA for this workload. Could be 2-8 hours.
   - Recommendation: Start the training run and monitor. Training can be interrupted gracefully (SIGINT handler saves emergency checkpoint).

2. **Will temporal variation ratio improve dramatically or modestly?**
   - What we know: v2.0 ratio is 0.060 (6% of target variation). The temporal diff loss, temporal var penalty, and SA-ConvLSTM are all designed to push this higher.
   - What's unclear: How much improvement given only 568 samples and MPS-based training.
   - Recommendation: Any improvement above 0.060 is a positive signal. Document honestly regardless of magnitude.

3. **MAE regression tradeoff?**
   - What we know: User explicitly said MAE/RMSE regression is acceptable if temporal metrics improve. Temporal weighting [1.0, 1.5, 2.0, 2.5] and temporal diff loss may sacrifice per-pixel accuracy for temporal fidelity.
   - What's unclear: Magnitude of tradeoff.
   - Recommendation: Report both dimensions neutrally with explicit tradeoff discussion in COMPARISON.md.

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest (no config file -- uses conftest.py markers) |
| Config file | tests/conftest.py (custom markers: mps, cuda) |
| Quick run command | `python -m pytest tests/ -x --timeout=60` |
| Full suite command | `python -m pytest tests/ -v` |

### Phase Requirements -> Test Map
Phase 11 has no formal requirement IDs (it validates all prior requirements end-to-end). The "tests" for this phase are the actual training runs and comparison report verification:

| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| SC-01 | 50-epoch training completes without errors/NaN | integration (manual) | `python main.py` | N/A -- uses existing main.py |
| SC-02 | Temporal variation ratio > v2.0 baseline (0.060) | integration (manual) | Check `outputs/test_results.json` | N/A -- checked in comparison script |
| SC-03 | CSI > v2.0 baseline (0.051) | integration (manual) | Check `outputs/test_results.json` | N/A -- checked in comparison script |
| SC-04 | Comparison report generated with all metrics | smoke | `python generate_comparison.py && test -f COMPARISON.md` | N/A -- Wave 0 |
| EXISTING | All 11 existing test files pass | unit | `python -m pytest tests/ -x` | Yes |

### Sampling Rate
- **Pre-training:** Run existing test suite to confirm no regressions: `python -m pytest tests/ -x`
- **Post-smoke-test:** Manual inspection of 3-epoch output (loss decreasing, no NaN, entropy stable)
- **Post-full-run:** Run comparison script, inspect COMPARISON.md for completeness
- **Phase gate:** COMPARISON.md exists with verdict, all charts generated, existing tests still pass

### Wave 0 Gaps
- [ ] `generate_comparison.py` -- standalone comparison report generator (new file)
- [ ] Verify existing test suite passes before any training: `python -m pytest tests/ -x`

*(No framework install needed -- pytest and all dependencies already available)*

## Sources

### Primary (HIGH confidence)
- **Codebase inspection** -- main.py, config.yaml, training/trainer.py, utils/metrics.py, utils/visualization.py, models/predictor.py, training/losses.py, models/attention.py (all read and analyzed)
- **v2.0 diagnostic results** -- `outputs copy/diagnostic_results.json` (verified baseline values match CONTEXT.md)
- **Test infrastructure** -- tests/conftest.py and 11 test files (verified existing coverage)

### Secondary (MEDIUM confidence)
- Training time estimates based on parameter count scaling (4x params, 2x epochs, MPS vs CUDA unknown)

### Tertiary (LOW confidence)
- None

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - no new dependencies, all libraries already in use
- Architecture: HIGH - no new code architecture, just running existing pipeline + comparison script
- Pitfalls: HIGH - derived from direct codebase analysis of all integration points
- Training outcome: LOW - actual results unknown until training completes (inherent to validation)

**Research date:** 2026-03-09
**Valid until:** No expiration -- this is project-specific validation research, not library version dependent
