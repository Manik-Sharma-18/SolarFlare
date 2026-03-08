# Phase 10: Architecture Scaling - Context

**Gathered:** 2026-03-08
**Status:** Ready for planning

<domain>
## Phase Boundary

Increase model representational capacity through SA-ConvLSTM cells with channel attention, temporal attention over encoder states, spatial attention gates on skip connections, wider channels [32,64,128], kernel size 5, learned delta head scaling, and MC Dropout 0.15. The model must train and evaluate without errors on CUDA, MPS, and CPU. Requirements: ARCH-01 through ARCH-07.

</domain>

<decisions>
## Implementation Decisions

### SA-ConvLSTM cell design (ARCH-01)
- Channel attention variant (NOT spatial self-attention) — avoids 24K x 24K attention matrices at latent resolution
- Self-Attention Memory (SAM) dimension = hidden_dim // 2 (e.g., c3=128 uses SAM dim=64)
- All 6 ConvLSTM modules replaced with SA-ConvLSTM: encoder_conv1, encoder_conv2, encoder_conv3, decoder_conv2, decoder_conv3, refine_conv
- ~25K total parameters added for SAM across all layers
- Manual attention implementation (bmm + softmax) for MPS compatibility — no F.scaled_dot_product_attention

### Temporal attention (ARCH-07)
- Separate standalone TemporalAttention module — NOT built into SA-ConvLSTM cells
- Encoder stores all T=10 hidden states from encoder_conv3 (latent layer only, c3=128)
- Decoder queries stored states at each decode step via scaled dot-product attention with global average pooling (temporal-only attention, ~8K params)
- Context injection: additive to decoder_conv3 output (dec_h3 = dec_h3 + context) — graceful degradation property, no architectural disruption
- Q/K/V projections via 1x1 Conv2d

### Spatial attention gate on skip connections (ARCH-03)
- Attention U-Net pattern: both encoder skip features AND decoder upsampled features jointly produce the gate
- Architecture: Conv2d(encoder) + Conv2d(decoder) -> ReLU -> Conv2d -> Sigmoid -> multiply with encoder skip
- The gate adapts per decoder timestep (decoder features change at each step)
- ~2,000-3,000 additional parameters

### Channel widening (ARCH-04) and kernel size (ARCH-05)
- Full [32, 64, 128] channels — go straight to target spec
- Kernel size 5 as specified
- Both remain configurable in config.yaml

### Delta head scaling (ARCH-02)
- Single scalar nn.Parameter initialized to 100.0 (inverse of typical ~0.01 delta magnitude)
- Applied as: delta = raw_delta * delta_scale
- Excluded from weight decay (decay pulls toward zero = "predict nothing")
- 1 parameter added, zero overfitting risk

### MC Dropout (ARCH-06)
- Dropout rate 0.15 as specified in requirements
- Existing Dropout2d infrastructure in predictor.py already handles this — just update config default

### Overfitting strategy
- Full [32,64,128] channels with regularization: MC Dropout 0.15, balanced augmentation (3x dataset ~1,704 samples), weight decay 1e-5, flare oversampling 3x
- Fallback priority order if overfitting detected:
  1. Reduce channels [32,64,128] -> [24,48,96] (biggest parameter lever)
  2. Increase dropout 0.15 -> 0.25
  3. Reduce kernel 5 -> 3
  4. Increase weight decay 1e-5 -> 1e-4
- Log attention entropy per epoch as overfitting diagnostic:
  - Temporal attention entropy (max = ln(10) = 2.3) — low entropy = fixated on one timestep (red flag)
  - Channel attention entropy (max = ln(C)) — low entropy = collapsed to few channels (red flag)

### Claude's Discretion
- SA-ConvLSTM internal gate design for combining ConvLSTM output with SAM memory
- TemporalAttention projection dimension (suggested: channels or channels // 2)
- Attention U-Net gate intermediate feature dimension (F_int)
- How attention entropy metrics are integrated into training loop logging
- Parameter group setup for excluding delta_scale from weight decay
- Config key naming and structure for new architecture parameters

</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets
- `models/convlstm.py:ConvLSTMCell` — base cell to wrap inside SA-ConvLSTMCell (composition, not replacement)
- `models/convlstm.py:ConvLSTM` — multi-layer wrapper; needs SA variant or parameterization to use SA cells
- `models/predictor.py:SolarFluxPredictor` — encoder-decoder with skip connection at line 263, output head at line 269
- Existing dropout infrastructure: `dropout_enc1`, `dropout_enc2`, `dropout_dec` (Dropout2d) already wired — just change config default to 0.15
- `models/uncertainty.py` — MC Dropout inference already implemented, works with any dropout_rate > 0

### Established Patterns
- ConvLSTM returns `(outputs, hidden_state)` where hidden_state is `List[Tuple[h, c]]` — SA-ConvLSTM must also return memory M_t
- Config-driven model construction via `main.py` reading config.yaml model section
- `_init_forget_bias()` pattern for custom weight initialization — follow for delta_scale init
- Encoder stores `h1_skip = h1_states[0][0]` for skip connection — extend to store all h3 states for temporal attention

### Integration Points
- `SolarFluxPredictor.__init__()` — instantiate SA-ConvLSTM cells, TemporalAttention module, AttentionGate module, delta_scale parameter
- `SolarFluxPredictor._encoder_forward()` — must collect and return all encoder_conv3 hidden states (currently only returns final)
- `SolarFluxPredictor.forward()` decoder loop — add temporal attention query + additive injection, add attention gate before skip concat
- `config.yaml` model section — add SA-ConvLSTM toggle, attention params, update channels/kernel/dropout defaults
- `training/trainer.py` — log attention entropy metrics during validation
- Optimizer setup in `main.py` — parameter groups to exclude delta_scale from weight decay

</code_context>

<specifics>
## Specific Ideas

- Channel attention chosen specifically because spatial self-attention produces infeasible 24K x 24K matrices at latent resolution (110x221) and risks spatial memorization on 568 samples
- Attention U-Net gate chosen because the problem structure (small flare regions in large solar maps) closely matches medical imaging segmentation (small organs in large scans) — the pattern's original use case (Oktay et al., 2018, 4800+ citations)
- Additive context injection chosen for graceful degradation: if temporal attention weights are near-zero, the model falls back to baseline behavior — temporal attention can only help, never hurt
- Delta scale init=100.0 based on research analysis that typical deltas are ~0.01 in normalized space; network outputs at O(1) scale, learned scale maps to actual magnitude

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 10-architecture-scaling*
*Context gathered: 2026-03-08*
