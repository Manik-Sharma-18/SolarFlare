# Deep Research: Temporal Architecture Approaches for Spatiotemporal Solar Flux Prediction

**Domain:** Spatiotemporal solar flux forecasting (ConvLSTM encoder-decoder)
**Researched:** 2026-03-07
**Overall Confidence:** MEDIUM-HIGH (verified against published architectures and PyTorch documentation)

---

## Executive Summary

The current SolarFlare model produces near-persistence predictions (6% temporal variation ratio, 3-9% skill over persistence). This is primarily an architectural and loss function problem, not a hyperparameter problem. The model's ConvLSTM encoder compresses 10 timesteps into a single hidden state, losing temporal resolution. The residual prediction scheme (`pred = input + delta`) with L1 loss creates a strong gradient attractor toward zero deltas.

This research evaluates seven architectural approaches ranked by expected impact, implementation complexity, overfitting risk with 568 samples, and MPS compatibility. The recommended strategy is a staged approach:

1. **Immediate (v3.0):** Temporal attention over encoder hidden states + delta head normalization -- moderate complexity, high impact, low overfitting risk
2. **If needed (v3.1):** Self-Attention Memory (SAM) ConvLSTM replacement -- medium complexity, proven architecture, replaces ConvLSTM cells
3. **Future (v4.0):** Multi-quantity input with magnetograms -- requires data pipeline work but highest long-term ceiling
4. **Avoid for now:** Full transformer/ViT approaches -- too parameter-hungry for 568 samples

---

## 1. Temporal Attention Over Encoder Hidden States

### The Problem Being Solved

The current encoder processes 10 frames through ConvLSTM and produces a single final hidden state. All temporal information is compressed into this bottleneck. The decoder initializes from this state and has no mechanism to revisit specific input timesteps. For solar flux prediction, the last 2-3 frames before a flare contain the most predictive signal, but the encoder treats all 10 frames with equal recency bias (ConvLSTM naturally favors recent inputs, but in a crude exponential-decay fashion).

### Approach: Scaled Dot-Product Attention (Recommended)

**Use scaled dot-product attention, not Bahdanau or Luong.**

Rationale for this choice:

| Mechanism | Parameters | Computation | Suitability for Spatial Data |
|-----------|-----------|-------------|------------------------------|
| Bahdanau (additive) | O(d^2) from MLP | Separate alignment network | Adds unnecessary parameters; MLP learns alignment independently |
| Luong (multiplicative) | O(d^2) from W_a matrix | Bilinear scoring | Simpler but still adds a full weight matrix |
| Scaled dot-product | O(0) attention params | Q*K^T / sqrt(d) | Most parameter-efficient; relies on learned representations |

With only 568 samples, every parameter matters. Scaled dot-product attention adds zero new attention parameters -- it uses the existing ConvLSTM hidden states as queries/keys/values. The 1x1 convolutions needed to project spatial hidden states into Q/K/V add minimal parameters (3 * C_in * C_proj parameters per attention head).

**Confidence: HIGH** -- scaled dot-product attention is the standard in modern architectures (transformers, SA-ConvLSTM). Bahdanau/Luong are legacy approaches from seq2seq NMT that added complexity because the representations being aligned were from different modalities (source/target languages). In our case, encoder and decoder operate on the same data domain.

### Architecture Design

```
Encoder outputs: H = [h_1, h_2, ..., h_10]  # Each h_t: (B, C, H_lat, W_lat)

At each decoder step t_dec:
  Q = W_q(decoder_hidden)          # (B, C_proj, H_lat, W_lat)
  K = W_k(encoder_outputs)         # (B, T_in, C_proj, H_lat, W_lat)
  V = W_v(encoder_outputs)         # (B, T_in, C_proj, H_lat, W_lat)

  # Flatten spatial dims, compute attention per spatial position
  # or pool spatial dims and compute temporal-only attention

  # Option A: Temporal-only attention (recommended for small dataset)
  Q_pool = global_avg_pool(Q)      # (B, C_proj)
  K_pool = global_avg_pool(K)      # (B, T_in, C_proj)
  attn_weights = softmax(Q_pool @ K_pool^T / sqrt(C_proj))  # (B, T_in)
  context = sum(attn_weights * encoder_outputs)  # (B, C, H_lat, W_lat)
```

**Option A (Temporal-only, recommended):** Pool spatial dimensions before computing attention. Produces a single attention weight per input timestep. Much fewer operations, no risk of spatial overfitting. The intuition: "which input timestep is most relevant?" not "which spatial position in which timestep?"

**Option B (Full spatiotemporal):** Compute attention at every spatial position. Produces a (T_in, H, W) attention map. Extremely expensive with 110x221 spatial dims at latent resolution. Risk of memorizing spatial positions in 568 samples.

### Implementation Complexity: LOW-MEDIUM

```python
class TemporalAttention(nn.Module):
    def __init__(self, channels, proj_dim=None):
        super().__init__()
        proj_dim = proj_dim or channels
        self.q_proj = nn.Conv2d(channels, proj_dim, 1)
        self.k_proj = nn.Conv2d(channels, proj_dim, 1)
        self.v_proj = nn.Conv2d(channels, proj_dim, 1)
        self.out_proj = nn.Conv2d(proj_dim, channels, 1)
        self.scale = proj_dim ** -0.5

    def forward(self, decoder_state, encoder_outputs):
        # decoder_state: (B, C, H, W)
        # encoder_outputs: list of T tensors, each (B, C, H, W)
        B, C, H, W = decoder_state.shape
        T = len(encoder_outputs)

        q = self.q_proj(decoder_state).mean(dim=(-2, -1))  # (B, proj_dim)
        keys = torch.stack([self.k_proj(e).mean(dim=(-2, -1)) for e in encoder_outputs], dim=1)  # (B, T, proj_dim)
        values = torch.stack(encoder_outputs, dim=1)  # (B, T, C, H, W)

        attn = torch.softmax(torch.bmm(q.unsqueeze(1), keys.transpose(1, 2)) * self.scale, dim=-1)  # (B, 1, T)
        context = torch.bmm(attn, values.reshape(B, T, -1)).reshape(B, C, H, W)
        return self.out_proj(context), attn.squeeze(1)
```

Parameters added: ~4 * C * proj_dim (with C=64, proj_dim=32: ~8K parameters). Negligible.

### MPS Compatibility: HIGH

Uses only standard operations: Conv2d, bmm, softmax, mean. No `scaled_dot_product_attention` call needed (manual implementation avoids MPS SDPA issues). All operations are well-supported on MPS.

### Expected Impact: MEDIUM-HIGH

Temporal attention directly addresses the information bottleneck. The decoder can selectively attend to the most informative input timesteps. Published results from RA-ConvLSTM (2025, space weather domain) show attention-augmented ConvLSTM achieving "higher accuracy" for ionospheric forecasting. ResConvLSTM-Att (2024) showed 6.5% RMSE reduction over standard ConvLSTM for forest cover prediction.

However, temporal attention alone will not fix the delta prediction problem -- the model still needs to learn to produce non-trivial deltas. Temporal attention improves *what information is available*; it does not fix *what the model does with that information*.

### Overfitting Risk: LOW

Temporal-only attention adds ~8K parameters. The attention weights are computed from global-average-pooled features, so there is no spatial memorization risk. With 10 timesteps, the attention distribution is over a very small discrete space.

### Verdict: IMPLEMENT in v3.0

This is the highest-impact, lowest-risk architectural change. Implement alongside the loss function improvements (temporal difference loss, delta normalization) already planned.

---

## 2. Spatiotemporal Transformers (TimeSformer/ViViT/VideoMAE)

### Architecture Overview

Video transformers tokenize spatiotemporal volumes into patch tokens and process them with self-attention:

- **TimeSformer** (Meta, 2021): Divided space-time attention. Each frame is split into non-overlapping patches (e.g., 16x16). Temporal attention across same-position patches, then spatial attention within each frame. Avoids O(T*N^2) cost.
- **ViViT** (Google, 2021): Four variants from full spatiotemporal attention to factorized encoder. Tubelet embedding extracts tokens from 3D volumes.
- **VideoMAE** (2022): Masked autoencoder for video pre-training. Masks 90%+ of tokens during pre-training, enabling efficient self-supervised learning.

### Feasibility Assessment for This Project

**Verdict: NOT FEASIBLE for v3.0. Consider for v4.0+ only with significantly more data.**

#### Patch Tokenization at 220x442

At 16x16 patches: 14 x 28 = 392 spatial tokens per frame
With 10 input frames: 3,920 tokens total
With 4 output frames: 5,488 tokens total

This is within the tractable range for attention (well under typical ViT sequence lengths). So computational feasibility is not the blocker.

#### The Data Bottleneck

This is the blocker. From research findings:

> "ViViT requires an extremely large dataset to achieve good performance... such a scale of dataset is often unavailable for videos." -- ViViT paper

Transformers lack the spatial inductive biases that ConvLSTM provides:
- No weight sharing across spatial positions (unlike convolutions)
- No built-in locality bias
- No recurrence-based temporal smoothness

With 568 samples, a transformer would need to learn both spatial feature extraction AND temporal dynamics from scratch. ConvLSTM provides spatial feature extraction "for free" via convolutional structure.

**Parameter counts tell the story:**

| Model | Parameters | Samples Needed |
|-------|-----------|----------------|
| Current ConvLSTM [16,32,64] | ~150K | 500-2K |
| ConvLSTM [32,64,128] | ~600K | 1K-5K |
| ViViT-Tiny (custom) | ~5M | 10K-50K |
| TimeSformer-Base | ~121M | 100K+ |

Even a heavily stripped-down transformer (4 layers, 4 heads, 128 dim) would be ~2-5M parameters -- 10-30x the current model. With 568 samples and batch size 1, this will massively overfit.

#### VideoMAE Pre-training Path

VideoMAE's masked autoencoder approach could theoretically work:
1. Pre-train on unlabeled solar data (reconstruct masked patches)
2. Fine-tune for prediction

But this requires abundant unlabeled data of the SAME type (winding flux maps). If such data existed, the user would have more training samples already.

### MPS Compatibility: MEDIUM

Standard SDPA works on MPS but has documented crash issues with grouped query attention and memory limits for long sequences. The 3,920-token sequence length is within safe bounds, but MPS lacks FlashAttention optimization, making attention 3-5x slower than CUDA.

### Verdict: DO NOT IMPLEMENT in v3.0

The data regime is fundamentally wrong for transformers. The ConvLSTM architecture provides essential inductive biases (spatial locality, temporal recurrence) that transformers must learn from data. With 568 samples, ConvLSTM wins.

**Revisit conditions:**
- Dataset grows to 5,000+ samples (25+ data cubes)
- Pre-training on related solar data becomes available
- Task shifts to one where global spatial attention matters more than local patterns

---

## 3. ConvLSTM + Transformer Hybrid

### Architecture Concept

Keep ConvLSTM for spatial encoding (preserving convolutional inductive bias) but replace or augment the temporal reasoning with transformer-style attention:

```
Input frames --> ConvLSTM encoder --> spatial feature maps per timestep
                                          |
                                     [Pool to tokens]
                                          |
                                  Transformer temporal encoder
                                  (self-attention across timesteps)
                                          |
                                     [Reshape back]
                                          |
                                  ConvLSTM decoder --> predictions
```

### Why This Is Attractive

The hypothesis: ConvLSTM is good at spatial feature extraction but weak at temporal reasoning. The fixed-size hidden state creates an information bottleneck. A transformer over the T=10 encoder outputs can model arbitrary temporal dependencies without compression.

### Specific Design: Temporal Transformer Block

```python
class TemporalTransformerBlock(nn.Module):
    def __init__(self, spatial_channels, d_model=128, nhead=4, num_layers=2):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)  # Pool spatial to avoid O(H*W*T^2)
        self.proj_in = nn.Linear(spatial_channels, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model*2,
            dropout=0.1, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.proj_out = nn.Linear(d_model, spatial_channels)
        self.pos_encoding = nn.Parameter(torch.randn(1, 10, d_model) * 0.02)

    def forward(self, encoder_outputs):
        # encoder_outputs: (B, C, T, H, W)
        B, C, T, H, W = encoder_outputs.shape
        # Pool spatial: (B, C, T)
        x = self.pool(encoder_outputs.permute(0, 2, 1, 3, 4).reshape(B*T, C, H, W))
        x = x.reshape(B, T, C)
        # Project and add positional encoding
        x = self.proj_in(x) + self.pos_encoding[:, :T]
        # Transformer
        x = self.transformer(x)
        # Project back and modulate encoder outputs
        weights = torch.sigmoid(self.proj_out(x))  # (B, T, C)
        # Apply as temporal gating
        return encoder_outputs * weights.permute(0, 2, 1).unsqueeze(-1).unsqueeze(-1)
```

### Parameter Cost

With d_model=128, nhead=4, 2 layers:
- Linear projections: 2 * C * 128 (~16K with C=64)
- TransformerEncoder: 2 * (4 * 128^2 + 128 * 256) ~200K
- Positional encoding: 10 * 128 = 1.3K
- Total: ~220K parameters

This is modest -- roughly doubling the current model's parameter count. Acceptable for 568 samples with proper regularization.

### Key Concern: Spatial Information Loss

The pooling step discards spatial information. The transformer reasons about "what is the average state at each timestep?" not "what is happening at position (x,y) at timestep t?". This is actually appropriate for our use case -- the temporal dynamics question is "when does flux change?" not "where at each timestep?"

The spatial detail is preserved in the encoder outputs and skip connections. The transformer only modulates the temporal weighting.

### MPS Compatibility: HIGH

`nn.TransformerEncoder` with batch_first=True works on MPS. The sequence length is T=10 (tiny), so no memory issues. No need for `scaled_dot_product_attention` directly -- the TransformerEncoderLayer handles this internally and falls back to the math kernel on MPS.

### Expected Impact: MEDIUM

Similar to temporal attention (#1) but with more capacity for complex temporal dependencies. The transformer can model interactions like "if timestep 3 and timestep 8 show similar patterns, attend to both" -- something simple attention cannot express.

However, with T=10, the benefit of self-attention over simple temporal attention is limited. Self-attention shines with long sequences (T>50). For T=10, the information capacity of a simple attention mechanism is already sufficient.

### Overfitting Risk: MEDIUM

~220K new parameters is significant relative to the current ~150K model. Dropout in the transformer layers (0.1-0.2) helps. The pooled representation prevents spatial memorization. Monitor train/val gap carefully.

### Verdict: CONSIDER for v3.1 if temporal attention (#1) is insufficient

The hybrid is a natural escalation if simple temporal attention does not achieve the desired improvement. It is not worth the complexity as a first step. If temporal attention + loss function changes achieve >15% variation ratio (up from 6%), the hybrid may not be needed.

---

## 4. Delta Prediction Improvements

### Root Cause Analysis

The current scheme:
```python
delta = self.output_conv(refined[:, :, 0])   # Conv2d output
pred_flux = input_flux + delta                 # Residual prediction
```

With L1 loss, the gradient for the delta output is:
```
d(L1)/d(delta) = sign(pred - target) = sign(delta - (target - input))
```

When the typical delta magnitude is ~0.01 in normalized space, the loss landscape is:
- Very flat around delta=0 (the persistence solution)
- The model gets almost no gradient signal to produce larger deltas
- L1 loss treats a delta of 0.01 and 0.001 with equal gradient magnitude

**This is the single most important problem to solve.** Without fixing delta prediction, temporal attention adds information that the model still cannot use.

### Approach 4A: Learned Delta Scaling (RECOMMENDED)

Add a learnable scale parameter to the output head:

```python
self.delta_scale = nn.Parameter(torch.tensor(1.0))

# In forward:
raw_delta = self.output_conv(refined[:, :, 0])
delta = raw_delta * self.delta_scale
```

Initialize `delta_scale` to the inverse of the typical delta standard deviation. If typical deltas are ~0.01, initialize to 100.0. This means the network outputs at O(1) scale, and the learned scale maps to actual delta magnitude.

**Better variant -- per-channel adaptive:**
```python
self.delta_scale = nn.Parameter(torch.ones(output_channels) * initial_scale)
self.delta_bias = nn.Parameter(torch.zeros(output_channels))
```

**Confidence: HIGH.** This is directly analogous to batch normalization's learnable affine parameters. Neural networks optimize much better when output targets are O(1). This was identified in the IMPROVEMENT_NOTES as item 7.8 and is well-motivated.

Parameters added: 1-2. Overfitting risk: zero.

### Approach 4B: Normalized Delta Targets

Instead of modifying the output head, normalize the target deltas during training:

```python
target_delta = target - input_frame
delta_mean = target_delta.mean()
delta_std = target_delta.std() + 1e-8
normalized_target = (target_delta - delta_mean) / delta_std

# Loss computed on normalized deltas
loss = L1(raw_delta, normalized_target)

# During inference, denormalize
pred = input + (raw_delta * delta_std + delta_mean)
```

This requires computing running statistics of delta distributions. More complex but more principled.

**Concern:** Delta statistics may vary significantly across samples (quiet sun vs. pre-flare). A single global scale may not be appropriate.

### Approach 4C: Separate Magnitude and Direction Heads

Split the output into two heads:
- **Direction head:** `sign_pred = tanh(head_direction(features))` -- predicts whether flux increases or decreases (-1 to +1)
- **Magnitude head:** `mag_pred = softplus(head_magnitude(features))` -- predicts the absolute change magnitude (always positive)
- **Delta:** `delta = sign_pred * mag_pred`

**Rationale:** Decouples the "which way?" question from the "how much?" question. The magnitude head uses softplus to ensure positive output, preventing the model from collapsing to zero magnitude.

**Concern:** Adds complexity and doubles the output head parameters. The multiplication of two predicted quantities can be unstable during training. Not recommended as a first approach.

### Approach 4D: Multi-Step Delta with Intermediate Supervision

Instead of predicting `delta_t = pred_t - input`, predict cumulative deltas from the last input frame:

```
delta_1 = pred_1 - input_last    (1-step change)
delta_2 = pred_2 - input_last    (2-step cumulative change)
delta_3 = pred_3 - input_last    (3-step cumulative change)
delta_4 = pred_4 - input_last    (4-step cumulative change)
```

All deltas are relative to the same reference frame. This avoids error compounding in autoregressive delta prediction (where each step's delta is relative to the previous step's potentially erroneous prediction).

**However:** This changes the autoregressive structure. Currently, each decoder step conditions on its own previous output. Multi-step delta from a fixed reference frame removes this coupling, which may lose temporal coherence.

### Verdict: IMPLEMENT 4A (Learned Delta Scaling) in v3.0

Learned delta scaling is the minimum viable fix. One parameter, zero overfitting risk, directly addresses the core numerical problem. Combine with temporal difference loss (already planned) for maximum effect.

If 4A + temporal difference loss are still insufficient, escalate to 4B (normalized targets).

Avoid 4C and 4D -- they add complexity without clear advantage over 4A + proper loss functions.

---

## 5. Temporal Convolutions (TCN-style)

### Architecture Concept

Replace recurrent temporal processing (ConvLSTM stepping through T=10) with 1D temporal convolutions:

```
Encoder features: (B, C, T=10, H, W)
  |
  Reshape to (B*H*W, C, T) -- treat each spatial position independently
  |
  1D temporal convolution stack:
    - Conv1d(C, C, kernel=3, dilation=1, causal_pad=2)
    - Conv1d(C, C, kernel=3, dilation=2, causal_pad=4)
    - Conv1d(C, C, kernel=3, dilation=4, causal_pad=8)
  |
  Reshape back to (B, C, T, H, W)
```

Dilated causal convolutions cover the entire temporal receptive field (T=10) in 3 layers:
- Layer 1: receptive field = 3
- Layer 2: receptive field = 7 (3 + 2*2)
- Layer 3: receptive field = 15 (7 + 2*4) -- covers all 10 timesteps

### Advantages Over ConvLSTM for Temporal Processing

| Property | ConvLSTM | Temporal Conv |
|----------|----------|---------------|
| Parallelism | Sequential (T steps) | Parallel (all T at once) |
| Gradient flow | Through T recurrent steps | Direct (no vanishing gradient) |
| Temporal receptive field | Theoretically infinite but decays | Fixed, controlled by dilation |
| Training speed | Slow (sequential) | Fast (parallelizable) |
| Memory | O(T) activations for backprop | O(T) but cacheable |

**Key insight from research:** TCNs avoid the gradient vanishing/explosion that plagues RNNs, and capture temporal patterns more effectively in many benchmarks (confirmed in 2024 comparative study: TCN outperformed LSTM on most time series tasks).

### Practical Concern: Spatial Independence Assumption

The reshape `(B, C, T, H, W) -> (B*H*W, C, T)` treats each spatial position independently. This means temporal convolutions cannot capture spatiotemporal interactions (e.g., "flux increasing at position A predicts flux decrease at neighboring position B").

ConvLSTM captures these interactions because it applies 2D spatial convolutions at each temporal step. Temporal convolutions alone lose this.

**Mitigation:** Use TCN as an additional module after ConvLSTM, not a replacement. ConvLSTM handles spatial-temporal coupling; TCN refines temporal patterns on the resulting feature maps.

### For This Project

With T=10 timesteps, the advantage of TCN over ConvLSTM is marginal:
- Gradient vanishing over 10 steps is not a severe problem
- Parallelism gain is small (10 steps is fast sequentially)
- The real bottleneck is information compression, not temporal processing speed

### MPS Compatibility: HIGH

Conv1d is fully supported on MPS with no known issues.

### Expected Impact: LOW-MEDIUM

TCN adds a complementary temporal view but does not address the core problems (information bottleneck from encoder, delta prediction collapse). It is a "nice to have" refinement, not a fundamental improvement.

### Overfitting Risk: LOW

3-layer dilated TCN with C=64 channels adds ~25K parameters. Residual connections prevent degradation.

### Verdict: DEFER to v4.0

TCN is a reasonable addition but not a priority. The simpler temporal attention mechanism (#1) provides similar benefits (temporal access to all encoder states) with better integration into the existing architecture. TCN would be more valuable if the model were scaled up to longer input sequences (T>20) where ConvLSTM gradient flow becomes a real issue.

---

## 6. Self-Attention ConvLSTM (SA-ConvLSTM)

### Architecture Overview

SA-ConvLSTM (Lin et al., AAAI 2020) introduces a Self-Attention Memory (SAM) module that runs parallel to the standard ConvLSTM cell. The key innovation is an auxiliary memory M_t that stores long-range spatiotemporal features via self-attention:

**Standard ConvLSTM cell:**
```
(h_t, c_t) = ConvLSTMCell(x_t, h_{t-1}, c_{t-1})
```

**SA-ConvLSTM cell (with SAM):**
```
(h_t, c_t) = ConvLSTMCell(x_t, h_{t-1}, c_{t-1})

# Self-Attention Memory update
Q_h = W_q(h_t)           # Query from current hidden state
K_h = W_k(h_t)           # Key from current hidden state
V_h = W_v(h_t)           # Value from current hidden state
Z_h = softmax(Q_h * K_h^T / sqrt(d)) * V_h   # Self-attention on hidden state

K_m = W_km(M_{t-1})      # Key from previous memory
V_m = W_vm(M_{t-1})      # Value from previous memory
Z_m = softmax(Q_h * K_m^T / sqrt(d)) * V_m    # Cross-attention with memory

# Gate-controlled memory update
gate = sigmoid(W_g([Z_h, Z_m]))
M_t = gate * tanh(W_m(Z_h)) + (1 - gate) * M_{t-1}

# Output combines ConvLSTM output and memory
h_t_out = combine(h_t, M_t)
```

### Why This Is Relevant

SA-ConvLSTM directly addresses the limitation that standard ConvLSTM loses long-range temporal dependencies. The SAM module maintains a separate memory that captures global spatial patterns across all timesteps. This is exactly the information that gets compressed out in the current bottleneck.

**Published results (AAAI 2020):**
- Moving MNIST: SSIM 0.868 vs 0.838 (ConvLSTM), 0.843 (PredRNN)
- KTH actions: SSIM 0.882 vs 0.863 (ConvLSTM)
- TaxiBJ traffic: MSE 47.95 vs 52.47 (ConvLSTM)
- Fewer parameters and higher efficiency than PredRNN and PredRNN++

**Confidence: HIGH** -- this is a well-established architecture with peer-reviewed results, multiple independent implementations, and follow-up work in space weather (RA-ConvLSTM for ionospheric prediction, 2025).

### Integration with Current Architecture

SA-ConvLSTM is a **drop-in replacement** for ConvLSTMCell. The ConvLSTM module interface stays the same. Integration:

```python
class SAConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, attn_dim=None):
        super().__init__()
        self.convlstm_cell = ConvLSTMCell(input_dim, hidden_dim, kernel_size)
        self.sam = SelfAttentionMemory(hidden_dim, attn_dim or hidden_dim // 2)

    def forward(self, x, h_prev, c_prev, m_prev):
        h, c = self.convlstm_cell(x, h_prev, c_prev)
        h_out, m = self.sam(h, m_prev)
        return h_out, c, m
```

**Key concern:** Self-attention over spatial dimensions. With latent resolution 110x221 = 24,310 spatial positions, the attention matrix would be 24,310 x 24,310 -- ~2.4GB per layer. This is **infeasible** at full resolution.

**Solution:** Use attention at the pooled/downsampled level only, or restrict attention to local windows (like Swin Transformer). The latent ConvLSTM (at 55x111 after encoder downsampling) is more tractable but still large.

**Alternative: Channel attention instead of spatial attention.** Compute attention over channels (C=64) rather than spatial positions. This is O(C^2) = O(4096) -- trivially cheap. Channel attention can capture "which feature maps are most informative at this timestep?" without the spatial cost.

### MPS Compatibility: MEDIUM-HIGH

If implemented with manual attention (bmm + softmax), fully compatible. If using PyTorch's SDPA, need to be cautious about tensor sizes on MPS. The channel-attention variant is completely safe.

### Expected Impact: HIGH

SA-ConvLSTM has been validated in spatiotemporal prediction across multiple domains. The combination of standard ConvLSTM temporal memory + self-attention long-range memory directly addresses the temporal information loss problem. Proven 5-10% improvement over standard ConvLSTM in published benchmarks.

### Overfitting Risk: MEDIUM

The SAM module adds ~4 * C * attn_dim parameters per ConvLSTM layer. With C=64, attn_dim=32: ~8K per layer, ~24K total for 3 layers. Combined with the 1x1 convolutions for Q/K/V projections: ~50K additional parameters. This is manageable with dropout.

The spatial self-attention variant is higher risk (dense attention learns spatial memorization patterns). Channel attention is much safer.

### Verdict: IMPLEMENT in v3.0 or v3.1

SA-ConvLSTM (channel attention variant) is the strongest architectural upgrade for temporal dynamics. It is a principled improvement over the simple temporal attention (#1), with published evidence of effectiveness. The channel attention variant avoids the spatial explosion problem.

**Recommended staging:**
1. v3.0: Implement temporal attention (#1) + delta scaling (#4A) -- simple, fast to validate
2. v3.0 or v3.1: If temporal attention is insufficient, upgrade ConvLSTM cells to SA-ConvLSTM with channel attention

---

## 7. Transfer Learning from Magnetogram Data

### Physical Relationship

Winding flux (the current prediction target) and magnetograms measure related but distinct physical quantities:

- **Magnetograms:** Measure the line-of-sight (or vector) magnetic field strength at the photosphere. Available from SDO/HMI at high cadence (~45 sec) and resolution (4096x4096 pixels, ~0.5 arcsec).
- **Winding flux:** Derived quantity measuring the topological winding of magnetic field lines. Computed from magnetogram data via spatial derivatives and cross-products. Captures magnetic complexity and helicity injection.

The relationship: winding flux = f(magnetogram, velocity_field). The winding is computed from the spatial structure of the magnetic field and its temporal evolution. A model that understands magnetogram evolution implicitly understands the driver of winding flux changes.

**Key references (2025):**
- Physics-informed deep learning for 12-hour prediction of solar vector magnetic field evolution (SSIM=0.85 for radial component)
- MAG2MAG (Ramunno et al., 2024): DDPM-based magnetogram forecasting, outperforms persistence at 24hr horizon
- SimSiam self-supervised pre-training on SHARP data for learning rotation/translation-invariant representations

### Transfer Learning Architecture

```
Phase 1: Pre-train encoder on magnetogram sequences
  Magnetogram(B,1,T,H,W) --> Encoder --> Latent --> Decoder --> Magnetogram(B,1,T_out,H,W)
  Loss: L1 + SSIM on reconstructed magnetograms

Phase 2: Fine-tune for winding flux
  Winding(B,1,T,H,W) --> [Pre-trained Encoder] --> Latent --> [New Decoder] --> Winding(B,1,T_out,H,W)
  Freeze encoder initially, then unfreeze with low LR
```

### Feasibility Assessment

**Data availability:** SDO/HMI magnetograms are freely available (2010-present) at very high cadence. Tens of thousands of full-disk magnetograms exist. This solves the data scarcity problem for pre-training.

**Domain gap:** Magnetograms and winding flux maps have very different value distributions and spatial patterns:
- Magnetograms: bipolar regions, smooth quiet sun, sharp polarity inversion lines
- Winding flux: highly localized peaks at sites of magnetic complexity, mostly zero elsewhere

The encoder would learn "where are the magnetically active regions and how are they evolving?" -- this transfers well. The decoder must learn the different output distribution -- this does not transfer.

**Spatial resolution mismatch:** SDO/HMI is 4096x4096. The current winding flux data is 440x884 (likely extracted from a subregion at lower resolution). The pre-trained encoder would need to handle the same spatial resolution, requiring either:
- Downsampling HMI magnetograms to match (losing detail)
- Training at full HMI resolution and fine-tuning at winding flux resolution (architecture mismatch)

**Temporal cadence mismatch:** HMI is ~45 sec cadence vs. winding flux at ~12 min cadence. Pre-training would need to subsample HMI to match.

### Expected Impact: MEDIUM-HIGH (long-term)

Transfer learning from magnetograms could be transformative IF:
1. The encoder learns generalizable temporal dynamics of magnetic field evolution
2. These dynamics transfer to winding flux prediction
3. The resolution and cadence mismatches are resolved

The physics supports this -- winding flux evolution is driven by magnetogram evolution. A model that understands how magnetic fields change should predict winding flux changes better.

### Implementation Complexity: HIGH

- Requires downloading and processing HMI magnetogram data
- Need to align spatial coordinates and temporal cadence
- Two-phase training pipeline (pre-train + fine-tune)
- Architecture must support encoder freezing/unfreezing
- Need to handle resolution mismatch

### Overfitting Risk: LOW (for pre-training), MEDIUM (for fine-tuning)

Pre-training on abundant magnetogram data eliminates overfitting concern for the encoder. Fine-tuning on 568 winding flux samples still has overfitting risk, mitigated by freezing most encoder weights.

### MPS Compatibility: HIGH

No new operations beyond what the base model uses.

### Verdict: DEFER to v4.0

Transfer learning has the highest potential ceiling but also the highest implementation cost. It requires a data pipeline for HMI magnetograms, spatial/temporal alignment, and a two-phase training system. This is a project-scale effort, not a feature addition.

**Recommended preparation for v3.0:** Ensure the encoder architecture is modular and can be easily swapped/frozen for future transfer learning. This means clean separation between encoder, latent, and decoder modules (already the case in the current architecture).

---

## 8. Multi-Quantity Input (Magnetograms as Additional Channel)

### Architecture Concept

Instead of transfer learning, use magnetograms as an additional input channel alongside winding flux:

```
Input: (B, 2, T, H, W)
  Channel 0: Winding flux (current data)
  Channel 1: Magnetogram (new data, co-registered)

Output: (B, 1, T_out, H, W)
  Channel 0: Predicted winding flux
```

### Physical Motivation

The magnetogram provides direct information about the magnetic field that DRIVES winding flux evolution. Currently, the model must infer the underlying field dynamics from the winding flux alone -- this is like predicting wave behavior from the foam patterns without seeing the water.

With magnetogram input:
- The model sees magnetic flux emergence (new field appearing at the surface)
- The model sees shearing motions (field lines being wound up)
- The model sees cancellation/reconnection (field topology simplifying)

All of these are precursors to winding flux changes and are visible in magnetograms before they manifest in winding flux.

### Data Requirements

**Co-registration:** The magnetogram data must be spatially aligned with the winding flux data. Since winding flux is computed from magnetograms, the spatial grid should be the same or derivable. This is a data preprocessing task, not an architectural one.

**Temporal alignment:** Magnetograms and winding flux must be at the same time cadence. Since winding flux is computed from magnetograms, they come from the same observation times.

**Storage/Loading:** Doubles the data volume (220x442 x 2 channels instead of 1). Manageable with existing lazy loading.

### Implementation Complexity: LOW-MEDIUM

The model already supports `input_channels > 1` (dual-channel mode with extreme indicator). Adding magnetograms:

1. Modify data pipeline to load co-registered magnetogram .npy files
2. Set `input_channels: 2` in config
3. The model architecture handles this natively (Conv2d input layer accepts any channel count)
4. Output remains `output_channels: 1` (winding flux only)

The main work is in data preparation, not model changes.

### Expected Impact: HIGH

This is the highest-impact approach if the data is available. It gives the model direct causal information about what drives winding flux evolution. No architectural cleverness can substitute for having the right input data.

**Analogy:** It is like giving a weather prediction model both temperature and pressure (the driver of temperature changes), instead of just temperature.

### Overfitting Risk: MEDIUM

Doubling input channels roughly doubles the first layer's parameters. With 568 samples, this is a concern but not prohibitive. The magnetogram channel may actually reduce overfitting by providing a more informative signal (the model does not need to memorize spatial patterns to infer the underlying field).

### MPS Compatibility: HIGH

No new operations. Just a wider input tensor.

### Key Question: Is the Data Available?

The user's winding flux data is computed from magnetograms. The magnetogram data therefore exists or can be obtained. The practical question is:
1. Are the magnetograms stored alongside the winding flux data cubes?
2. Are they at the same spatial resolution (440x884)?
3. Which magnetogram component? (Line-of-sight? Radial? All three vector components?)

If the magnetograms are available, this is the single highest-value improvement per unit of implementation effort.

### Verdict: INVESTIGATE DATA AVAILABILITY for v3.0, IMPLEMENT in v3.0 or v3.1

This should be the first question asked: "Do you have co-registered magnetogram data at the same resolution and cadence?" If yes, this jumps to the top of the priority list. If no, defer.

---

## Comparative Assessment

### Impact vs. Complexity Matrix

| Approach | Impact | Complexity | Overfitting Risk | MPS Safe | Recommendation |
|----------|--------|-----------|------------------|----------|----------------|
| 1. Temporal attention | HIGH | LOW | LOW | YES | v3.0 -- implement |
| 4A. Delta scaling | HIGH | MINIMAL | NONE | YES | v3.0 -- implement |
| 6. SA-ConvLSTM (channel attn) | HIGH | MEDIUM | MEDIUM | YES | v3.0/v3.1 -- implement |
| 8. Multi-quantity input | HIGHEST | LOW-MED | MEDIUM | YES | v3.0 -- if data available |
| 3. ConvLSTM+Transformer hybrid | MEDIUM | MEDIUM | MEDIUM | YES | v3.1 -- if needed |
| 5. Temporal convolutions | LOW-MED | LOW | LOW | YES | v4.0 -- defer |
| 2. Full transformer (ViT) | UNKNOWN | HIGH | VERY HIGH | MEDIUM | v4.0+ -- defer |
| 7. Transfer learning | HIGH (ceiling) | VERY HIGH | LOW | YES | v4.0 -- defer |

### Recommended Implementation Order

**Phase 1 (v3.0 -- immediate):**
1. Delta head learned scaling (4A) -- 1 parameter, fixes numerical problem
2. Temporal attention over encoder outputs (#1) -- ~8K parameters, unlocks temporal information
3. Store all encoder hidden states (currently discarded) -- prerequisite for attention

**Phase 2 (v3.0/v3.1 -- based on Phase 1 results):**
4. Multi-quantity magnetogram input (#8) -- if data is available
5. SA-ConvLSTM with channel attention (#6) -- if temporal attention + delta scaling achieve <20% variation ratio

**Phase 3 (v4.0 -- future):**
6. Transfer learning from magnetogram sequences (#7)
7. ConvLSTM+Transformer hybrid (#3) -- if dataset grows significantly
8. Temporal convolutions (#5) -- for longer input sequences

---

## Critical Dependencies Between Approaches

```
Delta scaling (4A) -----> Temporal attention (1) -----> SA-ConvLSTM (6)
     ^                         ^                            ^
     |                         |                            |
     +-- Required first        +-- Requires storing         +-- Replaces (1) or
         (fixes gradient            encoder hidden states       augments it
          signal before
          attention can help)

Multi-quantity input (8) -- Independent of all above, can be done in parallel
Transfer learning (7)   -- Requires (8)'s data pipeline as a prerequisite
```

**The most critical insight:** Delta scaling must come FIRST. Without it, the model cannot produce meaningful deltas regardless of how good the temporal attention is. Temporal attention provides better information; delta scaling ensures the model can act on it.

---

## Appendix: MPS Compatibility Notes

### Operations Confirmed Safe on MPS

| Operation | Status | Notes |
|-----------|--------|-------|
| Conv2d, Conv1d | SAFE | Core operations, fully optimized |
| ConvLSTM (manual) | SAFE | Uses Conv2d + element-wise ops |
| bmm (batch matrix multiply) | SAFE | Used in manual attention |
| softmax | SAFE | Standard operation |
| nn.TransformerEncoder | SAFE | Falls back to math kernel |
| AdaptiveAvgPool2d | SAFE | Standard pooling |

### Operations with Known Issues on MPS

| Operation | Issue | Workaround |
|-----------|-------|------------|
| F.scaled_dot_product_attention | Crashes with GQA (enable_gqa=True) | Use manual attention or equal head counts |
| F.scaled_dot_product_attention | Memory issues with seq_len > 12K | Not relevant for T=10 |
| Grouped Conv2d | MPS bugs | Channel-loop fallback (already in codebase) |

### Recommendation

For all attention implementations in v3.0, use **manual attention** (bmm + softmax + scaling) rather than `F.scaled_dot_product_attention`. This avoids all MPS compatibility issues with zero performance penalty at the sequence lengths involved (T=10).

---

## Sources

### Published Research
- [SA-ConvLSTM (AAAI 2020)](https://ojs.aaai.org/index.php/AAAI/article/view/6819) -- Self-Attention ConvLSTM architecture
- [RA-ConvLSTM (2025)](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2024SW004173) -- Attention ConvLSTM for ionospheric prediction
- [ResConvLSTM-Att (2024)](https://www.sciencedirect.com/science/article/pii/S1364815224003219) -- Residual ConvLSTM with attention
- [MAG2MAG (2024)](https://arxiv.org/abs/2407.11659) -- DDPM-based magnetogram forecasting
- [Physics-Informed Vector Field Prediction (2025)](https://www.researchsquare.com/article/rs-8931595/v1) -- 12-hour solar magnetic field prediction
- [Magnetic Winding in Spherical Coords (2023)](https://link.springer.com/article/10.1007/s11207-023-02211-9) -- Winding flux computation from SHARP data
- [Deep Multi-Scale Video Prediction (2015)](https://arxiv.org/abs/1511.05440) -- Blurry prediction problem and solutions
- [Fourier Amplitude and Correlation Loss (NeurIPS 2024)](https://proceedings.neurips.cc/paper_files/paper/2024/file/b54532b0e57eb963b19e00583376cda3-Paper-Conference.pdf) -- Frequency-domain loss for video

### Implementations
- [SA-ConvLSTM PyTorch](https://github.com/tsugumi-sys/SA-ConvLSTM-Pytorch) -- Community implementation
- [PredRNN Official](https://github.com/thuml/predrnn-pytorch) -- Official PredRNN implementation
- [PyTorch Video Prediction Models](https://github.com/Ibzie/PyTorch-Video-Prediction-Models) -- ConvLSTM + attention + transformer

### PyTorch MPS Compatibility
- [SDPA MPS Crash (Fixed)](https://github.com/pytorch/pytorch/issues/149132) -- GQA crash on MPS, fixed in PR #149147
- [MPS Memory Issues](https://github.com/pytorch/pytorch/issues/147443) -- Large tensor SDPA issues
- [PyTorch MPS Attention Optimization](https://medium.com/@rakshekaraj/optimizing-pytorch-mps-attention-memory-efficient-large-sequence-processing-without-accuracy-5239f565f07b)

### Domain Context
- [Extreme Flare Prediction with HMI](https://arxiv.org/html/2405.14750) -- Multi-channel magnetogram + intensitygram approach
- [Deep Learning in Earth System Science (Review)](https://www.tandfonline.com/doi/full/10.1080/17538947.2024.2391952) -- Comprehensive survey of spatiotemporal methods

---
*Research completed: 2026-03-07*
