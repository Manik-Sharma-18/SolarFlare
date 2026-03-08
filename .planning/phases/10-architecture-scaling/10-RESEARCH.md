# Phase 10: Architecture Scaling - Research

**Researched:** 2026-03-08
**Domain:** ConvLSTM encoder-decoder architecture upgrades -- SA-ConvLSTM, temporal attention, spatial attention gates, channel/kernel widening, delta head scaling, MC Dropout
**Confidence:** HIGH

## Summary

Phase 10 introduces seven coordinated architectural changes to the SolarFluxPredictor. The changes are well-specified by the CONTEXT.md decisions and can be grouped into three implementation categories: (1) new modules to create (SA-ConvLSTMCell, TemporalAttention, AttentionGate), (2) modifications to existing code (predictor.py forward pass, config.yaml defaults, optimizer parameter groups), and (3) infrastructure changes (attention entropy logging, config validation updates).

The existing codebase is well-structured for these changes. The ConvLSTMCell is already factored out as a standalone class, making composition with the SAM module straightforward. The encoder-decoder in predictor.py has clear integration points identified in CONTEXT.md. The config-driven construction pattern means channel/kernel changes only require config defaults plus passing new parameters. PyTorch 2.10.0 is installed, providing all needed operations.

**Primary recommendation:** Implement in dependency order -- new modules first (SA-ConvLSTMCell, TemporalAttention, AttentionGate), then predictor integration (wire modules, add delta_scale, update defaults), then infrastructure (optimizer param groups, entropy logging, config validation). All attention implementations MUST use manual bmm+softmax (not F.scaled_dot_product_attention) for MPS compatibility.

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- SA-ConvLSTM cell design (ARCH-01): Channel attention variant (NOT spatial self-attention) to avoid 24K x 24K attention matrices at latent resolution. SAM dimension = hidden_dim // 2. All 6 ConvLSTM modules replaced. ~25K total parameters for SAM. Manual attention implementation (bmm + softmax) for MPS compatibility.
- Temporal attention (ARCH-07): Separate standalone TemporalAttention module. Encoder stores all T=10 hidden states from encoder_conv3 (latent layer only, c3=128). Decoder queries via scaled dot-product attention with global average pooling (~8K params). Additive context injection to decoder_conv3 output. Q/K/V projections via 1x1 Conv2d.
- Spatial attention gate (ARCH-03): Attention U-Net pattern with both encoder skip features AND decoder upsampled features jointly producing the gate. Conv2d(encoder) + Conv2d(decoder) -> ReLU -> Conv2d -> Sigmoid -> multiply with encoder skip. ~2,000-3,000 additional parameters.
- Channel widening (ARCH-04): Full [32, 64, 128] channels, configurable in config.yaml.
- Kernel size (ARCH-05): Kernel size 5, configurable in config.yaml.
- Delta head scaling (ARCH-02): Single scalar nn.Parameter initialized to 100.0. Applied as delta = raw_delta * delta_scale. Excluded from weight decay. 1 parameter.
- MC Dropout (ARCH-06): Dropout rate 0.15. Existing Dropout2d infrastructure already handles this -- just update config default.
- Overfitting strategy: Full [32,64,128] with MC Dropout 0.15, balanced augmentation, weight decay 1e-5, flare oversampling 3x. Fallback priority defined. Attention entropy logging per epoch.

### Claude's Discretion
- SA-ConvLSTM internal gate design for combining ConvLSTM output with SAM memory
- TemporalAttention projection dimension (suggested: channels or channels // 2)
- Attention U-Net gate intermediate feature dimension (F_int)
- How attention entropy metrics are integrated into training loop logging
- Parameter group setup for excluding delta_scale from weight decay
- Config key naming and structure for new architecture parameters

### Deferred Ideas (OUT OF SCOPE)
None -- discussion stayed within phase scope.
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| ARCH-01 | SA-ConvLSTM cells replace standard ConvLSTM cells (channel-attention variant with Self-Attention Memory) | SAM module design verified against SA-ConvLSTM paper (AAAI 2020) and reference implementations. Channel attention avoids spatial explosion. Composition pattern wraps existing ConvLSTMCell. |
| ARCH-02 | Learned delta head scaling parameter (nn.Parameter, initialized to match typical delta magnitude) | Single nn.Parameter(100.0), excluded from weight decay via optimizer parameter groups. Standard PyTorch pattern. |
| ARCH-03 | Spatial attention gates on skip connections (Attention U-Net pattern: Conv2d + Sigmoid) | Verified against Attention U-Net reference (Oktay et al., 2018). Gate takes encoder + decoder features, produces sigmoid mask. |
| ARCH-04 | Model channels widened to [32, 64, 128] (configurable) | Config-driven via existing channels parameter. Just update default in config.yaml. |
| ARCH-05 | Kernel size increased to 5 (configurable) | Config-driven via existing kernel_size parameter. Just update default in config.yaml. |
| ARCH-06 | MC Dropout enabled at 0.15 for regularization | Existing Dropout2d infrastructure in predictor.py already supports this. Update config.yaml default from 0.0 to 0.15. |
| ARCH-07 | Encoder stores all hidden states (not just final) for attention access | Encoder _encoder_forward() must collect encoder_conv3 hidden states at each timestep. Currently only returns final h3_states. |
</phase_requirements>

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| PyTorch | 2.10.0 | All model components | Already installed, all needed ops available |
| torch.nn | 2.10.0 | Conv2d, Parameter, ModuleList, Dropout2d | Standard neural network building blocks |
| torch.bmm | 2.10.0 | Batch matrix multiply for attention | MPS-safe alternative to F.scaled_dot_product_attention |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| torch.softmax | 2.10.0 | Attention weight normalization | In all attention modules (SAM, temporal, spatial gate) |
| torch.sigmoid | 2.10.0 | Gating mechanisms | SAM memory gate, attention gate sigmoid output |
| math.log | stdlib | Entropy computation | Attention entropy = -sum(p * log(p)), max entropy reference values |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Manual bmm+softmax attention | F.scaled_dot_product_attention | SDPA has MPS compatibility issues (crashes with GQA, memory issues). Manual is safe everywhere. |
| BatchNorm in attention gate | No normalization | Context decision omits BatchNorm from the gate spec. Simpler, fewer parameters, avoids batch_size=1 instability. |
| nn.Dropout instead of Dropout2d | Dropout2d | Dropout2d drops entire channels which is correct for 4D spatial data. BUT current code applies Dropout2d to 5D ConvLSTM output causing PyTorch deprecation warning -- consider switching to nn.Dropout for the 5D application. |

**No new dependencies required.** All implementations use PyTorch built-ins.

## Architecture Patterns

### Recommended Project Structure
```
models/
  convlstm.py           # Existing ConvLSTMCell + ConvLSTM (keep unchanged)
  sa_convlstm.py         # NEW: SAConvLSTMCell + SelfAttentionMemory + SAConvLSTM wrapper
  attention.py            # NEW: TemporalAttention + AttentionGate modules
  predictor.py            # MODIFIED: Wire new modules, add delta_scale, update forward
  uncertainty.py          # Existing (unchanged)
  __init__.py             # MODIFIED: Export new classes
```

### Pattern 1: SA-ConvLSTM Cell (Composition over Inheritance)

**What:** Wrap existing ConvLSTMCell inside SAConvLSTMCell via composition. The SAM module runs after the ConvLSTM step and refines the hidden state.

**When to use:** All 6 ConvLSTM modules in the encoder-decoder.

**Design:**
```python
class SelfAttentionMemory(nn.Module):
    """Channel-attention Self-Attention Memory module (SA-ConvLSTM, AAAI 2020).

    Uses channel attention (not spatial) to avoid O(H*W * H*W) cost.
    At latent resolution 110x221, spatial attention would produce 24K x 24K matrices.
    Channel attention operates on C dimensions (32-128), which is trivially cheap.
    """
    def __init__(self, hidden_dim: int, attn_dim: int):
        super().__init__()
        # attn_dim = hidden_dim // 2 per CONTEXT decision
        # Q/K/V projections for hidden state attention (1x1 Conv2d)
        self.query_h = nn.Conv2d(hidden_dim, attn_dim, 1)
        self.key_h = nn.Conv2d(hidden_dim, attn_dim, 1)
        self.value_h = nn.Conv2d(hidden_dim, attn_dim, 1)

        # K/V projections for memory cross-attention
        self.key_m = nn.Conv2d(hidden_dim, attn_dim, 1)
        self.value_m = nn.Conv2d(hidden_dim, attn_dim, 1)

        # Gated combination of Z_h and Z_m
        self.gate = nn.Conv2d(attn_dim * 2, attn_dim, 1)

        # Output projection back to hidden_dim
        self.output_proj = nn.Conv2d(attn_dim, hidden_dim, 1)

        self.scale = attn_dim ** -0.5

    def forward(self, h: torch.Tensor, m_prev: torch.Tensor):
        """
        Args:
            h: Current hidden state (B, hidden_dim, H, W) from ConvLSTM
            m_prev: Previous memory (B, hidden_dim, H, W)
        Returns:
            h_out: Refined hidden state (B, hidden_dim, H, W)
            m_new: Updated memory (B, hidden_dim, H, W)
        """
        B, C, H, W = h.shape

        # Channel attention: pool spatial dims, attend over channels
        # Q from h, K/V from h (self-attention on current state)
        q_h = self.query_h(h)  # (B, attn_dim, H, W)
        k_h = self.key_h(h)
        v_h = self.value_h(h)

        # Global average pool to get channel descriptors
        q_pool = q_h.mean(dim=(-2, -1))  # (B, attn_dim)
        k_pool = k_h.mean(dim=(-2, -1))  # (B, attn_dim)

        # Channel attention weights
        attn_h = torch.softmax(
            torch.bmm(q_pool.unsqueeze(1), k_pool.unsqueeze(2)) * self.scale,
            dim=-1
        )  # (B, 1, 1) -- scalar attention weight
        # For channel attention: weight each channel
        # Reshape for proper channel weighting
        q_flat = q_pool.unsqueeze(2)  # (B, attn_dim, 1)
        k_flat = k_pool.unsqueeze(1)  # (B, 1, attn_dim)
        attn_weights_h = torch.softmax(q_flat * k_flat * self.scale, dim=-1)  # (B, attn_dim, attn_dim)
        v_flat = v_h.view(B, -1, H * W)  # (B, attn_dim, H*W)
        z_h = torch.bmm(attn_weights_h, v_flat).view(B, -1, H, W)  # (B, attn_dim, H, W)

        # Cross-attention with memory
        k_m = self.key_m(m_prev)
        v_m = self.value_m(m_prev)
        k_m_pool = k_m.mean(dim=(-2, -1))
        k_m_flat = k_m_pool.unsqueeze(1)
        attn_weights_m = torch.softmax(q_flat * k_m_flat * self.scale, dim=-1)
        v_m_flat = v_m.view(B, -1, H * W)
        z_m = torch.bmm(attn_weights_m, v_m_flat).view(B, -1, H, W)

        # Gate-controlled combination
        combined = torch.cat([z_h, z_m], dim=1)  # (B, 2*attn_dim, H, W)
        gate = torch.sigmoid(self.gate(combined))  # (B, attn_dim, H, W)

        z_fused = gate * z_h + (1 - gate) * z_m

        # Memory update: new memory preserves fused information
        m_new = z_fused  # Could add residual: m_new = z_fused + some_transform(m_prev)

        # Output: combine ConvLSTM output with memory via projection
        h_out = h + self.output_proj(z_fused)  # Residual connection

        return h_out, m_new


class SAConvLSTMCell(nn.Module):
    """SA-ConvLSTM cell: ConvLSTM + Self-Attention Memory.

    Wraps existing ConvLSTMCell via composition (not inheritance).
    Returns (h, c, m) instead of (h, c).
    """
    def __init__(self, input_dim, hidden_dim, kernel_size, attn_dim=None):
        super().__init__()
        attn_dim = attn_dim or hidden_dim // 2
        self.convlstm_cell = ConvLSTMCell(input_dim, hidden_dim, kernel_size)
        self.sam = SelfAttentionMemory(hidden_dim, attn_dim)
        self.hidden_dim = hidden_dim

    def forward(self, x, h_prev, c_prev, m_prev):
        h, c = self.convlstm_cell(x, h_prev, c_prev)
        h_out, m_new = self.sam(h, m_prev)
        return h_out, c, m_new
```

**Parameter count per cell:** For hidden_dim=C, attn_dim=C//2:
- Q/K/V for h: 3 * C * (C//2) = 1.5 * C^2
- K/V for m: 2 * C * (C//2) = C^2
- Gate: (C//2 * 2) * (C//2) = C^2 / 2
- Output proj: (C//2) * C = C^2 / 2
- Total SAM: 3.5 * C^2

With c1=32: 3,584 params. c2=64: 14,336. c3=128: 57,344. Total ~75K across 6 cells.

**NOTE:** The CONTEXT.md says ~25K total SAM parameters. At [32,64,128] with SAM dim = hidden_dim // 2, the actual count will be higher. The planner should verify actual parameter counts and consider whether SAM should only be applied to the latent layers (c3=128) to stay closer to the ~25K estimate. Alternative: reduce attn_dim further or apply SAM selectively.

### Pattern 2: SAConvLSTM Wrapper (Multi-step Processing)

**What:** The multi-step ConvLSTM wrapper must be updated to handle the extra memory state (m) in addition to (h, c).

**Design:**
```python
class SAConvLSTM(nn.Module):
    """Multi-step SA-ConvLSTM that processes a sequence of spatial inputs."""
    def __init__(self, input_dim, hidden_dim, kernel_size, num_layers=1, attn_dim=None):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        cells = []
        for layer_idx in range(num_layers):
            cur_input_dim = input_dim if layer_idx == 0 else hidden_dim
            cells.append(SAConvLSTMCell(cur_input_dim, hidden_dim, kernel_size, attn_dim))
        self.cell_list = nn.ModuleList(cells)

    def forward(self, x, hidden_state=None):
        """
        Args:
            x: (B, C, T, H, W)
            hidden_state: list of (h, c, m) tuples per layer
        Returns:
            outputs: (B, hidden_dim, T, H, W)
            last_state: list of (h, c, m) tuples
        """
        B, _, T, H, W = x.size()
        if hidden_state is None:
            hidden_state = self._init_hidden(B, H, W, x.device)

        outputs = []
        for t in range(T):
            x_t = x[:, :, t]
            for layer_idx, cell in enumerate(self.cell_list):
                h_prev, c_prev, m_prev = hidden_state[layer_idx]
                h_next, c_next, m_next = cell(x_t, h_prev, c_prev, m_prev)
                hidden_state[layer_idx] = (h_next, c_next, m_next)
                x_t = h_next
            outputs.append(h_next)

        outputs = torch.stack(outputs, dim=2)
        return outputs, hidden_state

    def _init_hidden(self, B, H, W, device):
        return [
            (
                torch.zeros(B, self.hidden_dim, H, W, device=device),
                torch.zeros(B, self.hidden_dim, H, W, device=device),
                torch.zeros(B, self.hidden_dim, H, W, device=device),  # memory M
            )
            for _ in range(self.num_layers)
        ]
```

**Critical:** The return signature changes from `List[Tuple[h, c]]` to `List[Tuple[h, c, m]]`. Every call site in predictor.py that unpacks hidden states must be updated. This includes:
- `decoder_state2 = [(h2_states[0][0].clone(), h2_states[0][1].clone())]` -- must add m state
- `refine_state = None` -- must handle 3-tuple initialization

### Pattern 3: Temporal Attention Module

**What:** Standalone module that queries encoder hidden states at each decoder step.

**Design:**
```python
class TemporalAttention(nn.Module):
    """Temporal attention over encoder hidden states.

    Uses global average pooling for temporal-only attention (no spatial attention).
    Q/K/V via 1x1 Conv2d projections. ~8K parameters with c3=128.
    """
    def __init__(self, channels: int, proj_dim: int = None):
        super().__init__()
        proj_dim = proj_dim or channels  # or channels // 2
        self.q_proj = nn.Conv2d(channels, proj_dim, 1)
        self.k_proj = nn.Conv2d(channels, proj_dim, 1)
        self.v_proj = nn.Conv2d(channels, proj_dim, 1)
        self.out_proj = nn.Conv2d(proj_dim, channels, 1)
        self.scale = proj_dim ** -0.5

    def forward(self, decoder_state, encoder_states):
        """
        Args:
            decoder_state: (B, C, H, W) -- current decoder hidden state
            encoder_states: list of T tensors, each (B, C, H, W)
        Returns:
            context: (B, C, H, W) -- weighted combination of encoder states
            attn_weights: (B, T) -- attention distribution (for logging)
        """
        B, C, H, W = decoder_state.shape
        T = len(encoder_states)

        # Query from decoder (pool spatial)
        q = self.q_proj(decoder_state).mean(dim=(-2, -1))  # (B, proj_dim)

        # Keys from all encoder states (pool spatial)
        keys = torch.stack([
            self.k_proj(e).mean(dim=(-2, -1)) for e in encoder_states
        ], dim=1)  # (B, T, proj_dim)

        # Values from encoder states (keep spatial)
        values = torch.stack([self.v_proj(e) for e in encoder_states], dim=1)
        # (B, T, proj_dim, H, W)

        # Attention weights: (B, 1, T)
        attn = torch.softmax(
            torch.bmm(q.unsqueeze(1), keys.transpose(1, 2)) * self.scale,
            dim=-1
        )

        # Weighted combination of values: (B, proj_dim, H, W)
        values_flat = values.view(B, T, -1)  # (B, T, proj_dim*H*W)
        context_flat = torch.bmm(attn, values_flat)  # (B, 1, proj_dim*H*W)
        context = context_flat.view(B, -1, H, W)  # (B, proj_dim, H, W)

        return self.out_proj(context), attn.squeeze(1)
```

**Integration point:** In decoder loop, after `dec_h3, decoder_state3 = self.decoder_conv3(dec_h2, decoder_state3)`:
```python
context, attn_weights = self.temporal_attention(
    decoder_state3[0][0], self.encoder_h3_states
)
dec_h3_out = dec_h3[:, :, 0] + context  # Additive injection
```

### Pattern 4: Attention Gate on Skip Connection

**What:** Gates the skip connection using both encoder and decoder features before concatenation.

**Design:**
```python
class AttentionGate(nn.Module):
    """Attention U-Net gate for skip connections.

    Produces a spatial attention mask from encoder skip features
    and decoder upsampled features.
    """
    def __init__(self, encoder_channels: int, decoder_channels: int, f_int: int = None):
        super().__init__()
        f_int = f_int or max(encoder_channels // 2, 8)

        self.W_g = nn.Conv2d(decoder_channels, f_int, kernel_size=1, bias=True)
        self.W_x = nn.Conv2d(encoder_channels, f_int, kernel_size=1, bias=True)
        self.psi = nn.Conv2d(f_int, 1, kernel_size=1, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, g: torch.Tensor, x: torch.Tensor):
        """
        Args:
            g: Gating signal from decoder (B, decoder_channels, H, W)
            x: Skip connection from encoder (B, encoder_channels, H, W)
        Returns:
            gated_x: Attention-weighted skip features (B, encoder_channels, H, W)
        """
        g1 = self.W_g(g)      # (B, f_int, H, W)
        x1 = self.W_x(x)      # (B, f_int, H, W)
        psi = self.relu(g1 + x1)
        psi = self.sigmoid(self.psi(psi))  # (B, 1, H, W)
        return x * psi
```

**Integration point:** Replace line 263 in predictor.py:
```python
# Before: dec_concat = torch.cat([dec_up, h1_skip], dim=1)
# After:
gated_skip = self.attention_gate(dec_up, h1_skip)
dec_concat = torch.cat([dec_up, gated_skip], dim=1)
```

**Parameter count:** With encoder_channels=c1=32, decoder_channels=c2=64, f_int=16:
- W_g: 64*16 + 16 = 1,040
- W_x: 32*16 + 16 = 528
- psi: 16*1 + 1 = 17
- Total: ~1,585 parameters

### Pattern 5: Delta Head Scaling

**What:** Single learnable scalar that maps network-scale outputs to delta-scale values.

```python
# In __init__:
self.delta_scale = nn.Parameter(torch.tensor(100.0))

# In forward (output head):
raw_delta = self.output_conv(refined[:, :, 0])
delta = raw_delta * self.delta_scale
```

### Pattern 6: Optimizer Parameter Groups (Exclude delta_scale from Weight Decay)

**What:** Use parameter groups to apply weight_decay=0 to delta_scale.

```python
# In main.py or trainer.py, replace simple model.parameters():
decay_params = []
no_decay_params = []
for name, param in model.named_parameters():
    if not param.requires_grad:
        continue
    if name == 'delta_scale' or 'bias' in name:
        no_decay_params.append(param)
    else:
        decay_params.append(param)

optimizer = torch.optim.AdamW([
    {'params': decay_params, 'weight_decay': weight_decay},
    {'params': no_decay_params, 'weight_decay': 0.0},
], lr=lr)
```

**Important:** The current trainer.py passes `model.parameters()` directly to AdamW. This must change to parameter groups. The simplest approach: only exclude `delta_scale` by name (not all biases, to avoid changing training dynamics for existing parameters).

### Pattern 7: Attention Entropy Logging

**What:** Log channel attention entropy and temporal attention entropy as overfitting diagnostics.

```python
def compute_attention_entropy(attn_weights: torch.Tensor, eps: float = 1e-8) -> float:
    """Compute Shannon entropy of attention distribution.

    Args:
        attn_weights: (B, T) or (B, C) attention probabilities (sum to 1 along last dim)
    Returns:
        Mean entropy across batch (scalar)
    """
    # H = -sum(p * log(p))
    log_attn = torch.log(attn_weights + eps)
    entropy = -(attn_weights * log_attn).sum(dim=-1)  # (B,)
    return entropy.mean().item()
```

Max entropy references (for logging context):
- Temporal attention (T=10): ln(10) = 2.303 (uniform over 10 timesteps)
- Channel attention with C=32: ln(32) = 3.466
- Channel attention with C=64: ln(64) = 4.159
- Channel attention with C=128: ln(128) = 4.852

### Anti-Patterns to Avoid

- **DO NOT use F.scaled_dot_product_attention:** Known MPS issues. Use manual bmm + softmax everywhere.
- **DO NOT apply Dropout2d to 5D tensors:** PyTorch 2.10 warns that Dropout2d on 5D input is deprecated and will error in future releases. The existing code triggers this warning. Use nn.Dropout instead for 5D ConvLSTM sequence outputs, or reshape to 4D before applying Dropout2d.
- **DO NOT use BatchNorm in the attention gate:** With batch_size=1, BatchNorm statistics are meaningless. The CONTEXT.md spec omits it correctly.
- **DO NOT inherit from ConvLSTMCell for SAConvLSTMCell:** Use composition. Inheritance would require duplicating the forward logic. Composition reuses the cell directly.
- **DO NOT modify ConvLSTMCell or ConvLSTM classes:** These should remain unchanged for backward compatibility. New SA variants go in a new file.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Attention weight computation | Custom attention loop | `torch.bmm` + `torch.softmax` | Standard, efficient, MPS-safe, readable |
| Gating mechanism | Custom gate logic | `torch.sigmoid` on learned conv output | Standard pattern from LSTM/attention literature |
| Parameter group exclusion | Manual optimizer manipulation | AdamW parameter_groups list | PyTorch built-in, documented, standard pattern |
| Entropy computation | Approximate methods | `-(p * log(p)).sum()` | Exact Shannon entropy, trivial to compute |

## Common Pitfalls

### Pitfall 1: Hidden State Shape Change Breaks Decoder Init
**What goes wrong:** SAConvLSTM returns (h, c, m) tuples instead of (h, c). The decoder initialization in forward() unpacks h2_states and h3_states as 2-tuples. After switching to SAConvLSTM, these become 3-tuples, causing index errors.
**Why it happens:** The predictor.py decoder init does `decoder_state2 = [(h2_states[0][0].clone(), h2_states[0][1].clone())]` -- this only copies h and c, missing m.
**How to avoid:** Update ALL state unpacking sites in predictor.py to handle 3-tuples. Search for every reference to `h2_states`, `h3_states`, `decoder_state2`, `decoder_state3`, `refine_state`.
**Warning signs:** RuntimeError about tuple index out of range.

### Pitfall 2: Encoder Hidden State Collection Memory
**What goes wrong:** Storing all T=10 encoder_conv3 hidden states for temporal attention consumes extra GPU memory. Each state is (B, 128, H_latent, W_latent). With B=1 and latent ~55x111, that is 10 * 128 * 55 * 111 * 4 bytes = ~31 MB. Manageable, but worth noting.
**Why it happens:** Currently only the final state is kept. Collecting all 10 requires a list accumulation in the encoder loop.
**How to avoid:** The memory cost is acceptable. Just ensure states are detached from the computation graph if gradient checkpointing is used (checkpoint will recompute them). Actually, since the states are needed for the decoder's temporal attention forward pass, they should NOT be detached -- they must be part of the gradient graph.
**Warning signs:** OOM on smaller GPUs (unlikely with B=1).

### Pitfall 3: Gradient Checkpointing Incompatibility
**What goes wrong:** The current `_encoder_forward` is wrapped with `torch.utils.checkpoint.checkpoint()`. If the encoder now returns a list of hidden states (for temporal attention), checkpointing may not handle variable-length list outputs correctly.
**Why it happens:** Gradient checkpointing serializes/deserializes return values. Lists of tensors are supported, but the increased number of returned tensors increases recomputation cost.
**How to avoid:** Return the encoder_h3_states list as a stacked tensor `torch.stack(states, dim=0)` rather than a Python list. Tensors are natively supported by checkpointing. Unstack after the checkpoint call.
**Warning signs:** Errors during backward pass when use_checkpointing=True.

### Pitfall 4: Spatial Dimension Mismatch in Attention Gate
**What goes wrong:** The decoder upsampled features `dec_up` may not match `h1_skip` spatial dimensions exactly due to stride-2 convolutions on odd-sized inputs. The existing code handles this with `F.interpolate`, but the attention gate must also handle it.
**Why it happens:** The existing code already has a dimension-matching check at line 259-260 of predictor.py. The attention gate must be applied AFTER this interpolation.
**How to avoid:** Apply attention gate after the `F.interpolate` size matching, before the concatenation. The gate sees spatially-aligned tensors.
**Warning signs:** RuntimeError about tensor size mismatch in the gate's addition.

### Pitfall 5: delta_scale Gradient Explosion
**What goes wrong:** Initializing delta_scale to 100.0 means gradients flowing through it are scaled by 100x. With grad_clip=0.5, this may cause the delta_scale parameter to barely update, or the output_conv weights to get clipped before they can learn.
**Why it happens:** `delta = raw_delta * delta_scale`. d(loss)/d(raw_delta) = d(loss)/d(delta) * delta_scale. If delta_scale=100, raw_delta gradients are 100x larger.
**How to avoid:** Monitor delta_scale value during training (log it). Consider a more moderate initialization (e.g., 10.0 or 50.0) if gradient clipping causes issues. Alternatively, apply the scale OUTSIDE the gradient computation (detach delta_scale), but this prevents learning, which defeats the purpose.
**Warning signs:** delta_scale not changing from initial value, high gradient norm warnings from trainer.

### Pitfall 6: Dropout2d 5D Deprecation Warning
**What goes wrong:** Current code applies Dropout2d to 5D ConvLSTM output (B, C, T, H, W), which triggers a PyTorch deprecation warning that will become an error in future PyTorch versions.
**Why it happens:** Dropout2d expects 3D or 4D input. The ConvLSTM output is 5D.
**How to avoid:** Apply dropout to individual timestep outputs (4D) within the ConvLSTM loop, or use nn.Dropout which accepts any dimension. Since we're refactoring the ConvLSTM to SAConvLSTM, this can be addressed naturally.
**Warning signs:** UserWarning about dropout2d receiving 5D input (already visible in test output).

## Code Examples

### Example 1: Encoder Collecting All h3 States
```python
def _encoder_forward(self, x_prep, T_in):
    h1_seq, h1_states = self.encoder_conv1(x_prep)
    h1_skip = h1_states[0][0]  # Still needed for skip connection
    h1_seq = self.dropout_enc1(h1_seq)

    h1_down = self.downsample1(h1_seq[:, :, -1])
    h1_down_expanded = h1_down.unsqueeze(2).expand(-1, -1, T_in, -1, -1)

    h2_seq, h2_states = self.encoder_conv2(h1_down_expanded)
    h2_seq = self.dropout_enc2(h2_seq)

    h3_seq, h3_states = self.encoder_conv3(h2_seq)

    # NEW: Collect all h3 hidden states for temporal attention
    # h3_seq is (B, c3, T, H_lat, W_lat) -- extract per-timestep
    encoder_h3_states = [h3_seq[:, :, t] for t in range(T_in)]

    return h1_skip, h2_states, h3_states, h1_down, encoder_h3_states
```

### Example 2: Decoder Loop with Temporal Attention and Attention Gate
```python
for t in range(self.t_out):
    # ... existing decoder input processing ...

    dec_h2, decoder_state2 = self.decoder_conv2(dec_down, decoder_state2)
    dec_h2 = self.dropout_dec(dec_h2)
    dec_h3, decoder_state3 = self.decoder_conv3(dec_h2, decoder_state3)

    # NEW: Temporal attention (additive context injection)
    dec_h3_frame = dec_h3[:, :, 0]
    context, attn_weights = self.temporal_attention(
        decoder_state3[0][0], encoder_h3_states
    )
    dec_h3_frame = dec_h3_frame + context

    # Upsample from latent
    dec_up = self.upsample(dec_h3_frame)

    # Dimension matching (existing)
    if dec_up.shape[2:] != h1_skip.shape[2:]:
        dec_up = F.interpolate(dec_up, size=h1_skip.shape[2:], mode='nearest')

    # NEW: Attention gate on skip connection
    gated_skip = self.attention_gate(dec_up, h1_skip)
    dec_concat = torch.cat([dec_up, gated_skip], dim=1)
    dec_concat = dec_concat.unsqueeze(2)

    refined, refine_state = self.refine_conv(dec_concat, refine_state)

    # NEW: Delta scaling
    raw_delta = self.output_conv(refined[:, :, 0])
    delta = raw_delta * self.delta_scale

    # ... rest unchanged ...
```

### Example 3: Config.yaml Model Section Updates
```yaml
model:
  input_channels: 2
  output_channels: 1
  channels: [32, 64, 128]          # ARCH-04: widened from [16, 32, 64]
  kernel_size: 5                    # ARCH-05: increased from 3
  downsample_input: true
  use_checkpointing: false
  dropout_rate: 0.15                # ARCH-06: enabled from 0.0
  use_sa_convlstm: true             # ARCH-01: SA-ConvLSTM toggle
  delta_scale_init: 100.0           # ARCH-02: initial delta scale value
  # Attention parameters
  temporal_attention: true           # ARCH-07: enable temporal attention
  attention_gate: true               # ARCH-03: enable skip attention gate
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Standard ConvLSTM | SA-ConvLSTM with channel attention | AAAI 2020 (Lin et al.) | 5-10% improvement in SSIM/MSE for spatiotemporal prediction |
| Plain skip connections | Attention U-Net gated skips | MIDL 2018 (Oktay et al.) | Better focus on small regions of interest, 4800+ citations |
| Fixed delta output | Learned delta scaling | Common practice | Resolves numerical issue where near-zero deltas are gradient attractors |
| F.scaled_dot_product_attention | Manual bmm+softmax | PyTorch 2.x MPS issues | Required for MPS compatibility, no performance penalty at T=10 |

## Open Questions

1. **SAM Parameter Count vs CONTEXT Estimate**
   - What we know: CONTEXT.md estimates ~25K SAM parameters total. With channels [32,64,128] and SAM dim = hidden_dim // 2 applied to all 6 cells, actual count is ~75K.
   - What's unclear: Whether to apply SAM to all 6 cells or only selected ones (e.g., latent layers only).
   - Recommendation: Apply to all 6 as specified. The 75K is still modest. Document actual count in implementation.

2. **Dropout2d vs Dropout for 5D Tensors**
   - What we know: PyTorch 2.10 warns that Dropout2d on 5D input is deprecated.
   - What's unclear: Whether to fix this in phase 10 or leave for later.
   - Recommendation: Fix it while touching the dropout code. Use nn.Dropout for 5D or reshape to 4D before Dropout2d.

3. **Gradient Checkpointing with Extended Encoder Returns**
   - What we know: _encoder_forward will return more data (encoder_h3_states list).
   - What's unclear: Whether checkpoint() handles the larger return cleanly.
   - Recommendation: Test with use_checkpointing=True during validation. Stack the list into a tensor for checkpoint compatibility.

4. **delta_scale Initialization Stability**
   - What we know: Init=100.0 based on analysis of typical delta magnitudes (~0.01).
   - What's unclear: Whether this causes gradient clipping issues in practice.
   - Recommendation: Log delta_scale value per epoch. Start with 100.0 as specified, reduce if gradient warnings appear.

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest (already configured) |
| Config file | tests/conftest.py (shared fixtures including base_config, device, tiny_model_config) |
| Quick run command | `python -m pytest tests/test_model.py -x -q` |
| Full suite command | `python -m pytest tests/ -x -q` |

### Phase Requirements to Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| ARCH-01 | SA-ConvLSTM cells produce correct output shapes and finite values | unit | `python -m pytest tests/test_sa_convlstm.py -x -q` | Wave 0 |
| ARCH-01 | SAM module produces attention weights that sum to 1 | unit | `python -m pytest tests/test_sa_convlstm.py::TestSAM -x -q` | Wave 0 |
| ARCH-02 | delta_scale parameter exists, initialized to 100.0, excluded from weight decay | unit | `python -m pytest tests/test_model.py::TestDeltaScale -x -q` | Wave 0 |
| ARCH-03 | AttentionGate produces correct shape, sigmoid output in [0,1], non-collapsed weights | unit | `python -m pytest tests/test_attention.py::TestAttentionGate -x -q` | Wave 0 |
| ARCH-04 | Model with channels=[32,64,128] produces correct output shape | unit | `python -m pytest tests/test_model.py::TestForwardShape::test_wider_channels -x -q` | Wave 0 |
| ARCH-05 | Model with kernel_size=5 produces correct output shape | unit | `python -m pytest tests/test_model.py::TestForwardShape::test_kernel_5 -x -q` | Wave 0 |
| ARCH-06 | Model with dropout_rate=0.15 produces different outputs across train-mode runs | unit | `python -m pytest tests/test_model.py::TestForwardOutput::test_forward_with_dropout -x -q` | Exists (update) |
| ARCH-07 | Encoder stores all T hidden states, TemporalAttention produces correct shape | unit | `python -m pytest tests/test_attention.py::TestTemporalAttention -x -q` | Wave 0 |
| ALL | Full model with all ARCH features produces finite output on CPU | integration | `python -m pytest tests/test_model.py::TestForwardOutput::test_full_arch_forward -x -q` | Wave 0 |
| ALL | Full model with all ARCH features runs on MPS without error | smoke | `python -m pytest tests/test_model.py::TestMPSSmoke::test_full_arch_mps -x -q` | Wave 0 |

### Sampling Rate
- **Per task commit:** `python -m pytest tests/test_model.py tests/test_sa_convlstm.py tests/test_attention.py -x -q`
- **Per wave merge:** `python -m pytest tests/ -x -q`
- **Phase gate:** Full suite green before `/gsd:verify-work`

### Wave 0 Gaps
- [ ] `tests/test_sa_convlstm.py` -- covers ARCH-01 (SAConvLSTMCell shape, SAM attention weights, parameter count)
- [ ] `tests/test_attention.py` -- covers ARCH-03, ARCH-07 (AttentionGate, TemporalAttention shapes and properties)
- [ ] Update `tests/test_model.py` -- add tests for ARCH-02 (delta_scale), ARCH-04 (wider channels), ARCH-05 (kernel 5), full architecture integration test, MPS smoke test for new architecture
- [ ] Update `tests/conftest.py` -- add fixture for SA model config with all ARCH features enabled

## Sources

### Primary (HIGH confidence)
- PyTorch 2.10.0 documentation -- nn.Parameter, Conv2d, bmm, softmax, AdamW parameter groups
- Existing codebase: models/convlstm.py, models/predictor.py, training/trainer.py, main.py, config.yaml
- CONTEXT.md decisions (locked by user, verified against research)

### Secondary (MEDIUM confidence)
- [SA-ConvLSTM (AAAI 2020)](https://ojs.aaai.org/index.php/AAAI/article/view/6819) -- Original SA-ConvLSTM paper, Self-Attention Memory architecture
- [SA-ConvLSTM PyTorch implementation](https://github.com/tsugumi-sys/SA-ConvLSTM-Pytorch) -- Reference implementation verified against paper
- [SA-ConvLSTM implementation (johnjaejunlee95)](https://github.com/johnjaejunlee95/SA_ConvLSTM) -- Second reference implementation
- [Attention U-Net (Oktay et al., 2018)](https://smcdonagh.github.io/papers/attention_u_net_learning_where_to_look_for_the_pancreas.pdf) -- Original attention gate paper
- [Attention U-Net PyTorch tutorial](https://idiotdeveloper.com/attention-unet-in-pytorch/) -- Verified gate implementation pattern
- Project research document: .planning/research/TEMPORAL_ARCHITECTURES.md -- Prior deep research on all approaches

### Tertiary (LOW confidence)
- None -- all findings verified against primary/secondary sources

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH -- pure PyTorch, all ops verified available in 2.10.0
- Architecture: HIGH -- SA-ConvLSTM, temporal attention, attention gate are all well-established published architectures with reference implementations
- Integration: HIGH -- existing codebase is well-structured with clear integration points identified in CONTEXT.md and verified by reading source
- Pitfalls: HIGH -- identified from actual codebase analysis (hidden state tuple change, Dropout2d 5D warning confirmed in test output, dimension mismatch handling)
- Parameter estimates: MEDIUM -- SAM parameter count differs from CONTEXT estimate, actual count needs verification during implementation

**Research date:** 2026-03-08
**Valid until:** 2026-04-08 (stable -- PyTorch APIs and architectural patterns are well-established)
