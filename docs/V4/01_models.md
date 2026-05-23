# V4 Model Architecture Reference

Module-level reference for the V4 (`Version_4` branch) ConvLSTM stack under
`models/`. Complements `docs/MODEL_ARCHITECTURE.md` (narrative end-to-end)
and `architecture.md` (high-level). API + invariants only — no pipeline prose.

Public package surface from `models/__init__.py` (re-exports only):
`ConvLSTMCell`, `ConvLSTM`, `SelfAttentionMemory`, `SAConvLSTMCell`,
`SAConvLSTM`, `TemporalAttention`, `AttentionGate`, `SolarFluxPredictor`,
`predict_with_uncertainty`, `predict_with_confidence_intervals`,
`uncertainty_weighted_loss`.

## `models/convlstm.py` — vanilla ConvLSTM

Purpose: spatial LSTM where the four gates are produced by a single fused 2D conv over `concat(x, h_prev)`.

`ConvLSTMCell(input_dim, hidden_dim, kernel_size, bias=True)`
- One `Conv2d(input_dim + hidden_dim → 4·hidden_dim)` for all gates; gate slice order `i, f, g, o`.
- Forget bias initialised to `1.0` in `_init_forget_bias` (slice `[hidden_dim:2·hidden_dim]`); other gate biases zero.
- `forward(x, h_prev, c_prev) → (h_next, c_next)` with all tensors `(B, ·, H, W)`. `c = f·c_prev + i·g`, `h = o·tanh(c)`.
- Params: `(input_dim + hidden_dim) · 4·hidden_dim · k² + 4·hidden_dim`.

`ConvLSTM(input_dim, hidden_dim, kernel_size=3, num_layers=1)`
- Stacks `num_layers` cells; layer 0 takes `input_dim`, later layers consume `hidden_dim` (uniform across stack).
- `forward(x, hidden_state=None) → (outputs, last_state)`; `x` is `(B, C, T, H, W)`, `outputs` is `(B, hidden_dim, T, H, W)`, `last_state` is `list[(h, c)]`.
- `_init_hidden` zeros on `x.device`. Gotcha: no dtype argument — always default float; mixed precision must cast externally.

## `models/sa_convlstm.py` — SA-ConvLSTM (channel-attention memory)

Purpose: ConvLSTM + Self-Attention Memory (Lin et al. AAAI 2020) but with channel attention to dodge the O((HW)²) cost spatial attention would incur at latent 110×221.

`SelfAttentionMemory(hidden_dim, attn_dim=None)`
- Default `attn_dim = hidden_dim // 2`. Six 1×1 convs: `query_h`, `key_h`, `value_h` (self), `key_m`, `value_m` (memory cross), plus `gate` (`2·attn_dim→attn_dim`), `output_proj`, `memory_proj` (`attn_dim→hidden_dim`).
- `forward(h, m_prev) → (h_out, m_new)`:
  1. Project Q/K/V; global-average-pool Q and K to `(B, attn_dim)`.
  2. Softmax of outer product `Q⊗K · scale` → `(B, attn_dim, attn_dim)`.
  3. Apply via `bmm` to flattened V; reshape spatial.
  4. Repeat against memory K/V for cross-attention `z_m`.
  5. Sigmoid `gate` fuses `z_h`, `z_m`; `m_new = memory_proj(z_fused)`; `h_out = h + output_proj(z_fused)` (residual).
- MPS-safe: hand-rolled `bmm + softmax`. `scale = attn_dim ** -0.5`.

`SAConvLSTMCell(input_dim, hidden_dim, kernel_size, attn_dim=None)`
- Composition (not inheritance) of `ConvLSTMCell` + `SelfAttentionMemory`.
- `forward(x, h_prev, c_prev, m_prev) → (h_out, c, m_new)` — 3-tuple; callers must thread `m`.

`SAConvLSTM(input_dim, hidden_dim, kernel_size=3, num_layers=1, attn_dim=None)`
- Same shape contract as `ConvLSTM` but `hidden_state` is `list[(h, c, m)]`. `_init_hidden` zeros all three.

## `models/attention.py` — encoder-decoder attention

Purpose: temporal attention over encoder hidden states and an Attention-U-Net gate on skip connections. MPS-safe (manual `bmm + softmax`).

`TemporalAttention(channels, proj_dim=None, t_max=20)`
- 1×1 Q/K/V projections + `out_proj` (all Conv2d). `scale = proj_dim**-0.5`.
- `pos_embed = nn.Parameter(torch.zeros(t_max, proj_dim))` — learnable PE. Zero init means the layer starts as vanilla content attention.
- `forward(decoder_state, encoder_states) → (context, attn_weights)`. `decoder_state` `(B, C, H, W)`; `encoder_states` is `list[(B, C, H, W)]` of length `T`. Runtime guard raises `ValueError` if `T > t_max`.
- Q and keys are spatially average-pooled; values keep spatial dims. `attn_weights` returned as `(B, T)`.

`AttentionGate(encoder_channels, decoder_channels, f_int=None)`
- `f_int` defaults to `max(encoder_channels // 2, 8)`. Three 1×1 convs: `W_g`, `W_x`, `psi`. No BatchNorm — V4 trains with `batch_size=1` and BN would be unstable (`attention.py:97`).
- `forward(g, x) → x · sigmoid(psi(relu(W_g(g) + W_x(x))))`.

## `models/predictor.py` — `SolarFluxPredictor`

Purpose: top-level model. ConvLSTM (or SA-ConvLSTM) encoder/decoder with skip connection, residual output head, and four ARCH feature flags.

`SolarFluxPredictor(input_channels=1, output_channels=1, t_out=3, channels=[16,32,64], kernel_size=3, downsample_input=True, use_checkpointing=False, dropout_rate=0.0, use_sa_convlstm=False, temporal_attention=False, attention_gate=False, delta_scale_init=0.0)`

Submodule wiring (`c1, c2, c3 = channels`):
- `input_down`: `Conv2d(in→c1, k=4, s=2) + ReLU` when `downsample_input` (default `False` in production after the 2026-05-21 review). Vestigial `input_up` was removed; the production path runs encoder/decoder at native window resolution (typically 128×128 spatial tiles via the sliding-window dataset).
- `preprocess`: `Conv2d(c1→c1, k=3) + ReLU`.
- Encoder: `encoder_conv1(c1→c1)`, `downsample1(c1→c2, s=2)`, `encoder_conv2(c2→c2)`, `encoder_conv3(c2→c3)`. `ConvLSTMClass = SAConvLSTM if use_sa_convlstm else ConvLSTM` is chosen once (`predictor.py:92`).
- Decoder: `decoder_input_conv(in→c1)` (used only when `downsample_input=False`), `decoder_proj(c1→c2, s=2)`, `decoder_conv2(c2→c2)`, `decoder_conv3(c2→c3)`, `upsample: ConvTranspose2d(c3→c2, s=2)`, `refine_conv((c2+c1)→c1)`.
- Output head: with `downsample_input`, `ConvTranspose2d(c1→c1, s=2) + ReLU + Conv2d(c1→output_channels)`. Else `Conv2d(c1→output_channels)`.
- MC Dropout: three `nn.Dropout` (`enc1`, `enc2`, `dec`) when `dropout_rate > 0`, else `nn.Identity` (zero overhead). `nn.Dropout` (not `Dropout2d`) avoids the 5D deprecation warning (`predictor.py:150`).
- ARCH-07 (`temporal_attention`): adds `TemporalAttention(c3, t_max=20)`.
- ARCH-03 (`attention_gate`): adds `AttentionGate(c1, c2)`.
- ARCH-02 (`delta_scale_init != 0.0`): scalar `nn.Parameter` multiplying the residual; init `0.0` leaves it `None` (multiplication skipped).

Helper at module scope: `_match_spatial(t, target_hw)` — center-crops or zero-pads a tensor's spatial dims; raises if mismatch exceeds 2 px (config bug, not stride rounding). Used to align decoder upsample output to the encoder skip and to align final delta to the input frame size.

`_encoder_forward(x_prep, T_in) → (h1_skip, h2_states, h3_states, h1_down, encoder_h3_packed)`
- `h1_skip = h1_states[0][0]` — last-timestep hidden state of `encoder_conv1`, reused per decoder step.
- `encoder_h3_packed` is `(B, T_in, c3, H_lat, W_lat)` when temporal attention is on (stacked tensor, not list, for `checkpoint` compatibility); `None` otherwise.

`forward(x, teacher_forcing_ratio=0.0, y_true=None) → (B, C, T_out, H, W)`
- Autoregressive loop over `t_out`: decoder ConvLSTMs → optional temporal attention (additive context onto `dec_h3[:,:,0]`) → `upsample` → optional `AttentionGate` on skip → `refine_conv` → output head → `pred_flux = input_flux + delta`. Residual computed on flux channel only (`flux_channel_idx = 0`).
- `_match_spatial()` (center-crop or zero-pad) reconciles ±2 px drift from stride-2 ops at the skip-connection (`predictor.py:330`) and final delta (`predictor.py:353`). Larger mismatch raises `RuntimeError`. Replaces the previous nearest-interp patch.
- Teacher forcing: when `C > output_channels` (e.g. flux + extreme indicator), non-flux channels of `input_frame` are reused in both branches; flux is overwritten with ground truth or prediction.
- Gradient checkpointing only fires when `self.training` and uses `use_reentrant=False`.

`count_parameters()` → trainable param sum.

Gotchas
- Both `ConvLSTM` and `SAConvLSTM` `_init_hidden` ignore dtype.
- Decoder always clones encoder final states (`h2_states[0][0].clone()` …) to avoid storage sharing; the 3-tuple branch handles SAM memory.
- Default config (`channels=[16,32,64]`, `t_out=3`, flags off, `dropout=0`) is the lean baseline.

## `models/uncertainty.py` — MC Dropout helpers

Purpose: O(1)-memory uncertainty via Welford's online algorithm.

- `_welford_update(count, mean, m2, new_value) → (count, mean, m2)` — one step.
- `predict_with_uncertainty(model, x, n_samples=20, teacher_forcing_ratio=0.0, y_true=None) → (mean, std)`. Forces `model.train()` to keep dropout active; restores prior mode in `finally`. Raises `ValueError` if `model.dropout_rate == 0.0`. `std = sqrt(M2/count + 1e-8)`.
- `predict_with_confidence_intervals(model, x, n_samples=20, confidence_level=0.95) → (mean, lower, upper)`. Gaussian approximation `mean ± z·std` with lookup for `{0.90, 0.95, 0.99}`. Arbitrary `confidence_level ∈ (0, 1)` falls back to `sqrt(2) · erfinv(c)` via `torch.erfinv` on a float64 CPU scalar; out-of-range raises `ValueError`. Avoids `torch.quantile` (incorrect on MPS — see comment, `uncertainty.py:115`).
- `uncertainty_weighted_loss(predictions, targets, uncertainty, base_loss_fn=nn.L1Loss(reduction='none'))` → scalar. Weights are `1/(uncertainty + 1e-6)`, mean-normalised before multiplying per-element base loss.

## Model composition

`SolarFluxPredictor.__init__` selects `ConvLSTMClass` once (line 92) and instantiates every recurrent block (3 encoder + 2 decoder + 1 refine = 6 stacks) of that class. Each `SAConvLSTM` stack composes `SAConvLSTMCell`, which composes `ConvLSTMCell` (vanilla) + one `SelfAttentionMemory`. `TemporalAttention` and `AttentionGate` are instantiated at most once at the top level and called inside the decoder loop. `models/__init__.py` is import-only; no construction at import time. `predict_with_uncertainty` wraps an already-built `SolarFluxPredictor` and only toggles `.train()` / `.eval()`.

## Files reference

| File:lines | Entity |
|---|---|
| `models/__init__.py:1-10` | Public re-exports |
| `models/convlstm.py:12-94` | `ConvLSTMCell` |
| `models/convlstm.py:47-53` | Forget-bias init |
| `models/convlstm.py:97-188` | `ConvLSTM` (stack + sequence loop) |
| `models/sa_convlstm.py:19-122` | `SelfAttentionMemory` |
| `models/sa_convlstm.py:125-174` | `SAConvLSTMCell` |
| `models/sa_convlstm.py:177-267` | `SAConvLSTM` |
| `models/attention.py:16-89` | `TemporalAttention` (learnable PE) |
| `models/attention.py:92-140` | `AttentionGate` |
| `models/predictor.py:26-174` | `SolarFluxPredictor.__init__` |
| `models/predictor.py:92` | `ConvLSTMClass` selection |
| `models/predictor.py:152-160` | Dropout / Identity branches |
| `models/predictor.py:176-223` | `_encoder_forward` |
| `models/predictor.py:225-390` | `forward` (autoregressive decode) |
| `models/predictor.py:392-394` | `count_parameters` |
| `models/uncertainty.py:25-37` | `_welford_update` |
| `models/uncertainty.py:40-100` | `predict_with_uncertainty` |
| `models/uncertainty.py:103-173` | `predict_with_confidence_intervals` |
| `models/uncertainty.py:176-210` | `uncertainty_weighted_loss` |
