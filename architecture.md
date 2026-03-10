# SolarFlare Architecture Diagrams

---

## 1. High-Level System Overview

> **What** SolarFlare does, **why** it exists, and **how** it works at the highest level.

```mermaid
flowchart TD
    subgraph WHY["WHY: Space Weather Forecasting"]
        W1["Solar flares & CMEs release<br/>massive energy bursts"]
        W2["Damage satellites, GPS,<br/>power grids, astronauts"]
        W3["Need: predict magnetic flux<br/>evolution hours ahead"]
        W1 --> W2 --> W3
    end

    subgraph WHAT["WHAT: Spatiotemporal Deep Learning Predictor"]
        A1["Input: 10 consecutive frames<br/>of solar magnetic winding flux<br/>(B, C, 10, 440, 884)"]
        A2["ConvLSTM Encoder-Decoder<br/>learns spatial patterns +<br/>temporal dynamics jointly"]
        A3["Output: 4 predicted future<br/>frames of magnetic flux<br/>(B, 1, 4, 440, 884)"]
        A1 --> A2 --> A3
    end

    subgraph HOW["HOW: End-to-End ML Pipeline"]
        H1["Load raw .npy<br/>structured arrays"] --> H2["Asinh normalize<br/>+ dual channel"]
        H2 --> H3["Sliding windows<br/>+ augmentation"]
        H3 --> H4["Train ConvLSTM<br/>encoder-decoder"]
        H4 --> H5["Evaluate: MAE,<br/>CSI, HSS, SSIM"]
    end

    WHY --> WHAT --> HOW

    style WHY fill:#2d1b69,stroke:#7c3aed,color:#e0d4fc
    style WHAT fill:#1b3a69,stroke:#3a7aed,color:#d4e4fc
    style HOW fill:#1b694a,stroke:#3aed7c,color:#d4fce4
```

---

## 2. Data Pipeline

> From raw solar observations to GPU-ready training batches.

```mermaid
flowchart TD
    RAW["Raw .npy Structured Arrays<br/>Fields: X, Y, windTotal, time<br/>Sparse coordinate format"]

    RAW --> SCAN["Pre-flight Scan<br/>Validate required fields exist<br/>Abort if more than 10% corrupt"]

    SCAN --> CUBE["Sparse-to-Dense Conversion<br/>Map X,Y coords to grid indices<br/>Output: (T, H, W) dense cubes<br/>Spatial: ~440 x 884 pixels"]

    CUBE --> STATS["Compute Normalization Stats<br/>99th percentile of |flux| values<br/>Median, interquartile range"]

    STATS --> ASINH["Asinh Normalization<br/>normalized = arcsinh(val / 1000) / scale<br/>Linear below softening, log above<br/>Symmetric for +/- flux, no clipping"]

    ASINH --> CH1["Channel 1: Normalized Flux<br/>Full spatial field values<br/>Range compressed to ~[-1, +1]"]
    ASINH --> CH2["Channel 2: Extreme Indicator<br/>sigmoid( (|flux| - p99) / (p99 * 0.5) )<br/>Smooth 0-1 mask highlighting<br/>flare-active regions"]

    CH1 & CH2 --> SPLIT["Temporal Split (Deterministic Seed)<br/>70% train / 20% test / 10% val<br/>Whole-file assignment prevents<br/>temporal data leakage"]

    SPLIT --> WINDOW["Sliding Window Construction<br/>10 input + 4 output frames per sample<br/>Configurable stride (1 = max overlap)"]

    WINDOW --> AUG["Augmentation (Pre-computed Codes)<br/>Balanced: H-flip, V-flip (50% each)<br/>Aggressive: + 90/180/270 rotations"]

    AUG --> LOADER["PyTorch DataLoader<br/>Lazy per-worker memory-mapped access<br/>Optional 3x flare sequence oversampling<br/>Pin memory for CUDA transfers"]

    style RAW fill:#1a1a2e,stroke:#4a4a6e,color:#c0c0e0
    style ASINH fill:#2e1a2e,stroke:#6e4a6e,color:#e0c0e0
    style CH1 fill:#1a2e2e,stroke:#4a6e6e,color:#c0e0e0
    style CH2 fill:#1a2e2e,stroke:#4a6e6e,color:#c0e0e0
    style LOADER fill:#2e2e1a,stroke:#6e6e4a,color:#e0e0c0
```

### Data Pipeline Equations

**Asinh Normalization** (preserves extreme values, handles negatives symmetrically):

$$\hat{v} = \frac{\text{arcsinh}(v / s)}{S_{max}}, \quad \text{arcsinh}(x) = \ln(x + \sqrt{x^2 + 1})$$

Where $v$ is the raw flux value, $s = 1000.0$ is the softening parameter (typical non-flare magnitude), and $S_{max}$ is computed so that $\hat{v} \in [-1, 1]$.

**Behavior:** For $|v| \ll s$: $\hat{v} \approx v/s$ (linear). For $|v| \gg s$: $\hat{v} \approx \text{sgn}(v) \cdot \ln(2|v|/s)$ (logarithmic compression).

**Extreme Indicator Channel:**

$$\text{Ch}_2(v) = \sigma\left(\frac{|v| - \tau}{\tau / 2}\right) = \frac{1}{1 + \exp\left(-\frac{|v| - \tau}{\tau / 2}\right)}$$

Where $\tau = P_{99}(|v|)$ is the 99th percentile of absolute flux values. Produces a smooth 0-1 mask: background $\approx 0$, flare regions $\approx 1$.

**Residual Prediction:**

$$\hat{y}_t = x_{last} + \Delta_t, \quad \Delta_t = \gamma \cdot f_\theta(h_t^{dec})$$

Where $x_{last}$ is the last input flux frame, $f_\theta$ is the output Conv2d head, and $\gamma$ is the optional learnable delta scale (initialized near 1.0).

---

## 3. Model Architecture Overview

> The full encoder-decoder pipeline from input to predicted frames.

```mermaid
flowchart TD
    INPUT["Input: (B, C, 10, H, W)<br/>C=1 flux or C=2 flux+extreme<br/>H=440, W=884"]

    INPUT --> INDOWN["Input Downsampling (Optional)<br/>Conv2d(C, 16, k=4, stride=2) + ReLU<br/>Halves spatial dims: H/2, W/2<br/>Saves ~4x GPU memory"]

    INDOWN --> PREPROC["Preprocess Conv<br/>Conv2d(16, 16, k=3, pad=1) + ReLU<br/>Per-frame feature extraction"]

    PREPROC --> ENC1["Encoder ConvLSTM-1 (16ch, k=5)<br/>Process all 10 frames sequentially<br/>Output: h1_seq (B, 16, 10, H/2, W/2)<br/>Dropout 15% applied to output"]

    ENC1 --> SDOWN["Spatial Downsample<br/>Conv2d(16, 32, k=3, stride=2, pad=1)<br/>Further halves: H/4, W/4"]

    SDOWN --> ENC2["Encoder ConvLSTM-2 (32ch, k=5)<br/>10 frames at reduced resolution<br/>Output: h2_seq (B, 32, 10, H/4, W/4)<br/>Dropout 15% applied to output"]

    ENC2 --> ENC3["Encoder ConvLSTM-3 (64ch, k=5)<br/>LATENT BOTTLENECK<br/>Output: h3_seq (B, 64, 10, H/4, W/4)<br/>Final encoder states saved"]

    ENC3 --> DECODER["AUTOREGRESSIVE DECODER<br/>Loops t = 0, 1, 2, 3<br/>Each step predicts one future frame<br/>See Diagram 4 for decoder detail"]

    DECODER --> OUTPUT["Output: (B, 1, 4, H, W)<br/>4 predicted future flux frames<br/>at original resolution"]

    ENC1 -. "h1_skip: (B, 16, H/2, W/2)<br/>Skip connection preserves<br/>fine spatial details" .-> DECODER

    ENC2 -. "h2_states: initialize<br/>decoder ConvLSTM-2" .-> DECODER

    ENC3 -. "h3_states: initialize<br/>decoder ConvLSTM-3" .-> DECODER

    ENC3 -. "h3 per timestep:<br/>10 states for temporal<br/>attention queries" .-> DECODER

    style INPUT fill:#1a1a2e,stroke:#4a4a6e,color:#c0c0e0
    style ENC1 fill:#1b2d4a,stroke:#3a5a8a,color:#c0d4f0
    style ENC2 fill:#1b2d4a,stroke:#3a5a8a,color:#c0d4f0
    style ENC3 fill:#152848,stroke:#2a4a78,color:#b0c4e8
    style DECODER fill:#4a1b2d,stroke:#8a3a5a,color:#f0c0d4
    style OUTPUT fill:#2d4a1b,stroke:#5a8a3a,color:#d4f0c0
```

---

## 4. Decoder Detail (One Autoregressive Step)

> What happens inside the decoder at each of the 4 prediction steps.

```mermaid
flowchart TD
    FRAME_IN["Current Input Frame<br/>(B, C, H, W)<br/>t=0: last observed frame<br/>t>0: previous prediction"]

    FRAME_IN --> DEC_DOWN["Downsample to Latent<br/>input_down (stride=2) then<br/>decoder_proj Conv2d(16, 32, stride=2)<br/>Result: (B, 32, H/4, W/4)"]

    DEC_DOWN --> DCONV2["Decoder ConvLSTM-2 (32ch)<br/>Continues encoder h2 hidden state<br/>Single timestep processing"]

    DCONV2 --> DROP["MC Dropout (15%)<br/>Kept active at inference for<br/>uncertainty quantification"]

    DROP --> DCONV3["Decoder ConvLSTM-3 (64ch)<br/>Continues encoder h3 hidden state<br/>Output: decoder h3 (B, 64, H/4, W/4)"]

    DCONV3 --> TA_CHECK{"Temporal<br/>Attention<br/>enabled?"}

    TA_CHECK -->|Yes| TEMP_ATTN["Temporal Attention Module<br/>See Diagram 5 for mechanism<br/>Adds context from encoder history"]
    TA_CHECK -->|No| PASSTHRU["Pass through unchanged"]

    TEMP_ATTN --> UP
    PASSTHRU --> UP

    UP["Upsample<br/>ConvTranspose2d(64, 32, k=4, stride=2)<br/>Result: (B, 32, H/2, W/2)"]

    UP --> AG_CHECK{"Attention<br/>Gate<br/>enabled?"}

    AG_CHECK -->|Yes| ATTN_GATE["Attention Gate (ARCH-03)<br/>See Diagram 7 for mechanism<br/>Spatially gates the skip features"]
    AG_CHECK -->|No| RAW_SKIP["Raw concatenation"]

    ATTN_GATE --> CONCAT["Concatenate: gated_skip + upsampled<br/>(B, 16+32, H/2, W/2)"]
    RAW_SKIP --> CONCAT

    CONCAT --> REFINE["Refine ConvLSTM (48ch in, 16ch out)<br/>Polishes combined encoder+decoder<br/>features into coherent prediction"]

    REFINE --> OUTCONV["Output Head<br/>ConvTranspose2d(16, 16, stride=2) + ReLU<br/>Conv2d(16, 1, k=1)<br/>Produces residual delta"]

    OUTCONV --> SCALE{"Learnable<br/>delta scale?"}
    SCALE -->|Yes| SCALED["delta = raw_delta * learned_param<br/>Initialized near 1.0"]
    SCALE -->|No| UNSCALED["delta = raw_delta"]

    SCALED --> RESIDUAL
    UNSCALED --> RESIDUAL

    RESIDUAL["Residual Prediction<br/>pred_flux = input_flux + delta<br/>Learns small changes, not absolutes<br/>Stabilizes training, prevents divergence"]

    RESIDUAL --> TF{"Teacher Forcing?<br/>Probability decays<br/>0.5 at epoch 1<br/>0.0 by final epoch"}

    TF -->|"Yes: use ground truth"| NEXT_GT["Next input = y_true[:,:,t]"]
    TF -->|"No: use prediction"| NEXT_PRED["Next input = pred_flux"]

    NEXT_GT --> LOOP["Feed into next<br/>decoder step (t+1)"]
    NEXT_PRED --> LOOP

    style FRAME_IN fill:#1a1a2e,stroke:#4a4a6e,color:#c0c0e0
    style DCONV2 fill:#4a1b2d,stroke:#8a3a5a,color:#f0c0d4
    style DCONV3 fill:#4a1b2d,stroke:#8a3a5a,color:#f0c0d4
    style TEMP_ATTN fill:#3d1b4a,stroke:#6a3a8a,color:#d8c0f0
    style ATTN_GATE fill:#3d1b4a,stroke:#6a3a8a,color:#d8c0f0
    style RESIDUAL fill:#2d4a1b,stroke:#5a8a3a,color:#d4f0c0
    style REFINE fill:#2d2d1b,stroke:#5a5a3a,color:#e0e0c0
```

---

## 5. Temporal Attention Mechanism (ARCH-07)

> How the decoder queries its memory of the encoder's history at each prediction step.
> This is NOT multi-head attention. It uses a single attention head with Q/K/V projections.

```mermaid
flowchart TD
    subgraph INPUTS["Inputs to Temporal Attention"]
        DEC_H3["Decoder h3 state (current step)<br/>(B, 64, H/4, W/4)<br/>The decoder's current understanding"]
        ENC_STATES["Encoder h3 states (all 10 steps)<br/>List of 10 tensors, each (B, 64, H/4, W/4)<br/>The full temporal history"]
    end

    subgraph QUERY["Query: What does the decoder need?"]
        DEC_H3 --> Q_PROJ["Q = Conv2d(64, 64, k=1)<br/>Project decoder state"]
        Q_PROJ --> Q_POOL["Global Average Pool<br/>mean over (H, W) spatial dims<br/>Result: q (B, 64)<br/>Collapses space, keeps channel info"]
    end

    subgraph KEYS["Keys: What does each encoder step offer?"]
        ENC_STATES --> K_PROJ["K = Conv2d(64, 64, k=1)<br/>Project each encoder state"]
        K_PROJ --> K_POOL["Global Average Pool per state<br/>Result: keys (B, 10, 64)<br/>Each timestep summarized as vector"]
    end

    subgraph VALUES["Values: Full spatial detail from encoder"]
        ENC_STATES --> V_PROJ["V = Conv2d(64, 64, k=1)<br/>Project each encoder state<br/>Keeps full spatial resolution"]
        V_PROJ --> V_STACK["Stack all values<br/>Result: (B, 10, 64, H/4, W/4)<br/>Spatial features preserved"]
    end

    subgraph ATTEND["Scaled Dot-Product Attention (Single Head)"]
        Q_POOL --> SCORES["scores = bmm(q, keys^T) * scale<br/>q: (B, 1, 64) x keys^T: (B, 64, 10)<br/>Result: (B, 1, 10) raw scores<br/>scale = 1/sqrt(64) = 0.125"]
        K_POOL --> SCORES
        SCORES --> SOFTMAX["attn_weights = softmax(scores)<br/>Result: (B, 1, 10)<br/>Probability distribution over time<br/>e.g. [0.05, 0.03, ..., 0.40, 0.15]"]
    end

    subgraph COMBINE["Weighted Combination"]
        SOFTMAX --> WEIGHTED["context = bmm(attn_weights, values)<br/>attn: (B, 1, 10) x values: (B, 10, 64*H*W)<br/>Weighted sum of all encoder spatial features<br/>Result: (B, 64, H/4, W/4)"]
        V_STACK --> WEIGHTED
        WEIGHTED --> OUT_PROJ["out = Conv2d(64, 64, k=1)<br/>Project context back to hidden dim"]
    end

    subgraph INJECT["Additive Injection"]
        OUT_PROJ --> ADD["decoder_h3 = decoder_h3 + context<br/>Additive, not replacement<br/>Near-zero init = graceful degradation<br/>Model works fine without attention,<br/>attention adds refinement on top"]
    end

    style QUERY fill:#2d1b4a,stroke:#5a3a8a,color:#d8c0f0
    style KEYS fill:#1b2d4a,stroke:#3a5a8a,color:#c0d4f0
    style VALUES fill:#1b2d4a,stroke:#3a5a8a,color:#c0d4f0
    style ATTEND fill:#4a1b1b,stroke:#8a3a3a,color:#f0c0c0
    style COMBINE fill:#2d4a1b,stroke:#5a8a3a,color:#d4f0c0
    style INJECT fill:#1b4a2d,stroke:#3a8a5a,color:#c0f0d4
```

### Temporal Attention Equations

Given decoder hidden state $h^{dec} \in \mathbb{R}^{B \times C \times H \times W}$ and encoder hidden states $\{h^{enc}_1, \ldots, h^{enc}_T\}$:

**Projections** (1x1 Conv2d, learned):

$$Q = W_Q \ast h^{dec}, \quad K_t = W_K \ast h^{enc}_t, \quad V_t = W_V \ast h^{enc}_t$$

**Global Average Pooling** (collapse spatial dims for lightweight attention):

$$\bar{q} = \frac{1}{HW}\sum_{i,j} Q_{:,:,i,j} \in \mathbb{R}^{B \times d}, \quad \bar{k}_t = \frac{1}{HW}\sum_{i,j} K_{t,:,:,i,j} \in \mathbb{R}^{B \times d}$$

**Scaled Dot-Product Attention** (single head, not multi-head):

$$\alpha_t = \text{softmax}\left(\frac{\bar{q} \cdot \bar{k}_t}{\sqrt{d}}\right) \in \mathbb{R}^{B \times T}$$

**Weighted Combination** (values retain full spatial resolution):

$$\text{context} = W_{out} \ast \left(\sum_{t=1}^{T} \alpha_t \cdot V_t\right) \in \mathbb{R}^{B \times C \times H \times W}$$

**Additive Injection** (not replacement - graceful degradation):

$$h^{dec}_{out} = h^{dec} + \text{context}$$

Where $d = C = 64$ (proj_dim equals channel count), $T = 10$ input timesteps, and $\ast$ denotes convolution.

**Why temporal attention?** The decoder needs to "look back" at the full encoder history to decide which past patterns are relevant for the current prediction step. Without it, the decoder only has the final encoder hidden state (a lossy summary). With it, the decoder can selectively retrieve spatial features from any of the 10 input timesteps - e.g., attending heavily to a timestep where a flux rope was forming.

---

## 6. Self-Attention Memory / SA-ConvLSTM (ARCH-01)

> An enhanced ConvLSTM cell that maintains a separate attention-refined memory state M
> alongside the standard LSTM states (h, c). Uses **channel attention** (not spatial attention)
> to keep computation tractable at the model's spatial resolution.

```mermaid
flowchart TD
    subgraph STANDARD["Standard ConvLSTM Cell (runs first)"]
        X_IN["Input: x_t (B, in, H, W)"] --> CELL["ConvLSTMCell<br/>Concat [x, h_prev] -> Conv2d -> 4 gates<br/>See Diagram 8 for gate details"]
        H_PREV["h_prev (B, hid, H, W)"] --> CELL
        C_PREV["c_prev (B, hid, H, W)"] --> CELL
        CELL --> H_RAW["h (B, hid, H, W)<br/>Raw hidden state"]
        CELL --> C_OUT["c (B, hid, H, W)<br/>Cell state (passed through)"]
    end

    subgraph SAM["Self-Attention Memory Module"]
        direction TB

        subgraph SELF_ATTN["Self-Attention on Hidden State h"]
            H_RAW --> QH["Q_h = Conv2d(hid, hid/2, k=1)"]
            H_RAW --> KH["K_h = Conv2d(hid, hid/2, k=1)"]
            H_RAW --> VH["V_h = Conv2d(hid, hid/2, k=1)"]

            QH --> QH_POOL["Global Avg Pool -> (B, hid/2)"]
            KH --> KH_POOL["Global Avg Pool -> (B, hid/2)"]

            QH_POOL --> CH_ATTN_H["Channel Attention:<br/>outer_product(q, k) * scale<br/>softmax -> (B, hid/2, hid/2)<br/>Captures channel interdependencies"]
            KH_POOL --> CH_ATTN_H

            CH_ATTN_H --> ZH["z_h = bmm(attn, V_h_flat)<br/>Reshape to (B, hid/2, H, W)<br/>Self-refined features"]
            VH --> ZH
        end

        subgraph CROSS_ATTN["Cross-Attention with Memory M"]
            M_PREV["M_prev (B, hid, H, W)<br/>Previous memory state"] --> KM["K_m = Conv2d(hid, hid/2, k=1)"]
            M_PREV --> VM["V_m = Conv2d(hid, hid/2, k=1)"]

            KM --> KM_POOL["Global Avg Pool -> (B, hid/2)"]

            QH_POOL --> CH_ATTN_M["Channel Attention:<br/>outer_product(q_h, k_m) * scale<br/>softmax -> (B, hid/2, hid/2)<br/>How h relates to stored memory"]
            KM_POOL --> CH_ATTN_M

            CH_ATTN_M --> ZM["z_m = bmm(attn, V_m_flat)<br/>Reshape to (B, hid/2, H, W)<br/>Memory-refined features"]
            VM --> ZM
        end

        subgraph GATE_FUSE["Gated Fusion"]
            ZH --> CAT["Concatenate z_h, z_m<br/>(B, hid, H, W)"]
            ZM --> CAT
            CAT --> GATE["gate = sigmoid(Conv2d(hid, hid/2, k=1))<br/>Learned gate: how much from self<br/>vs how much from memory"]
            GATE --> FUSED["z_fused = gate * z_h + (1-gate) * z_m<br/>Blended representation"]
        end

        subgraph OUTPUTS["SA-ConvLSTM Outputs"]
            FUSED --> H_OUT["h_out = h + out_proj(z_fused)<br/>Residual: original h + attention refinement<br/>(B, hid, H, W)"]
            FUSED --> M_OUT["M_new = mem_proj(z_fused)<br/>Updated memory state<br/>(B, hid, H, W)"]
        end
    end

    H_OUT --> NEXT_H["Output: h_out, c, M_new<br/>3-tuple (vs 2-tuple for standard)"]
    C_OUT --> NEXT_H
    M_OUT --> NEXT_H

    style STANDARD fill:#1b2d4a,stroke:#3a5a8a,color:#c0d4f0
    style SAM fill:#2e1a2e,stroke:#6e4a6e,color:#e0c0e0
    style SELF_ATTN fill:#3d1b4a,stroke:#6a3a8a,color:#d8c0f0
    style CROSS_ATTN fill:#4a1b3d,stroke:#8a3a6a,color:#f0c0e0
    style GATE_FUSE fill:#2d2d1b,stroke:#5a5a3a,color:#e0e0c0
    style OUTPUTS fill:#1b4a2d,stroke:#3a8a5a,color:#c0f0d4
```

### SA-ConvLSTM Equations

Given hidden state $h \in \mathbb{R}^{B \times C \times H \times W}$ and previous memory $M_{t-1} \in \mathbb{R}^{B \times C \times H \times W}$:

**Step 1: Standard ConvLSTM** (produces raw $h$, $c$):

$$h, c = \text{ConvLSTMCell}(x_t, h_{t-1}, c_{t-1})$$

**Step 2: Self-Attention on $h$** (channel attention, NOT spatial):

$$Q_h = W_Q \ast h, \quad K_h = W_K \ast h, \quad V_h = W_V \ast h \quad \text{(1x1 Conv2d, } C \to C/2\text{)}$$

$$\bar{q}_h = \text{GAP}(Q_h) \in \mathbb{R}^{B \times C/2}, \quad \bar{k}_h = \text{GAP}(K_h) \in \mathbb{R}^{B \times C/2}$$

$$A_h = \text{softmax}\left(\bar{q}_h \cdot \bar{k}_h^\top \cdot \frac{1}{\sqrt{C/2}}\right) \in \mathbb{R}^{B \times C/2 \times C/2}$$

$$z_h = A_h \cdot \text{reshape}(V_h) \in \mathbb{R}^{B \times C/2 \times H \times W}$$

**Step 3: Cross-Attention with Memory $M_{t-1}$** (same query $\bar{q}_h$, keys/values from $M$):

$$K_m = W_{K_m} \ast M_{t-1}, \quad V_m = W_{V_m} \ast M_{t-1}$$

$$A_m = \text{softmax}\left(\bar{q}_h \cdot \text{GAP}(K_m)^\top \cdot \frac{1}{\sqrt{C/2}}\right) \in \mathbb{R}^{B \times C/2 \times C/2}$$

$$z_m = A_m \cdot \text{reshape}(V_m) \in \mathbb{R}^{B \times C/2 \times H \times W}$$

**Step 4: Gated Fusion** (learned gate balances self vs memory):

$$g = \sigma\left(W_g \ast [z_h \| z_m]\right) \in [0, 1]^{B \times C/2 \times H \times W}$$

$$z_{fused} = g \odot z_h + (1 - g) \odot z_m$$

**Step 5: Output** (residual connection on $h$, fresh projection for $M$):

$$h_{out} = h + W_{out} \ast z_{fused}, \quad M_t = W_{mem} \ast z_{fused}$$

Where GAP = Global Average Pooling, $\sigma$ = sigmoid, $\|$ = channel concatenation, $\odot$ = element-wise multiply.

**Why channel attention instead of spatial?** At the latent resolution (110x221), spatial attention would create 24,310 x 24,310 = ~591M element attention matrices per sample. Channel attention operates on $C/2$ channels (16-64), producing tiny matrices (e.g. 32x32 = 1,024 elements) that are trivially cheap.

**Why a separate memory M?** The LSTM cell state $c$ is tightly coupled to the forget/input gating mechanism. Adding a separate memory $M$ gives the model an independent long-term storage that is updated purely through attention, allowing it to retain patterns across many timesteps without gate interference.

---

## 7. Attention Gate on Skip Connections (ARCH-03)

> Based on Attention U-Net (Oktay et al., 2018). Produces a spatial attention mask
> that filters the encoder's skip features before they reach the decoder.

```mermaid
flowchart TD
    subgraph INPUTS_AG["Inputs"]
        G["Gating signal g<br/>from decoder upsample<br/>(B, 32, H/2, W/2)<br/>Decoder knows WHAT it needs"]
        X["Skip features x<br/>from encoder ConvLSTM-1<br/>(B, 16, H/2, W/2)<br/>Encoder has fine spatial detail"]
    end

    subgraph GATE_COMPUTE["Attention Gate Computation"]
        G --> WG["W_g = Conv2d(32, 8, k=1)<br/>Project decoder signal<br/>(B, 8, H/2, W/2)"]
        X --> WX["W_x = Conv2d(16, 8, k=1)<br/>Project encoder features<br/>(B, 8, H/2, W/2)"]

        WG --> ADDITION["Element-wise addition<br/>W_g(g) + W_x(x)<br/>Combine decoder intent<br/>with encoder content"]
        WX --> ADDITION

        ADDITION --> RELU["ReLU activation<br/>Suppress mismatched regions<br/>where decoder and encoder disagree"]

        RELU --> PSI["psi = Conv2d(8, 1, k=1)<br/>Collapse to single channel<br/>(B, 1, H/2, W/2)"]

        PSI --> SIGMOID["sigmoid(psi)<br/>Spatial attention mask<br/>alpha in [0, 1] per pixel"]
    end

    subgraph APPLY["Apply Gate"]
        SIGMOID --> MULTIPLY["gated_x = x * alpha<br/>Element-wise multiply<br/>Suppress irrelevant regions<br/>Amplify relevant features"]
        X --> MULTIPLY

        MULTIPLY --> OUT_AG["Output: gated skip features<br/>(B, 16, H/2, W/2)<br/>Only relevant details pass through"]
    end

    style INPUTS_AG fill:#1a1a2e,stroke:#4a4a6e,color:#c0c0e0
    style GATE_COMPUTE fill:#3d1b4a,stroke:#6a3a8a,color:#d8c0f0
    style APPLY fill:#2d4a1b,stroke:#5a8a3a,color:#d4f0c0
```

### Attention Gate Equations

Given decoder gating signal $g \in \mathbb{R}^{B \times C_d \times H \times W}$ and encoder skip features $x \in \mathbb{R}^{B \times C_e \times H \times W}$:

$$\psi = \sigma\left(W_\psi \ast \text{ReLU}\left(W_g \ast g + W_x \ast x\right)\right) \in [0, 1]^{B \times 1 \times H \times W}$$

$$\hat{x} = \psi \odot x$$

Where $W_g: C_d \to F_{int}$, $W_x: C_e \to F_{int}$, $W_\psi: F_{int} \to 1$ are 1x1 Conv2d projections, $F_{int} = \max(C_e / 2, 8) = 8$, $\sigma$ is sigmoid, and $\odot$ is element-wise multiplication. The output $\hat{x}$ has the same shape as $x$ but with irrelevant spatial regions suppressed toward zero.

**Why gate skip connections?** Without gating, the skip connection dumps ALL encoder features into the decoder equally. Most of these features are background solar surface with no flare activity. The attention gate lets the decoder signal which spatial regions are important for the current prediction, suppressing background noise and focusing reconstruction effort on active regions.

---

## 8. ConvLSTM Cell Internals

> The fundamental building block. Replaces LSTM's matrix multiplications with 2D convolutions
> so spatial structure (H, W) is preserved while temporal dynamics are modeled.

```mermaid
flowchart TD
    subgraph INPUTS_CELL["Inputs at Timestep t"]
        XT["x_t: Input frame<br/>(B, input_dim, H, W)"]
        HT["h_(t-1): Previous hidden<br/>(B, hidden_dim, H, W)"]
        CT["c_(t-1): Previous cell<br/>(B, hidden_dim, H, W)"]
    end

    subgraph GATES["Gate Computation (Single Conv2d)"]
        XT --> CONCAT_CELL["Concatenate along channels<br/>[x_t, h_(t-1)]<br/>(B, input+hidden, H, W)"]
        HT --> CONCAT_CELL

        CONCAT_CELL --> CONV_CELL["Single Conv2d<br/>(input+hidden) -> 4*hidden<br/>kernel_size=5, padding=2<br/>Computes ALL 4 gates at once"]

        CONV_CELL --> SPLIT_CELL["Split output into 4 equal chunks"]

        SPLIT_CELL --> I_GATE["Input Gate (i)<br/>sigmoid -> [0, 1]<br/>Controls: how much NEW<br/>information to write"]
        SPLIT_CELL --> F_GATE["Forget Gate (f)<br/>sigmoid -> [0, 1]<br/>bias initialized to 1.0<br/>Controls: how much OLD<br/>information to retain"]
        SPLIT_CELL --> G_GATE["Cell Gate (g)<br/>tanh -> [-1, +1]<br/>Candidate new values<br/>to potentially store"]
        SPLIT_CELL --> O_GATE["Output Gate (o)<br/>sigmoid -> [0, 1]<br/>Controls: what to expose<br/>to the next layer"]
    end

    subgraph UPDATE["State Update"]
        CT --> FORGET["f * c_(t-1)<br/>Selectively forget"]
        F_GATE --> FORGET

        I_GATE --> WRITE["i * g<br/>Selectively write"]
        G_GATE --> WRITE

        FORGET --> C_NEXT["c_t = f*c_(t-1) + i*g<br/>New cell state:<br/>filtered old memory<br/>+ gated new candidates"]
        WRITE --> C_NEXT

        C_NEXT --> TANH_C["tanh(c_t)<br/>Normalize cell state"]
        O_GATE --> H_NEXT["h_t = o * tanh(c_t)<br/>New hidden state:<br/>gated view of cell"]
        TANH_C --> H_NEXT
    end

    subgraph OUTPUTS_CELL["Outputs at Timestep t"]
        H_NEXT --> OUT_H["h_t: Hidden state<br/>(B, hidden_dim, H, W)<br/>Passed to next layer AND<br/>next timestep"]
        C_NEXT --> OUT_C["c_t: Cell state<br/>(B, hidden_dim, H, W)<br/>Internal memory, only<br/>passed to next timestep"]
    end

    style INPUTS_CELL fill:#1a1a2e,stroke:#4a4a6e,color:#c0c0e0
    style GATES fill:#1b2d4a,stroke:#3a5a8a,color:#c0d4f0
    style UPDATE fill:#2e1a2e,stroke:#6e4a6e,color:#e0c0e0
    style OUTPUTS_CELL fill:#2d4a1b,stroke:#5a8a3a,color:#d4f0c0
```

### ConvLSTM Cell Equations

Given input $x_t \in \mathbb{R}^{B \times C_{in} \times H \times W}$, previous hidden state $h_{t-1} \in \mathbb{R}^{B \times C_h \times H \times W}$, and previous cell state $c_{t-1} \in \mathbb{R}^{B \times C_h \times H \times W}$:

**Single convolution computes all four gates:**

$$[i \; f \; g \; o] = \text{Conv2d}_{k \times k}\left([x_t \| h_{t-1}]\right) \in \mathbb{R}^{B \times 4C_h \times H \times W}$$

Where $\|$ denotes channel-wise concatenation, input channels = $C_{in} + C_h$, and $k = 5$.

**Gate activations:**

$$i_t = \sigma\left(W_i \ast [x_t \| h_{t-1}]\right) \quad \text{(Input gate: what to write)}$$

$$f_t = \sigma\left(W_f \ast [x_t \| h_{t-1}] + b_f\right) \quad \text{(Forget gate: what to keep, } b_f = 1.0\text{)}$$

$$g_t = \tanh\left(W_g \ast [x_t \| h_{t-1}]\right) \quad \text{(Cell gate: candidate values)}$$

$$o_t = \sigma\left(W_o \ast [x_t \| h_{t-1}]\right) \quad \text{(Output gate: what to expose)}$$

**State updates:**

$$c_t = f_t \odot c_{t-1} + i_t \odot g_t$$

$$h_t = o_t \odot \tanh(c_t)$$

Where $\sigma$ is sigmoid, $\odot$ is element-wise (Hadamard) product, and $\ast$ denotes 2D convolution. All operations preserve the spatial dimensions $(H, W)$.

**Why Conv instead of Dense?** A standard LSTM flattens the 2D image into a 1D vector, destroying spatial relationships. ConvLSTM applies 2D convolutions, so the gate computations happen locally at each spatial position. A pixel's input gate is influenced by its 5x5 neighborhood, naturally modeling local magnetic field interactions.

**Why $b_f = 1.0$?** Initializing the forget gate bias to 1.0 means $f_t$ starts near $\sigma(1) \approx 0.73$, biased toward remembering. This prevents the common failure mode of LSTMs forgetting everything early in training before useful gradients arrive.

---

## 9. Composite Loss Function

> 6 loss components, each targeting a different failure mode. The total is their weighted sum.

```mermaid
flowchart TD
    PRED["Predictions (B, 1, 4, H, W)"] --> PHASE1
    TARGET["Ground Truth (B, 1, 4, H, W)"] --> PHASE1

    subgraph PHASE1["Phase 1: Temporal Losses (on 5D tensor)"]
        TW_L1["Timestep-Weighted L1<br/>weight: 1.0<br/>Per-element |pred - target|<br/>weighted [1.0, 1.5, 2.0, 2.5] per step<br/>Later frames penalized more"]

        TD_LOSS["Temporal Difference<br/>weight: 1.0<br/>L1 on (frame[t] - frame[t-1])<br/>pred vs target diff sequences<br/>Matches rate of change"]

        TV_LOSS["Temporal Variance Penalty<br/>lambda: 0.1<br/>-lambda * min(pred_var, target_var)<br/>NEGATIVE: rewards dynamics up to<br/>target level, caps to prevent jitter"]
    end

    PRED --> PHASE2
    TARGET --> PHASE2

    subgraph PHASE2["Phase 2: Spatial Losses (on 4D, flattened across time)"]
        SSIM_L["MS-SSIM Loss<br/>weight: 0.3<br/>1 - MultiScaleSSIM(pred, target)<br/>5 scales: [0.04, 0.29, 0.30, 0.24, 0.13]<br/>Preserves edges and structure"]

        EXT_L["Weighted MAE (Extreme Focus)<br/>weight: 3.0<br/>Normal pixels: weight=1.0<br/>Extreme pixels (|target|>0.277): weight=3.0<br/>Binary threshold, not gradient"]

        ASYM_L["Asymmetric Extreme<br/>weight: 0.5<br/>Above threshold:<br/>underestimate penalty = 2x overestimate<br/>Below threshold: symmetric MAE"]
    end

    TW_L1 --> TOTAL
    TD_LOSS --> TOTAL
    TV_LOSS --> TOTAL
    SSIM_L --> TOTAL
    EXT_L --> TOTAL
    ASYM_L --> TOTAL

    TOTAL["Total = 1.0*L1 + 0.3*SSIM + 3.0*Extreme<br/>+ 1.0*TempDiff + TempVar<br/>+ 0.5*Asymmetric"]

    TOTAL --> BACKWARD["Backpropagation<br/>Gradient clipping: max_norm=1.0<br/>Optional AMP (mixed precision)<br/>NaN/Inf detection: skip batch if found"]

    style PHASE1 fill:#1b2d4a,stroke:#3a5a8a,color:#c0d4f0
    style PHASE2 fill:#4a1b2d,stroke:#8a3a5a,color:#f0c0d4
    style TOTAL fill:#2d4a1b,stroke:#5a8a3a,color:#d4f0c0
```

### Loss Function Equations

**1. Timestep-Weighted L1 (MAE):**

$$\mathcal{L}_{L1} = \frac{1}{BTHW} \sum_{b,t,h,w} w_t \cdot |\hat{y}_{b,t,h,w} - y_{b,t,h,w}|$$

Where $w_t = [1.0, 1.5, 2.0, 2.5]$ are per-timestep weights (later predictions penalized more).

**2. Multi-Scale SSIM:**

$$\mathcal{L}_{SSIM} = 1 - \prod_{s=1}^{5} \text{SSIM}_s(\hat{y}, y)^{\beta_s}$$

$$\text{SSIM}(x, y) = \frac{(2\mu_x\mu_y + C_1)(2\sigma_{xy} + C_2)}{(\mu_x^2 + \mu_y^2 + C_1)(\sigma_x^2 + \sigma_y^2 + C_2)}$$

Where $C_1 = (0.01 \cdot R)^2$, $C_2 = (0.03 \cdot R)^2$, $R = 2.0$ (data range), $\beta = [0.0448, 0.2856, 0.3001, 0.2363, 0.1333]$. Statistics computed via Gaussian-windowed Conv2d ($k=11$, $\sigma=1.5$).

**3. Weighted MAE (Extreme Focus):**

$$\mathcal{L}_{extreme} = \frac{1}{N} \sum_{i} w_i \cdot |\hat{y}_i - y_i|, \quad w_i = \begin{cases} 3.0 & \text{if } |y_i| > \tau \\ 1.0 & \text{otherwise} \end{cases}$$

Where $\tau = 0.277$ (extreme threshold in normalized space).

**4. Temporal Difference:**

$$\mathcal{L}_{td} = \frac{1}{(T-1)HW} \sum_{t=2}^{T} \left| (\hat{y}_t - \hat{y}_{t-1}) - (y_t - y_{t-1}) \right|$$

**5. Temporal Variance Penalty:**

$$\mathcal{L}_{tv} = -\lambda \cdot \min\left(\text{Var}_{pred}, \text{Var}_{target}\right)$$

$$\text{Var} = \frac{1}{(T-1)HW} \sum_{t=2}^{T} |y_t - y_{t-1}|$$

Where $\lambda = 0.1$. The negative sign rewards variation; capping at target variance prevents rewarding jitter.

**6. Asymmetric Extreme:**

$$\mathcal{L}_{asym} = \frac{1}{N} \sum_{i} \begin{cases} \alpha \cdot (y_i - \hat{y}_i)^+ + ({\hat{y}_i - y_i})^+ & \text{if } |y_i| > \tau \\ |\hat{y}_i - y_i| & \text{otherwise} \end{cases}$$

Where $\alpha = 2.0$ (underestimation penalty multiplier), $(x)^+ = \max(0, x)$.

**Total Composite Loss:**

$$\mathcal{L} = 1.0 \cdot \mathcal{L}_{L1} + 0.3 \cdot \mathcal{L}_{SSIM} + 3.0 \cdot \mathcal{L}_{extreme} + 1.0 \cdot \mathcal{L}_{td} + \mathcal{L}_{tv} + 0.5 \cdot \mathcal{L}_{asym}$$

**Why 6 losses?** Each addresses a specific failure mode:
- **L1 alone** produces blurry averages (minimizes pixel error by predicting the mean)
- **SSIM** forces structural preservation (edges, contrast, luminance)
- **Extreme weight** prevents the model from ignoring rare flare pixels (which are <1% of area)
- **Temporal diff** prevents static predictions that just copy the last frame
- **Temporal var** actively rewards dynamics (L1+SSIM can still prefer frozen frames)
- **Asymmetric** encodes the domain truth: missing a flare is worse than a false alarm

---

## 10. Training Loop & Evaluation

> The full training cycle including teacher forcing schedule and evaluation metrics.

```mermaid
flowchart TD
    subgraph TRAIN["Training Phase"]
        EPOCH["For each epoch (1 to 50+)"]
        EPOCH --> TF_SCHED["Compute teacher forcing ratio<br/>tf = max(0, 0.5 * (1 - epoch/20))<br/>Epoch 1: 50% ground truth fed<br/>Epoch 20+: 0% (fully autoregressive)"]

        TF_SCHED --> BATCH["For each batch:"]
        BATCH --> FWD["Forward pass with teacher forcing<br/>AMP autocast if enabled"]
        FWD --> LOSS_COMP["Compute composite loss<br/>+ per-component tracking"]
        LOSS_COMP --> NAN{"NaN or Inf<br/>in loss?"}
        NAN -->|Yes| SKIP["Skip batch, log warning<br/>Zero gradients, continue"]
        NAN -->|No| BACKWARD["Backward pass<br/>GradScaler if AMP"]
        BACKWARD --> CLIP["Gradient clipping<br/>max_norm = 1.0"]
        CLIP --> OPTIM["Optimizer step (AdamW)<br/>Update model weights"]
        OPTIM --> BATCH
    end

    TRAIN --> VAL

    subgraph VAL["Validation Phase (every epoch)"]
        EVAL_MODE["model.eval(), torch.no_grad()"]
        EVAL_MODE --> PRED_VAL["Run predictions on val set<br/>No teacher forcing, no dropout"]
        PRED_VAL --> METRICS

        subgraph METRICS["Evaluation Metrics"]
            MAE["MAE per timestep<br/>mean(|pred - target|) at t=1,2,3,4<br/>Shows error growth over horizon"]
            RMSE["RMSE per timestep<br/>sqrt(mean((pred-target)^2))<br/>Penalizes large errors more"]
            CSI_M["CSI (Critical Success Index)<br/>TP / (TP + FP + FN)<br/>Flare detection skill:<br/>correct detections vs total events"]
            HSS_M["HSS (Heidke Skill Score)<br/>Measures skill beyond random chance<br/>HSS=0: no skill, HSS=1: perfect"]
            SSIM_M["SSIM per timestep<br/>Structural similarity index<br/>Measures edge/texture preservation"]
            PERSIST["Persistence Skill Score<br/>1 - (MAE_model / MAE_persist)<br/>persist = repeat last frame forever<br/>Positive = model beats naive baseline"]
            TVR["Temporal Variation Ratio<br/>pred_variation / target_variation<br/>1.0 = realistic dynamics<br/>0.0 = frozen predictions"]
        end
    end

    VAL --> BEST{"val_loss improved?"}
    BEST -->|Yes| SAVE["Save checkpoint<br/>best_model.pt<br/>model + optimizer + epoch + metrics"]
    BEST -->|No| CONTINUE["Continue to next epoch"]
    SAVE --> CONTINUE

    CONTINUE --> FINAL["After all epochs:<br/>Load best checkpoint<br/>Run final test evaluation<br/>Generate prediction visualizations"]

    style TRAIN fill:#1a2e1a,stroke:#4a6e4a,color:#c0e0c0
    style VAL fill:#1a1a2e,stroke:#4a4a6e,color:#c0c0e0
    style METRICS fill:#2e2e1a,stroke:#6e6e4a,color:#e0e0c0
```

### Teacher Forcing Schedule

$$p_{tf}(e) = \max\left(0, \; 0.5 \cdot \left(1 - \frac{e}{E_{decay}}\right)\right)$$

Where $e$ is the current epoch, $E_{decay} = 20$ is the decay horizon. At each decoder step, with probability $p_{tf}$ the ground truth frame is fed as input instead of the model's own prediction. This decays linearly from 50% to 0% over the first 20 epochs, after which the decoder runs fully autoregressively.

### Evaluation Metric Equations

**MAE** (Mean Absolute Error, per timestep):

$$\text{MAE}_t = \frac{1}{BHW}\sum_{b,h,w} |\hat{y}_{b,t,h,w} - y_{b,t,h,w}|$$

**RMSE** (Root Mean Square Error):

$$\text{RMSE}_t = \sqrt{\frac{1}{BHW}\sum_{b,h,w} (\hat{y}_{b,t,h,w} - y_{b,t,h,w})^2}$$

**CSI** (Critical Success Index) for flare detection at threshold $\tau$:

$$\text{CSI} = \frac{TP}{TP + FP + FN}$$

Where $TP$ = correctly predicted extreme pixels, $FP$ = false alarms, $FN$ = missed extremes.

**HSS** (Heidke Skill Score) - skill relative to random chance:

$$\text{HSS} = \frac{2(TP \cdot TN - FP \cdot FN)}{(TP+FN)(FN+TN) + (TP+FP)(FP+TN)}$$

$\text{HSS} = 0$: no skill (random), $\text{HSS} = 1$: perfect.

**Persistence Skill Score** - improvement over naive "repeat last frame" baseline:

$$\text{PSS} = 1 - \frac{\text{MAE}_{model}}{\text{MAE}_{persistence}}, \quad \text{where } \hat{y}^{persist}_t = x_{T_{in}} \;\forall\; t$$

$\text{PSS} > 0$: model is better than persistence. $\text{PSS} < 0$: model is worse.

**Temporal Variation Ratio:**

$$\text{TVR} = \frac{\sum_t |\hat{y}_t - \hat{y}_{t-1}|}{\sum_t |y_t - y_{t-1}|}$$

$\text{TVR} = 1.0$: realistic dynamics. $\text{TVR} \to 0$: static/frozen predictions.

---

## 11. Module Dependency Map

> How the source files connect. Arrows show import/usage dependencies.

```mermaid
flowchart TD
    MAIN["main.py<br/>Entry point, orchestrates<br/>load -> train -> evaluate"]

    MAIN --> CONFIG["config.yaml<br/>All hyperparameters:<br/>model, data, training, loss"]

    MAIN --> LOADER["solarflare_data/loader.py<br/>load_and_prepare_data()<br/>load_preprocessed_data()<br/>Normalization, splitting"]

    MAIN --> TRAINER["training/trainer.py<br/>train_epoch(), validate()<br/>train_model() main loop"]

    LOADER --> DATASET["solarflare_data/dataset.py<br/>SolarFluxDataset<br/>__getitem__: lazy mmap + augment<br/>+ extreme channel computation"]

    TRAINER --> PREDICTOR["models/predictor.py<br/>SolarFluxPredictor<br/>Encoder-decoder forward()"]

    TRAINER --> LOSSES["training/losses.py<br/>CompositeLoss (6 components)<br/>WeightedMAE, AsymmetricExtreme<br/>ssim(), ms_ssim()"]

    TRAINER --> METR["utils/metrics.py<br/>compute_metrics, CSI, HSS<br/>RMSE, SSIM, persistence<br/>temporal variation ratio"]

    TRAINER --> VIS["utils/visualization.py<br/>plot_training_history()<br/>visualize_predictions()<br/>visualize_with_uncertainty()"]

    TRAINER --> CKPT["utils/checkpoint.py<br/>save_checkpoint()<br/>load_checkpoint()"]

    PREDICTOR --> CLSTM["models/convlstm.py<br/>ConvLSTMCell: single timestep<br/>ConvLSTM: sequence wrapper"]

    PREDICTOR --> SACLSTM["models/sa_convlstm.py<br/>SelfAttentionMemory<br/>SAConvLSTMCell: cell + SAM<br/>SAConvLSTM: sequence wrapper"]

    PREDICTOR --> ATTN["models/attention.py<br/>TemporalAttention: Q/K/V over time<br/>AttentionGate: U-Net skip gating"]

    TRAINER --> DEVICE["utils/device.py<br/>resolve_device(): CUDA/MPS/CPU<br/>get_amp_context(): autocast<br/>get_grad_scaler(): AMP scaling"]

    LOSSES --> MPS_OPS["utils/mps_ops.py<br/>safe_outer(): MPS-compatible<br/>torch.outer replacement"]

    MAIN --> OUTPUTS["outputs/<br/>best_model.pt, metadata.json<br/>training_history.json/.png<br/>test_results.json, predictions.png"]

    style MAIN fill:#4a1b4a,stroke:#8a3a8a,color:#f0c0f0
    style PREDICTOR fill:#1b4a4a,stroke:#3a8a8a,color:#c0f0f0
    style CLSTM fill:#1b3a4a,stroke:#3a6a8a,color:#c0d8f0
    style SACLSTM fill:#1b3a4a,stroke:#3a6a8a,color:#c0d8f0
    style ATTN fill:#3d1b4a,stroke:#6a3a8a,color:#d8c0f0
    style TRAINER fill:#1b4a1b,stroke:#3a8a3a,color:#c0f0c0
    style LOSSES fill:#4a1b1b,stroke:#8a3a3a,color:#f0c0c0
    style LOADER fill:#4a4a1b,stroke:#8a8a3a,color:#f0f0c0
    style DATASET fill:#4a4a1b,stroke:#8a8a3a,color:#f0f0c0
```

---

## 12. Design Decision Tree

> Traces from the core problem to each architectural choice, explaining **why**.

```mermaid
flowchart TD
    PROBLEM["PROBLEM: Predict solar magnetic<br/>flux evolution in 2D space over time"]

    PROBLEM --> SPATIAL{"Spatial patterns?<br/>Sunspots, flux ropes,<br/>active regions"}
    SPATIAL -->|"Yes: local neighborhood<br/>interactions matter"| CNN["Use 2D Convolutions<br/>Local receptive field<br/>Translation equivariant"]

    PROBLEM --> TEMPORAL{"Temporal dynamics?<br/>Evolution over<br/>10 timesteps"}
    TEMPORAL -->|"Yes: need sequential<br/>memory and gating"| RNN["Use LSTM Recurrence<br/>Forget/input/output gates<br/>Selective memory"]

    CNN --> CONVLSTM["CONVLSTM<br/>Conv inside LSTM gates<br/>Spatial + temporal in one cell"]
    RNN --> CONVLSTM

    CONVLSTM --> ENCODE{"How to compress<br/>spatiotemporal input?"}
    ENCODE --> ENCODER["3-Layer Encoder<br/>Progressive downsampling<br/>16ch -> 32ch -> 64ch"]

    CONVLSTM --> DECODE{"How to predict<br/>multiple future frames?"}
    DECODE --> AUTOREGRESSIVE["Autoregressive Decoder<br/>Each prediction feeds next step<br/>Maintains physical consistency<br/>between consecutive frames"]

    AUTOREGRESSIVE --> STABILITY{"How to stabilize<br/>autoregressive training?"}
    STABILITY --> RESIDUAL_D["Residual Prediction<br/>pred = input + small_delta<br/>Easier to learn, smaller gradients"]
    STABILITY --> TF_D["Teacher Forcing Schedule<br/>50% ground truth at start<br/>Decays to 0% by epoch 20<br/>Curriculum: easy then hard"]
    STABILITY --> GRAD_CLIP["Gradient Clipping (norm 1.0)<br/>Prevents exploding gradients<br/>from long backprop chains"]

    PROBLEM --> RANGE{"Extreme dynamic range?<br/>Flux: -36M to +70M<br/>Flares are rare outliers"}
    RANGE --> ASINH_D["Asinh Normalization<br/>Linear below softening (1000)<br/>Logarithmic above softening<br/>No clipping, symmetric +/-"]
    RANGE --> DUAL_CH["Dual Channel Input<br/>Ch1: normalized flux<br/>Ch2: soft extreme indicator<br/>Explicit flare attention signal"]

    PROBLEM --> RARE{"How to learn<br/>rare flare events?<br/>(<1% of pixels)"}
    RARE --> EXTREME_LOSS["Extreme-Weighted MAE<br/>3x penalty for |target|>0.277"]
    RARE --> ASYM_LOSS["Asymmetric Loss<br/>2x for underestimation<br/>Missing flares is worse"]
    RARE --> OVERSAMPLE["3x Flare Oversampling<br/>Sequences with flares appear<br/>3x more often in training"]

    PROBLEM --> BLUR{"How to prevent<br/>blurry predictions?<br/>(L1 favors mean)"}
    BLUR --> SSIM_LOSS["SSIM Loss<br/>Preserves edges, contrast,<br/>structural patterns"]
    BLUR --> TEMPVAR["Temporal Variance Penalty<br/>Actively rewards dynamics<br/>Capped at target level"]

    PROBLEM --> DETAIL{"How to preserve<br/>fine spatial detail<br/>through bottleneck?"}
    DETAIL --> SKIP_CONN["Skip Connections<br/>h1_skip from encoder to decoder<br/>Bypasses information bottleneck"]
    DETAIL --> ATTN_GATE_D["Attention Gate<br/>Decoder-guided spatial filtering<br/>Only passes relevant features"]

    PROBLEM --> MEMORY{"How to access<br/>full encoder history<br/>during decoding?"}
    MEMORY --> TEMP_ATTN_D["Temporal Attention<br/>Decoder queries all 10 encoder states<br/>Selective retrieval by relevance"]
    MEMORY --> SA_MEM["SA-ConvLSTM Memory<br/>Separate attention-refined M state<br/>Channel attention for efficiency"]

    style PROBLEM fill:#4a1b1b,stroke:#8a3a3a,color:#f0c0c0
    style CONVLSTM fill:#1b4a4a,stroke:#3a8a8a,color:#c0f0f0
    style RESIDUAL_D fill:#2d4a1b,stroke:#5a8a3a,color:#d4f0c0
    style ASINH_D fill:#2e1a2e,stroke:#6e4a6e,color:#e0c0e0
    style EXTREME_LOSS fill:#4a4a1b,stroke:#8a8a3a,color:#f0f0c0
    style TEMP_ATTN_D fill:#3d1b4a,stroke:#6a3a8a,color:#d8c0f0
```
