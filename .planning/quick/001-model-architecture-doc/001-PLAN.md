---
phase: 001-model-architecture-doc
plan: 01
type: execute
wave: 1
depends_on: []
files_modified:
  - docs/MODEL_ARCHITECTURE.md
autonomous: true
requirements: [DOC-01]

must_haves:
  truths:
    - "A physics professor with no ML background can read the document and understand what the model does, how data flows through it, and why each component exists"
    - "All five major pipeline stages are covered: data ingestion, normalization, model architecture, training, and inference"
    - "Mermaid diagrams render correctly on GitHub and illustrate the architecture visually"
    - "Physics analogies are used throughout to bridge ML concepts to familiar physics concepts"
    - "The planned improvements section faithfully represents all 23 items from IMPROVEMENT_NOTES.md"
  artifacts:
    - path: "docs/MODEL_ARCHITECTURE.md"
      provides: "Complete model architecture documentation with Mermaid diagrams"
      min_lines: 400
  key_links:
    - from: "docs/MODEL_ARCHITECTURE.md"
      to: ".planning/IMPROVEMENT_NOTES.md"
      via: "Improvement roadmap section"
      pattern: "Improvement|Roadmap|Phase [A-G]"
---

<objective>
Create a comprehensive, accessible model architecture document explaining the SolarFlare prediction pipeline end-to-end. The document targets a Physics Professor with no ML background and uses physics analogies throughout.

Purpose: Make the model's design decisions, data flow, and architecture understandable to domain scientists who will use and evaluate the predictions but have no deep learning experience.

Output: `docs/MODEL_ARCHITECTURE.md` -- a single self-contained document with Mermaid diagrams, physics analogies, and a planned improvements roadmap.
</objective>

<execution_context>
@/Users/manik/.claude/get-shit-done/workflows/execute-plan.md
@/Users/manik/.claude/get-shit-done/templates/summary.md
</execution_context>

<context>
@models/predictor.py
@models/convlstm.py
@solarflare_data/dataset.py
@solarflare_data/loader.py
@training/losses.py
@training/trainer.py
@config.yaml
@.planning/IMPROVEMENT_NOTES.md
</context>

<tasks>

<task type="auto">
  <name>Task 1: Write the complete MODEL_ARCHITECTURE.md document</name>
  <files>docs/MODEL_ARCHITECTURE.md</files>
  <action>
Create `docs/` directory if it does not exist. Write `docs/MODEL_ARCHITECTURE.md` with the following structure and content. The entire document must be written for a physics professor with NO machine learning background. Use physics analogies extensively (listed below for each section).

## Document Structure

### 1. Title and Overview (~30 lines)
- Title: "SolarFlare Model Architecture: A Complete Guide"
- One-paragraph executive summary: what the system does (predicts future solar magnetic winding flux maps from historical observations), what kind of model it is (a neural network that learns spatiotemporal dynamics), and the key result (4-step-ahead predictions at 4-6 hour cadence).
- A top-level Mermaid flowchart showing the full pipeline: Raw .npy Files -> Normalization -> Sliding Windows -> ConvLSTM Encoder -> ConvLSTM Decoder -> Residual Prediction -> Output Frames
- Physics analogy: "Think of this as a numerical weather model, but instead of solving PDEs on a grid, the model *learns* the evolution operator from data."

### 2. Data Pipeline (~80 lines)
Cover three sub-stages:

**2a. Raw Data Ingestion**
- Data format: structured .npy arrays with fields (X, Y, time, windTotal) representing winding flux on a spatial grid over time. Each file is one active region observation campaign.
- Conversion to dense (T, H, W) cubes via `_structured_to_cube()` in `loader.py`.
- Pre-flight validation: memory-mapped scan to catch corrupted files before training.
- Physics analogy: "Each .npy file is like a time-lapse movie of the magnetic winding flux across one active region, sampled every 4-6 hours."

**2b. Normalization (asinh transform)**
- Explain WHY normalization is needed: raw flux values span orders of magnitude (~-40k to +40k). Neural networks learn best when inputs are O(1).
- asinh transform: `normalized = arcsinh(raw / softening) / scale` where softening=1000. Explain that arcsinh behaves like log for large values but passes through zero smoothly (unlike log). Compresses dynamic range while preserving sign.
- Normalization computed from TRAINING data only (to prevent data leakage).
- Extreme threshold: 99.5th percentile of |flux| marks "extreme" regions.
- Include a Mermaid diagram showing the normalization pipeline.
- Physics analogy: "The asinh transform is analogous to a logarithmic detector response -- it compresses the dynamic range so that both quiet-sun regions and intense flare sites are represented on a comparable scale, much like a CCD with a logarithmic response curve."

**2c. Sliding Windows and Dual-Channel Input**
- Sliding window: from a cube of T timesteps, extract overlapping windows of (t_in + t_out) = 14 consecutive frames. Stride=1 for maximum data usage.
- Whole-file splitting: entire files go to train/val/test (70/20/10). Never split a file across sets (prevents temporal leakage).
- Dual-channel mode: Channel 1 = normalized flux. Channel 2 = extreme event indicator (soft sigmoid highlighting |flux| > threshold). This gives the model an explicit "attention hint" about where extreme activity is occurring.
- Augmentation: spatial flips (horizontal, vertical) to increase effective dataset 3x. Rotation-invariant because flux evolution does not depend on spatial orientation.
- Physics analogy: "The sliding window is like moving a temporal observation window across a long time series -- analogous to selecting 10-frame segments from a continuous solar observation for analysis. The dual channel is like giving the model both the raw magnetogram AND a binary mask highlighting strong-field regions."

### 3. Model Architecture (~150 lines)
This is the core section. Cover:

**3a. Overview: Encoder-Decoder Architecture**
- High-level concept: The encoder reads 10 input frames and compresses the spatiotemporal information into a latent representation (the model's "understanding" of the current state and recent dynamics). The decoder then generates 4 future frames one at a time.
- Mermaid diagram: Full encoder-decoder architecture showing:
  - Input (B, 2, 10, H, W) -> Input Downsampling -> Preprocessing Conv
  - Encoder path: ConvLSTM1 (c1) -> Spatial Downsample -> ConvLSTM2 (c2) -> ConvLSTM3 (c3)
  - Skip connection from ConvLSTM1 to decoder refinement
  - Decoder path (autoregressive loop): Input Conv -> Downsample -> ConvLSTM2 -> ConvLSTM3 -> Upsample -> Concat with Skip -> Refinement ConvLSTM -> Output Conv -> Residual Add
- Physics analogy: "The encoder-decoder structure is analogous to a two-step physical process: (1) analyze the current state by extracting the essential degrees of freedom (encoder = projection onto relevant modes), then (2) propagate forward in time using those modes (decoder = time evolution operator). The skip connection is like retaining high-frequency spatial detail that the coarse-grained representation would lose -- similar to how a multi-scale simulation preserves fine-scale features via a coupling term."

**3b. ConvLSTM Cell: The Core Building Block**
- Start with what a convolution does: a spatial filter that detects local patterns (like a finite-difference stencil, but with learned coefficients instead of prescribed ones). Kernel size 3 means each output pixel depends on a 3x3 neighborhood.
- Then LSTM concept: a recurrent unit with "memory" (cell state) that can selectively remember, forget, and output information across timesteps. Four gates:
  - Forget gate (f): "What fraction of the old memory to keep" -- analogous to exponential decay of a field.
  - Input gate (i): "What fraction of new information to store" -- analogous to a source term.
  - Cell gate (g): "What new information to compute" -- the actual new value.
  - Output gate (o): "What to expose as the current state" -- analogous to an observable vs. internal state.
- ConvLSTM = convolution + LSTM: gates are computed via 2D convolutions instead of matrix multiplications. This means the memory is a 2D spatial field, not a single vector.
- Cell state update: c_next = f * c_prev + i * g (linear combination = superposition of old and new states).
- Hidden state: h_next = o * tanh(c_next).
- Forget bias initialization to 1.0: ensures the network starts by remembering everything (stable initial dynamics).
- Include a Mermaid diagram showing data flow through one ConvLSTMCell.
- Physics analogy: "The cell state is like a 'memory field' -- a 2D field that evolves over time according to learned dynamics, similar to how a scalar field (e.g., temperature) evolves under a PDE. The gates are spatially-varying coefficients that control the evolution at each grid point independently. The convolution is essentially a local spatial operator (like a discrete Laplacian or gradient), but instead of having fixed stencil weights, the network learns the optimal stencil from data."

**3c. Channel Hierarchy and Spatial Downsampling**
- Channels [16, 32, 64]: increasing channels at lower spatial resolution. More channels = more "features" the model tracks at each spatial point.
- Spatial downsampling (stride-2 conv): reduces spatial resolution by 2x at each level. This creates a multi-scale representation.
- Physics analogy: "This is directly analogous to a multi-resolution analysis or wavelet decomposition. The first ConvLSTM operates at the full resolution capturing fine spatial structure. After downsampling, the second operates at half resolution capturing larger-scale dynamics. The deepest level captures the broadest spatial correlations with the most feature channels -- like going from local vorticity to synoptic-scale flow patterns."

**3d. Autoregressive Decoding**
- The decoder generates frames ONE AT A TIME. Each predicted frame becomes the input for predicting the next frame. This is called "autoregressive" decoding.
- Decoder state initialization: encoder's final hidden states are passed to the decoder (the decoder "inherits" the encoder's understanding of the input sequence).
- For each output step t:
  1. Take the last predicted frame (or the final input frame for t=0)
  2. Process through decoder ConvLSTMs (which have persistent state from previous steps)
  3. Upsample back to original resolution
  4. Concatenate with skip connection from encoder
  5. Refine through a final ConvLSTM
  6. Predict a RESIDUAL (delta) via output conv
  7. Add delta to previous frame: pred[t] = prev_frame + delta
- Physics analogy: "Autoregressive decoding is like a time-stepping scheme in numerical integration. Each step uses the result of the previous step as the initial condition -- exactly like a Runge-Kutta integrator where each step depends on the previous solution. The residual prediction (pred = prev + delta) is analogous to computing perturbations rather than absolute values -- a technique used throughout physics to improve numerical stability (e.g., perturbation theory, incremental stress formulations)."

**3e. Skip Connections**
- The encoder's first-layer hidden state is concatenated with the decoder's upsampled features before refinement.
- This preserves fine spatial details that would otherwise be lost in the encode-decode bottleneck.
- Physics analogy: "Skip connections serve the same purpose as correction terms in a coarse-grained simulation. The encoder's deep layers capture large-scale dynamics but lose spatial detail. The skip connection 'injects' the original high-resolution spatial structure back into the reconstruction -- analogous to subgrid-scale models in LES turbulence simulations that add back the effect of small scales."

**3f. Input Downsampling and Upsampling**
- Optional 2x downsampling at input (stride-2 conv) reduces spatial dims from HxW to H/2 x W/2 for all internal processing.
- Output head upsamples back to original resolution via transposed convolution.
- Trade-off: 4x fewer spatial computations at the cost of some spatial resolution in internal representations.
- Physics analogy: "This is equivalent to solving the dynamics on a coarser grid for computational efficiency, then interpolating back to the fine grid for output -- a standard technique in adaptive mesh refinement."

### 4. Training Process (~80 lines)

**4a. Composite Loss Function**
- Three terms combined: Total = 1.0 * L1 + 0.5 * (1 - MS-SSIM) + 1.0 * WeightedMAE
- L1 (Mean Absolute Error): pixel-by-pixel error. Simple, robust, penalizes all errors equally.
- MS-SSIM (Multi-Scale Structural Similarity): measures structural patterns (edges, textures) at multiple scales. Prevents blurry predictions by rewarding sharp structures.
- WeightedMAE: MAE with extra weight on high-flux regions. Pixels with larger |flux| get higher penalty. This counteracts the natural tendency of the model to focus on quiet-sun (which covers most of the image area).
- Include a Mermaid diagram showing the three loss components feeding into total loss.
- Physics analogy: "The composite loss is like a multi-objective cost function in optimization. L1 ensures global accuracy (like minimizing total energy error). MS-SSIM ensures structural fidelity (like preserving the topology of field lines). WeightedMAE ensures extreme regions are captured (like adding a constraint on peak field strength). No single metric captures all aspects of a good prediction."

**4b. Optimizer and Scheduling**
- AdamW optimizer: adaptive learning rate per parameter with weight decay (L2 regularization).
- Teacher forcing: during early training, the decoder sometimes receives the GROUND TRUTH previous frame instead of its own prediction. Ratio starts at 0.5 and decays linearly to 0.
- Physics analogy: "Teacher forcing is like training wheels. Early in training, the model's predictions are poor, so feeding them back would compound errors (like numerical instability in an explicit time-stepping scheme). By occasionally providing the correct answer, we stabilize early learning. As training progresses, we remove this support so the model learns to be robust to its own errors."

**4c. Training Logistics**
- Gradient clipping (max norm 0.5): prevents exploding gradients.
- NaN detection and abort: if 10 consecutive batches produce NaN loss, training aborts with an emergency checkpoint.
- Early stopping (patience=8): if validation loss doesn't improve for 8 epochs, stop.
- Checkpointing: best model and latest model saved. Emergency checkpoints on crash/interrupt.
- Graceful shutdown: SIGINT/SIGTERM caught; current epoch completes and checkpoint is saved before exit.

### 5. Inference and Residual Prediction (~40 lines)
- At inference time: teacher forcing = 0 (model uses only its own predictions).
- The model predicts RESIDUALS (changes) not absolute values. Final prediction: pred[t] = input_frame + cumulative_deltas.
- Multi-channel handling: model accepts 2-channel input (flux + extreme indicator) but outputs only 1 channel (flux). The extreme indicator for the predicted flux is recomputed for the next autoregressive step.
- Uncertainty quantification (MC Dropout): with dropout_rate > 0, running inference multiple times with dropout active produces a distribution of predictions. The spread indicates model uncertainty.
- Physics analogy: "Residual prediction is analogous to perturbation theory in physics: rather than computing the full solution from scratch at each timestep, we compute the deviation from the previous state. This is numerically more stable and leverages the fact that consecutive frames are highly correlated (the 'background' changes slowly). MC Dropout uncertainty is analogous to ensemble forecasting in weather prediction -- by introducing controlled stochastic perturbations, we sample the space of possible predictions to estimate confidence."

### 6. Current Configuration Summary (~30 lines)
A clean table summarizing all key hyperparameters from config.yaml:
- Data: t_in=10, t_out=4, dual_channel=true, augmentation=none, stride=1
- Normalization: asinh, softening=1000, extreme_percentile=99.5
- Model: channels=[16,32,64], kernel_size=3, downsample=true, dropout=0.0
- Training: batch_size=1, epochs=25, lr=1e-4, weight_decay=1e-5, tf_start=0.5, patience=8, grad_clip=0.5
- Loss: composite (L1=1.0, SSIM=0.5, extreme=1.0)

### 7. Planned Improvements (~100 lines)
Reproduce the full improvement roadmap from `.planning/IMPROVEMENT_NOTES.md`, organized by the 7 phases (A through G). For each improvement, include:
- What it changes
- Why it matters (1-2 sentences)
- Priority level

Present as a structured list with the 7 phase groupings. Include the priority order at the end.

### 8. Glossary (~20 lines)
Define key ML terms used in the document with physics analogies:
- Epoch, Batch, Gradient, Loss, Convolution, LSTM, Latent space, Skip connection, Residual, Teacher forcing, Early stopping

## Writing Style Guidelines
- Use active voice and direct language
- Define every ML term on first use with a physics analogy in parentheses
- Use "the model" not "our model"
- Avoid jargon without explanation
- Each Mermaid diagram should have a caption explaining what it shows
- Use footnotes or inline parenthetical explanations for mathematical notation
  </action>
  <verify>
1. File exists: `ls docs/MODEL_ARCHITECTURE.md`
2. File is substantial: `wc -l docs/MODEL_ARCHITECTURE.md` should show 400+ lines
3. Contains Mermaid diagrams: `grep -c 'mermaid' docs/MODEL_ARCHITECTURE.md` should be >= 4
4. Contains physics analogies: `grep -ci 'analogy\|analogous\|physics\|like a\|similar to' docs/MODEL_ARCHITECTURE.md` should be >= 15
5. Contains all 7 improvement phases: `grep -c 'Phase [A-G]' docs/MODEL_ARCHITECTURE.md` should be 7
6. Contains glossary: `grep -c 'Glossary' docs/MODEL_ARCHITECTURE.md` should be >= 1
7. Mermaid syntax valid: no opening ``` blocks without closing (check manually)
  </verify>
  <done>
`docs/MODEL_ARCHITECTURE.md` exists with 400+ lines covering all 8 sections: overview, data pipeline, model architecture (with ConvLSTM explanation, encoder-decoder, skip connections, autoregressive decoding), training process, inference, current config, planned improvements (all 23 items across 7 phases from IMPROVEMENT_NOTES.md), and glossary. Contains 4+ Mermaid diagrams. Physics analogies are used throughout every section. Readable by a non-ML physicist.
  </done>
</task>

</tasks>

<verification>
- Document renders correctly on GitHub (Mermaid diagrams, headings, tables)
- All code references match actual file paths in the repository
- Improvement roadmap matches IMPROVEMENT_NOTES.md faithfully (23 items, 7 phases)
- A non-ML reader could follow the document from start to finish without external references
</verification>

<success_criteria>
1. `docs/MODEL_ARCHITECTURE.md` exists with 400+ lines
2. Contains 4+ Mermaid diagrams illustrating pipeline, architecture, ConvLSTM cell, and loss function
3. Every ML concept is introduced with a physics analogy
4. All 8 sections present and substantive
5. Planned improvements section covers all 23 items from IMPROVEMENT_NOTES.md
6. Document is self-contained -- no required reading of source code to understand the architecture
</success_criteria>

<output>
After completion, create `.planning/quick/001-model-architecture-doc/001-SUMMARY.md`
</output>
