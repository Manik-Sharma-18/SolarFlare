# Technology Stack Research: v3.0 Temporal Dynamics & Flare Detection

**Project:** SolarFlare v3.0
**Researched:** 2026-03-07
**Confidence:** HIGH (all features use PyTorch built-ins)

## Stack Assessment

### No New Runtime Dependencies Required

All v3.0 features are implementable with existing PyTorch + NumPy. No new packages needed.

| Feature | Implementation | PyTorch Built-in? |
|---------|---------------|-------------------|
| Temporal difference loss | Custom loss term (L1 on consecutive frame diffs) | Yes - tensor ops |
| Temporal weighting | Per-timestep weight vector in loss | Yes - tensor ops |
| Asymmetric loss | Conditional penalty (torch.where) | Yes - tensor ops |
| Spatial attention gate | Conv2d + Sigmoid (Attention U-Net pattern) | Yes - nn.Conv2d, nn.Sigmoid |
| Temporal attention | Linear + Softmax over encoder outputs | Yes - nn.Linear, F.softmax |
| CSI / HSS metrics | TP/FP/FN counting on binarized tensors | Yes - tensor comparisons |
| Persistence baseline | Copy last input frame, compute MAE | Yes - tensor slicing |
| WeightedRandomSampler | torch.utils.data.WeightedRandomSampler | Yes - built-in class |
| Cosine LR scheduler | torch.optim.lr_scheduler.CosineAnnealingLR | Yes - built-in class |
| MC Dropout | Already implemented (dropout_rate > 0) | Yes - existing code |
| Progressive curriculum | Config-driven t_out changes + checkpoint resume | Yes - existing infra |
| Delta head normalization | Learnable nn.Parameter scale factor | Yes - nn.Parameter |

### Config Changes Required

```yaml
# New fields for v3.0
model:
  channels: [32, 64, 128]        # Was [16, 32, 64]
  kernel_size: 5                   # Was 3
  dropout_rate: 0.15               # Was 0.0
  use_spatial_attention: true      # New
  use_temporal_attention: true     # New
  delta_head_scale: true           # New

loss:
  extreme_weight: 3.0              # Was 1.0
  temporal_diff_weight: 1.0        # New
  temporal_var_weight: 0.1         # New
  asymmetric_alpha: 2.0            # New
  timestep_weights: [1.0, 1.5, 2.0, 2.5]  # New

training:
  tf_start: 0.0                    # Was 0.5
  scheduler:
    type: "cosine"                 # Was "none"

data:
  augmentation: "balanced"         # Was "none"
  oversample_extreme: true         # New
  oversample_factor: 3             # New
```

### MPS Compatibility Notes

All new features use standard ops (Conv2d, Linear, element-wise) that are fully MPS-compatible. No new MPS edge cases introduced beyond what v2.0 already handles. Spatial attention uses standard Conv2d (not grouped), temporal attention uses Linear layers -- both safe on MPS.

### What NOT to Add

- **Transformer architectures** -- Overkill for 10-frame sequences; ConvLSTM + attention is sufficient
- **External metric libraries (torchmetrics)** -- CSI/HSS are simple enough to implement in ~20 lines
- **Optuna/Ray Tune** -- Defer hyperparameter search to v4.0
- **torch.compile** -- Still poor MPS support, risk of subtle bugs

---
*Research completed: 2026-03-07*
