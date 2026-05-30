"""Generate the 10-arm V4 ablation matrix.

Reads `configs/finetune_winding_flux.yaml` as the baseline and writes
one variant YAML per ablation arm to `configs/ablations/`. Each variant
is a full, standalone config (no merge needed at runtime) with one or
two keys mutated against the baseline.

Run:
    python3 configs/ablations/build_configs.py
"""
from __future__ import annotations
import copy
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]
BASE = REPO / "configs" / "finetune_winding_flux.yaml"
OUT = REPO / "configs" / "ablations"

# Each entry: (arm_id, short_desc, mutator)
# mutator takes a deep-copied config dict and mutates in place.

def _mut_baseline(cfg): pass

def _mut_win64(cfg):
    cfg["data"]["window_size"] = 64
    cfg["data"]["window_stride"] = 32

def _mut_win192(cfg):
    cfg["data"]["window_size"] = 192
    cfg["data"]["window_stride"] = 96

def _mut_signed_asinh(cfg):
    cfg["normalization"]["method"] = "signed_asinh"

def _mut_no_sa(cfg):
    cfg["model"]["use_sa_convlstm"] = False

def _mut_no_tempattn(cfg):
    cfg["model"]["temporal_attention"] = False

def _mut_no_attngate(cfg):
    cfg["model"]["attention_gate"] = False

def _mut_channels_small(cfg):
    cfg["model"]["channels"] = [16, 32, 64]

def _mut_kernel3(cfg):
    cfg["model"]["kernel_size"] = 3

def _mut_tstride1(cfg):
    cfg["data"]["stride"] = 1

def _mut_aug_aggressive(cfg):
    cfg["data"]["augmentation"] = "aggressive"

def _mut_convlstm_pure(cfg):
    # All three attention add-ons off → vanilla ConvLSTM encoder/decoder.
    # delta_scale (learnable residual scalar) and dropout (MC-uncertainty)
    # are NOT attention, so they stay — one architectural variable vs A0.
    cfg["model"]["use_sa_convlstm"] = False
    cfg["model"]["temporal_attention"] = False
    cfg["model"]["attention_gate"] = False

def _mut_simple_convlstm(cfg):
    # Canonical minimal baseline: a brand-new 2-layer SimpleConvLSTM (not the
    # 6-layer SolarFluxPredictor). Single channel, flat hidden 64, k3, plain L1,
    # no AMP/attention/delta/dropout. Tractable on Mac Mini MPS.
    cfg["model"] = {
        "kind": "simple_convlstm",
        "input_channels": 1,
        "output_channels": 1,
        "hidden_dim": 64,
        "num_layers": 2,
        "kernel_size": 3,
    }
    cfg["data"]["dual_channel"] = False        # flux only
    cfg["data"]["augmentation"] = "none"       # no D4 aug — fewer windows, faster epoch
    cfg["data"]["window_size"] = 64            # Keras-scale frames, lighter MPS step
    cfg["data"]["window_stride"] = 64          # no overlap → fewer windows
    cfg["loss"] = {"type": "l1"}               # one term, no weights to tune
    cfg["training"]["use_amp"] = False         # MPS DummyGradScaler caused NaN-skip spam
    cfg["training"]["batch_size"] = 8          # batch32 was only ~8% faster (ConvLSTM is step-bound)
                                               # but 4x fewer grad steps → slower convergence; not worth it
    cfg["training"]["lr"] = 1.0e-3             # canonical Adam LR for ConvLSTM
    cfg["training"]["epochs"] = 15
    cfg["training"]["patience"] = 3
    cfg["evaluation"] = {"extreme_threshold": cfg.get("evaluation", {}).get("extreme_threshold", 0.528),
                         "verbose_metrics": False}

def _mut_simple_convlstm_residual(cfg):
    # S0 + residual decode: frame = prev + Δ. Anchors to persistence (Δ→0 =
    # copy last frame) so the model can't collapse to ≈0 like S0/S1 did
    # (which scored CSI 0 vs persistence CSI 0.29).
    _mut_simple_convlstm(cfg)
    cfg["model"]["residual"] = True

def _mut_simple_convlstm_residual_tf(cfg):
    # S2 (residual) + teacher forcing. tf_start=1.0 decays linearly to 0 over
    # training, so early epochs see ground-truth context (curbs the
    # autoregressive error explosion S2 showed); later epochs run free.
    _mut_simple_convlstm_residual(cfg)
    cfg["training"]["tf_start"] = 1.0

def _mut_simple_convlstm_alldata(cfg):
    # S4 (residual + TF) trained on ~all data: 90/10 train/val, no test holdout.
    # norm_type=group (the BatchNorm-in-loop inplace fix) so the multi-step
    # backward can't crash like S4 did at epoch 5.
    _mut_simple_convlstm_residual_tf(cfg)
    cfg["model"]["norm_type"] = "group"
    # ~all data in train. Validator + end-of-run test_eval need test>0 and
    # val>0, so 90/5/5 (≈19 train / 1 val / 1 test of 21 cubes) — the
    # closest to "all data" the pipeline allows.
    cfg["data"]["split_ratios"] = [0.9, 0.05, 0.05]

def _mut_simple_convlstm_composite(cfg):
    # S5 (residual + TF + GroupNorm, ~all data) + composite loss instead of L1.
    # The targeted fix for flare CSI≈0: L1 on a zero-mean heavy-tailed field makes
    # "predict the smooth mean" optimal, so |pred| never crosses the extreme
    # threshold (TP=0). Composite adds extreme_pixel_weight (heavier gradient on
    # extreme regions) + asymmetric extreme loss (penalise underestimation harder)
    # + SSIM (fight blur) + temporal weighting. Watch var_ratio — reweighting can
    # trade CSI for the explosion S2 showed.
    _mut_simple_convlstm_alldata(cfg)
    cfg["loss"] = {
        "type": "composite",
        "l1_weight": 1.0,
        "ssim_weight": 0.3,
        "extreme_weight": 3.0,
        "extreme_pixel_weight": 25.0,
        "use_ms_ssim": False,
        "ssim_data_range": 2.0,
        "ssim_tiling_threshold": 256,
        "temporal_diff_weight": 1.0,
        "temporal_var_lambda": 0.1,
        "temporal_weights": [1.0, 1.5, 2.0, 2.5],
        "asymmetric_weight": 0.5,
        "asymmetric_alpha": 5.0,
        "extreme_threshold": 0.528,
    }

# Fixed, informative test set (cubes with the highest extreme-pixel rate, per
# data/_extreme_rate_per_cube.json) — pins flare CSI comparability across arms
# instead of split-luck. harp_8 excluded (pathological 1.68e10 pixels).
TEST_CUBES = ["harp_245", "harp_274", "harp_49"]


def _mut_simple_convlstm_zscore_fixedtest(cfg):
    # S5 (L1 + zscore, residual+TF+GroupNorm, all data) but on the FIXED
    # informative test set. The matched control for S7: S7 vs S8 isolates the
    # normalization change (zscore→signed_asinh) on the same test cubes, since
    # S5 itself ran on a split-luck single (flareless) test cube.
    _mut_simple_convlstm_alldata(cfg)
    cfg["data"]["test_cubes"] = list(TEST_CUBES)

def _mut_simple_convlstm_signed_asinh(cfg):
    # S8 + signed_asinh normalization (heavy-tail compression at the
    # representation level): sign·asinh(|x|/softening)/scale instead of linear
    # zscore. Hypothesis: compressing the tail lets L1 spend relatively more
    # error budget on extremes, so the field stays sharp where S5's zscore+L1
    # smooths it to the mean. Orthogonal to the loss lever (S3/S6).
    #
    # extreme_threshold=0.0346 ≡ same-tail-fraction match to S5's 0.528 in
    # zscore (1.66% tail on the test cubes under global-fallback norm); it is
    # NOT the same physical pixels. Comparability convention: equal base rate.
    _mut_simple_convlstm_zscore_fixedtest(cfg)
    cfg["normalization"]["method"] = "signed_asinh"
    cfg["normalization"]["signed_asinh_softening"] = 1.0e6
    cfg["evaluation"]["extreme_threshold"] = 0.0346

def _mut_simple_convlstm_s4_fixedtest(cfg):
    # "S4 on the fixed test set." S4 (residual+TF) used BatchNorm and looked
    # sharpest of all S-arms on harp_11930 (retained GT-like texture where the
    # GroupNorm arms smoothed it). This isolates that: S9 = S8 in every respect
    # (L1, zscore, residual, TF, fixed informative test, thr 0.528) EXCEPT
    # norm_type=batch instead of group. S9 vs S8 ⇒ is BatchNorm the sharpness
    # driver, and does that sharpness beat persistence where S8 lost? NOTE:
    # BatchNorm-in-loop can hit the inplace-autograd crash S4 saw at ep5; best
    # checkpoint saves before then and is retestable.
    _mut_simple_convlstm_zscore_fixedtest(cfg)
    cfg["model"]["norm_type"] = "batch"

def _mut_simple_convlstm_fast_tf(cfg):
    # S8 (GroupNorm, L1, zscore, residual+TF, fixed test) but with the teacher-
    # forcing curriculum COMPLETED before early-stop. Diagnosis: TF decays over
    # `epochs` (15) while early-stop fires ~ep2-8, so every prior arm trained at
    # TF 0.6-0.9 the whole time and NEVER saw the free-running regime inference
    # uses → exposure-bias drift (t+3/t+4 polarity collapse). Fix: tf_decay_epochs=5
    # (TF→0 by ep5) + patience 8 so the model trains free-running before stopping.
    # GroupNorm base (not S9 BatchNorm) to avoid the ep5 inplace-autograd crash
    # confounding the TF effect.
    _mut_simple_convlstm_zscore_fixedtest(cfg)
    cfg["training"]["tf_decay_epochs"] = 5
    cfg["training"]["patience"] = 8

# --- Structural pivot: dual-head BCE classifier ---------------------------
# After S0..S11 mapped the loss / norm / TF / representation levers and ALL
# lost to persistence (test CSI 0.043) on the fixed test, the only untried
# lever is the OUTPUT PARAMETRIZATION. Diagnostic (89% of test extreme pixels
# at t+1..t+4 are *new*, not inherited from input) showed persistence is a
# weak baseline with huge headroom — model loses by smoothing to ~0 (var_ratio
# 0.036) rather than by hitting a hard ceiling. CSI is binary against
# |x|>0.528; the dual-head adds a per-pixel `P(|x|>thr)` classifier head on
# the last decoder activation, with class imbalance handled by BCE pos_weight
# (1.66% positive ⇒ pos_weight≈60) or focal loss.

def _mut_simple_convlstm_dual_head(cfg):
    """S12 baseline. S10 recipe (GroupNorm + fast-TF + fixed test) + classifier
    head + BCE loss with pos_weight=60. α=β=1: regression and classification
    contribute equally; the classifier alone drives CSI at eval."""
    _mut_simple_convlstm_fast_tf(cfg)
    cfg["model"]["enable_classifier_head"] = True
    cfg["loss"] = {
        "type": "dual_head",
        "alpha": 1.0,
        "beta": 1.0,
        "pos_weight": 60.0,
        "extreme_threshold": 0.528,
        "classification_loss": "bce",
    }

def _mut_simple_convlstm_dual_posweight100(cfg):
    """S13 = S12 + stronger pos_weight 100 (test sensitivity to class
    imbalance). If S12 still under-predicts positives, pushing pos_weight up
    forces more recall at the cost of precision."""
    _mut_simple_convlstm_dual_head(cfg)
    cfg["loss"]["pos_weight"] = 100.0

def _mut_simple_convlstm_dual_focal(cfg):
    """S14 = S12 but focal loss (γ=2, α_focal=0.25) instead of pos_weight BCE.
    Focal down-weights easy negatives by ``(1-p)^γ`` rather than uniformly
    re-weighting positives — often more stable when many negatives are easy
    and a minority of positives are hard."""
    _mut_simple_convlstm_dual_head(cfg)
    cfg["loss"]["classification_loss"] = "focal"
    cfg["loss"]["focal_gamma"] = 2.0
    cfg["loss"]["focal_alpha"] = 0.25

def _mut_simple_convlstm_dual_classifier_dominant(cfg):
    """S15 = S12 with α=0.1, β=1 (regression as a small regularizer only).
    Tests whether the joint regression objective helps or hurts the classifier
    — if CSI is higher than S12, the L1 term was diluting the classification
    signal."""
    _mut_simple_convlstm_dual_head(cfg)
    cfg["loss"]["alpha"] = 0.1
    cfg["loss"]["beta"] = 1.0

def _mut_simple_convlstm_fasttf_extreme(cfg):
    # The combination arm. S10 showed completing the TF curriculum de-smooths
    # transiently (var_ratio→0.75 @ep7) but L1 drags it back (0.028 @ep15);
    # S6 showed extreme_pixel_weight is the only loss term that lifts CSI. The
    # two levers are orthogonal and each alone loses to persistence — S11 runs
    # both at once: fast-TF (curriculum completes, less drift) + extreme-only
    # composite (cross the threshold) on the fixed informative test set.
    _mut_simple_convlstm_fast_tf(cfg)   # GroupNorm, fixed test, tf_decay 5, patience 8
    cfg["loss"] = {                     # S6's extreme-only composite (one active term)
        "type": "composite",
        "l1_weight": 1.0,
        "ssim_weight": 0.0,
        "extreme_weight": 3.0,
        "extreme_pixel_weight": 25.0,
        "use_ms_ssim": False,
        "ssim_data_range": 2.0,
        "ssim_tiling_threshold": 256,
        "temporal_diff_weight": 0.0,
        "temporal_var_lambda": 0.0,
        "temporal_weights": [1.0, 1.0, 1.0, 1.0],
        "asymmetric_weight": 0.0,
        "asymmetric_alpha": 5.0,
        "extreme_threshold": 0.528,
    }

def _mut_simple_convlstm_extremeonly(cfg):
    # Decomposition ablation for S3: S5 + composite loss with ONLY the
    # extreme_pixel_weight term active (SSIM, asymmetric, temporal all zeroed).
    # S3 changed 4 loss terms at once vs S5's pure L1; its 12x CSI lift could be
    # any one of them. This isolates whether per-pixel extreme reweighting
    # (WeightedMAE, extreme_pixel_weight=25) is the active ingredient. Three-way:
    # S5 (pure L1) vs S6 (L1 + extreme only) vs S3 (all terms) → attributes the lift.
    _mut_simple_convlstm_alldata(cfg)
    cfg["loss"] = {
        "type": "composite",
        "l1_weight": 1.0,
        "ssim_weight": 0.0,            # OFF
        "extreme_weight": 3.0,
        "extreme_pixel_weight": 25.0,  # the term under test
        "use_ms_ssim": False,
        "ssim_data_range": 2.0,
        "ssim_tiling_threshold": 256,
        "temporal_diff_weight": 0.0,   # OFF
        "temporal_var_lambda": 0.0,    # OFF
        "temporal_weights": [1.0, 1.0, 1.0, 1.0],
        "asymmetric_weight": 0.0,      # OFF
        "asymmetric_alpha": 5.0,
        "extreme_threshold": 0.528,
    }

def _mut_simple_convlstm_noaug(cfg):
    # S0 minus augmentation, same 0.7/0.2/0.1 split (apples-to-apples vs S0).
    # balanced=[none,hflip,vflip] (3x) → no-aug cuts train windows ~3x:
    # ~20.5k iters/epoch @ batch8 → ~44 min/epoch on CUDA. 4 epochs ⇒ ~3h.
    _mut_simple_convlstm(cfg)
    cfg["data"]["augmentation"] = "none"
    cfg["training"]["epochs"] = 4
    cfg["training"]["patience"] = 4   # >= epochs ⇒ early-stop won't fire


ARMS = [
    ("A0_baseline",        "production config (win128, zscore_per_cube, SA + tempattn + attngate, k5, [32,64,128])", _mut_baseline),
    ("A1_win64",           "spatial scale: 64x64 windows (4x more samples, smaller receptive field)", _mut_win64),
    ("A2_win192",          "spatial scale: 192x192 windows (fewer samples, larger field, drops smaller cubes)", _mut_win192),
    ("A3_signed_asinh",    "norm: per-cube signed_asinh (heavy-tail compression vs linear zscore)", _mut_signed_asinh),
    ("A4_no_sa_convlstm",  "ablate SA-ConvLSTM memory (use vanilla ConvLSTM)", _mut_no_sa),
    ("A5_no_temporal_attn","ablate TemporalAttention over encoder hidden states", _mut_no_tempattn),
    ("A6_no_attn_gate",    "ablate AttentionGate on skip connection", _mut_no_attngate),
    ("A7_channels_small",  "smaller model: channels [16,32,64] (~4x fewer params)", _mut_channels_small),
    ("A8_kernel3",         "smaller spatial receptive field per cell (kernel 3 vs 5)", _mut_kernel3),
    ("A9_tstride1",        "denser temporal sampling: temporal stride 1 (4x more train samples)", _mut_tstride1),
    ("A10_aug_aggressive", "all 6 D4 augmentations incl rotations (chirality-aware sign flips)", _mut_aug_aggressive),
    ("B0_convlstm_pure",   "pure ConvLSTM: all 3 attention add-ons off (SA + tempattn + attngate), delta_scale + dropout kept", _mut_convlstm_pure),
    ("S0_simple_convlstm", "minimal 2-layer SimpleConvLSTM, 1ch flux, hidden64 k3, L1, no AMP — canonical nowcasting baseline", _mut_simple_convlstm),
    ("S1_simple_convlstm_noaug", "S0 minus augmentation, 4 epochs (~3h on CUDA) — isolates the value of D4 augmentation", _mut_simple_convlstm_noaug),
    ("S2_simple_convlstm_residual", "S0 + residual decode (frame=prev+Δ, persistence anchor) — fixes the collapse-to-0 that gave CSI 0", _mut_simple_convlstm_residual),
    ("S4_simple_convlstm_residual_tf", "S2 + teacher forcing (tf_start=1.0 decaying) — curb the autoregressive error explosion", _mut_simple_convlstm_residual_tf),
    ("S5_simple_convlstm_alldata", "S4 (residual+TF, GroupNorm) on ~all data (90/10 split, no test) — does more data lift flare CSI?", _mut_simple_convlstm_alldata),
    ("S3_simple_convlstm_composite", "S5 + composite loss (extreme_pixel_weight 25 + asymmetric α5 + SSIM) — targeted fix for flare CSI≈0 under L1 mean-collapse", _mut_simple_convlstm_composite),
    ("S6_simple_convlstm_extremeonly", "S5 + composite loss with ONLY extreme_pixel_weight 25 (SSIM/asymmetric/temporal off) — isolates which S3 term drives the CSI lift", _mut_simple_convlstm_extremeonly),
    ("S8_simple_convlstm_zscore_fixedtest", "S5 (L1+zscore) on the FIXED informative test set [harp_245,274,49] — matched control for S7 (isolates the test-set change from the norm change)", _mut_simple_convlstm_zscore_fixedtest),
    ("S7_simple_convlstm_signed_asinh", "S8 + signed_asinh normalization (heavy-tail compression) — does representation-level tail compression keep the field sharp where zscore+L1 smooths it?", _mut_simple_convlstm_signed_asinh),
    ("S9_simple_convlstm_s4_fixedtest", "S8 but norm_type=batch (= 'S4 on fixed test') — is BatchNorm S4's sharpness driver, and does it beat persistence where S8/S7 lost?", _mut_simple_convlstm_s4_fixedtest),
    ("S10_simple_convlstm_fast_tf", "S8 + tf_decay_epochs=5 + patience 8 — completes the TF curriculum before early-stop so the model trains free-running; does it cure the t+3/t+4 autoregressive drift?", _mut_simple_convlstm_fast_tf),
    ("S11_simple_convlstm_fasttf_extreme", "S10 (fast-TF) + S6 extreme_pixel_weight loss — both levers at once (completed curriculum + extreme reweighting); does the combo finally beat persistence_csi 0.043?", _mut_simple_convlstm_fasttf_extreme),
    ("S12_simple_convlstm_dual_head", "S10 + dual-head classifier (per-pixel BCE with pos_weight=60) — first structural pivot; CSI from classifier head directly. Does it finally beat persistence_csi 0.043?", _mut_simple_convlstm_dual_head),
    ("S13_simple_convlstm_dual_posweight100", "S12 with pos_weight=100 — test sensitivity to class-imbalance correction (1.66% positive rate)", _mut_simple_convlstm_dual_posweight100),
    ("S14_simple_convlstm_dual_focal", "S12 with focal loss (γ=2, α_focal=0.25) instead of pos_weight BCE — alternative class-imbalance handling", _mut_simple_convlstm_dual_focal),
    ("S15_simple_convlstm_dual_classifier_dominant", "S12 with α=0.1 β=1 (regression as small regularizer) — does the joint L1 dilute the classifier or help it?", _mut_simple_convlstm_dual_classifier_dominant),
]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    base = yaml.safe_load(BASE.read_text())
    rows = []
    for arm_id, desc, mut in ARMS:
        cfg = copy.deepcopy(base)
        # Per-arm output dir so checkpoints + history don't collide
        cfg["output"]["save_dir"] = f"./outputs/ablations/{arm_id}"
        # No transfer_learning for ablations — train from scratch so each
        # arm reflects the architecture/policy, not pretrain weights.
        cfg["transfer_learning"] = None
        cfg.pop("resume_from", None)
        cfg["resume_from"] = None
        mut(cfg)
        p = OUT / f"{arm_id}.yaml"
        header = (
            f"# Ablation {arm_id}\n"
            f"# {desc}\n"
            f"# Baseline: configs/finetune_winding_flux.yaml\n"
            f"# Auto-generated by configs/ablations/build_configs.py — do not edit by hand.\n\n"
        )
        p.write_text(header + yaml.safe_dump(cfg, sort_keys=False, default_flow_style=False))
        rows.append((arm_id, desc))
        print(f"wrote {p.relative_to(REPO)}")

    # README matrix
    readme = OUT / "README.md"
    lines = [
        "# V4 Ablation Matrix",
        "",
        f"{len(ARMS)} arms. Each is a standalone YAML (no merge at runtime). All",
        "share the production baseline except the listed delta. Each",
        "writes to its own `outputs/ablations/<arm_id>/`.",
        "",
        "Generated by `build_configs.py`.",
        "",
        "| Arm | Δ from baseline | Tests |",
        "|---|---|---|",
    ]
    tests_per_arm = {
        "A0_baseline": "—",
        "A1_win64": "smaller spatial scale beat baseline?",
        "A2_win192": "larger spatial scale needed?",
        "A3_signed_asinh": "heavy-tail compression helps over linear zscore?",
        "A4_no_sa_convlstm": "is SA-ConvLSTM memory net positive at this scale?",
        "A5_no_temporal_attn": "TemporalAttention worth its params?",
        "A6_no_attn_gate": "AttentionGate worth its params?",
        "A7_channels_small": "does the small model underfit or close the gap?",
        "A8_kernel3": "is k=5 needed or does k=3 suffice?",
        "A9_tstride1": "more samples → better val? (overfit risk)",
        "A10_aug_aggressive": "rotations + chirality flip improve generalisation?",
        "B0_convlstm_pure": "does pure ConvLSTM beat the full attention stack? (clean arch-only A0 vs B0; NOT the v2/v3 11-02 number — that compared different pipelines/epochs)",
        "S0_simple_convlstm": "can a minimal 2-layer ConvLSTM (canonical recipe) match the deep model? simplest viable baseline",
        "S1_simple_convlstm_noaug": "does D4 augmentation help the simple model? S0 vs S1 (no-aug, 4ep, ~3h)",
        "S2_simple_convlstm_residual": "does residual/persistence anchoring beat collapse? target CSI > persistence 0.29",
        "S4_simple_convlstm_residual_tf": "does teacher forcing stop S2's autoregressive explosion? CSI vs persistence 0.29",
        "S5_simple_convlstm_alldata": "does training on ~all data (90/10) lift flare CSI over S4's 0.099?",
        "S3_simple_convlstm_composite": "does composite loss (tail reweight + asymmetric + SSIM) lift flare CSI past S5 L1 collapse without var_ratio explosion?",
        "S6_simple_convlstm_extremeonly": "is extreme_pixel_weight alone the active ingredient in S3's CSI lift? (S5 L1 vs S6 extreme-only vs S3 full composite)",
        "S8_simple_convlstm_zscore_fixedtest": "L1+zscore reference on the fixed informative test set — anchor for S7 (and a meaningful CSI vs S5's flareless-cube ~0)",
        "S7_simple_convlstm_signed_asinh": "does signed_asinh tail-compression lift flare CSI / keep var_ratio toward 1 vs S8's zscore+L1? (same test, same base rate)",
        "S9_simple_convlstm_s4_fixedtest": "is BatchNorm (vs S8's GroupNorm) the driver of S4's visual sharpness, and does it beat persistence_csi 0.043 on the fixed test?",
        "S10_simple_convlstm_fast_tf": "does completing the TF curriculum (decay→0 by ep5, patience 8) cure autoregressive t+3/t+4 drift + lift CSI vs S8?",
        "S11_simple_convlstm_fasttf_extreme": "do both levers together (fast-TF + extreme_pixel_weight) beat persistence_csi 0.043 where each alone (S10, S6) failed?",
        "S12_simple_convlstm_dual_head": "structural pivot — does a per-pixel BCE classifier head (binary metric ↔ binary head) finally beat persistence_csi 0.043?",
        "S13_simple_convlstm_dual_posweight100": "is pos_weight=60 too low? does 100 lift recall (and therefore CSI) at acceptable precision cost?",
        "S14_simple_convlstm_dual_focal": "does focal loss (down-weights easy negatives) beat pos_weight BCE on this imbalance?",
        "S15_simple_convlstm_dual_classifier_dominant": "is the joint L1 diluting the classifier? does α=0.1 (classifier-dominant) give a higher CSI?",
    }
    for arm_id, desc in rows:
        lines.append(f"| `{arm_id}` | {desc} | {tests_per_arm.get(arm_id, '?')} |")
    lines.extend([
        "",
        "## Run",
        "",
        "```bash",
        "python3 main.py --config configs/ablations/A0_baseline.yaml",
        "# repeat per arm",
        "```",
        "",
        "## Compare runs",
        "",
        "Each arm writes `outputs/ablations/<arm_id>/{best_model.pt,",
        "training_history.json, test_results.json}`. Aggregate with",
        "`generate_comparison.py` or a custom script that pulls val MAE,",
        "CSI, HSS, SSIM per arm.",
        "",
        "## Run order",
        "",
        "1. **A0** first — establishes baseline.",
        "2. **A4, A5, A6** in parallel — cheapest ablations (single feature off).",
        "3. **A1, A2** — spatial-scale sweep (different sample counts).",
        "4. **A3** — orthogonal normalisation choice.",
        "5. **A7, A8** — architecture down-scaling (cheap).",
        "6. **A9, A10** — sample-count + augmentation sweeps (longest).",
        "",
        "If wall-clock is constrained, **A0 + A4 + A5 + A6** is the",
        "minimum viable matrix for the v3.0→v4.0 ConvLSTM+attention story.",
    ])
    readme.write_text("\n".join(lines) + "\n")
    print(f"wrote {readme.relative_to(REPO)}")


if __name__ == "__main__":
    main()
