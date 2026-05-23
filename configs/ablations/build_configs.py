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
    cfg["data"]["window_size"] = 64            # Keras-scale frames, lighter MPS step
    cfg["data"]["window_stride"] = 64          # no overlap → fewer windows
    cfg["loss"] = {"type": "l1"}               # one term, no weights to tune
    cfg["training"]["use_amp"] = False         # MPS DummyGradScaler caused NaN-skip spam
    cfg["training"]["batch_size"] = 8
    cfg["training"]["lr"] = 1.0e-3             # canonical Adam LR for ConvLSTM
    cfg["training"]["epochs"] = 15
    cfg["training"]["patience"] = 6
    cfg["evaluation"] = {"extreme_threshold": cfg.get("evaluation", {}).get("extreme_threshold", 0.528),
                         "verbose_metrics": False}


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
        "11 arms. Each is a standalone YAML (no merge at runtime). All",
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
