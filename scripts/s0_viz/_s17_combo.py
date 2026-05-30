"""S17 inference combo: S11 regression × S13 classifier on harp_11930.

Tests 3 fusion variants vs S11/S13 alone, no retraining:
  A_gate:  S11_pred * (sigmoid(S13_logits) > 0.5)
  B_soft:  S11_pred * (1 + 2 * sigmoid(S13_logits))
  C_blend: S13_pred where sigmoid>0.5 else S11_pred

Metric: integrated winding flux MAE (sum over space, MAE over time).

Usage: python3 -u -m scripts.s0_viz._s17_combo
"""
import sys
from pathlib import Path
import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from solarflare_data.loader import assign_files_to_splits
from scripts.s0_viz.cube_io import load_cube, cube_stats, asinh_scale, normalize, denormalize
from scripts.s0_viz.windows import find_active_windows
from scripts.s0_viz.infer import load_s0_model

S11 = "S11_simple_convlstm_fasttf_extreme"
S13 = "S13_simple_convlstm_dual_posweight100"
CUBE_NAME = "harp_11930"
WIN = 64
STRIDE = 32


def device():
    if torch.backends.mps.is_available(): return torch.device("mps")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def predict_full_dual(model, cube_norm, t_start, t_in, t_out, win, stride, dev, want_logits=False):
    """Like predict_full_frame but optionally returns (pred, logits) tiled."""
    T, H, W = cube_norm.shape
    wy, wx = win, win
    ys = list(range(0, H - wy + 1, stride))
    if ys[-1] != H - wy: ys.append(H - wy)
    xs = list(range(0, W - wx + 1, stride))
    if xs[-1] != W - wx: xs.append(W - wx)
    tiles, coords = [], []
    for y in ys:
        for x in xs:
            tiles.append(cube_norm[t_start : t_start + t_in, y : y + wy, x : x + wx])
            coords.append((y, x))
    batch = torch.from_numpy(np.ascontiguousarray(np.stack(tiles))).unsqueeze(1).float()
    accum_p = np.zeros((t_out, H, W), dtype=np.float64)
    accum_l = np.zeros((t_out, H, W), dtype=np.float64) if want_logits else None
    count = np.zeros((H, W), dtype=np.float64)
    for i in range(0, batch.shape[0], 8):
        chunk = batch[i : i + 8].to(dev)
        out = model(chunk, teacher_forcing_ratio=0.0)
        if isinstance(out, tuple):
            pred, logits = out
        else:
            pred, logits = out, None
        pred = pred[:, 0].cpu().numpy()
        if want_logits and logits is not None:
            logits = logits[:, 0].cpu().numpy()
        for j in range(pred.shape[0]):
            y, x = coords[i + j]
            accum_p[:, y : y + wy, x : x + wx] += pred[j]
            if want_logits and logits is not None:
                accum_l[:, y : y + wy, x : x + wx] += logits[j]
            count[y : y + wy, x : x + wx] += 1.0
    count = np.maximum(count, 1e-9)
    pred_full = (accum_p / count[None]).astype(np.float32)
    if want_logits:
        return pred_full, (accum_l / count[None]).astype(np.float32) if accum_l is not None else None
    return pred_full


def main():
    dev = device()
    cfg11 = yaml.safe_load(Path(f"configs/ablations/{S11}.yaml").read_text())
    cfg13 = yaml.safe_load(Path(f"configs/ablations/{S13}.yaml").read_text())
    t_in, t_out = cfg11["data"]["t_in"], cfg11["data"]["t_out"]
    data_dir = Path(cfg11["data"]["data_dir"])

    zarr_dirs = sorted(p for p in data_dir.glob("*.zarr") if p.is_dir())
    zp = next(p for p in zarr_dirs if p.stem == CUBE_NAME)
    cube_arr, _ = load_cube(zp)
    sets_meta = find_active_windows(cube_arr, t_in, t_out, WIN, n_sets=3)
    print(f"cube {CUBE_NAME}: T={cube_arr.shape[0]} HxW={cube_arr.shape[1]}x{cube_arr.shape[2]}; {len(sets_meta)} sets")

    nm11 = cfg11.get("normalization", {}).get("method", "zscore_per_cube")
    nm13 = cfg13.get("normalization", {}).get("method", "zscore_per_cube")
    mu11, s11_ = (asinh_scale(cube_arr) if nm11 == "signed_asinh" else cube_stats(cube_arr))
    mu13, s13_ = (asinh_scale(cube_arr) if nm13 == "signed_asinh" else cube_stats(cube_arr))
    cube_n11 = normalize(cube_arr, mu11, s11_, nm11)
    cube_n13 = normalize(cube_arr, mu13, s13_, nm13)

    m11 = load_s0_model(Path(f"outputs/ablations/{S11}/checkpoints/best_model.pt"), cfg11["model"], t_out, dev)
    m13 = load_s0_model(Path(f"outputs/ablations/{S13}/checkpoints/best_model.pt"), cfg13["model"], t_out, dev)

    rows = []
    for set_i, (ts, _y, _x) in enumerate(sets_meta, 1):
        p11_n = predict_full_dual(m11, cube_n11, ts, t_in, t_out, WIN, STRIDE, dev, want_logits=False)
        p13_n, l13_n = predict_full_dual(m13, cube_n13, ts, t_in, t_out, WIN, STRIDE, dev, want_logits=True)
        gt_n = cube_n11[ts + t_in : ts + t_in + t_out]

        p11 = denormalize(p11_n, mu11, s11_, nm11)
        p13 = denormalize(p13_n, mu13, s13_, nm13)
        gt = denormalize(gt_n, mu11, s11_, nm11)
        sig13 = 1.0 / (1.0 + np.exp(-l13_n))  # classifier prob from S13 logits
        mask = (sig13 > 0.5).astype(np.float32)

        # Variants in normalized space, then denorm via S11 stats (regression-comparable)
        v_gate_n = p11_n * mask
        v_soft_n = p11_n * (1.0 + 2.0 * sig13)
        v_blend_n = np.where(mask > 0.5, p13_n, p11_n)
        v_gate = denormalize(v_gate_n, mu11, s11_, nm11)
        v_soft = denormalize(v_soft_n, mu11, s11_, nm11)
        v_blend = denormalize(v_blend_n, mu11, s11_, nm11)

        gt_int = gt.sum(axis=(1, 2))
        per_variant = {
            "S11_only": p11.sum(axis=(1, 2)),
            "S13_only": p13.sum(axis=(1, 2)),
            "A_gate(S11*mask)": v_gate.sum(axis=(1, 2)),
            "B_soft(S11*(1+2σ))": v_soft.sum(axis=(1, 2)),
            "C_blend(S13|mask;S11)": v_blend.sum(axis=(1, 2)),
            "Persistence_zero": np.zeros_like(gt_int),
        }
        row = {"set": set_i, "gt_int": gt_int}
        for k, v in per_variant.items():
            row[k] = float(np.mean(np.abs(v - gt_int)))
        rows.append(row)
        mask_rate = float(mask.mean())
        print(f"set{set_i} t0={ts}: classifier mask coverage {mask_rate:.3%}")

    print("\n" + "=" * 110)
    print(f"INTEGRATED WINDING FLUX MAE on {CUBE_NAME} (lower = better)")
    print("=" * 110)
    variants = ["S11_only", "S13_only", "A_gate(S11*mask)", "B_soft(S11*(1+2σ))", "C_blend(S13|mask;S11)", "Persistence_zero"]
    hdr = f"{'variant':<28}" + "".join(f"{'set'+str(i+1):>10}" for i in range(len(rows))) + f"{'mean':>10}"
    print(hdr)
    print("-" * 110)
    summary = []
    for v in variants:
        per = [r[v] for r in rows]
        m = float(np.mean(per))
        summary.append((v, m))
        print(f"{v:<28}" + "".join(f"{x:>10.3f}" for x in per) + f"{m:>10.3f}")

    summary.sort(key=lambda x: x[1])
    print("\n--- RANKED ---")
    for i, (v, m) in enumerate(summary, 1):
        print(f"  {i}. {v:<28} {m:.3f}")

    gt_int_avg = float(np.mean([np.mean(np.abs(r["gt_int"])) for r in rows]))
    print(f"\nGT |integrated flux| avg = {gt_int_avg:.3f}  (persistence_zero MAE ≈ this)")


if __name__ == "__main__":
    main()
