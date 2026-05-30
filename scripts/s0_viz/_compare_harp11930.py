"""Per-arm integrated-flux MAE on harp_11930. Runs same viz pipeline,
dumps pred/gt integrated flux per set+t, prints MAE table.

Usage: python3 -m scripts.s0_viz._compare_harp11930
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
from scripts.s0_viz.fullframe import predict_full_frame

ARMS = [
    "S0_simple_convlstm", "S3_simple_convlstm_composite",
    "S4_simple_convlstm_residual_tf", "S5_simple_convlstm_alldata",
    "S6_simple_convlstm_extremeonly", "S7_simple_convlstm_signed_asinh",
    "S8_simple_convlstm_zscore_fixedtest", "S9_simple_convlstm_s4_fixedtest",
    "S10_simple_convlstm_fast_tf", "S11_simple_convlstm_fasttf_extreme",
    "S13_simple_convlstm_dual_posweight100",
]
CUBE_NAME = "harp_11930"
WIN = 64
STRIDE = 32


def device():
    if torch.backends.mps.is_available(): return torch.device("mps")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    dev = device()
    results = {}
    cube_arr = None
    sets_meta = None

    for arm in ARMS:
        cfg = yaml.safe_load(Path(f"configs/ablations/{arm}.yaml").read_text())
        ckpt = Path(f"outputs/ablations/{arm}/checkpoints/best_model.pt")
        if not ckpt.exists():
            print(f"SKIP {arm}: no ckpt"); continue

        norm_method = cfg.get("normalization", {}).get("method", "zscore_per_cube")
        t_in, t_out = cfg["data"]["t_in"], cfg["data"]["t_out"]
        ratios = cfg["data"].get("split_ratios", [0.7, 0.2, 0.1])
        seed = cfg.get("seed", 42)
        data_dir = Path(cfg["data"]["data_dir"])

        zarr_dirs = sorted(p for p in data_dir.glob("*.zarr") if p.is_dir())
        idx = next(i for i, p in enumerate(zarr_dirs) if p.stem == CUBE_NAME)
        zp = zarr_dirs[idx]

        if cube_arr is None:
            cube_arr, _ = load_cube(zp)
            sets_meta_ws = find_active_windows(cube_arr, t_in, t_out, WIN, n_sets=3)
            sets_meta = sets_meta_ws
            print(f"cube {CUBE_NAME}: T={cube_arr.shape[0]} HxW={cube_arr.shape[1]}x{cube_arr.shape[2]}; {len(sets_meta)} active sets")

        mu, sigma = (asinh_scale(cube_arr) if norm_method == "signed_asinh"
                     else cube_stats(cube_arr))
        cube_norm = normalize(cube_arr, mu, sigma, norm_method)

        model = load_s0_model(ckpt, cfg["model"], t_out, dev)

        arm_sets = []
        for (ts, y0, x0) in sets_meta:
            pn = predict_full_frame(model, cube_norm, ts, t_in, t_out, WIN, STRIDE, dev)
            gn = cube_norm[ts + t_in : ts + t_in + t_out]
            pd = denormalize(pn, mu, sigma, norm_method)
            gd = denormalize(gn, mu, sigma, norm_method)
            arm_sets.append({"pred_int": pd.sum(axis=(1, 2)),
                             "gt_int": gd.sum(axis=(1, 2))})
        results[arm] = arm_sets
        print(f"  {arm}: done")

    print("\n" + "=" * 90)
    print(f"INTEGRATED WINDING FLUX MAE on {CUBE_NAME} (lower = better)")
    print("=" * 90)
    print(f"{'arm':<40} {'set1':>10} {'set2':>10} {'set3':>10} {'mean':>10}")
    print("-" * 90)
    table = []
    for arm, sets in results.items():
        per_set = [float(np.mean(np.abs(s["pred_int"] - s["gt_int"]))) for s in sets]
        mean = float(np.mean(per_set))
        table.append((arm, per_set, mean))
        print(f"{arm:<40} " + " ".join(f"{v:>10.3f}" for v in per_set) + f" {mean:>10.3f}")

    table.sort(key=lambda x: x[2])
    print("\n--- RANKED (best mean integrated-flux MAE first) ---")
    for i, (arm, _, mean) in enumerate(table, 1):
        print(f"  {i:2d}. {arm:<42} mean MAE = {mean:.3f}")

    # GT magnitude reference
    gt_mag = float(np.mean([np.mean(np.abs(s["gt_int"])) for s in results[ARMS[0]] if "gt_int" in s]))
    print(f"\nGT integrated-flux mean |value| (set-avg): {gt_mag:.3f}")
    print(f"Persistence baseline (pred = 0): MAE = {gt_mag:.3f}  (each arm must beat this)")


if __name__ == "__main__":
    main()
