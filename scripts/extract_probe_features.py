"""Extract spatially-pooled per-frame encoder features from a frozen V5 JEPA ckpt.

For every cube in `data/manifest.json`:
  1. Load full cube via solarflare_data.zarr_loader (lazy zarr).
  2. Apply per-cube z-score (μ, σ from cube_norm_stats, matches encoder training norm).
  3. Run frozen InputAdapter + ViTEncoder per frame (chunked to bound memory).
  4. Spatial pool tokens with (valid pixel-token AND non-pad token) mask → [D].
  5. Stack to [T, D] fp32 and save to `data/<harp_id>_feat_<tag>.npy`.
  6. Also save `data/<harp_id>_feat_<tag>_valid.npy` — bool [T] (Time>0).

Frames with Time==0 (sentinel) get a NaN feature row + valid=False so the probe
dataset can drop them; preserves index alignment with cached wind_1d targets.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from models.v5.input_adapter import InputAdapter, valid_pixel_to_token_mask
from models.v5.vit_encoder import ViTEncoder
from solarflare_data.zarr_loader import (
    WIND_FLUX_CLIP,
    cube_norm_stats,
    open_cube,
)


def _strip_prefix(sd: dict, prefix: str) -> dict:
    n = len(prefix)
    return {k[n:]: v for k, v in sd.items() if k.startswith(prefix)}


def load_frozen_modules(ckpt_path: Path, device: torch.device) -> tuple[InputAdapter, ViTEncoder, dict]:
    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    sd_raw = ckpt["model_state_dict"]
    sd = {k.replace("._orig_mod.", "."): v for k, v in sd_raw.items()}
    cfg = ckpt["config"]
    enc_cfg = cfg["model"]["encoder"]
    patch = int(cfg["model"]["input_adapter"]["patch_size"])
    in_ch = int(cfg["model"]["input_adapter"]["in_ch"])
    out_ch = int(cfg["model"]["input_adapter"]["out_ch"])

    adapter = InputAdapter(in_ch=in_ch, out_ch=out_ch, patch_size=patch).to(device)
    encoder = ViTEncoder(
        in_ch=out_ch, embed_dim=int(enc_cfg["dim"]), patch_size=patch,
        layers=int(enc_cfg["layers"]), heads=int(enc_cfg["heads"]),
        mlp_ratio=int(enc_cfg["mlp_ratio"]), dropout=0.0, drop_path=0.0,
        use_grad_checkpoint=False,
    ).to(device)

    adapter.load_state_dict(_strip_prefix(sd, "adapter."))
    encoder.load_state_dict(_strip_prefix(sd, "encoder."))
    adapter.eval(); encoder.eval()
    for p in adapter.parameters(): p.requires_grad_(False)
    for p in encoder.parameters(): p.requires_grad_(False)
    return adapter, encoder, cfg


def _read_frames_chunk(cube, t_idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Read chunk of frames by index. Returns (wind [n,H,W], valid_pixel [n,H,W])."""
    h, w, _ = cube.shape
    n = len(t_idx)
    wind = np.empty((n, h, w), dtype=np.float32)
    valid = np.empty((n, h, w), dtype=bool)
    for i, t in enumerate(t_idx):
        raw = np.asarray(cube.wind[:, :, int(t)], dtype=np.float32)
        v = np.isfinite(raw) & (np.abs(raw) <= WIND_FLUX_CLIP)
        wind[i] = np.where(v, raw, 0.0)
        valid[i] = v
    return wind, valid


@torch.no_grad()
def encode_cube(adapter: InputAdapter, encoder: ViTEncoder, cube_path: Path,
                device: torch.device, chunk: int = 16,
                threshold: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    """Encode all frames of a cube. Returns (feats [T, D], valid_frames [T] bool)."""
    cube = open_cube(cube_path)
    T = cube.shape[2]
    valid_frames = cube.valid_frames.copy()
    valid_idx = np.where(valid_frames)[0]
    mu, sigma = cube_norm_stats(cube)
    D = encoder.embed_dim

    feats = np.full((T, D), np.nan, dtype=np.float32)

    for start in range(0, valid_idx.size, chunk):
        idx = valid_idx[start:start + chunk]
        wind, valid_px = _read_frames_chunk(cube, idx)               # [n,H,W]
        wind_z = ((wind - mu) / sigma).astype(np.float32, copy=False)
        # Build [B=1, T=n, 1, H, W] and per-pixel valid mask matching shape.
        x = torch.from_numpy(wind_z)[None, :, None].to(device)        # [1,n,1,H,W]
        v = torch.from_numpy(valid_px)[None, :, None].to(device)      # [1,n,1,H,W]

        x_adapt, token_pad_mask, (pad_h, pad_w) = adapter(x)          # [1,n,13,Hp,Wp]
        Hp_pad, Wp_pad = x_adapt.shape[-2], x_adapt.shape[-1]
        hp = Hp_pad // adapter.patch_size
        wp = Wp_pad // adapter.patch_size

        z = encoder.encode_frames(x_adapt)                            # [1,n,hp,wp,D]
        valid_tok = valid_pixel_to_token_mask(
            v, hp, wp, Hp_pad, Wp_pad, adapter.patch_size, threshold, device,
        )                                                             # [1,n,hp,wp]
        pool_mask = valid_tok & token_pad_mask[None, None]            # [1,n,hp,wp]
        m = pool_mask.float().unsqueeze(-1)                           # [1,n,hp,wp,1]
        denom = m.sum(dim=(2, 3)).clamp(min=1.0)                      # [1,n,1]
        pooled = (z * m).sum(dim=(2, 3)) / denom                      # [1,n,D]
        pooled_np = pooled.squeeze(0).detach().to("cpu").numpy().astype(np.float32)
        feats[idx] = pooled_np

        # Frames where pool_mask was empty (all padded/invalid) → mark invalid.
        empty = (pool_mask.sum(dim=(2, 3)) == 0).squeeze(0).cpu().numpy()
        if empty.any():
            for j, t in enumerate(idx):
                if empty[j]:
                    feats[int(t)] = np.nan
                    valid_frames[int(t)] = False
    return feats, valid_frames


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="JEPA checkpoint (e.g. outputs_v5_mini_mask_on_slow/best.pt)")
    ap.add_argument("--manifest", default="data/manifest.json")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--tag", default="E09", help="Suffix for cache files: <harp>_feat_<tag>.npy")
    ap.add_argument("--device", default="mps", choices=["mps", "cpu", "cuda"])
    ap.add_argument("--chunk", type=int, default=16)
    ap.add_argument("--cube", default=None, help="Single cube (smoke). Default: all.")
    ap.add_argument("--threshold", type=float, default=0.5)
    args = ap.parse_args()

    device = torch.device(args.device)
    print(f"[info] device={device} ckpt={args.ckpt}")
    adapter, encoder, cfg = load_frozen_modules(Path(args.ckpt), device)
    enc_dim = encoder.embed_dim
    print(f"[info] encoder loaded — dim={enc_dim} layers={len(encoder.blocks)} patch={adapter.patch_size}")

    manifest = json.loads(Path(args.manifest).read_text())
    cubes = manifest["cubes"]
    if args.cube is not None:
        cubes = [c for c in cubes if c["harp_id"] == args.cube]
        if not cubes:
            raise SystemExit(f"cube {args.cube} not in manifest")

    out_dir = Path(args.data_dir)
    for entry in cubes:
        hid = entry["harp_id"]
        cube_path = Path(entry["path"])
        feats_out = out_dir / f"{hid}_feat_{args.tag}.npy"
        valid_out = out_dir / f"{hid}_feat_{args.tag}_valid.npy"
        if feats_out.exists() and valid_out.exists():
            print(f"[skip] {hid} already cached → {feats_out.name}")
            continue
        print(f"[extract] {hid} ({entry.get('shape_hwt', '?')})")
        feats, valid = encode_cube(adapter, encoder, cube_path, device, chunk=args.chunk, threshold=args.threshold)
        np.save(feats_out, feats)
        np.save(valid_out, valid.astype(bool))
        n_valid = int(valid.sum())
        print(f"  → {feats.shape} dtype={feats.dtype} valid_frames={n_valid}/{valid.size} saved")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
