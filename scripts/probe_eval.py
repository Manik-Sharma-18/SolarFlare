"""Evaluate a trained probe ckpt: per-cube R²/Pearson r in original units +
1-frame persistence baseline + overlay PNGs + backbone JEPA curve.

See submodules: probe_eval_metrics.py, probe_eval_plots.py.

Outputs (under <out_dir>):
  probe_metrics.md   markdown table (persistence vs linear vs MLP)
  probe_metrics.json same numbers, machine-readable
  overlay_<harp>.png true vs predictions vs persistence
  e09_jepa_curve.png JEPA val_loss vs epoch (sanity: backbone converged)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml

from models.v5.wind_probe_head import build_probe
from solarflare_data.probe_dataset import TargetStats, split_cubes_for_probe
from scripts.probe_eval_metrics import evaluate_set, write_metrics_md
from scripts.probe_eval_plots import plot_overlays, plot_jepa_curve


def load_probe(path: Path, device: torch.device) -> tuple[torch.nn.Module, str, TargetStats, dict]:
    payload = torch.load(str(path), map_location=device, weights_only=False)
    cfg = payload["config"]
    kind = payload["kind"]
    dim = int(payload["dim"])
    stats = TargetStats.from_dict(payload["stats"])
    head = build_probe(kind, dim=dim,
                       hidden=int(cfg["probe"].get("hidden", 256)),
                       dropout=float(cfg["probe"].get("dropout", 0.1)))
    head.load_state_dict(payload["state_dict"])
    head.to(device).eval()
    return head, kind, stats, cfg


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/probe_e09.yaml")
    ap.add_argument("--linear", default="outputs_probe/linear/best.pt")
    ap.add_argument("--mlp", default="outputs_probe/mlp/best.pt")
    ap.add_argument("--out-dir", default="outputs_probe/eval")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--jepa-run", default="outputs_v5_mini_mask_on_slow/run.jsonl")
    ap.add_argument("--overlay-cubes", nargs="*", default=None,
                    help="Cubes to plot overlays. Default: val cube + 2 novel.")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    device = torch.device(args.device)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    heads: dict[str, tuple[torch.nn.Module, TargetStats]] = {}
    for k, p in (("linear", args.linear), ("mlp", args.mlp)):
        if Path(p).exists():
            head, kind, stats, _ = load_probe(Path(p), device)
            heads[k] = (head, stats)
            print(f"[load] {k} ← {p}")
        else:
            print(f"[skip] {k} ckpt missing: {p}")
    if not heads:
        raise SystemExit("No probe ckpts found.")

    data_dir = Path(cfg["data"]["data_dir"])
    manifest_path = Path(cfg["data"]["manifest_path"])
    feat_tag = cfg["encoder"]["feat_tag"]
    abs_value = bool(cfg["data"]["target"]["abs_value"])
    clip = float(cfg["data"]["target"]["clip"])

    if cfg["data"].get("train_cubes") and cfg["data"].get("val_cubes"):
        train_harps = sorted(cfg["data"]["train_cubes"])
        val_harps = sorted(cfg["data"]["val_cubes"])
    else:
        train_harps, val_harps = split_cubes_for_probe(
            manifest_path, val_frac=float(cfg["data"]["val_fraction"]),
            seed=int(cfg["data"]["split_seed"]),
            allowlist=cfg["data"].get("cube_allowlist"),
        )
    all_ids = sorted(json.loads(manifest_path.read_text())["cubes"], key=lambda e: e["harp_id"])
    novel_harps = [e["harp_id"] for e in all_ids
                   if e["harp_id"] not in set(train_harps) | set(val_harps)]

    eval_sets = []
    eval_sets.append(evaluate_set("encoder-val (held-out, in-allowlist)", val_harps, heads,
                                  data_dir, feat_tag, abs_value, clip, device))
    eval_sets.append(evaluate_set("novel cubes (encoder never saw)", novel_harps, heads,
                                  data_dir, feat_tag, abs_value, clip, device))
    eval_sets.append(evaluate_set("encoder-train (sanity, in-train)", train_harps, heads,
                                  data_dir, feat_tag, abs_value, clip, device))

    write_metrics_md(out_dir / "probe_metrics.md", eval_sets, list(heads.keys()))
    (out_dir / "probe_metrics.json").write_text(json.dumps(eval_sets, indent=2, default=float))

    overlay = args.overlay_cubes or (val_harps + novel_harps[:2])
    overlay = [h for h in overlay if any(h in s["per_cube"] for s in eval_sets)]
    plot_overlays(out_dir, overlay, data_dir, feat_tag, abs_value, clip, heads, device)
    plot_jepa_curve(out_dir, Path(args.jepa_run))
    print(f"[done] metrics + plots → {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
