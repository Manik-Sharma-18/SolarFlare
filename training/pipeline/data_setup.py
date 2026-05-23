"""Dataset + DataLoader construction.

Dispatches on ``config['data']['loader']``:

- ``harp_zarr`` → :func:`solarflare_data.load_harp_zarr_data` (V4 default)
- ``preprocessed`` (or ``use_preprocessed=True``) → :func:`solarflare_data.load_preprocessed_data`
- otherwise → :func:`solarflare_data.load_and_prepare_data`

All branches return ``(train_dataset, val_dataset, test_dataset, metadata)``,
then we wrap them with :func:`solarflare_data.loader.create_dataloaders`
(which handles flare oversampling).
"""
from typing import Any, Dict, Tuple

import torch

from solarflare_data import (
    load_and_prepare_data,
    load_preprocessed_data,
    load_harp_zarr_data,
)
from solarflare_data.loader import create_dataloaders


def _resolve_flare_threshold(config: Dict[str, Any], oversample_weight: float):
    if oversample_weight <= 1.0:
        return None
    return config.get("evaluation", {}).get("extreme_threshold", 0.277)


def _build_datasets(config: Dict[str, Any]) -> Tuple[Any, Any, Any, Dict[str, Any]]:
    data_cfg = config["data"]
    loader_kind = data_cfg.get("loader", "auto")
    failure_threshold = config.get("error_handling", {}).get(
        "data_failure_threshold", 0.1
    )
    seed = config.get("seed", 42)
    oversample = data_cfg.get("flare_oversample_weight", 1.0)
    common = dict(
        t_in=data_cfg["t_in"],
        t_out=data_cfg["t_out"],
        split_ratios=data_cfg.get("split_ratios", [0.7, 0.2, 0.1]),
        stride=data_cfg.get("stride", 1),
        augmentation=data_cfg.get("augmentation", "none"),
        dual_channel=data_cfg.get("dual_channel", False),
        seed=seed,
        flare_extreme_threshold=_resolve_flare_threshold(config, oversample),
        crop_size=tuple(data_cfg["crop_size"]) if data_cfg.get("crop_size") else None,
        flare_density_threshold=data_cfg.get("flare_density_threshold", 0.02),
    )

    if loader_kind == "harp_zarr":
        return load_harp_zarr_data(
            data_dir=data_cfg["data_dir"],
            norm_method=config["normalization"]["method"],
            norm_config=config["normalization"],
            cube_allowlist=data_cfg.get("cube_allowlist"),
            clip=float(data_cfg.get("clip", 1e8)),
            window_size=data_cfg.get("window_size"),
            window_stride=data_cfg.get("window_stride"),
            **common,
        )
    if data_cfg.get("use_preprocessed", False):
        return load_preprocessed_data(
            preprocessed_dir=data_cfg["preprocessed_dir"],
            failure_threshold=failure_threshold,
            **common,
        )
    return load_and_prepare_data(
        data_dir=data_cfg["data_dir"],
        norm_method=config["normalization"]["method"],
        norm_config=config["normalization"],
        failure_threshold=failure_threshold,
        **common,
    )


def build_loaders(
    config: Dict[str, Any], device: torch.device
) -> Tuple[Any, Any, Any, Any, Dict[str, Any]]:
    """Build train/val/test loaders + return ``test_dataset`` and ``metadata``.

    Returns:
        ``(train_loader, val_loader, test_loader, test_dataset, metadata)``.
    """
    train_dataset, val_dataset, test_dataset, metadata = _build_datasets(config)

    train_flare_flags = metadata.get("train_flare_flags")
    oversample = config["data"].get("flare_oversample_weight", 1.0)

    train_loader, val_loader, test_loader = create_dataloaders(
        train_dataset, val_dataset, test_dataset,
        batch_size=config["training"]["batch_size"],
        num_workers=config["data"].get("num_workers", 0),
        device=device,
        seed=config.get("seed", 42),
        train_flare_flags=train_flare_flags,
        flare_oversample_weight=oversample,
    )

    if train_flare_flags and oversample > 1.0:
        n_flare = sum(train_flare_flags)
        n_total = len(train_flare_flags)
        print(
            f"  Flare sampling: {n_flare}/{n_total} sequences contain flares "
            f"({100 * n_flare / n_total:.1f}%), oversample weight: {oversample}x"
        )
    return train_loader, val_loader, test_loader, test_dataset, metadata
