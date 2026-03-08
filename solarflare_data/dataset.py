"""
Memory-mapped Dataset for solar flux prediction.

Uses lazy-open mmap handles per worker, precomputed sliding window index
with configurable stride, and deterministic augmentation via index
multiplication (no randomness in __getitem__).

Supports dual-channel mode for background + extreme event detection.
"""
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Augmentation constants -- used as integer codes in the precomputed index
# ---------------------------------------------------------------------------
AUG_NONE = 0
AUG_HFLIP = 1
AUG_VFLIP = 2
AUG_ROT90 = 3
AUG_ROT180 = 4
AUG_ROT270 = 5

_BALANCED_AUGS = [AUG_NONE, AUG_HFLIP, AUG_VFLIP]
_AGGRESSIVE_AUGS = [AUG_NONE, AUG_HFLIP, AUG_VFLIP, AUG_ROT90, AUG_ROT180, AUG_ROT270]


class SolarFluxDataset(Dataset):
    """
    PyTorch Dataset for solar flux prediction with sliding windows.

    Stores only file paths and a precomputed index of
    ``(file_idx, window_start, aug_type)`` tuples.  Mmap handles are opened
    lazily on first access so each DataLoader worker gets its own file
    descriptor (safe for both *spawn* and *fork* start methods).

    Dual-channel mode:
        - Channel 1: Flux values (optionally normalized on-the-fly)
        - Channel 2: Extreme event indicator (|flux| > threshold)
    """

    def __init__(
        self,
        file_paths: List[str],
        index: List[Tuple[int, int, int]],
        t_in: int = 8,
        t_out: int = 3,
        dual_channel: bool = False,
        extreme_threshold: Optional[float] = None,
        norm_params: Optional[Dict] = None,
        norm_method: Optional[str] = None,
    ):
        """
        Args:
            file_paths: Absolute paths to ``.npy`` flux cubes, each ``(T, H, W)``.
            index: Precomputed list of ``(file_idx, window_start, aug_type)``
                tuples.  Use :func:`build_index` to generate this.
            t_in: Number of input timesteps per window.
            t_out: Number of output (target) timesteps per window.
            dual_channel: If True, output 2 channels (flux + extreme indicator).
            extreme_threshold: Threshold for extreme events (in raw/normalized space).
            norm_params: Optional dict of normalisation parameters applied
                on-the-fly.  Keys depend on *norm_method*:
                - ``asinh``: ``{"asinh_softening": float, "scale": float}``
                - ``linear``: ``{"center": float, "scale": float}``
            norm_method: ``"asinh"`` or ``"linear"`` (required when
                *norm_params* is provided).
        """
        # Serializable attributes only -- no numpy arrays or mmap handles
        self.file_paths = list(file_paths)
        self.index = list(index)
        self.t_in = t_in
        self.t_out = t_out
        self.dual_channel = dual_channel
        self.extreme_threshold = extreme_threshold
        self.norm_params = norm_params
        self.norm_method = norm_method

        # Populated lazily per worker via _get_mmap
        self._mmap_cache: Dict[int, np.ndarray] = {}

    # ------------------------------------------------------------------
    # Mmap management
    # ------------------------------------------------------------------

    def _get_mmap(self, file_idx: int) -> np.ndarray:
        """Return a read-only mmap handle for *file_idx*, opening on first use.

        Each DataLoader worker process calls this independently, so every
        worker ends up with its own file descriptor -- no cross-process
        sharing of mmap pages.
        """
        if file_idx not in self._mmap_cache:
            self._mmap_cache[file_idx] = np.load(
                self.file_paths[file_idx], mmap_mode="r"
            )
        return self._mmap_cache[file_idx]

    # ------------------------------------------------------------------
    # Augmentation
    # ------------------------------------------------------------------

    @staticmethod
    def _apply_augmentation(data: np.ndarray, aug_type: int) -> np.ndarray:
        """Apply a deterministic spatial augmentation and return a contiguous copy.

        Args:
            data: Array of shape ``(T, H, W)``.
            aug_type: One of the ``AUG_*`` constants.

        Returns:
            Contiguous ``(T, H, W)`` array with the augmentation applied.
        """
        if aug_type == AUG_NONE:
            return data.copy()
        if aug_type == AUG_HFLIP:
            return np.flip(data, axis=2).copy()
        if aug_type == AUG_VFLIP:
            return np.flip(data, axis=1).copy()
        if aug_type == AUG_ROT90:
            return np.rot90(data, k=1, axes=(1, 2)).copy()
        if aug_type == AUG_ROT180:
            return np.rot90(data, k=2, axes=(1, 2)).copy()
        if aug_type == AUG_ROT270:
            return np.rot90(data, k=3, axes=(1, 2)).copy()
        # Fallback -- treat unknown codes as no augmentation
        return data.copy()

    # ------------------------------------------------------------------
    # Extreme-event indicator (preserved from original implementation)
    # ------------------------------------------------------------------

    def _compute_extreme_channel(self, flux: np.ndarray) -> np.ndarray:
        """Compute extreme event indicator channel.

        Uses a soft sigmoid-like activation to highlight regions with
        extreme flux values, providing gradient information for training.

        Args:
            flux: Flux values ``(T, H, W)``.

        Returns:
            Extreme indicator ``(T, H, W)`` in range ``[0, 1]``.
        """
        abs_flux = np.abs(flux)
        scaled = (abs_flux - self.extreme_threshold) / (
            self.extreme_threshold * 0.5 + 1e-6
        )
        extreme = 1.0 / (1.0 + np.exp(-scaled * 2))
        return extreme.astype(np.float32)

    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(
        self, idx: int
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor, Tuple[int, int]]]:
        """Return a single ``(X_in, Y_out, info)`` sample.

        Returns:
            X_in: Input tensor ``(C, T_in, H, W)`` where ``C=1`` or ``2``.
            Y_out: Target tensor ``(C, T_out, H, W)``.
            info: ``(file_idx, window_start)`` for provenance tracking.
            *None* on unrecoverable read error (use a custom collate_fn
            to skip ``None`` samples).
        """
        file_idx, window_start, aug_type = self.index[idx]

        try:
            mmap_data = self._get_mmap(file_idx)

            # Slice windows from mmap view
            X_in = mmap_data[window_start : window_start + self.t_in]
            Y_out = mmap_data[
                window_start + self.t_in : window_start + self.t_in + self.t_out
            ]

            # CRITICAL: copy before any modification (mmap is read-only)
            X_in = X_in.copy()
            Y_out = Y_out.copy()
        except Exception:
            logger.warning(
                "Failed to read sample idx=%d file=%s window_start=%d",
                idx,
                self.file_paths[file_idx],
                window_start,
                exc_info=True,
            )
            return None

        # On-the-fly normalization
        if self.norm_params is not None:
            if self.norm_method == "asinh":
                softening = self.norm_params["asinh_softening"]
                scale = self.norm_params["scale"]
                X_in = np.arcsinh(X_in / softening) / scale
                Y_out = np.arcsinh(Y_out / softening) / scale
            else:
                center = self.norm_params["center"]
                scale = self.norm_params["scale"]
                X_in = (X_in - center) / scale
                Y_out = (Y_out - center) / scale

        # Deterministic augmentation
        X_in = self._apply_augmentation(X_in, aug_type)
        Y_out = self._apply_augmentation(Y_out, aug_type)

        # Dual-channel extreme indicator
        if self.dual_channel and self.extreme_threshold is not None:
            X_extreme = self._compute_extreme_channel(X_in)
            Y_extreme = self._compute_extreme_channel(Y_out)
            X_in = np.stack([X_in, X_extreme], axis=0)  # (2, T, H, W)
            Y_out = np.stack([Y_out, Y_extreme], axis=0)
            X_in = torch.from_numpy(X_in.copy()).float()
            Y_out = torch.from_numpy(Y_out.copy()).float()
        else:
            X_in = torch.from_numpy(X_in).float().unsqueeze(0)  # (1, T, H, W)
            Y_out = torch.from_numpy(Y_out).float().unsqueeze(0)

        return X_in, Y_out, (file_idx, window_start)


# -----------------------------------------------------------------------
# Index builder
# -----------------------------------------------------------------------


def build_index(
    file_paths: List[str],
    file_assignments: Dict[str, List[int]],
    t_in: int,
    t_out: int,
    stride: int = 1,
    augmentation: str = "none",
    split: str = "train",
    extreme_threshold: Optional[float] = None,
) -> Tuple[List[Tuple[int, int, int]], List[bool]]:
    """Build a precomputed sample index for a given data split.

    For each file assigned to *split*, generates sliding-window start
    positions with the given *stride*.  When *augmentation* is ``"balanced"``
    or ``"aggressive"`` **and** *split* is ``"train"``, each window is
    multiplied by the corresponding set of augmentation codes.  Validation
    and test splits always receive ``AUG_NONE`` only.

    When *extreme_threshold* is provided, scans **output frames only** for
    each window to detect extreme flux events (any pixel > threshold).
    This enables flare-aware weighted sampling downstream.

    Args:
        file_paths: Ordered list of ``.npy`` file paths.
        file_assignments: Mapping from split name (``"train"``, ``"val"``,
            ``"test"``) to lists of file indices into *file_paths*.
        t_in: Number of input timesteps.
        t_out: Number of target timesteps.
        stride: Step size between consecutive window starts.  Defaults to 1.
        augmentation: One of ``"none"``, ``"balanced"``, ``"aggressive"``.
        split: Split name to build the index for.
        extreme_threshold: If provided, flag windows whose output frames
            contain any pixel above this value.  ``None`` disables flare
            detection (all flags ``False``).

    Returns:
        ``(index, flare_flags)`` where *index* is a list of
        ``(file_idx, window_start, aug_type)`` tuples and *flare_flags* is
        a parallel list of booleans indicating extreme-event windows.
    """
    # Determine which augmentation codes to apply
    if split == "train" and augmentation == "balanced":
        aug_codes = _BALANCED_AUGS
    elif split == "train" and augmentation == "aggressive":
        aug_codes = _AGGRESSIVE_AUGS
    else:
        aug_codes = [AUG_NONE]

    index: List[Tuple[int, int, int]] = []
    flare_flags: List[bool] = []

    for file_idx in file_assignments.get(split, []):
        # Open briefly to read shape -- mmap so we only touch metadata
        mmap = np.load(file_paths[file_idx], mmap_mode="r")
        T = mmap.shape[0]

        max_start = T - t_in - t_out + 1
        if max_start <= 0:
            logger.warning(
                "File %s has only %d timesteps, need %d; skipping.",
                file_paths[file_idx],
                T,
                t_in + t_out,
            )
            continue

        for window_start in range(0, max_start, stride):
            # Detect extreme values in OUTPUT frames only
            if extreme_threshold is not None:
                output_frames = mmap[
                    window_start + t_in : window_start + t_in + t_out
                ]
                is_flare = bool(np.any(output_frames > extreme_threshold))
            else:
                is_flare = False

            # Append one entry per augmentation code; all share same flare flag
            for aug in aug_codes:
                index.append((file_idx, window_start, aug))
                flare_flags.append(is_flare)

    return index, flare_flags
