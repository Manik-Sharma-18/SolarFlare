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
        index: List[Tuple],
        t_in: int = 8,
        t_out: int = 3,
        dual_channel: bool = False,
        extreme_threshold: Optional[float] = None,
        norm_params: Optional[Dict] = None,
        norm_method: Optional[str] = None,
        window_size: Optional[int] = None,
        per_cube_norm: Optional[Dict[int, Dict[str, float]]] = None,
        is_pseudoscalar: bool = False,
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
            norm_method: ``"asinh"``, ``"linear"``, ``"zscore_per_cube"``,
                or ``"signed_asinh"`` (required when *norm_params* or
                *per_cube_norm* is provided).
            window_size: If set, the dataset uses spatial sliding-window
                indexing (5-tuple index). Window is square, ``window_size ×
                window_size``. If ``None``, full-frame slicing (3-tuple
                index, legacy behaviour).
            per_cube_norm: Optional mapping ``{file_idx: {mu, sigma}}`` for
                ``zscore_per_cube`` / ``signed_asinh`` methods. When set,
                *norm_params* may be a fallback for cubes missing from the
                dict.
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
        self.window_size = window_size
        self.per_cube_norm = per_cube_norm
        self.is_pseudoscalar = is_pseudoscalar

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
    def _apply_augmentation(data: np.ndarray, aug_type: int,
                            is_pseudoscalar: bool = False) -> np.ndarray:
        """Apply a deterministic spatial augmentation and return a contiguous copy.

        Args:
            data: Array of shape ``(T, H, W)``.
            aug_type: One of the ``AUG_*`` constants.
            is_pseudoscalar: When ``True`` (winding flux), parity-odd
                transforms (H-flip, V-flip, 90°, 270°) are paired with a
                sign flip; the identity and 180° rotation preserve sign.
                This is the D4 chirality rule from
                ``archive/v5_jepa/docs/V5_JEPA/06_data.md`` §11.4.

        Returns:
            Contiguous ``(T, H, W)`` array with the augmentation applied.
        """
        if aug_type == AUG_NONE:
            return data.copy()
        if aug_type == AUG_HFLIP:
            out = np.flip(data, axis=2).copy()
            return -out if is_pseudoscalar else out
        if aug_type == AUG_VFLIP:
            out = np.flip(data, axis=1).copy()
            return -out if is_pseudoscalar else out
        if aug_type == AUG_ROT90:
            out = np.rot90(data, k=1, axes=(1, 2)).copy()
            return -out if is_pseudoscalar else out
        if aug_type == AUG_ROT180:
            return np.rot90(data, k=2, axes=(1, 2)).copy()
        if aug_type == AUG_ROT270:
            out = np.rot90(data, k=3, axes=(1, 2)).copy()
            return -out if is_pseudoscalar else out
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
        entry = self.index[idx]
        if len(entry) == 5:
            file_idx, window_start, y_start, x_start, aug_type = entry
        else:
            file_idx, window_start, aug_type = entry
            y_start = x_start = None

        try:
            mmap_data = self._get_mmap(file_idx)

            t_end = window_start + self.t_in
            y_end_t = window_start + self.t_in + self.t_out
            if y_start is None:
                X_in = mmap_data[window_start:t_end]
                Y_out = mmap_data[t_end:y_end_t]
            else:
                win = self.window_size
                X_in = mmap_data[window_start:t_end,
                                 y_start:y_start + win,
                                 x_start:x_start + win]
                Y_out = mmap_data[t_end:y_end_t,
                                  y_start:y_start + win,
                                  x_start:x_start + win]

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

        # On-the-fly normalisation
        if self.per_cube_norm is not None and file_idx in self.per_cube_norm:
            stats = self.per_cube_norm[file_idx]
            mu = stats["mu"]
            sigma = stats["sigma"]
            if self.norm_method == "signed_asinh":
                softening = stats.get("softening",
                                      self.norm_params.get("softening", 1.0)
                                      if self.norm_params else 1.0)
                scale = stats.get("scale", sigma)
                X_in = np.sign(X_in) * np.arcsinh(np.abs(X_in) / softening) / scale
                Y_out = np.sign(Y_out) * np.arcsinh(np.abs(Y_out) / softening) / scale
            else:
                # zscore_per_cube
                X_in = (X_in - mu) / sigma
                Y_out = (Y_out - mu) / sigma
        elif self.norm_params is not None:
            if self.norm_method == "asinh":
                softening = self.norm_params["asinh_softening"]
                scale = self.norm_params["scale"]
                X_in = np.arcsinh(X_in / softening) / scale
                Y_out = np.arcsinh(Y_out / softening) / scale
            elif self.norm_method == "zscore_per_cube":
                # Fallback for val/test cubes: harp_loader populates
                # norm_params with the global (train-only) mu/sigma.
                mu = self.norm_params["mu"]
                sigma = self.norm_params["sigma"]
                X_in = (X_in - mu) / sigma
                Y_out = (Y_out - mu) / sigma
            elif self.norm_method == "signed_asinh":
                softening = self.norm_params.get("softening", 1.0)
                scale = self.norm_params.get("scale", self.norm_params.get("sigma", 1.0))
                X_in = np.sign(X_in) * np.arcsinh(np.abs(X_in) / softening) / scale
                Y_out = np.sign(Y_out) * np.arcsinh(np.abs(Y_out) / softening) / scale
            else:
                center = self.norm_params["center"]
                scale = self.norm_params["scale"]
                X_in = (X_in - center) / scale
                Y_out = (Y_out - center) / scale

        # Deterministic augmentation (sign-flip on chirality-flipping ops
        # when is_pseudoscalar is set, per V5 §11.4)
        X_in = self._apply_augmentation(X_in, aug_type, self.is_pseudoscalar)
        Y_out = self._apply_augmentation(Y_out, aug_type, self.is_pseudoscalar)

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
    flare_density_threshold: float = 0.02,
) -> Tuple[List[Tuple[int, int, int]], List[bool]]:
    """Build a precomputed sample index for a given data split.

    For each file assigned to *split*, generates sliding-window start
    positions with the given *stride*.  When *augmentation* is ``"balanced"``
    or ``"aggressive"`` **and** *split* is ``"train"``, each window is
    multiplied by the corresponding set of augmentation codes.  Validation
    and test splits always receive ``AUG_NONE`` only.

    When *extreme_threshold* is provided, scans **output frames only** for
    each window to detect extreme flux events using a spatial density
    criterion: the fraction of pixels with ``|value| > threshold`` must
    exceed *flare_density_threshold* (default 2%).  This enables
    flare-aware weighted sampling downstream.

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
            have spatial density of extreme pixels exceeding
            *flare_density_threshold*.  ``None`` disables flare detection
            (all flags ``False``).
        flare_density_threshold: Fraction of pixels above threshold to
            flag a window as containing a flare (default 0.02 = 2%).

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
            # Detect extreme values in OUTPUT frames only via spatial density
            if extreme_threshold is not None:
                output_frames = mmap[
                    window_start + t_in : window_start + t_in + t_out
                ]
                extreme_pixels = np.abs(output_frames) > extreme_threshold
                extreme_fraction = extreme_pixels.mean()  # across all output pixels
                is_flare = bool(extreme_fraction > flare_density_threshold)
            else:
                is_flare = False

            # Append one entry per augmentation code; all share same flare flag
            for aug in aug_codes:
                index.append((file_idx, window_start, aug))
                flare_flags.append(is_flare)

    return index, flare_flags


def _spatial_window_starts(extent: int, win: int, stride: int) -> List[int]:
    """Return window start positions over [0, extent) covering full extent.

    Stride < win produces overlapping windows. When ``(extent - win) % stride
    != 0``, an extra flush-right window at ``extent - win`` is appended so the
    boundary is covered without zero-padding leakage.
    """
    if extent < win:
        return []
    starts = list(range(0, extent - win + 1, stride))
    if starts[-1] != extent - win:
        starts.append(extent - win)
    return starts


def build_spatial_index(
    file_paths: List[str],
    file_assignments: Dict[str, List[int]],
    t_in: int,
    t_out: int,
    window_size: int,
    t_stride: int = 1,
    s_stride: Optional[int] = None,
    augmentation: str = "none",
    split: str = "train",
    extreme_threshold: Optional[float] = None,
    flare_density_threshold: float = 0.02,
) -> Tuple[List[Tuple[int, int, int, int, int]], List[bool]]:
    """Build a precomputed 5-tuple sample index with spatial sliding window.

    Each entry is ``(file_idx, t_start, y_start, x_start, aug_type)``. Window
    is square (``window_size × window_size``) so D4 rotations are legal.
    ``s_stride`` defaults to ``window_size // 2`` (50% overlap).

    Cubes whose smaller spatial axis is below ``window_size`` are skipped
    with a warning — train at the resolution you can support.

    Args:
        file_paths: Ordered list of densified cube ``.npy`` paths.
        file_assignments: Mapping split → list of file indices.
        t_in, t_out: Window temporal lengths.
        window_size: Spatial window side. Must be ``> 0``.
        t_stride: Temporal step between consecutive windows.
        s_stride: Spatial step. Defaults to ``window_size // 2``.
        augmentation, split, extreme_threshold, flare_density_threshold:
            Same as :func:`build_index`.

    Returns:
        ``(index, flare_flags)``.
    """
    if window_size is None or window_size <= 0:
        raise ValueError("window_size must be a positive integer")
    if s_stride is None:
        s_stride = max(1, window_size // 2)

    if split == "train" and augmentation == "balanced":
        aug_codes = _BALANCED_AUGS
    elif split == "train" and augmentation == "aggressive":
        aug_codes = _AGGRESSIVE_AUGS
    else:
        aug_codes = [AUG_NONE]

    index: List[Tuple[int, int, int, int, int]] = []
    flare_flags: List[bool] = []

    for file_idx in file_assignments.get(split, []):
        mmap = np.load(file_paths[file_idx], mmap_mode="r")
        T, H, W = mmap.shape
        max_t_start = T - t_in - t_out + 1
        if max_t_start <= 0:
            logger.warning(
                "File %s has only %d timesteps, need %d; skipping.",
                file_paths[file_idx], T, t_in + t_out,
            )
            continue
        if min(H, W) < window_size:
            logger.warning(
                "File %s has shape (%d, %d) smaller than window_size=%d; "
                "skipping.",
                file_paths[file_idx], H, W, window_size,
            )
            continue

        y_starts = _spatial_window_starts(H, window_size, s_stride)
        x_starts = _spatial_window_starts(W, window_size, s_stride)

        for t_start in range(0, max_t_start, t_stride):
            if extreme_threshold is not None:
                out_frames = mmap[t_start + t_in:t_start + t_in + t_out]
            for y in y_starts:
                for x in x_starts:
                    if extreme_threshold is not None:
                        tile = out_frames[:, y:y + window_size,
                                          x:x + window_size]
                        extreme_pixels = np.abs(tile) > extreme_threshold
                        is_flare = bool(
                            extreme_pixels.mean() > flare_density_threshold
                        )
                    else:
                        is_flare = False
                    for aug in aug_codes:
                        index.append((file_idx, t_start, y, x, aug))
                        flare_flags.append(is_flare)

    return index, flare_flags
