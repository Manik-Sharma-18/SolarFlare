"""
Config validation for SolarFlare training pipeline.

Validates config.yaml structure and values at startup, before any data loading
or model creation. Reports ALL errors at once so users can fix everything in
one pass.
"""
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class ConfigValidationError(Exception):
    """Raised when config validation finds one or more errors."""

    def __init__(self, errors: List[str]):
        self.errors = errors
        bullet_list = "\n".join(f"  - {e}" for e in errors)
        super().__init__(f"Config validation failed with {len(errors)} error(s):\n{bullet_list}")


def _get_nested(config: dict, dotted_key: str, default=None):
    """Retrieve a nested value using dotted notation (e.g. 'training.lr')."""
    keys = dotted_key.split(".")
    current = config
    for k in keys:
        if not isinstance(current, dict):
            return default
        current = current.get(k, default)
        if current is default:
            return default
    return current


def validate_config(config: dict) -> None:
    """
    Validate the full training config dict.

    Accumulates all errors and raises a single ConfigValidationError listing
    every problem. Warnings are logged but do not abort.

    Args:
        config: Parsed config dict (from yaml.safe_load on config.yaml).

    Raises:
        ConfigValidationError: If one or more validation errors are found.
    """
    errors: List[str] = []
    warnings: List[str] = []

    # ------------------------------------------------------------------ #
    # Helper closures
    # ------------------------------------------------------------------ #
    def _require_type(key: str, expected_type, value, label: Optional[str] = None):
        """Check type; return True if OK."""
        label = label or key
        if not isinstance(value, expected_type):
            type_name = expected_type.__name__ if not isinstance(expected_type, tuple) else "/".join(
                t.__name__ for t in expected_type)
            errors.append(f"'{label}' must be {type_name}, got {type(value).__name__}: {value!r}")
            return False
        return True

    # ------------------------------------------------------------------ #
    # Top-level fields
    # ------------------------------------------------------------------ #

    # device
    device = config.get("device")
    valid_devices = ("auto", "cuda", "mps", "cpu")
    if device is None:
        errors.append("'device' is required")
    elif not isinstance(device, str):
        errors.append(f"'device' must be a string, got {type(device).__name__}")
    elif device not in valid_devices:
        errors.append(f"'device' must be one of {valid_devices}, got '{device}'")

    # seed
    seed = config.get("seed")
    if seed is None:
        errors.append("'seed' is required")
    elif _require_type("seed", int, seed):
        if seed < 0:
            errors.append(f"'seed' must be >= 0, got {seed}")

    # ------------------------------------------------------------------ #
    # data section
    # ------------------------------------------------------------------ #
    data = config.get("data")
    if not isinstance(data, dict):
        errors.append("'data' section is required and must be a mapping")
    else:
        # At least one directory
        has_data_dir = isinstance(data.get("data_dir"), str)
        has_preprocessed_dir = isinstance(data.get("preprocessed_dir"), str)
        if not has_data_dir and not has_preprocessed_dir:
            errors.append("'data.data_dir' or 'data.preprocessed_dir' must be a non-empty string")

        for field in ("t_in", "t_out"):
            val = data.get(field)
            if val is None:
                errors.append(f"'data.{field}' is required")
            elif _require_type(f"data.{field}", int, val):
                if val <= 0:
                    errors.append(f"'data.{field}' must be positive, got {val}")

        # split_ratios (replaces train_split/val_split)
        split_ratios = data.get("split_ratios")
        old_train_split = data.get("train_split")
        old_val_split = data.get("val_split")

        if split_ratios is None:
            # Backward compat: convert old train_split/val_split to split_ratios
            if old_train_split is not None and old_val_split is not None:
                if isinstance(old_train_split, (int, float)) and isinstance(old_val_split, (int, float)):
                    test_split = 1.0 - old_train_split - old_val_split
                    data["split_ratios"] = [old_train_split, test_split, old_val_split]
                    logger.warning(
                        "Config deprecation: 'data.train_split' and 'data.val_split' are deprecated. "
                        "Use 'data.split_ratios: [%.2f, %.2f, %.2f]' instead.",
                        old_train_split, test_split, old_val_split,
                    )
                    split_ratios = data["split_ratios"]
            else:
                # Default split ratios
                data["split_ratios"] = [0.7, 0.2, 0.1]
                split_ratios = data["split_ratios"]

        if split_ratios is not None:
            if not isinstance(split_ratios, list) or len(split_ratios) != 3:
                errors.append("'data.split_ratios' must be a list of 3 numbers, e.g. [0.7, 0.2, 0.1]")
            else:
                all_numeric = True
                for i, r in enumerate(split_ratios):
                    if not isinstance(r, (int, float)):
                        errors.append(f"'data.split_ratios[{i}]' must be a number, got {type(r).__name__}")
                        all_numeric = False
                    elif r <= 0:
                        errors.append(f"'data.split_ratios[{i}]' must be > 0, got {r}")
                        all_numeric = False
                if all_numeric and len(split_ratios) == 3:
                    ratio_sum = sum(split_ratios)
                    if abs(ratio_sum - 1.0) > 0.01:
                        errors.append(
                            f"'data.split_ratios' must sum to ~1.0 (tolerance 0.01), got {split_ratios} = {ratio_sum}"
                        )

        # augmentation (replaces augment boolean)
        augmentation = data.get("augmentation")
        old_augment = data.get("augment")

        if augmentation is None:
            if old_augment is not None:
                # Backward compat: convert old boolean to mode string
                if old_augment is True:
                    data["augmentation"] = "balanced"
                else:
                    data["augmentation"] = "none"
                logger.warning(
                    "Config deprecation: 'data.augment' (bool) is deprecated. "
                    "Use 'data.augmentation: \"%s\"' instead.",
                    data["augmentation"],
                )
                augmentation = data["augmentation"]
            else:
                data["augmentation"] = "none"
                augmentation = "none"

        valid_augmentations = ("none", "balanced", "aggressive")
        if isinstance(augmentation, str) and augmentation not in valid_augmentations:
            errors.append(
                f"'data.augmentation' must be one of {valid_augmentations}, got '{augmentation}'"
            )
        elif not isinstance(augmentation, str):
            errors.append(f"'data.augmentation' must be a string, got {type(augmentation).__name__}")

        # stride
        stride = data.get("stride")
        if stride is None:
            data["stride"] = 1
        elif _require_type("data.stride", int, stride):
            if stride <= 0:
                errors.append(f"'data.stride' must be a positive integer, got {stride}")

        # num_workers
        num_workers = data.get("num_workers")
        if num_workers is None:
            data["num_workers"] = 0
        elif _require_type("data.num_workers", int, num_workers):
            if num_workers < 0:
                errors.append(f"'data.num_workers' must be a non-negative integer, got {num_workers}")

    # ------------------------------------------------------------------ #
    # model section
    # ------------------------------------------------------------------ #
    model = config.get("model")
    if not isinstance(model, dict):
        errors.append("'model' section is required and must be a mapping")
    else:
        kind = model.get("kind", "solar_flux")

        # input_channels
        ic = model.get("input_channels")
        if ic is None:
            errors.append("'model.input_channels' is required")
        elif _require_type("model.input_channels", int, ic):
            if ic <= 0:
                errors.append(f"'model.input_channels' must be positive, got {ic}")

        # channels list — only the deep SolarFluxPredictor uses a channel
        # pyramid. SimpleConvLSTM has a flat hidden_dim instead.
        if kind == "simple_convlstm":
            hd = model.get("hidden_dim")
            if hd is not None and (not isinstance(hd, int) or hd <= 0):
                errors.append(f"'model.hidden_dim' must be a positive int, got {hd!r}")
            nl = model.get("num_layers")
            if nl is not None and (not isinstance(nl, int) or nl <= 0):
                errors.append(f"'model.num_layers' must be a positive int, got {nl!r}")
        else:
            channels = model.get("channels")
            if channels is None:
                errors.append("'model.channels' is required")
            elif not isinstance(channels, list) or len(channels) < 1:
                errors.append("'model.channels' must be a list of positive ints with length >= 1")
            else:
                for i, ch in enumerate(channels):
                    if not isinstance(ch, int) or ch <= 0:
                        errors.append(f"'model.channels[{i}]' must be a positive int, got {ch!r}")

        # kernel_size
        ks = model.get("kernel_size")
        if ks is None:
            errors.append("'model.kernel_size' is required")
        elif _require_type("model.kernel_size", int, ks):
            if ks <= 0 or ks % 2 == 0:
                errors.append(f"'model.kernel_size' must be a positive odd int, got {ks}")

        # v3.0 architecture features (optional, safe defaults)
        for bool_key in ("use_sa_convlstm", "temporal_attention", "attention_gate"):
            val = model.get(bool_key)
            if val is not None and not isinstance(val, bool):
                errors.append(
                    f"'model.{bool_key}' must be a bool, got {type(val).__name__}: {val!r}"
                )

        dsi = model.get("delta_scale_init")
        if dsi is not None:
            if not isinstance(dsi, (int, float)):
                errors.append(
                    f"'model.delta_scale_init' must be a number, "
                    f"got {type(dsi).__name__}: {dsi!r}"
                )

    # ------------------------------------------------------------------ #
    # training section
    # ------------------------------------------------------------------ #
    training = config.get("training")
    if not isinstance(training, dict):
        errors.append("'training' section is required and must be a mapping")
    else:
        for field in ("batch_size", "epochs", "patience"):
            val = training.get(field)
            if val is None:
                errors.append(f"'training.{field}' is required")
            elif _require_type(f"training.{field}", int, val):
                if val <= 0:
                    errors.append(f"'training.{field}' must be positive, got {val}")

        for field in ("lr", "grad_clip"):
            val = training.get(field)
            if val is None:
                errors.append(f"'training.{field}' is required")
            elif _require_type(f"training.{field}", (int, float), val):
                if val <= 0:
                    errors.append(f"'training.{field}' must be positive, got {val}")

        use_amp = training.get("use_amp")
        if use_amp is not None and not isinstance(use_amp, bool):
            errors.append(f"'training.use_amp' must be a bool, got {type(use_amp).__name__}")

        # scheduler
        scheduler = training.get("scheduler")
        if isinstance(scheduler, dict):
            stype = scheduler.get("type")
            valid_schedulers = ("cosine", "step", "constant", "none", "cosine_warmup")
            if stype is not None and stype not in valid_schedulers:
                errors.append(f"'training.scheduler.type' must be one of {valid_schedulers}, got '{stype}'")

    # ------------------------------------------------------------------ #
    # loss section
    # ------------------------------------------------------------------ #
    loss = config.get("loss")
    if isinstance(loss, dict):
        ltype = loss.get("type")
        valid_loss_types = ("l1", "composite", "weighted", "dual_head", "quantile")
        if ltype is not None and ltype not in valid_loss_types:
            errors.append(f"'loss.type' must be one of {valid_loss_types}, got '{ltype}'")

        # ssim_tiling_threshold (optional, default 256)
        ssim_tile = loss.get("ssim_tiling_threshold")
        if ssim_tile is not None:
            if _require_type("loss.ssim_tiling_threshold", int, ssim_tile):
                if ssim_tile < 32:
                    errors.append(
                        f"'loss.ssim_tiling_threshold' must be >= 32 (SSIM window_size=11), got {ssim_tile}"
                    )

        # temporal_diff_weight (optional, default 1.0)
        tdw = loss.get("temporal_diff_weight")
        if tdw is not None:
            if _require_type("loss.temporal_diff_weight", (int, float), tdw):
                if tdw < 0:
                    errors.append(f"'loss.temporal_diff_weight' must be >= 0, got {tdw}")

        # temporal_var_lambda (optional, default 0.1)
        tvl = loss.get("temporal_var_lambda")
        if tvl is not None:
            if _require_type("loss.temporal_var_lambda", (int, float), tvl):
                if tvl < 0:
                    errors.append(f"'loss.temporal_var_lambda' must be >= 0, got {tvl}")

        # temporal_weights (optional, default [1.0, 1.5, 2.0, 2.5])
        tw = loss.get("temporal_weights")
        if tw is not None:
            if not isinstance(tw, list):
                errors.append(
                    f"'loss.temporal_weights' must be a list of numbers, "
                    f"got {type(tw).__name__}: {tw!r}"
                )
            else:
                for i, w in enumerate(tw):
                    if not isinstance(w, (int, float)):
                        errors.append(
                            f"'loss.temporal_weights[{i}]' must be a number, "
                            f"got {type(w).__name__}: {w!r}"
                        )

        # asymmetric_weight (optional, default 0.5)
        aw = loss.get("asymmetric_weight")
        if aw is not None:
            if _require_type("loss.asymmetric_weight", (int, float), aw):
                if aw < 0:
                    errors.append(f"'loss.asymmetric_weight' must be >= 0, got {aw}")

        # asymmetric_alpha (optional, default 2.0)
        aa = loss.get("asymmetric_alpha")
        if aa is not None:
            if _require_type("loss.asymmetric_alpha", (int, float), aa):
                if aa < 1.0:
                    errors.append(f"'loss.asymmetric_alpha' must be >= 1.0, got {aa}")

        # extreme_pixel_weight (optional, default 3.0)
        epw = loss.get("extreme_pixel_weight")
        if epw is not None:
            if _require_type("loss.extreme_pixel_weight", (int, float), epw):
                if epw < 1.0:
                    errors.append(f"'loss.extreme_pixel_weight' must be >= 1.0, got {epw}")

        # extreme_threshold (optional, default 0.277)
        et_loss = loss.get("extreme_threshold")
        if et_loss is not None:
            if _require_type("loss.extreme_threshold", (int, float), et_loss):
                if et_loss <= 0:
                    errors.append(f"'loss.extreme_threshold' must be > 0, got {et_loss}")

        # Cross-check: warn if loss.extreme_threshold differs from evaluation.extreme_threshold
        eval_section = config.get("evaluation", {})
        if isinstance(eval_section, dict) and et_loss is not None:
            eval_et = eval_section.get("extreme_threshold")
            if eval_et is not None and isinstance(eval_et, (int, float)):
                if abs(et_loss - eval_et) > 1e-6:
                    warnings.append(
                        f"loss.extreme_threshold ({et_loss}) differs from "
                        f"evaluation.extreme_threshold ({eval_et}); "
                        f"consider keeping them consistent"
                    )

    # ------------------------------------------------------------------ #
    # evaluation section (optional)
    # ------------------------------------------------------------------ #
    evaluation = config.get("evaluation")
    if isinstance(evaluation, dict):
        # extreme_threshold
        et = evaluation.get("extreme_threshold")
        if et is not None:
            if _require_type("evaluation.extreme_threshold", (int, float), et):
                if et <= 0:
                    errors.append(
                        f"'evaluation.extreme_threshold' must be positive, got {et}"
                    )
                elif et > 1.0:
                    warnings.append(
                        f"evaluation.extreme_threshold is {et} (> 1.0); "
                        f"ensure this is correct for your normalized data space"
                    )
                elif et < 0.01:
                    warnings.append(
                        f"evaluation.extreme_threshold is {et} (< 0.01); "
                        f"very low threshold may classify most values as extreme"
                    )

        # verbose_metrics
        vm = evaluation.get("verbose_metrics")
        if vm is not None and not isinstance(vm, bool):
            errors.append(
                f"'evaluation.verbose_metrics' must be a bool, got {type(vm).__name__}: {vm!r}"
            )

    # ------------------------------------------------------------------ #
    # normalization section
    # ------------------------------------------------------------------ #
    norm = config.get("normalization")
    if isinstance(norm, dict):
        nmethod = norm.get("method")
        valid_norm = ("asinh", "robust", "fixed", "zscore_per_cube", "signed_asinh")
        if nmethod is not None and nmethod not in valid_norm:
            errors.append(f"'normalization.method' must be one of {valid_norm}, got '{nmethod}'")

    # ------------------------------------------------------------------ #
    # Cross-check: flare_oversample_weight + augmentation
    # ------------------------------------------------------------------ #
    data_section = config.get("data", {})
    if isinstance(data_section, dict):
        fow = data_section.get("flare_oversample_weight")
        aug_mode = data_section.get("augmentation", "none")
        if isinstance(fow, (int, float)) and fow > 1.0:
            if aug_mode == "none":
                warnings.append(
                    f"flare_oversample_weight is {fow} but augmentation is 'none'; "
                    f"consider enabling augmentation ('balanced') to increase "
                    f"diversity of oversampled flare sequences"
                )

    # ------------------------------------------------------------------ #
    # Cross-field validation (errors)
    # ------------------------------------------------------------------ #
    data = config.get("data", {})
    model = config.get("model", {})
    training = config.get("training", {})

    # dual_channel vs input_channels
    dual_channel = data.get("dual_channel", False)
    input_channels = model.get("input_channels")
    if dual_channel and isinstance(input_channels, int) and input_channels < 2:
        errors.append(
            f"dual_channel is enabled but input_channels is {input_channels} (must be >= 2)"
        )

    # AMP on CPU
    use_amp = training.get("use_amp", False)
    device_val = config.get("device", "auto")
    if use_amp and isinstance(device_val, str) and device_val == "cpu":
        errors.append("AMP enabled but device is 'cpu'; AMP requires CUDA or MPS")

    # ------------------------------------------------------------------ #
    # resume_from validation
    # ------------------------------------------------------------------ #
    resume_from = config.get("resume_from")
    if resume_from is not None and resume_from != "":
        resume_path = Path(resume_from)
        if not resume_path.exists():
            errors.append(f"resume_from: file not found: {resume_from}")
        if not str(resume_from).endswith(".pt"):
            logger.warning("Config warning: resume_from does not end with .pt: %s", resume_from)

    # ------------------------------------------------------------------ #
    # transfer_learning section (optional)
    # ------------------------------------------------------------------ #
    transfer = config.get("transfer_learning")
    if transfer is not None:
        if not isinstance(transfer, dict):
            errors.append("'transfer_learning' must be a mapping")
        else:
            ptc = transfer.get("pretrained_checkpoint")
            if ptc is not None:
                if not isinstance(ptc, str):
                    errors.append("'transfer_learning.pretrained_checkpoint' must be a string")
                elif not Path(ptc).exists():
                    errors.append(f"transfer_learning.pretrained_checkpoint not found: {ptc}")

            mode = transfer.get("mode", "finetune")
            if mode not in ("finetune", "feature_extract"):
                errors.append(
                    f"'transfer_learning.mode' must be 'finetune' or 'feature_extract', got '{mode}'"
                )

            uae = transfer.get("unfreeze_after_epochs")
            if uae is not None and (not isinstance(uae, int) or uae < 0):
                errors.append("'transfer_learning.unfreeze_after_epochs' must be a non-negative int")

            lrsp = transfer.get("lr_scale_pretrained")
            if lrsp is not None:
                if not isinstance(lrsp, (int, float)) or lrsp <= 0 or lrsp > 1.0:
                    errors.append("'transfer_learning.lr_scale_pretrained' must be in (0, 1.0]")

            # Mutual exclusion with resume_from
            if resume_from is not None and resume_from != "" and ptc is not None:
                errors.append(
                    "Cannot use both 'resume_from' and 'transfer_learning.pretrained_checkpoint'. "
                    "Use resume_from to continue training, or transfer_learning to fine-tune."
                )

    # ------------------------------------------------------------------ #
    # Warnings (do not abort)
    # ------------------------------------------------------------------ #
    lr = training.get("lr")
    if isinstance(lr, (int, float)) and lr > 0.01:
        warnings.append(f"Learning rate {lr} is unusually high")

    batch_size = training.get("batch_size")
    if isinstance(batch_size, int) and batch_size > 16:
        warnings.append(f"Batch size {batch_size} is high for spatiotemporal models; may cause OOM")

    patience = training.get("patience")
    epochs = training.get("epochs")
    if isinstance(patience, int) and isinstance(epochs, int) and patience > epochs:
        warnings.append(f"patience ({patience}) exceeds epochs ({epochs}); early stopping will never trigger")

    # uncertainty section
    uncertainty = config.get("uncertainty", {})
    uncertainty_enabled = uncertainty.get("enabled", False) if isinstance(uncertainty, dict) else False

    if isinstance(uncertainty, dict):
        n_samples = uncertainty.get("n_samples")
        if n_samples is not None:
            if _require_type("uncertainty.n_samples", int, n_samples):
                if n_samples < 2:
                    errors.append(
                        f"'uncertainty.n_samples' must be >= 2, got {n_samples}"
                    )
                elif n_samples > 100:
                    warnings.append(
                        f"uncertainty.n_samples is {n_samples}; values > 100 are very slow"
                    )

    # dropout + uncertainty
    dropout_rate = model.get("dropout_rate", 0.0)
    if isinstance(dropout_rate, (int, float)) and dropout_rate > 0 and not uncertainty_enabled:
        warnings.append("Dropout enabled but uncertainty estimation disabled")

    # ------------------------------------------------------------------ #
    # Emit warnings
    # ------------------------------------------------------------------ #
    for w in warnings:
        logger.warning("Config warning: %s", w)

    # ------------------------------------------------------------------ #
    # Raise if errors
    # ------------------------------------------------------------------ #
    if errors:
        raise ConfigValidationError(errors)
