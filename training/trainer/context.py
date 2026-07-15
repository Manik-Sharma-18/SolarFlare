"""Pre-loop assembly: parse config and build everything train_model's loop needs."""
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import torch

from utils.device import get_grad_scaler
from ..losses import get_loss_function
from .setup import (
    build_optimizer, build_scheduler, init_history, maybe_resume,
    apply_transfer_learning,
)
from .ema import build_ema, ModelEMA
from .reporting import print_config_banner


@dataclass
class TrainContext:
    """Everything the epoch loop reads (config-derived) or starts from (working state)."""
    # config-derived (read-only during the loop)
    epochs: int
    lr: float
    weight_decay: float
    tf_start: float
    tf_decay_epochs: int
    patience: int
    use_amp: bool
    grad_clip: float
    save_dir: Path
    show_progress: bool
    output_channels: int
    max_consecutive_nan: int
    grad_norm_warning_threshold: float
    loss_fn: torch.nn.Module
    scheduler_config: Dict[str, Any]
    scheduler_enabled: bool
    scheduler_type: str
    extreme_threshold: float
    verbose_metrics: bool
    ssim_data_range: float
    transfer_config: Optional[dict]
    checkpoints_dir: Path
    ema_use_for_eval: bool
    ema_use_for_checkpoint: bool
    # working state (mutated by the loop)
    optimizer: torch.optim.Optimizer
    scheduler: Any
    scaler: Any
    best_val_loss: float
    patience_counter: int
    start_epoch: int
    history: Dict[str, list]
    normalization_params: Optional[dict]
    unfreeze_epoch: Optional[int]
    ema: Optional[ModelEMA]


def prepare_training(model, config, device, normalization_params) -> TrainContext:
    """Parse config, build optimizer/scheduler/scaler/loss, resume + transfer learn."""
    epochs = config.get('epochs', 25)
    lr = config.get('lr', 1e-3)
    weight_decay = config.get('weight_decay', 1e-5)
    use_amp = config.get('use_amp', True)
    save_dir = Path(config.get('save_dir', './outputs'))
    save_dir.mkdir(parents=True, exist_ok=True)

    error_handling = config.get('error_handling', {})
    max_consecutive_nan = error_handling.get('max_consecutive_nan', 10)
    grad_norm_warning_threshold = error_handling.get('grad_norm_warning_threshold', 100.0)

    # Loss, optimizer, scheduler, scaler
    loss_config = config.get('loss', {'type': 'l1'})
    loss_fn = get_loss_function(loss_config).to(device)
    optimizer = build_optimizer(model, lr, weight_decay)
    scheduler_config = config.get('scheduler', {'type': 'cosine'})
    scheduler, scheduler_enabled, scheduler_type = build_scheduler(
        optimizer, scheduler_config, epochs
    )
    scaler = get_grad_scaler(use_amp, device)

    # Training state
    best_val_loss = float('inf')
    patience_counter = 0
    history = init_history()

    # Resume from checkpoint if requested
    resume_from = config.get('resume_from')
    start_epoch, best_val_loss, patience_counter, history, normalization_params = (
        maybe_resume(
            model, optimizer, scheduler, scaler, config, device,
            best_val_loss, patience_counter, history, normalization_params,
        )
    )

    # Transfer learning from pretrained checkpoint
    transfer_config = config.get('transfer_learning')
    unfreeze_epoch = None
    if transfer_config and transfer_config.get('pretrained_checkpoint'):
        optimizer, scheduler, scaler, unfreeze_epoch = apply_transfer_learning(
            model, transfer_config, optimizer, scheduler, scaler,
            lr=lr, weight_decay=weight_decay, epochs=epochs, use_amp=use_amp,
            start_epoch=start_epoch, device=device,
            scheduler_config=scheduler_config, scheduler_type=scheduler_type,
            scheduler_enabled=scheduler_enabled,
        )

    # EMA shadow (built after transfer-learning may have mutated params)
    ema_config = config.get('ema', {})
    ema = build_ema(model, ema_config)
    ema_use_for_eval = bool(ema_config.get('use_for_eval', True))
    ema_use_for_checkpoint = bool(ema_config.get('use_for_checkpoint', True))

    checkpoints_dir = save_dir / 'checkpoints'
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    print_config_banner(
        epochs, device, use_amp, loss_config, scheduler_type, config.get('tf_start', 0.5),
        max_consecutive_nan, grad_norm_warning_threshold, resume_from, start_epoch,
    )

    return TrainContext(
        epochs=epochs, lr=lr, weight_decay=weight_decay,
        tf_start=config.get('tf_start', 0.5),
        tf_decay_epochs=config.get('tf_decay_epochs', epochs),
        patience=config.get('patience', 8),
        use_amp=use_amp, grad_clip=config.get('grad_clip', 1.0), save_dir=save_dir,
        show_progress=config.get('show_progress', True),
        output_channels=config.get('output_channels', 1),
        max_consecutive_nan=max_consecutive_nan,
        grad_norm_warning_threshold=grad_norm_warning_threshold,
        loss_fn=loss_fn, scheduler_config=scheduler_config,
        scheduler_enabled=scheduler_enabled, scheduler_type=scheduler_type,
        extreme_threshold=config.get('evaluation', {}).get('extreme_threshold', 0.277),
        verbose_metrics=config.get('evaluation', {}).get('verbose_metrics', False),
        ssim_data_range=config.get('loss', {}).get('ssim_data_range', 2.0),
        transfer_config=transfer_config, checkpoints_dir=checkpoints_dir,
        optimizer=optimizer, scheduler=scheduler, scaler=scaler,
        best_val_loss=best_val_loss, patience_counter=patience_counter,
        start_epoch=start_epoch, history=history,
        normalization_params=normalization_params, unfreeze_epoch=unfreeze_epoch,
        ema=ema, ema_use_for_eval=ema_use_for_eval,
        ema_use_for_checkpoint=ema_use_for_checkpoint,
    )
