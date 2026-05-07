"""V5 JEPA model (Path B): ViT context encoder + EMA target encoder + block-causal predictor.

Forward contract (training):
    x:     [B, T_total=t_in+t_out, 1, H, W]   z-scored wind frames
    valid: [B, T_total, 1, H, W] bool          NaN-origin mask (currently unused; reserved)
Returns dict with:
    z_pred:   [B, n_steps, tokens_per_frame, encoder_dim]   predictor output for future frames
    z_target: [B, n_steps, tokens_per_frame, encoder_dim]   EMA-target embedding of true future
    loss:     smooth-L1 between z_pred and z_target

Anti-collapse: target_encoder is an EMA copy of context encoder, gradient-detached.
Trainer must call `update_target_ema(decay)` after each optimizer step.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .input_adapter import InputAdapter
from .predictor import BlockCausalPredictor
from .vit_encoder import ViTEncoder


@dataclass
class JEPAConfig:
    t_in: int
    t_out: int
    patch_size: int = 16
    cadence_min: float = 12.0
    pixel_scale_mm: float = 0.364
    encoder_dim: int = 384            # ViT-Small default
    encoder_layers: int = 12
    encoder_heads: int = 6
    encoder_mlp_ratio: int = 4
    predictor_hidden: int = 384
    predictor_layers: int = 6
    predictor_heads: int = 6
    predictor_mlp_ratio: int = 4
    dropout: float = 0.0
    drop_path: float = 0.1
    target_ema_decay: float = 0.996
    grad_checkpoint: bool = False


class V5JEPAModel(nn.Module):
    def __init__(self, cfg: JEPAConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.adapter = InputAdapter(in_ch=1, out_ch=13, patch_size=cfg.patch_size)
        self.encoder = ViTEncoder(
            in_ch=13,
            embed_dim=cfg.encoder_dim,
            patch_size=cfg.patch_size,
            layers=cfg.encoder_layers,
            heads=cfg.encoder_heads,
            mlp_ratio=cfg.encoder_mlp_ratio,
            dropout=cfg.dropout,
            drop_path=cfg.drop_path,
            use_grad_checkpoint=cfg.grad_checkpoint,
        )
        self.target_encoder = copy.deepcopy(self.encoder)
        for p in self.target_encoder.parameters():
            p.requires_grad_(False)
        self.target_encoder.eval()

        self.predictor = BlockCausalPredictor(
            hidden=cfg.predictor_hidden,
            layers=cfg.predictor_layers,
            heads=cfg.predictor_heads,
            mlp_ratio=cfg.predictor_mlp_ratio,
            dropout=cfg.dropout,
            drop_path=cfg.drop_path,
            encoder_dim=cfg.encoder_dim,
            use_grad_checkpoint=cfg.grad_checkpoint,
        )

    @torch.no_grad()
    def update_target_ema(self, decay: float | None = None) -> None:
        d = self.cfg.target_ema_decay if decay is None else decay
        self.target_encoder.update_ema_from(self.encoder, d)

    def forward(
        self,
        x: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
        rollout_steps: int | None = None,
    ) -> dict:
        """Autoregressive rollout in embedding space.

        - context_encoder runs on input frames [0:t_in] (gradient-trained).
        - target_encoder runs on full sequence under no_grad → slice future for z_target.
        - predictor extends embeddings autoregressively; last-position output = next-frame pred.
        """
        cfg = self.cfg
        if x.shape[1] != cfg.t_in + cfg.t_out:
            raise ValueError(f"Expected T={cfg.t_in + cfg.t_out}, got {x.shape[1]}")
        n_steps = rollout_steps if rollout_steps is not None else cfg.t_out
        n_steps = max(1, min(n_steps, cfg.t_out))

        x_adapt, token_pad_mask, _ = self.adapter(x)              # [B, T, 13, Hp, Wp]

        with torch.no_grad():
            z_target_full = self.target_encoder.encode_frames(
                x_adapt, cadence_min=cfg.cadence_min, pixel_scale_mm=cfg.pixel_scale_mm,
            )                                                     # [B, T, hp, wp, D]
        z_ctx = self.encoder.encode_frames(
            x_adapt[:, :cfg.t_in], cadence_min=cfg.cadence_min, pixel_scale_mm=cfg.pixel_scale_mm,
        )                                                         # [B, t_in, hp, wp, D]

        b, _, hp, wp, d = z_target_full.shape
        tpf = hp * wp
        z_target = z_target_full[:, cfg.t_in:cfg.t_in + n_steps].reshape(b, n_steps, tpf, d).detach()
        z_seq = z_ctx.reshape(b, cfg.t_in * tpf, d)

        preds = []
        for step in range(n_steps):
            cur_t = cfg.t_in + step
            z_out = self.predictor(
                z_seq, t_total=cur_t, hp=hp, wp=wp, patch=cfg.patch_size,
                cadence_min=cfg.cadence_min, pixel_scale_mm=cfg.pixel_scale_mm,
                token_pad_mask=token_pad_mask,
            )
            z_last = z_out.reshape(b, cur_t, tpf, d)[:, -1:]      # [B, 1, tpf, D]
            preds.append(z_last)
            z_seq = torch.cat([z_seq, z_last.reshape(b, tpf, d)], dim=1)

        z_pred = torch.cat(preds, dim=1)                          # [B, n_steps, tpf, D]
        loss = F.smooth_l1_loss(z_pred, z_target)

        return {
            "z_pred": z_pred,
            "z_target": z_target,
            "loss": loss,
            "rollout_steps": n_steps,
            "token_pad_mask": token_pad_mask,
            "tokens_hw": (hp, wp),
        }
