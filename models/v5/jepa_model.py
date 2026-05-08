"""V5 JEPA model (Path B): ViT context + EMA target + block-causal predictor.

Two forward paths, dispatched by `mask` arg:
  - mask=None  → autoregressive rollout in embedding space (sanity / no-mask).
  - mask given → MAE-style zero-token pretext: post-adapter zero at masked
                 patches, full-T context + clean full-T target encoder, single
                 predictor pass, smooth-L1 only at (mask & valid & non-pad).

Trainer must call `update_target_ema(decay)` after each optimizer step.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .input_adapter import InputAdapter, valid_pixel_to_token_mask
from .predictor import BlockCausalPredictor
from .vit_encoder import ViTEncoder


@dataclass
class JEPAConfig:
    t_in: int
    t_out: int
    patch_size: int = 16
    cadence_min: float = 12.0
    pixel_scale_mm: float = 0.364
    encoder_dim: int = 384
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
    valid_token_threshold: float = 0.5


class V5JEPAModel(nn.Module):
    def __init__(self, cfg: JEPAConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.adapter = InputAdapter(in_ch=1, out_ch=13, patch_size=cfg.patch_size)
        self.encoder = ViTEncoder(
            in_ch=13, embed_dim=cfg.encoder_dim, patch_size=cfg.patch_size,
            layers=cfg.encoder_layers, heads=cfg.encoder_heads,
            mlp_ratio=cfg.encoder_mlp_ratio, dropout=cfg.dropout,
            drop_path=cfg.drop_path, use_grad_checkpoint=cfg.grad_checkpoint,
        )
        self.target_encoder = copy.deepcopy(self.encoder)
        for p in self.target_encoder.parameters():
            p.requires_grad_(False)
        self.target_encoder.eval()
        self.predictor = BlockCausalPredictor(
            hidden=cfg.predictor_hidden, layers=cfg.predictor_layers,
            heads=cfg.predictor_heads, mlp_ratio=cfg.predictor_mlp_ratio,
            dropout=cfg.dropout, drop_path=cfg.drop_path,
            encoder_dim=cfg.encoder_dim, use_grad_checkpoint=cfg.grad_checkpoint,
        )

    @torch.no_grad()
    def update_target_ema(self, decay: float | None = None) -> None:
        d = self.cfg.target_ema_decay if decay is None else decay
        self.target_encoder.update_ema_from(self.encoder, d)

    def forward(
        self,
        x: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        rollout_steps: int | None = None,
    ) -> dict:
        if mask is not None:
            return self._forward_masked(x, valid_mask=valid_mask, mask=mask)
        return self._forward_rollout(x, rollout_steps=rollout_steps)

    def _forward_rollout(self, x: torch.Tensor, rollout_steps: int | None = None) -> dict:
        cfg = self.cfg
        if x.shape[1] != cfg.t_in + cfg.t_out:
            raise ValueError(f"Expected T={cfg.t_in + cfg.t_out}, got {x.shape[1]}")
        n_steps = rollout_steps if rollout_steps is not None else cfg.t_out
        n_steps = max(1, min(n_steps, cfg.t_out))

        x_adapt, token_pad_mask, _ = self.adapter(x)
        with torch.no_grad():
            z_target_full = self.target_encoder.encode_frames(
                x_adapt, cadence_min=cfg.cadence_min, pixel_scale_mm=cfg.pixel_scale_mm,
            )
        z_ctx = self.encoder.encode_frames(
            x_adapt[:, :cfg.t_in], cadence_min=cfg.cadence_min, pixel_scale_mm=cfg.pixel_scale_mm,
        )

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
            z_last = z_out.reshape(b, cur_t, tpf, d)[:, -1:]
            preds.append(z_last)
            z_seq = torch.cat([z_seq, z_last.reshape(b, tpf, d)], dim=1)

        z_pred = torch.cat(preds, dim=1)
        loss = F.smooth_l1_loss(z_pred, z_target)
        return {
            "z_pred": z_pred, "z_target": z_target, "loss": loss,
            "rollout_steps": n_steps, "token_pad_mask": token_pad_mask,
            "tokens_hw": (hp, wp),
        }

    def _forward_masked(
        self, x: torch.Tensor, valid_mask: torch.Tensor | None, mask: torch.Tensor,
    ) -> dict:
        cfg = self.cfg
        B, T = x.shape[0], x.shape[1]
        x_adapt, token_pad_mask, _ = self.adapter(x)
        Hp_pad, Wp_pad = x_adapt.shape[-2], x_adapt.shape[-1]
        hp, wp = mask.shape[-2:]
        if hp != Hp_pad // cfg.patch_size or wp != Wp_pad // cfg.patch_size:
            raise ValueError(
                f"mask grid {(hp, wp)} != padded patch grid "
                f"{(Hp_pad // cfg.patch_size, Wp_pad // cfg.patch_size)}"
            )

        # Post-adapter zero (pre-adapter would leak the 1×1 conv bias).
        m_pix = mask.float().reshape(B * T, 1, hp, wp)
        m_pix = F.interpolate(m_pix, scale_factor=cfg.patch_size, mode="nearest").bool()
        m_pix = m_pix.reshape(B, T, 1, Hp_pad, Wp_pad)
        x_ctx_in = x_adapt * (~m_pix).to(x_adapt.dtype)

        with torch.no_grad():
            z_target = self.target_encoder.encode_frames(
                x_adapt, cadence_min=cfg.cadence_min, pixel_scale_mm=cfg.pixel_scale_mm,
            )
        z_ctx = self.encoder.encode_frames(
            x_ctx_in, cadence_min=cfg.cadence_min, pixel_scale_mm=cfg.pixel_scale_mm,
        )

        b, _, _, _, d = z_target.shape
        tpf = hp * wp
        z_out = self.predictor(
            z_ctx.reshape(b, T * tpf, d), t_total=T, hp=hp, wp=wp, patch=cfg.patch_size,
            cadence_min=cfg.cadence_min, pixel_scale_mm=cfg.pixel_scale_mm,
            token_pad_mask=token_pad_mask,
        )
        z_pred = z_out.reshape(b, T, tpf, d)
        z_target = z_target.reshape(b, T, tpf, d).detach()

        valid_tok = valid_pixel_to_token_mask(
            valid_mask, hp, wp, Hp_pad, Wp_pad, cfg.patch_size,
            cfg.valid_token_threshold, x.device,
        )
        loss_mask = mask & valid_tok.expand(B, T, hp, wp) & token_pad_mask[None, None]
        flat = loss_mask.reshape(b, T, tpf)
        if flat.any():
            sel = flat.unsqueeze(-1).expand_as(z_pred)
            loss = F.smooth_l1_loss(z_pred[sel], z_target[sel])
        else:
            loss = z_pred.sum() * 0.0
        return {
            "z_pred": z_pred, "z_target": z_target, "loss": loss,
            "mask": mask, "loss_mask": loss_mask,
            "token_pad_mask": token_pad_mask, "tokens_hw": (hp, wp),
        }
