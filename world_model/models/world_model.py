"""Latent world model (Sec. 4.2, App. B.1).

A frozen V-JEPA 2 ViT-L encoder and an action-conditioned predictor trained
from scratch, following V-JEPA 2-AC. Each 2 Hz latent is encoded from a tubelet
of two consecutive frames ``[I_{t-0.25s}, I_t]`` so that it carries the motion
of the surrounding traffic. The predictor consumes interleaved action, state and
latent tokens; the ego state is only given at ``t = 0`` (later states are
zeroed) because the downstream VLA never provides future states.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import partial
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from world_model.vjepa2.models import ac_predictor as _ac
from world_model.vjepa2.models import vision_transformer as _vit


@dataclass
class WorldModelConfig:
    # Frozen encoder (V-JEPA 2 pretrained)
    encoder_arch: str = "vit_large"
    encoder_embed_dim: int = 1024
    encoder_ckpt: Optional[str] = None
    img_h: int = 256
    img_w: int = 256
    patch_size: int = 16
    num_frames: int = 12      # encoder frames per clip = number of 2 Hz latents x tubelet_size
    tubelet_size: int = 2     # two consecutive 4 Hz frames per latent
    # Action-conditioned predictor (trained from scratch, ~25M parameters)
    predictor_embed_dim: int = 512
    predictor_depth: int = 8
    predictor_num_heads: int = 8
    predictor_mlp_ratio: float = 4.0
    action_dim: int = 3       # (dx, dy, dyaw) ego-frame pose delta over 0.5 s
    state_dim: int = 4        # (speed, a_long, a_lat, yaw rate)


def build_encoder(cfg: WorldModelConfig) -> nn.Module:
    if cfg.encoder_arch not in _vit.__dict__:
        raise ValueError(f"unknown encoder arch {cfg.encoder_arch}")
    encoder = _vit.__dict__[cfg.encoder_arch](
        patch_size=cfg.patch_size,
        img_size=(cfg.img_h, cfg.img_w),
        num_frames=cfg.num_frames,
        tubelet_size=cfg.tubelet_size,
        use_sdpa=True,
        use_SiLU=False,
        wide_SiLU=True,
        uniform_power=False,
        use_rope=True,
    )
    if cfg.encoder_ckpt is None:
        return encoder
    if not os.path.isfile(cfg.encoder_ckpt):
        raise FileNotFoundError(f"V-JEPA 2 encoder checkpoint not found: {cfg.encoder_ckpt}")
    state = torch.load(cfg.encoder_ckpt, map_location="cpu")
    key = "target_encoder" if "target_encoder" in state else "encoder"
    state_dict = {k.replace("module.", "").replace("backbone.", ""): v for k, v in state[key].items()}
    missing, unexpected = encoder.load_state_dict(state_dict, strict=False)
    print(f"[world model] loaded encoder {cfg.encoder_ckpt} ({key}): missing={len(missing)} unexpected={len(unexpected)}")
    return encoder


def build_predictor(cfg: WorldModelConfig) -> nn.Module:
    predictor = _ac.VisionTransformerPredictorAC(
        img_size=(cfg.img_h, cfg.img_w),
        patch_size=cfg.patch_size,
        num_frames=cfg.num_frames,
        tubelet_size=cfg.tubelet_size,
        embed_dim=cfg.encoder_embed_dim,
        predictor_embed_dim=cfg.predictor_embed_dim,
        depth=cfg.predictor_depth,
        num_heads=cfg.predictor_num_heads,
        mlp_ratio=cfg.predictor_mlp_ratio,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        use_rope=True,
        is_frame_causal=True,
        action_embed_dim=max(cfg.action_dim, cfg.state_dim),
    )
    # The upstream predictor ties the action and state encoders to one width;
    # size them for the driving action and state separately.
    predictor.action_encoder = nn.Linear(cfg.action_dim, cfg.predictor_embed_dim, bias=True)
    predictor.state_encoder = nn.Linear(cfg.state_dim, cfg.predictor_embed_dim, bias=True)
    nn.init.trunc_normal_(predictor.action_encoder.weight, std=0.02)
    nn.init.zeros_(predictor.action_encoder.bias)
    nn.init.trunc_normal_(predictor.state_encoder.weight, std=0.02)
    nn.init.zeros_(predictor.state_encoder.bias)
    return predictor


@torch.no_grad()
def _channel_std_ratio(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean over channels of std(pred) / std(target); < 1 indicates collapse."""
    d = pred.size(-1)
    return (pred.float().reshape(-1, d).std(dim=0) / (target.float().reshape(-1, d).std(dim=0) + 1e-8)).mean()


class WorldModel(nn.Module):
    """Frozen encoder + trainable action-conditioned predictor."""

    def __init__(self, cfg: WorldModelConfig):
        super().__init__()
        self.cfg = cfg
        self.encoder = build_encoder(cfg)
        self.predictor = build_predictor(cfg)
        for p in self.encoder.parameters():
            p.requires_grad_(False)
        self.encoder.eval()
        self.tokens_per_frame = (cfg.img_h // cfg.patch_size) * (cfg.img_w // cfg.patch_size)
        self.num_latents = cfg.num_frames // cfg.tubelet_size

    def train(self, mode: bool = True):
        super().train(mode)
        self.encoder.eval()
        return self

    @torch.no_grad()
    def encode(self, video: torch.Tensor) -> torch.Tensor:
        """``[B, 3, T * ts, H, W]`` -> ``[B, T * HW, D]``.

        Every tubelet is encoded independently, so a target latent never leaks
        into the context latents.
        """
        b, c, t_enc, h, w = video.shape
        ts, t = self.cfg.tubelet_size, self.num_latents
        assert t_enc == t * ts, f"expected {t * ts} frames, got {t_enc}"
        x = video.view(b, c, t, ts, h, w).permute(0, 2, 1, 3, 4, 5).contiguous().reshape(b * t, c, ts, h, w)
        out = self.encoder(x)
        return out.view(b, t * self.tokens_per_frame, out.size(-1))

    def forward_predictor(self, ctx: torch.Tensor, actions: torch.Tensor, states: torch.Tensor) -> torch.Tensor:
        """Predict the next latents; only the state at ``t = 0`` is visible."""
        if states.size(1) > 1:
            masked = torch.zeros_like(states)
            masked[:, 0] = states[:, 0]
            states = masked
        return self.predictor(ctx, actions, states)

    def forward(self, video: torch.Tensor, actions: torch.Tensor, states: torch.Tensor, auto_steps: int = 2):
        """Teacher-forced + autoregressive losses (per-token L1 on
        LayerNorm-normalized latents, equal weights).

        Teacher-forced: ground-truth latents ``z_0 .. z_{T-2}`` as context,
        supervise all ``T - 1`` next-step predictions in parallel.
        Autoregressive: start from ``z_0`` and roll ``auto_steps`` steps,
        feeding predictions back into the predictor.
        """
        tpf, t = self.tokens_per_frame, self.num_latents
        h = self.encode(video)
        h = F.layer_norm(h, (h.size(-1),))

        pred_tf = self.forward_predictor(h[:, : (t - 1) * tpf], actions[:, : t - 1], states[:, : t - 1])
        pred_tf = F.layer_norm(pred_tf, (pred_tf.size(-1),))
        target_tf = h[:, tpf: t * tpf]
        loss_tf = (pred_tf - target_tf).abs().mean()
        std_ratio_tf = _channel_std_ratio(pred_tf, target_tf)

        auto_steps = min(auto_steps, t - 1)
        loss_ar = video.new_zeros(())
        std_ratio_ar = video.new_zeros(())
        if auto_steps > 0:
            cur = h[:, :tpf].clone()
            losses, ratios = [], []
            for _ in range(auto_steps):
                t_ctx = cur.size(1) // tpf
                out = self.forward_predictor(cur, actions[:, :t_ctx], states[:, :t_ctx])
                last = F.layer_norm(out[:, -tpf:], (out.size(-1),))
                target = h[:, t_ctx * tpf: (t_ctx + 1) * tpf]
                losses.append((last - target).abs().mean())
                ratios.append(_channel_std_ratio(last, target))
                cur = torch.cat([cur, last], dim=1)
            loss_ar = torch.stack(losses).mean()
            std_ratio_ar = torch.stack(ratios).mean()

        return {
            "loss": loss_tf + loss_ar,
            "loss_tf": loss_tf.detach(),
            "loss_ar": loss_ar.detach(),
            "std_ratio_tf": std_ratio_tf.detach(),
            "std_ratio_ar": std_ratio_ar.detach(),
        }
