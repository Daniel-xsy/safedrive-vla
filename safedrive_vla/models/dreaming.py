"""Navigation-conditioned world-model dreaming (Sec. 4.2, App. B.2).

The frozen latent world model encodes the current two-frame tubelet into
``z_0`` and rolls it forward ``K`` steps (0.5 s each) under an action sequence,
by default the action anchor of the meta-command implied by the navigation
signal. The trainable world state projector compresses each dreamed latent
into one VLM token.
"""

from __future__ import annotations

import os
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from world_model.models.world_model import WorldModel, WorldModelConfig, build_predictor


class WorldStateProjector(nn.Module):
    """Dreamed latent ``[HW, D_enc]`` -> one world token ``[D_LM]`` (Fig. B.2)."""

    def __init__(self, d_enc: int, d_lm: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(d_enc, hidden * 2, kernel_size=1),
            nn.GroupNorm(8, hidden * 2),
            nn.GELU(),
            nn.Conv2d(hidden * 2, hidden, kernel_size=1),
            nn.GroupNorm(8, hidden),
            nn.GELU(),
        )
        self.out = nn.Linear(hidden, d_lm)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n, hw, d = x.shape
        side = int(hw ** 0.5)
        if side * side != hw:
            raise ValueError(f"non-square latent grid: {hw}")
        grid = x.permute(0, 2, 1).contiguous().view(n, d, side, side)
        return self.out(self.net(grid).mean(dim=(2, 3)))


class WorldModelDreamer(nn.Module):
    """Frozen latent world model that dreams ``K`` future latents."""

    def __init__(
        self,
        encoder_ckpt: Optional[str],
        predictor_ckpt: Optional[str],
        rollout_steps: int = 5,
        encoder_arch: str = "vit_large",
        encoder_embed_dim: int = 1024,
        img_size: int = 256,
        patch_size: int = 16,
        tubelet_size: int = 2,
        predictor_embed_dim: int = 512,
        predictor_depth: int = 8,
        predictor_num_heads: int = 8,
        predictor_mlp_ratio: float = 4.0,
    ):
        super().__init__()
        arch = dict(
            encoder_arch=encoder_arch,
            encoder_embed_dim=encoder_embed_dim,
            img_h=img_size,
            img_w=img_size,
            patch_size=patch_size,
            tubelet_size=tubelet_size,
            predictor_embed_dim=predictor_embed_dim,
            predictor_depth=predictor_depth,
            predictor_num_heads=predictor_num_heads,
            predictor_mlp_ratio=predictor_mlp_ratio,
        )
        # The encoder sees one tubelet; the predictor's frame-causal attention
        # covers the initial latent plus the K dreamed ones.
        self.world_model = WorldModel(WorldModelConfig(num_frames=tubelet_size, encoder_ckpt=encoder_ckpt, **arch))
        self.world_model.predictor = build_predictor(
            WorldModelConfig(num_frames=(rollout_steps + 1) * tubelet_size, **arch)
        )
        if predictor_ckpt is not None:
            if not os.path.isfile(predictor_ckpt):
                raise FileNotFoundError(f"World-model checkpoint not found: {predictor_ckpt}")
            state = torch.load(predictor_ckpt, map_location="cpu")
            self.world_model.predictor.load_state_dict(state["predictor"])
        for p in self.world_model.parameters():
            p.requires_grad_(False)
        self.world_model.eval()
        self.rollout_steps = int(rollout_steps)
        self.d_enc = int(encoder_embed_dim)

    def train(self, mode: bool = True):
        super().train(mode)
        self.world_model.eval()
        return self

    def _apply(self, fn, recurse: bool = True):
        # Keep the frozen world model in fp32 under bf16 training: mixed
        # dtypes break the rotary attention of the V-JEPA 2 blocks.
        super()._apply(fn, recurse=recurse)
        self.world_model.to(torch.float32)
        return self

    @torch.no_grad()
    def forward(self, tubelet: torch.Tensor, actions: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        """Dream ``K`` latents.

        Args:
            tubelet: ``[B, 2, 3, H, W]`` uint8 frames ``t - 0.25 s`` and ``t``.
            actions: ``[B, K, 3]`` body-frame ``(dx, dy, dyaw)`` per 0.5 s.
            state:   ``[B, 4]`` ego state at ``t``.

        Returns:
            ``[B, K, HW, D_enc]`` LayerNorm-normalized latents (fp32).
        """
        if actions.size(1) != self.rollout_steps:
            raise ValueError(f"expected {self.rollout_steps} actions, got {actions.size(1)}")
        wm = self.world_model
        tpf = wm.tokens_per_frame
        with torch.amp.autocast("cuda", enabled=False):
            video = tubelet.float() / 255.0 if tubelet.dtype == torch.uint8 else tubelet.float()
            z = wm.encode(video.permute(0, 2, 1, 3, 4).contiguous())
            z = F.layer_norm(z, (z.size(-1),))

            states = z.new_zeros((z.size(0), self.rollout_steps, state.size(-1)))
            states[:, 0] = state.to(z.dtype)
            actions = actions.to(z.dtype)
            latents = []
            for _ in range(self.rollout_steps):
                t_ctx = z.size(1) // tpf
                out = wm.forward_predictor(z, actions[:, :t_ctx], states[:, :t_ctx])
                last = F.layer_norm(out[:, -tpf:], (out.size(-1),))
                latents.append(last)
                z = torch.cat([z, last], dim=1)
        return torch.stack(latents, dim=1)
