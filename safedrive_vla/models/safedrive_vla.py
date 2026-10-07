"""SafeDriveVLA (Sec. 4).

An InternVL3 VLM that reads the front-camera frame, the navigation signal and
the dreamed world tokens, and autoregressively emits

    <MODE> <ACTIONS> a_0 ... a_9

i.e. a driving-mode token followed by discrete action tokens. A lightweight
path head regresses 20 path points from the hidden state at ``<ACTIONS>``
for lateral control. Training loss (App. C.1):

    L = L_action + L_mode + L_path + 0.5 * L_text
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import pytorch_lightning as pl
import torch
from peft import LoraConfig, get_peft_model
from torch import Tensor, nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from safedrive_vla.action_tokenizer import action_token_strings
from safedrive_vla.constants import (
    ACTIONS_TOKEN,
    MODE_TOKENS,
    TARGET_POINT_TOKEN,
    WORLD_BEGIN_TOKEN,
    WORLD_END_TOKEN,
    WORLD_TOKEN,
)
from safedrive_vla.models.backbone import lm_hidden_size, load_vlm
from safedrive_vla.models.dreaming import WorldModelDreamer, WorldStateProjector

VISION_MODULE_NAMES = {"visual", "vision_tower", "vision_model", "vision_encoder"}


def _is_vision_module(name: str) -> bool:
    return any(part in VISION_MODULE_NAMES for part in name.split("."))


class TargetPointEncoder(nn.Module):
    """Target waypoint ``(x, y)`` -> VLM token embedding (SimLingo-style MLP)."""

    def __init__(self, token_size: int, hidden_size: int = 256, hidden_size2: int = 512):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(2, hidden_size),
            nn.ReLU(True),
            nn.Linear(hidden_size, hidden_size2),
            nn.ReLU(True),
            nn.Linear(hidden_size2, token_size),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.mlp(x)


class PathHead(nn.Module):
    """Regresses ``num_points`` ego-frame path points from one hidden state."""

    def __init__(self, hidden_size: int, num_points: int = 20, head_hidden: int = 512):
        super().__init__()
        self.num_points = int(num_points)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, head_hidden),
            nn.GELU(),
            nn.Linear(head_hidden, head_hidden),
            nn.GELU(),
            nn.Linear(head_hidden, self.num_points * 2),
        )

    def forward(self, hidden: Tensor) -> Tensor:
        return self.mlp(hidden).view(hidden.size(0), self.num_points, 2)


class SafeDriveVLA(pl.LightningModule):
    def __init__(
        self,
        processor,
        variant: str = "OpenGVLab/InternVL3-1B",
        lora_r: int = 32,
        lora_alpha: int = 64,
        lora_dropout: float = 0.1,
        gradient_checkpointing: bool = True,
        use_mode_token: bool = True,
        action_loss_weight: float = 1.0,
        mode_loss_weight: float = 1.0,
        path_loss_weight: float = 1.0,
        text_loss_weight: float = 0.5,
        target_point_hidden: int = 256,
        target_point_hidden2: int = 512,
        path_head: Optional[dict] = None,
        lr: float = 3e-5,
        weight_decay: float = 0.1,
        betas: Tuple[float, float] = (0.9, 0.999),
        pct_start: float = 0.05,
        action_token: Optional[dict] = None,
        world_model: Optional[dict] = None,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["processor"])
        self.processor = processor
        self.tokenizer = processor.tokenizer
        self.use_mode_token = bool(use_mode_token)
        self.loss_weights = {
            "action": float(action_loss_weight),
            "mode": float(mode_loss_weight),
            "path": float(path_loss_weight),
            "text": float(text_loss_weight),
        }
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.betas = tuple(betas)
        self.pct_start = float(pct_start)
        path_head = dict(path_head or {})
        world_model = dict(world_model or {})

        # ---- VLM with the extended vocabulary -----------------------------
        self.vlm, hf_config = load_vlm(variant)
        num_actions = int((action_token or {}).get("num_tokens", 2048))
        self.vlm.resize_token_embeddings(len(self.tokenizer))
        self.action_start_id = int(self.tokenizer.convert_tokens_to_ids(action_token_strings(1)[0]))
        self.action_end_id = self.action_start_id + num_actions
        token_id = self.tokenizer.convert_tokens_to_ids
        self.actions_token_id = int(token_id(ACTIONS_TOKEN))
        mode_token_ids = [int(token_id(t)) for t in MODE_TOKENS] if self.use_mode_token else []
        self.register_buffer("mode_token_ids", torch.tensor(mode_token_ids, dtype=torch.long), persistent=False)
        world_token_ids = [int(token_id(t)) for t in (WORLD_BEGIN_TOKEN, WORLD_END_TOKEN, WORLD_TOKEN)]

        # ---- LoRA on the language model; the vision encoder is fully fine-tuned
        lora_targets = [
            name for name, module in self.vlm.named_modules()
            if isinstance(module, nn.Linear) and not _is_vision_module(name)
            # tied embeddings are trained through the gradient mask below
            and not any(name == e or name.endswith(f".{e}") for e in ("lm_head", "embed_tokens"))
        ]
        self.vlm = get_peft_model(
            self.vlm,
            LoraConfig(r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout, bias="none",
                       task_type="CAUSAL_LM", target_modules=lora_targets),
        )
        for name, param in self.vlm.named_parameters():
            if _is_vision_module(name) and "lora_" not in name:
                param.requires_grad = True
        # Train only the embedding / LM-head rows of the added tokens.
        trainable_rows = (self.action_start_id, self.action_end_id,
                          tuple([self.actions_token_id, *mode_token_ids, *world_token_ids]))

        def _mask_new_token_rows(grad, rows=trainable_rows):
            mask = torch.zeros_like(grad)
            mask[rows[0]:rows[1]] = 1.0
            for tid in rows[2]:
                mask[tid] = 1.0
            return grad * mask

        for name, param in self.vlm.named_parameters():
            if "embed_tokens" in name or "lm_head" in name:
                param.requires_grad = True
                param.register_hook(_mask_new_token_rows)
        if gradient_checkpointing:
            self.vlm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

        # ---- token embeddings replaced by learned inputs ------------------
        hidden_size = lm_hidden_size(hf_config)
        self.target_point_encoder = TargetPointEncoder(hidden_size, target_point_hidden, target_point_hidden2)
        # (token id, encoder, attribute holding the per-batch payload)
        self._embedding_overrides: List[Tuple[int, nn.Module, str]] = [
            (int(token_id(TARGET_POINT_TOKEN)), self.target_point_encoder, "_target_points"),
        ]
        self._target_points: Optional[Tensor] = None
        self._world_tokens: Optional[Tensor] = None

        self.world_model_enabled = bool(world_model.get("enabled", False))
        self.dreamer: Optional[WorldModelDreamer] = None
        self.world_state_projector: Optional[WorldStateProjector] = None
        if self.world_model_enabled:
            self.dreamer = WorldModelDreamer(
                encoder_ckpt=world_model.get("encoder_ckpt"),
                predictor_ckpt=world_model.get("predictor_ckpt"),
                rollout_steps=int(world_model.get("rollout_steps", 5)),
                **{k: world_model[k] for k in _DREAMER_ARCH_KEYS if k in world_model},
            )
            self.world_state_projector = WorldStateProjector(
                d_enc=self.dreamer.d_enc, d_lm=hidden_size, hidden=int(world_model.get("projector_hidden", 256))
            )
            self._embedding_overrides.append((int(token_id(WORLD_TOKEN)), nn.Identity(), "_world_tokens"))
        self.vlm.get_input_embeddings().register_forward_hook(self._override_embeddings)

        self.path_head = PathHead(hidden_size, int(path_head.get("num_points", 20)), int(path_head.get("hidden", 512)))

    # ------------------------------------------------------------------
    # learned token embeddings
    # ------------------------------------------------------------------
    def _override_embeddings(self, _module, inputs, output):
        """Replace the embeddings of ``<TARGET_POINT>`` / ``<WM_LATENT>``
        tokens, in order, with the encoded target points / world tokens."""
        ids = inputs[0] if inputs else None
        if ids is None or ids.dim() < 2:
            return output
        new_output = None
        for tok_id, encoder, attr in self._embedding_overrides:
            payload = getattr(self, attr)
            if payload is None:
                continue
            mask = ids == tok_id
            if not bool(mask.any()):
                continue
            payload = payload.to(device=output.device, dtype=output.dtype)
            if new_output is None:
                new_output = output.clone()
            for b in range(ids.size(0)):
                positions = mask[b].nonzero(as_tuple=False).flatten()
                k = min(positions.numel(), payload.size(1))
                if k > 0:
                    new_output[b, positions[:k]] = encoder(payload[b, :k]).to(output.dtype)
        return new_output if new_output is not None else output

    def set_conditioning(
        self,
        target_points: Optional[Tensor] = None,
        wm_tubelet: Optional[Tensor] = None,
        wm_actions: Optional[Tensor] = None,
        wm_state: Optional[Tensor] = None,
    ) -> None:
        """Set the inputs substituted into the next forward pass: target points
        ``[B, 2, 2]`` and, with the world model, its dreamed world tokens."""
        self._target_points = target_points
        self._world_tokens = None
        if self.world_model_enabled and wm_tubelet is not None:
            latents = self.dreamer(wm_tubelet, wm_actions, wm_state)       # [B, K, HW, D_enc]
            b, k, hw, d = latents.shape
            proj_dtype = next(self.world_state_projector.parameters()).dtype
            tokens = self.world_state_projector(latents.reshape(b * k, hw, d).to(proj_dtype))
            self._world_tokens = tokens.view(b, k, -1)

    def clear_conditioning(self) -> None:
        self._target_points = None
        self._world_tokens = None

    # ------------------------------------------------------------------
    # forward / losses
    # ------------------------------------------------------------------
    def forward_vlm(self, input_ids: Tensor, attention_mask: Tensor, pixel_values: Tensor):
        """Returns the logits and the last-layer hidden states."""
        out = self.vlm(input_ids=input_ids, attention_mask=attention_mask, pixel_values=pixel_values,
                       output_hidden_states=True)
        return out.logits, out.hidden_states[-1]

    def predict_path(self, input_ids: Tensor, hidden_states: Tensor) -> Tuple[Tensor, Tensor]:
        """Path-head prediction at the first ``<ACTIONS>`` token of each row.
        Returns ``(path [N, P, 2], row indices [N])`` for the rows that contain it."""
        mask = input_ids == self.actions_token_id
        rows = torch.where(mask.any(dim=1))[0]
        pos = mask.float().argmax(dim=1)[rows]
        return self.path_head(hidden_states[rows, pos]), rows

    def compute_losses(self, batch) -> Dict[str, Tensor]:
        self.set_conditioning(batch.target_points, batch.wm_tubelet, batch.wm_actions, batch.wm_state)
        try:
            logits, hidden_states = self.forward_vlm(batch.input_ids, batch.attention_mask, batch.pixel_values)
        finally:
            self.clear_conditioning()

        vocab = logits.size(-1)
        flat_logits = logits[:, :-1, :].contiguous().view(-1, vocab)
        flat_labels = batch.labels[:, 1:].contiguous().view(-1)
        valid = flat_labels != -100
        is_action = valid & (flat_labels >= self.action_start_id) & (flat_labels < self.action_end_id)
        is_mode = valid & torch.isin(flat_labels, self.mode_token_ids)
        is_text = valid & ~is_action & ~is_mode
        ce = nn.functional.cross_entropy(flat_logits, flat_labels.clamp(min=0), reduction="none") * valid.float()

        losses = {
            "action_loss": (ce * is_action).sum() / is_action.sum().clamp(min=1),
            "mode_loss": (ce * is_mode).sum() / is_mode.sum().clamp(min=1),
            "text_loss": (ce * is_text).sum() / is_text.sum().clamp(min=1),
        }
        pred_path, rows = self.predict_path(batch.input_ids, hidden_states)
        losses["path_loss"] = ((pred_path - batch.path.to(pred_path)[rows]) ** 2).mean()
        losses["loss"] = (
            self.loss_weights["action"] * losses["action_loss"]
            + self.loss_weights["text"] * losses["text_loss"]
            + self.loss_weights["mode"] * losses["mode_loss"]
            + self.loss_weights["path"] * losses["path_loss"]
        )
        return losses

    def training_step(self, batch, _batch_idx: int = 0):
        losses = self.compute_losses(batch)
        for key, value in losses.items():
            self.log(f"train/{key}", value, on_step=True, on_epoch=True, prog_bar=key == "loss")
        return losses["loss"]

    def validation_step(self, batch, _batch_idx: int = 0):
        losses = self.compute_losses(batch)
        for key, value in losses.items():
            self.log(f"val/{key}", value, on_step=False, on_epoch=True, prog_bar=key == "loss", sync_dist=True)
        return losses["loss"]

    def configure_optimizers(self):
        optimizer = AdamW([p for p in self.parameters() if p.requires_grad],
                          lr=self.lr, betas=self.betas, weight_decay=self.weight_decay)
        total_steps = int(self.trainer.estimated_stepping_batches)
        warmup_steps = int(total_steps * self.pct_start)

        def lr_lambda(step: int) -> float:  # linear warmup, then linear decay
            if step < warmup_steps:
                return float(step + 1) / float(warmup_steps)
            return max(0.0, 1.0 - float(step - warmup_steps) / float(max(1, total_steps - warmup_steps)))

        return {"optimizer": optimizer,
                "lr_scheduler": {"scheduler": LambdaLR(optimizer, lr_lambda), "interval": "step", "frequency": 1}}


_DREAMER_ARCH_KEYS = (
    "encoder_arch",
    "encoder_embed_dim",
    "img_size",
    "patch_size",
    "tubelet_size",
    "predictor_embed_dim",
    "predictor_depth",
    "predictor_num_heads",
    "predictor_mlp_ratio",
)
