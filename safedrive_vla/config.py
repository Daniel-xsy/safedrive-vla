"""Hydra config schema of SafeDriveVLA training (defaults = paper recipe, App. C.1)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

from hydra.core.config_store import ConfigStore


@dataclass
class ActionTokenConfig:
    codebook_path: str = "safedrive_vla/assets/action_codebook.pkl"
    num_tokens: int = 2048   # codebook size N
    hz: float = 4.0
    horizon: int = 10        # T action tokens (2.5 s)


@dataclass
class WorldModelConfig:
    encoder_ckpt: Optional[str] = "ckpts/vjepa2/vitl.pt"
    predictor_ckpt: Optional[str] = None
    # Action sequence the world model dreams under (Fig. 3c):
    #   action_anchor:     meta-command action anchor (navigation-conditioned dreaming)
    #   ego_extrapolation: the ego keeps its past 2.5 s mean speed and yaw rate
    action_input: str = "action_anchor"
    action_anchor_path: Optional[str] = "safedrive_vla/assets/action_anchors.npz"
    rollout_steps: int = 5   # K dreamed latents at 2 Hz
    projector_hidden: int = 256
    # Architecture of the pre-trained world model (world_model/configs).
    encoder_arch: str = "vit_large"
    encoder_embed_dim: int = 1024
    img_size: int = 256
    patch_size: int = 16
    tubelet_size: int = 2
    predictor_embed_dim: int = 512
    predictor_depth: int = 8
    predictor_num_heads: int = 8
    predictor_mlp_ratio: float = 4.0


@dataclass
class PathHeadConfig:
    num_points: int = 20
    hidden: int = 512


@dataclass
class ModelConfig:
    _target_: str = "safedrive_vla.models.safedrive_vla.SafeDriveVLA"
    variant: str = "OpenGVLab/InternVL3-1B"
    lora_r: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.1
    gradient_checkpointing: bool = True
    use_mode_token: bool = True
    action_loss_weight: float = 1.0
    mode_loss_weight: float = 1.0
    path_loss_weight: float = 1.0
    text_loss_weight: float = 0.5
    target_point_hidden: int = 256
    target_point_hidden2: int = 512
    path_head: PathHeadConfig = field(default_factory=PathHeadConfig)
    lr: float = 3e-5
    weight_decay: float = 0.1
    betas: Tuple[float, float] = (0.9, 0.999)
    pct_start: float = 0.05
    action_token: ActionTokenConfig = field(default_factory=ActionTokenConfig)
    # navigation-conditioned world-model dreaming; leave unset for a model without it
    world_model: Optional[WorldModelConfig] = None


@dataclass
class DataConfig:
    data_root: str = "database/simlingo"
    data_scale: float = 1.0
    batch_size: int = 12     # per GPU
    num_workers: int = 8
    # Sampling weights of SimLingo's scenario buckets ("all" = every frame).
    train_partitions: Dict[str, float] = field(default_factory=lambda: {"all": 1.0})
    train_partitions_action_dreaming: Dict[str, float] = field(default_factory=lambda: {"all": 1.0})


@dataclass
class TrainConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    name: str = "safedrive_vla"
    seed: int = 9876
    gpus: int = 8
    max_epochs: int = 14
    precision: str = "bf16-mixed"
    gradient_clip_val: float = 0.3
    accumulate_grad_batches: int = 1
    val_every_n_epochs: int = 1
    resume_path: Optional[str] = None   # DeepSpeed checkpoint directory to resume from
    wandb_project: Optional[str] = None  # log to Weights & Biases; CSV logs otherwise


ConfigStore.instance().store(name="train_base", node=TrainConfig)
