"""Train SafeDriveVLA.

Usage (from the repository root)::

    python -m safedrive_vla.train experiment=safedrive_vla_x0.2 [hydra overrides]

Outputs go to ``work_dirs/safedrive_vla/<name>/<timestamp>/`` (``.hydra/``
config and ``checkpoints/epoch=XXX.ckpt``).
"""

import os

os.environ.setdefault("NCCL_ASYNC_ERROR_HANDLING", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from datetime import timedelta

import hydra
import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, ModelSummary
from pytorch_lightning.loggers import CSVLogger, WandbLogger

from safedrive_vla.config import TrainConfig  # noqa: F401  (registers the config schema)
from safedrive_vla.data.datamodule import SafeDriveDataModule
from safedrive_vla.models.backbone import load_processor


@hydra.main(config_path="configs", config_name="config", version_base="1.1")
def main(cfg: TrainConfig) -> None:
    torch.set_float32_matmul_precision("high")
    pl.seed_everything(cfg.seed, workers=True)
    print(OmegaConf.to_yaml(cfg))

    # Hydra runs inside the output directory; resolve paths against the launch directory.
    root = hydra.utils.get_original_cwd()

    def abspath(path):
        return path if path is None or os.path.isabs(path) else os.path.join(root, path)

    cfg.model.action_token.codebook_path = abspath(cfg.model.action_token.codebook_path)
    for key in ("encoder_ckpt", "predictor_ckpt", "action_anchor_path"):
        cfg.model.world_model[key] = abspath(cfg.model.world_model[key])
    if cfg.resume_path:
        cfg.resume_path = abspath(cfg.resume_path)

    processor, action_start_id = load_processor(cfg.model.variant, cfg.model.action_token.num_tokens)
    data = SafeDriveDataModule(
        processor,
        data_root=abspath(cfg.data.data_root),
        data_scale=cfg.data.data_scale,
        codebook_path=cfg.model.action_token.codebook_path,
        action_start_id=action_start_id,
        num_action_tokens=cfg.model.action_token.num_tokens,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        train_partitions=OmegaConf.to_container(cfg.data.train_partitions),
        train_partitions_action_dreaming=OmegaConf.to_container(cfg.data.train_partitions_action_dreaming),
        use_mode_token=cfg.model.use_mode_token,
        world_model=OmegaConf.to_container(cfg.model.world_model, resolve=True),
    )
    model = hydra.utils.instantiate(cfg.model, processor=processor, _recursive_=False)
    if cfg.resume_path:
        # DeepSpeed checkpoints only store the trainable parameters.
        model.strict_loading = False

    if cfg.wandb_project:
        logger = WandbLogger(project=cfg.wandb_project, name=cfg.name,
                             config=OmegaConf.to_container(cfg, resolve=True))
    else:
        logger = CSVLogger(save_dir=".", name="logs")

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=cfg.gpus,
        strategy=pl.strategies.DeepSpeedStrategy(
            stage=2,
            loss_scale=32.0,
            logging_batch_size_per_gpu=cfg.data.batch_size,
            timeout=timedelta(seconds=int(os.environ.get("NCCL_TIMEOUT_SEC", "1800"))),
            exclude_frozen_parameters=True,
        ),
        precision=cfg.precision,
        max_epochs=cfg.max_epochs,
        gradient_clip_val=cfg.gradient_clip_val,
        accumulate_grad_batches=cfg.accumulate_grad_batches,
        check_val_every_n_epoch=cfg.val_every_n_epochs,
        benchmark=True,
        logger=logger,
        callbacks=[
            ModelCheckpoint(dirpath="./checkpoints", filename="{epoch:03d}", save_top_k=-1, save_last=True,
                            every_n_epochs=cfg.val_every_n_epochs),
            ModelSummary(max_depth=3),
            LearningRateMonitor(logging_interval="step"),
        ],
    )
    trainer.fit(model, data, ckpt_path=cfg.resume_path)


if __name__ == "__main__":
    main()
