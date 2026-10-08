"""Data module: bucketed sampling of driving + action-dreaming samples and
chat-template collation for the InternVL3 processor."""

from __future__ import annotations

import itertools
import os
from typing import Any, Dict, List, NamedTuple, Optional

import numpy as np
import torch
from PIL import Image
from pytorch_lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, WeightedRandomSampler

from safedrive_vla.constants import world_block_text
from safedrive_vla.data.dataset import ActionDreamingDataset, DrivingDataset, DrivingSample

IGNORE_INDEX = -100
MAX_LENGTH = 2048
LMDRIVE_TEMPLATES = os.path.join(os.path.dirname(os.path.dirname(__file__)), "assets", "lmdrive_commands.json")


class TrainingBatch(NamedTuple):
    input_ids: torch.Tensor          # [B, L] prompt + target, left padded
    attention_mask: torch.Tensor     # [B, L]
    labels: torch.Tensor             # [B, L], IGNORE_INDEX on prompt and padding
    pixel_values: torch.Tensor       # processor image tiles
    target_points: torch.Tensor      # [B, 2, 2]
    path: torch.Tensor               # [B, 20, 2] path-head target
    wm_tubelet: Optional[torch.Tensor] = None   # [B, 2, 3, H, W] uint8
    wm_state: Optional[torch.Tensor] = None     # [B, 4]
    wm_actions: Optional[torch.Tensor] = None   # [B, K, 3]


class SafeDriveDataModule(LightningDataModule):
    """Mixes driving and action-dreaming samples 50/50. Within each, samples
    are drawn from SimLingo's scenario buckets with the given ``train_partitions``
    weights (``all`` = every frame)."""

    def __init__(
        self,
        processor,
        data_root: str,
        data_scale: float,
        codebook_path: str,
        action_start_id: int,
        num_action_tokens: int,
        batch_size: int,
        num_workers: int,
        train_partitions: Dict[str, float],
        train_partitions_action_dreaming: Dict[str, float],
        use_mode_token: bool = True,
        world_model: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        self.processor = processor
        self.tokenizer = processor.tokenizer
        self.action_ids = (int(action_start_id), int(action_start_id) + int(num_action_tokens))
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.partitions = [dict(train_partitions), dict(train_partitions_action_dreaming)]
        self.world_model = dict(world_model or {})
        self.wm_enabled = bool(self.world_model)
        self.dataset_kwargs = dict(
            data_root=data_root,
            data_scale=data_scale,
            lmdrive_templates=LMDRIVE_TEMPLATES,
            codebook_path=codebook_path,
            use_mode_token=use_mode_token,
            world_model=self.world_model,
        )

    def setup(self, stage: Optional[str] = None) -> None:
        bucket_names: List[str] = []
        bucket_weights: List[float] = []
        datasets = {}
        for dataset_cls, partitions in zip((DrivingDataset, ActionDreamingDataset), self.partitions):
            total = sum(partitions.values())
            suffix = "_action_dreaming" if dataset_cls is ActionDreamingDataset else ""
            for bucket, weight in partitions.items():
                ds = dataset_cls(split="train", bucket_name=bucket, **self.dataset_kwargs)
                if len(ds) == 0:
                    continue
                datasets[bucket + suffix] = ds
                bucket_names.append(bucket + suffix)
                bucket_weights.append(float(weight) / total * 0.5)

        self.train_dataset = ConcatDataset([datasets[b] for b in bucket_names])
        per_sample = itertools.chain.from_iterable([w] * len(datasets[b]) for b, w in zip(bucket_names, bucket_weights))
        num_samples = int(min(len(datasets[b]) / max(w, 1e-9) for b, w in zip(bucket_names, bucket_weights)))
        self.train_sampler = WeightedRandomSampler(list(per_sample), num_samples=num_samples, replacement=True)
        print(f"[data] train buckets {dict(zip(bucket_names, bucket_weights))}, {num_samples} samples per epoch")

        self.val_dataset = ConcatDataset([
            DrivingDataset(split="val", bucket_name="all", **self.dataset_kwargs),
            ActionDreamingDataset(split="val", bucket_name="all", **self.dataset_kwargs),
        ])

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, sampler=self.train_sampler,
                          num_workers=self.num_workers, drop_last=True, pin_memory=True, collate_fn=self.collate)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False,
                          num_workers=self.num_workers, drop_last=True, pin_memory=True, collate_fn=self.collate)

    # ------------------------------------------------------------------
    def collate(self, samples: List[DrivingSample]) -> TrainingBatch:
        images = [Image.fromarray(np.transpose(s.image[0], (1, 2, 0))) for s in samples]
        conversations = []
        for s in samples:
            conv = [dict(turn, content=[dict(part) for part in turn["content"]]) for turn in s.conversation]
            if self.wm_enabled:  # world tokens go after the user prompt
                conv[0]["content"][0]["text"] += world_block_text(int(self.world_model.get("rollout_steps", 5)))
            conversations.append(conv)
        prompts = [[turn for turn in conv if turn["role"] == "user"] for conv in conversations]

        def encode(convs, add_generation_prompt: bool):
            text = self.processor.apply_chat_template(convs, add_generation_prompt=add_generation_prompt, tokenize=False)
            return self.processor(text=text, images=images, padding=True, return_tensors="pt",
                                  truncation=True, max_length=MAX_LENGTH)

        full = encode(conversations, add_generation_prompt=False)
        prompt_only = encode(prompts, add_generation_prompt=True)

        # Supervise only the assistant tokens.
        input_ids = full["input_ids"]
        labels = input_ids.clone()
        pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
        for b in range(len(samples)):
            non_pad = (input_ids[b] != pad_id).nonzero(as_tuple=False).flatten()
            if non_pad.numel() == 0:
                continue
            prompt_len = int(prompt_only["attention_mask"][b].sum().item())
            labels[b, : int(non_pad[0].item()) + prompt_len] = IGNORE_INDEX
            labels[b, input_ids[b] == pad_id] = IGNORE_INDEX
            if not bool(((labels[b] >= self.action_ids[0]) & (labels[b] < self.action_ids[1])).any()):
                raise RuntimeError("Sample truncated before its action tokens; increase MAX_LENGTH.")

        wm = {}
        if self.wm_enabled:
            wm = dict(
                wm_tubelet=torch.from_numpy(np.stack([s.wm_tubelet for s in samples])),
                wm_state=torch.from_numpy(np.stack([s.wm_state for s in samples]).astype(np.float32)),
                wm_actions=torch.from_numpy(np.stack([s.wm_actions for s in samples]).astype(np.float32)),
            )
        return TrainingBatch(
            input_ids=input_ids,
            attention_mask=full["attention_mask"],
            labels=labels,
            pixel_values=full["pixel_values"],
            target_points=torch.tensor(np.stack([s.target_points for s in samples]), dtype=torch.float32),
            path=torch.tensor(np.stack([s.path for s in samples]), dtype=torch.float32),
            **wm,
        )
