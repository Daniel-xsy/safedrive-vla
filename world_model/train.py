"""Latent world-model pre-training (App. B.1).

Recipe (V-JEPA 2-AC / DROID, with the CARLA-specific choices of the paper):
  * frozen V-JEPA 2 ViT-L encoder, ~25M action-conditioned predictor;
  * 6 latents at 2 Hz (2.5 s), each encoded from two consecutive frames;
  * teacher-forced + autoregressive L1 losses on LayerNorm-normalized latents,
    with the autoregressive depth ramped from 2 to 5 over the first 40% of training;
  * AdamW, warmup-stable-decay learning rate, cosine weight-decay schedule.

Usage (single node, N GPUs)::

    torchrun --standalone --nproc_per_node=N -m world_model.train \\
        --config world_model/configs/world_model.yaml

Training resumes automatically from ``<work_dir>/latest.pt``.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import random
import re
import time

import numpy as np
import torch
import torch.distributed as dist
import yaml
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from world_model.data.carla_clips import CarlaClipDataset, collate, find_routes
from world_model.models.world_model import WorldModel, WorldModelConfig
from world_model.vjepa2.utils.schedulers import CosineWDSchedule, WSDSchedule

_TOWN_RE = re.compile(r"(Town\d+)")


# ---------------------------------------------------------------------------
# distributed helpers
# ---------------------------------------------------------------------------

def init_distributed() -> tuple[int, int, int]:
    """(world_size, rank, local_rank) from torchrun, or a single process."""
    if "RANK" not in os.environ or int(os.environ.get("WORLD_SIZE", 1)) <= 1:
        return 1, 0, 0
    rank, world_size = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    return world_size, rank, local_rank


def reduce_mean(value: float, device: torch.device, world_size: int) -> float:
    if world_size <= 1:
        return value
    t = torch.tensor(value, device=device, dtype=torch.float64)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return float((t / world_size).item())


def log0(rank: int, message: str) -> None:
    if rank == 0:
        print(message, flush=True)


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------

def _routes_of_splits(root: str, splits) -> list[tuple[str, str]]:
    pairs = []
    for split in splits:
        pairs.extend((split, r) for r in find_routes(os.path.join(root, split)))
    return sorted(set(pairs))


def _data_scale_routes(pairs: list[tuple[str, str]], root: str, frac: float, seed: int) -> list[str]:
    """Deterministic route subset, balanced over (split, town)."""
    if frac >= 1.0:
        return [r for _, r in pairs]
    groups: dict[tuple[str, str], list[str]] = {}
    for split, route in pairs:
        town = _TOWN_RE.search(os.path.basename(os.path.normpath(route)))
        groups.setdefault((split, town.group(1) if town else "unknown"), []).append(route)
    selected = []
    for routes in groups.values():
        routes = sorted(routes, key=lambda r: hashlib.sha1(f"{seed}:{os.path.relpath(r, root)}".encode()).hexdigest())
        selected.extend(routes[: max(1, round(frac * len(routes)))])
    return sorted(selected)


def build_loaders(cfg: dict, rank: int, world_size: int):
    dcfg, bs = cfg["data"], cfg["train"]["batch_size"]
    root = os.path.abspath(dcfg["root"])

    def make(routes):
        return CarlaClipDataset(
            routes, num_frames=dcfg["num_frames"], tubelet_size=dcfg["tubelet_size"],
            img_hw=(dcfg["img_h"], dcfg["img_w"]), stride=dcfg["stride"],
        )

    pairs = _routes_of_splits(root, dcfg["splits"])
    train_set = make(_data_scale_routes(pairs, root, float(dcfg.get("data_scale", 1.0)), int(dcfg.get("data_scale_seed", 0))))
    log0(rank, f"[data] {len(train_set)} training clips")
    sampler = DistributedSampler(train_set, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True) if world_size > 1 else None
    train_loader = DataLoader(
        train_set, batch_size=bs, shuffle=sampler is None, sampler=sampler, num_workers=dcfg["num_workers"],
        collate_fn=collate, drop_last=True, pin_memory=True,
    )

    # Validation: a fixed random fraction of the held-out SimLingo validation routes.
    val_routes = [r for _, r in _routes_of_splits(root, dcfg["val_splits"])]
    rng = random.Random(int(dcfg.get("val_seed", 0)))
    rng.shuffle(val_routes)
    val_routes = sorted(val_routes[: max(1, int(float(dcfg["val_route_frac"]) * len(val_routes)))])
    val_set = make(val_routes)
    log0(rank, f"[data] {len(val_set)} validation clips")
    val_sampler = DistributedSampler(val_set, num_replicas=world_size, rank=rank, shuffle=False) if world_size > 1 else None
    val_loader = DataLoader(
        val_set, batch_size=bs, shuffle=False, sampler=val_sampler, num_workers=dcfg["num_workers"],
        collate_fn=collate, drop_last=False, pin_memory=True,
    )
    return train_loader, sampler, val_loader


# ---------------------------------------------------------------------------
# optimization
# ---------------------------------------------------------------------------

def auto_steps_at(epoch: int, tcfg: dict) -> int:
    """Autoregressive depth, linearly ramped over the first ``ramp_frac`` epochs."""
    start, end = tcfg["auto_steps_start"], tcfg["auto_steps_end"]
    ramp_epochs = max(1, int(tcfg["auto_steps_ramp_frac"] * tcfg["num_epochs"]))
    if epoch >= ramp_epochs:
        return end
    return start + round((end - start) * epoch / ramp_epochs)


def build_optimizer(predictor: torch.nn.Module, tcfg: dict) -> torch.optim.Optimizer:
    """AdamW; biases and 1-D parameters are excluded from weight decay."""
    decay = [p for n, p in predictor.named_parameters() if p.requires_grad and "bias" not in n and p.dim() != 1]
    no_decay = [p for n, p in predictor.named_parameters() if p.requires_grad and ("bias" in n or p.dim() == 1)]
    return torch.optim.AdamW(
        [{"params": decay, "weight_decay": tcfg["weight_decay"]},
         {"params": no_decay, "weight_decay": 0.0, "WD_exclude": True}],
        betas=(0.9, tcfg["beta2"]), eps=tcfg["eps"],
    )


def build_schedulers(tcfg: dict, optimizer: torch.optim.Optimizer, iters_per_epoch: int):
    total = tcfg["num_epochs"] * iters_per_epoch
    lr_sched = WSDSchedule(
        optimizer,
        warmup_steps=tcfg["warmup_epochs"] * iters_per_epoch,
        anneal_steps=tcfg["anneal_epochs"] * iters_per_epoch,
        T_max=total,
        start_lr=tcfg["start_lr"],
        ref_lr=tcfg["lr"],
        final_lr=tcfg["final_lr"],
    )
    wd_sched = CosineWDSchedule(optimizer, ref_wd=tcfg["weight_decay"], final_wd=tcfg["final_weight_decay"], T_max=total)
    return lr_sched, wd_sched


# ---------------------------------------------------------------------------
# training
# ---------------------------------------------------------------------------

@torch.no_grad()
def validate(model, loader, device, world_size, rank, auto_steps: int) -> dict:
    model.eval()
    sums = {"loss": 0.0, "loss_tf": 0.0, "loss_ar": 0.0, "std_ratio_tf": 0.0, "std_ratio_ar": 0.0}
    n = 0
    for batch in loader:
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        with torch.amp.autocast(device.type, dtype=torch.bfloat16):
            out = model(batch["video"], batch["actions"], batch["states"], auto_steps=auto_steps)
        for k in sums:
            sums[k] += float(out[k].item())
        n += 1
    metrics = {k: reduce_mean(v / max(1, n), device, world_size) for k, v in sums.items()}
    log0(rank, "[val] " + " | ".join(f"{k} {v:.5f}" for k, v in metrics.items()))
    return metrics


def run(cfg: dict, work_dir: str) -> None:
    tcfg = cfg["train"]
    world_size, rank, local_rank = init_distributed()
    random.seed(tcfg["seed"])
    np.random.seed(tcfg["seed"])
    torch.manual_seed(tcfg["seed"])
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda", local_rank)

    model = WorldModel(WorldModelConfig(**cfg["model"])).to(device)
    predictor = model.predictor
    log0(rank, f"[model] predictor {sum(p.numel() for p in predictor.parameters()) / 1e6:.1f}M parameters")
    optimizer = build_optimizer(predictor, tcfg)
    if world_size > 1:
        model = DistributedDataParallel(
            model, device_ids=[local_rank], output_device=local_rank,
            find_unused_parameters=True, broadcast_buffers=False,
        )

    train_loader, sampler, val_loader = build_loaders(cfg, rank, world_size)
    ipe = len(train_loader)
    scaler = torch.amp.GradScaler("cuda")
    lr_sched, wd_sched = build_schedulers(tcfg, optimizer, ipe)

    start_epoch = 0
    latest = os.path.join(work_dir, "latest.pt")
    if os.path.isfile(latest):
        ckpt = torch.load(latest, map_location="cpu")
        predictor.load_state_dict(ckpt["predictor"])
        optimizer.load_state_dict(ckpt["opt"])
        scaler.load_state_dict(ckpt["scaler"])
        start_epoch = ckpt["epoch"]
        for _ in range(start_epoch * ipe):
            lr_sched.step()
            wd_sched.step()
        log0(rank, f"[ckpt] resumed from {latest} (epoch {start_epoch})")

    wandb_run = None
    if rank == 0 and cfg.get("wandb", {}).get("enable", False):
        import wandb
        wandb_run = wandb.init(project=cfg["wandb"].get("project", "safedrive-vla"), name=os.path.basename(work_dir),
                               dir=work_dir, config=cfg, resume="allow")

    num_epochs = tcfg["num_epochs"]
    global_step = start_epoch * ipe
    for epoch in range(start_epoch, num_epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)
        model.train()
        auto_steps = auto_steps_at(epoch, tcfg)
        loss_sum = 0.0
        t_epoch = time.time()
        for itr, batch in enumerate(train_loader):
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            lr = lr_sched.step()
            wd = wd_sched.step()
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device.type, dtype=torch.bfloat16):
                out = model(batch["video"], batch["actions"], batch["states"], auto_steps=auto_steps)
            scaler.scale(out["loss"]).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_([p for p in predictor.parameters() if p.requires_grad], 1.0)
            scaler.step(optimizer)
            scaler.update()

            loss = reduce_mean(float(out["loss"].item()), device, world_size)
            loss_sum += loss
            global_step += 1
            if itr % tcfg["log_every"] == 0 or itr == ipe - 1:
                log0(rank, f"[ep {epoch + 1}/{num_epochs} itr {itr}/{ipe}] loss {loss:.5f} | lr {lr:.2e} | wd {wd:.2e} "
                           f"| auto_steps {auto_steps} | {time.time() - t_epoch:.0f}s")
                if wandb_run is not None:
                    wandb_run.log({"train/loss": loss, "train/loss_tf": float(out["loss_tf"]), "train/loss_ar": float(out["loss_ar"]),
                                   "train/lr": lr, "train/auto_steps": auto_steps}, step=global_step)
        log0(rank, f"[ep {epoch + 1}] mean loss {loss_sum / max(1, ipe):.5f}")

        metrics = validate(model, val_loader, device, world_size, rank, auto_steps=tcfg["auto_steps_end"])
        if wandb_run is not None:
            wandb_run.log({f"val/{k}": v for k, v in metrics.items()}, step=global_step)

        if rank == 0:
            state = {"predictor": predictor.state_dict(), "opt": optimizer.state_dict(), "scaler": scaler.state_dict(),
                     "epoch": epoch + 1}
            torch.save(state, latest)
            if (epoch + 1) % tcfg["save_every"] == 0:
                torch.save(state, os.path.join(work_dir, f"epoch_{epoch + 1:04d}.pt"))
        if world_size > 1:
            dist.barrier()

    if wandb_run is not None:
        wandb_run.finish()


def main() -> None:
    parser = argparse.ArgumentParser(description="Pre-train the latent world model.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--work-dir", default=None, help="output directory (default: work_dirs/<config name>)")
    args = parser.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    work_dir = args.work_dir or os.path.join("work_dirs", os.path.splitext(os.path.basename(args.config))[0])
    os.makedirs(work_dir, exist_ok=True)
    try:
        run(cfg, work_dir)
    finally:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
