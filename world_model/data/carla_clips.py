"""CARLA clips for world-model pre-training (App. B.1).

A clip covers 2.5 s at 2 Hz (``num_frames = 6`` latents). Each latent is
encoded from two consecutive 4 Hz frames ``[I_{t-0.25s}, I_t]``. Outputs:

* ``video``:   ``[3, num_frames * tubelet_size, H, W]`` in ``[0, 1]``, ordered
  ``[prev_0, frame_0, prev_1, frame_1, ...]``;
* ``actions``: ``[num_frames, 3]``, ``a_t`` = ego-frame pose delta from latent
  ``t`` to ``t + 1`` (``a_{T-1}`` is zero);
* ``states``:  ``[num_frames, 4]``, ``(speed, a_long, a_lat, yaw rate)``; only
  ``states[0]`` is visible to the predictor.
"""

from __future__ import annotations

import glob
import gzip
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


def _load_measurement(path: str) -> dict:
    with gzip.open(path, "rb") as f:
        return json.load(f)


def _load_rgb(path: str, out_hw: Tuple[int, int]) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    if img.size != (out_hw[1], out_hw[0]):
        img = img.resize((out_hw[1], out_hw[0]), Image.BILINEAR)
    return np.asarray(img, dtype=np.uint8)


def find_routes(root: str) -> List[str]:
    """Route directories under ``root`` that contain ``rgb/`` and ``measurements/``."""
    routes = []
    for m in glob.glob(os.path.join(os.path.abspath(root), "**", "measurements"), recursive=True):
        route = os.path.dirname(m)
        if os.path.isdir(os.path.join(route, "rgb")):
            routes.append(route)
    return sorted(routes)


def _ego_matrix(pos_xy: Sequence[float], theta: float) -> np.ndarray:
    c, s = math.cos(theta), math.sin(theta)
    m = np.eye(4, dtype=np.float64)
    m[0, 0], m[0, 1] = c, -s
    m[1, 0], m[1, 1] = s, c
    m[0, 3], m[1, 3] = pos_xy[0], pos_xy[1]
    return m


def _inv_se3(m: np.ndarray) -> np.ndarray:
    out = np.eye(4, dtype=m.dtype)
    out[:3, :3] = m[:3, :3].T
    out[:3, 3] = -m[:3, :3].T @ m[:3, 3]
    return out


def _delta_ego(m_from: dict, m_to: dict) -> np.ndarray:
    """``(dx, dy, dyaw)`` of ``m_to`` in the ego frame of ``m_from``."""
    rel = _inv_se3(_ego_matrix(m_from["pos_global"], m_from["theta"])) @ _ego_matrix(m_to["pos_global"], m_to["theta"])
    dpsi = float(m_to["theta"]) - float(m_from["theta"])
    dpsi = (dpsi + math.pi) % (2 * math.pi) - math.pi
    return np.array([float(rel[0, 3]), float(rel[1, 3]), dpsi], dtype=np.float32)


@dataclass
class ClipSpec:
    route_dir: str
    frame_indices: List[int]  # all num_frames * tubelet_size raw frames, in temporal order


class CarlaClipDataset(Dataset):
    """Clips of consecutive 4 Hz frames with aligned actions and states.

    Args:
        routes: route directories (see ``find_routes``).
        num_frames: number of 2 Hz latents per clip.
        tubelet_size: raw frames per latent.
        img_hw: resize resolution of the encoder input.
        stride: stride (in raw frames) between the starts of consecutive clips.
    """

    def __init__(
        self,
        routes: Sequence[str],
        num_frames: int = 6,
        tubelet_size: int = 2,
        img_hw: Tuple[int, int] = (256, 256),
        stride: int = 4,
    ):
        self.num_frames = num_frames
        self.tubelet_size = tubelet_size
        self.img_hw = tuple(img_hw)
        self.clips: List[ClipSpec] = []
        window = num_frames * tubelet_size
        for route in routes:
            frame_ids = [int(Path(f).stem) for f in sorted(glob.glob(os.path.join(route, "rgb", "*.jpg")))]
            for start in range(0, len(frame_ids) - window + 1, stride):
                raw = frame_ids[start: start + window]
                # Require consecutive frames so every tubelet spans exactly 0.25 s.
                if raw[-1] - raw[0] == window - 1:
                    self.clips.append(ClipSpec(route, raw))

    def __len__(self) -> int:
        return len(self.clips)

    def _actions_and_states(self, route: str, latent_frame_ids: List[int]) -> Tuple[np.ndarray, np.ndarray]:
        meas = [_load_measurement(os.path.join(route, "measurements", f"{fid:04d}.json.gz")) for fid in latent_frame_ids]
        t = self.num_frames
        actions = np.zeros((t, 3), dtype=np.float32)
        for i in range(t - 1):
            actions[i] = _delta_ego(meas[i], meas[i + 1])

        states = np.zeros((t, 4), dtype=np.float32)
        for i in range(t):
            speed = float(meas[i].get("speed", 0.0))
            a_long, yaw_rate = 0.0, 0.0
            if 0 < i < t - 1:  # central differences over the neighbouring latents
                a_long = (float(meas[i + 1].get("speed", speed)) - float(meas[i - 1].get("speed", speed))) / (2 * 0.5)
                dth = float(meas[i + 1].get("theta", meas[i]["theta"])) - float(meas[i - 1].get("theta", meas[i]["theta"]))
                yaw_rate = ((dth + math.pi) % (2 * math.pi) - math.pi) / (2 * 0.5)
            states[i] = [speed, a_long, speed * yaw_rate, yaw_rate]
        return actions, states

    def __getitem__(self, idx: int):
        clip = self.clips[idx]
        frames = [_load_rgb(os.path.join(clip.route_dir, "rgb", f"{fid:04d}.jpg"), self.img_hw) for fid in clip.frame_indices]
        video = torch.from_numpy(np.stack(frames, axis=0)).float().div_(255.0).permute(3, 0, 1, 2).contiguous()
        latent_ids = clip.frame_indices[self.tubelet_size - 1:: self.tubelet_size]
        actions, states = self._actions_and_states(clip.route_dir, latent_ids)
        return {"video": video, "actions": torch.from_numpy(actions), "states": torch.from_numpy(states)}


def collate(batch):
    return {k: torch.stack([b[k] for b in batch], dim=0) for k in ("video", "actions", "states")}
