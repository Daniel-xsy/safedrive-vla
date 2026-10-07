"""Discrete action tokens (Sec. 4.3).

Each codebook entry is a single-step motion primitive ``(dx, dy, dtheta)`` in
the ego frame of the previous step, built offline by K-disk clustering
(``safedrive_vla/tools/build_codebook.py``). A future trajectory of ``T`` steps
at 4 Hz is encoded as ``T`` tokens ``<action_0> ... <action_{N-1}>``.
"""

from __future__ import annotations

import pickle
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np

ACTION_TOKEN_FORMAT = "<action_{idx}>"


def action_token_strings(num_tokens: int) -> List[str]:
    return [ACTION_TOKEN_FORMAT.format(idx=i) for i in range(num_tokens)]


@dataclass
class CodebookMeta:
    num_tokens: int
    hz: float
    dt: float
    num_steps: int
    repr: str
    source: str = ""


class ActionTokenizer:
    """Convert between ego-frame trajectories and action-token indices."""

    def __init__(self, codebook: np.ndarray, meta: Optional[CodebookMeta] = None):
        if codebook.ndim != 2 or codebook.shape[1] != 3:
            raise ValueError(f"Expected codebook of shape [N, 3], got {codebook.shape}")
        self.codebook = codebook.astype(np.float32)
        self.num_tokens = int(codebook.shape[0])
        self.meta = meta

    @classmethod
    def from_file(cls, path: Union[str, Path]) -> "ActionTokenizer":
        with open(path, "rb") as f:
            payload = pickle.load(f)
        codebook = np.asarray(payload["codebook"], dtype=np.float32)
        meta = payload.get("meta", {})
        return cls(
            codebook=codebook,
            meta=CodebookMeta(
                num_tokens=int(meta.get("num_tokens", codebook.shape[0])),
                hz=float(meta.get("hz", 4.0)),
                dt=float(meta.get("dt", 0.25)),
                num_steps=int(meta.get("num_steps", 1)),
                repr=str(meta.get("repr", "dxdydtheta")),
                source=str(meta.get("source", "")),
            ),
        )

    @staticmethod
    def save(path: Union[str, Path], codebook: np.ndarray, meta: CodebookMeta) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"codebook": codebook.astype(np.float32), "meta": asdict(meta)}, f)

    @staticmethod
    def encode_to_string(indices: Iterable[int]) -> str:
        return "".join(ACTION_TOKEN_FORMAT.format(idx=int(i)) for i in indices)

    @staticmethod
    def derive_heading_from_waypoints(waypoints: np.ndarray, init_heading: float = 0.0) -> np.ndarray:
        """Per-waypoint heading from the direction of successive segments."""
        waypoints = np.asarray(waypoints, dtype=np.float32)
        headings = np.zeros(waypoints.shape[0], dtype=np.float32)
        prev_pos = np.zeros(2, dtype=np.float32)
        prev_head = float(init_heading)
        for i in range(waypoints.shape[0]):
            dx = float(waypoints[i, 0] - prev_pos[0])
            dy = float(waypoints[i, 1] - prev_pos[1])
            if dx * dx + dy * dy > 1e-6:
                prev_head = float(np.arctan2(dy, dx))
            headings[i] = prev_head
            prev_pos = waypoints[i]
        return headings

    def match_trajectory(
        self,
        waypoints: np.ndarray,
        vehicle_width: float = 2.0,
        vehicle_length: float = 4.8,
    ) -> np.ndarray:
        """Greedy nearest-neighbour tokenization of a ``[T, 2]`` trajectory.

        At every step, each codebook entry is rolled out from the current
        *token-decoded* pose and compared with the ground-truth pose through the
        four corners of the vehicle bounding box, so heading errors are weighted
        by the vehicle geometry. The rolling pose advances with the selected
        token, which keeps the quantisation error consistent with inference.
        """
        waypoints = np.asarray(waypoints, dtype=np.float32)
        if waypoints.ndim != 2 or waypoints.shape[1] != 2:
            raise ValueError(f"Expected waypoints shape [T, 2], got {waypoints.shape}")
        headings = self.derive_heading_from_waypoints(waypoints)
        half_w, half_l = vehicle_width / 2.0, vehicle_length / 2.0

        indices = np.zeros(waypoints.shape[0], dtype=np.int64)
        pos = np.zeros(2, dtype=np.float32)
        theta = 0.0
        book = self.codebook
        for t in range(waypoints.shape[0]):
            c, s = float(np.cos(theta)), float(np.sin(theta))
            cand_x = pos[0] + c * book[:, 0] - s * book[:, 1]
            cand_y = pos[1] + s * book[:, 0] + c * book[:, 1]
            cand_theta = theta + book[:, 2]
            cand_contours = _contours(cand_x, cand_y, cand_theta, half_w, half_l)  # [N, 4, 2]
            gt_contour = _contour(
                float(waypoints[t, 0]), float(waypoints[t, 1]), float(headings[t]), half_w, half_l
            )  # [4, 2]
            dist = np.sqrt(((cand_contours - gt_contour[None]) ** 2).sum(axis=-1)).sum(axis=-1)
            idx = int(np.argmin(dist))
            indices[t] = idx

            dx, dy, dtheta = (float(v) for v in book[idx])
            pos = pos + np.array([c * dx - s * dy, s * dx + c * dy], dtype=np.float32)
            theta = _wrap_angle(theta + dtheta)
        return indices

    def rollout_xy(self, indices: Sequence[int]) -> np.ndarray:
        """Chain token indices into a ``[T, 2]`` ego-frame trajectory."""
        pos = np.zeros(2, dtype=np.float32)
        theta = 0.0
        out = []
        for idx in indices:
            idx = int(idx)
            if idx < 0 or idx >= self.num_tokens:
                idx = 0
            dx, dy, dtheta = (float(v) for v in self.codebook[idx])
            c, s = float(np.cos(theta)), float(np.sin(theta))
            pos = pos + np.array([c * dx - s * dy, s * dx + c * dy], dtype=np.float32)
            theta = _wrap_angle(theta + dtheta)
            out.append([pos[0], pos[1]])
        return np.asarray(out, dtype=np.float32)


def _wrap_angle(a: float) -> float:
    return float((a + np.pi) % (2.0 * np.pi) - np.pi)


def _contour(x: float, y: float, theta: float, half_w: float, half_l: float) -> np.ndarray:
    """Bounding-box corners (LF, RF, RB, LB) of one pose: ``[4, 2]``."""
    c, s = np.cos(theta), np.sin(theta)
    lc, ls = half_l * c, half_l * s
    wc, ws = half_w * c, half_w * s
    return np.array([
        [x + lc - ws, y + ls + wc],
        [x + lc + ws, y + ls - wc],
        [x - lc + ws, y - ls - wc],
        [x - lc - ws, y - ls + wc],
    ], dtype=np.float32)


def _contours(x, y, theta, half_w: float, half_l: float) -> np.ndarray:
    """Vectorized bounding-box corners of ``N`` poses: ``[N, 4, 2]``."""
    c, s = np.cos(theta), np.sin(theta)
    lc, ls = half_l * c, half_l * s
    wc, ws = half_w * c, half_w * s
    return np.stack([
        np.stack([x + lc - ws, y + ls + wc], axis=-1),
        np.stack([x + lc + ws, y + ls - wc], axis=-1),
        np.stack([x - lc + ws, y - ls - wc], axis=-1),
        np.stack([x - lc - ws, y - ls + wc], axis=-1),
    ], axis=-2)
