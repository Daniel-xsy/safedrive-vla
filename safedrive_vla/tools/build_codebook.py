"""Build the action-token codebook (Sec. 4.3).

Collects every 0.25 s ego step ``(dx, dy, dtheta)`` of the PDM-Lite routes,
selects ``N`` well-spread motion primitives with K-disk clustering, and
reserves one entry for the zero action (the most redundant primitive is
replaced by ``(0, 0, 0)``).

    python -m safedrive_vla.tools.build_codebook \\
        --data-root database/simlingo/data/simlingo \\
        --output safedrive_vla/assets/action_codebook.pkl

The released codebook is ``safedrive_vla/assets/action_codebook.pkl``.
"""

from __future__ import annotations

import argparse
import glob
import gzip
import math
import os
from typing import List, Tuple

import numpy as np
import ujson
from tqdm import tqdm

from safedrive_vla.action_tokenizer import ActionTokenizer, CodebookMeta


def _pose(measurement: dict) -> Tuple[float, float, float]:
    m = np.asarray(measurement["ego_matrix"], dtype=np.float64)
    return float(m[0, 3]), float(m[1, 3]), float(math.atan2(m[1, 0], m[0, 0]))


def collect_steps(data_root: str, max_steps: int, min_speed: float) -> np.ndarray:
    """Local-frame step between consecutive frames of every route."""
    steps: List[Tuple[float, float, float]] = []
    route_dirs = sorted(glob.glob(os.path.join(data_root, "**", "measurements"), recursive=True))
    for route_dir in tqdm(route_dirs, desc="routes"):
        prev = None
        for path in sorted(glob.glob(os.path.join(route_dir, "*.json.gz"))):
            try:
                with gzip.open(path, "rt") as f:
                    meas = ujson.load(f)
            except Exception:
                prev = None
                continue
            if "ego_matrix" not in meas:
                prev = None
                continue
            pose = _pose(meas)
            if prev is not None:
                c, s = math.cos(prev[2]), math.sin(prev[2])
                dxw, dyw = pose[0] - prev[0], pose[1] - prev[1]
                dx, dy = c * dxw + s * dyw, -s * dxw + c * dyw
                dth = (pose[2] - prev[2] + math.pi) % (2.0 * math.pi) - math.pi
                # Skip teleports and idle samples.
                teleport = abs(dx) > 15.0 or abs(dy) > 15.0 or abs(dth) > math.pi / 2
                idle = float(meas.get("speed", 0.0)) < min_speed and abs(dx) < 1e-3 and abs(dy) < 1e-3
                if not teleport and not idle:
                    steps.append((dx, dy, dth))
                    if len(steps) >= max_steps:
                        return np.asarray(steps, dtype=np.float32)
            prev = pose
    return np.asarray(steps, dtype=np.float32)


def kdisk_cluster(samples: np.ndarray, num_clusters: int, tolerance: float, seed: int) -> np.ndarray:
    """K-disk clustering: repeatedly pick a random sample, replace it by the
    mean of the samples within ``tolerance``, and keep it if it is not a
    near-duplicate of an existing center."""
    rng = np.random.default_rng(seed)
    centers = np.zeros((num_clusters, samples.shape[1]), dtype=np.float32)
    n_found, attempts = 0, 0
    with tqdm(total=num_clusters, desc="k-disk") as bar:
        while n_found < num_clusters and attempts < num_clusters * 500:
            attempts += 1
            seed_point = samples[int(rng.integers(0, samples.shape[0]))]
            diff = samples - seed_point[None]
            members = np.einsum("ij,ij->i", diff, diff) <= tolerance * tolerance
            if not members.any():
                continue
            mean = samples[members].mean(axis=0)
            if n_found > 0 and np.linalg.norm(centers[:n_found] - mean[None], axis=1).min() < tolerance:
                continue
            centers[n_found] = mean
            n_found += 1
            bar.update(1)
    return centers[:n_found]


def reserve_zero_action(codebook: np.ndarray, top_k: int = 5) -> np.ndarray:
    """Replace the most redundant entry (smallest mean distance to its
    ``top_k`` nearest neighbours) by the zero action."""
    if np.linalg.norm(codebook, axis=-1).min() <= 1e-3:
        return codebook
    dist = np.sqrt(((codebook[:, None] - codebook[None]) ** 2).sum(-1))
    np.fill_diagonal(dist, np.inf)
    redundancy = np.partition(dist, top_k, axis=-1)[:, :top_k].mean(axis=-1)
    out = codebook.copy()
    out[int(np.argmin(redundancy))] = 0.0
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--num-tokens", type=int, default=2048)
    parser.add_argument("--tolerance", type=float, default=0.05)
    parser.add_argument("--max-steps", type=int, default=2_000_000)
    parser.add_argument("--min-speed", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    steps = collect_steps(args.data_root, args.max_steps, args.min_speed)
    print(f"collected {len(steps):,} steps")
    codebook = reserve_zero_action(kdisk_cluster(steps, args.num_tokens, args.tolerance, args.seed))
    meta = CodebookMeta(num_tokens=int(codebook.shape[0]), hz=4.0, dt=0.25, num_steps=1, repr="dxdydtheta",
                        source="K-disk clustering of PDM-Lite ego motion with a reserved zero action")
    ActionTokenizer.save(args.output, codebook, meta)
    print(f"wrote {codebook.shape[0]} tokens to {args.output}")


if __name__ == "__main__":
    main()
