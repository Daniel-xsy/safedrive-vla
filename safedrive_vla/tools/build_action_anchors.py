"""Build the meta-command action anchors (App. C.4).

For each meta-command, average the body-frame deltas ``(dx, dy, dyaw)`` at
0.5 s spacing of the expert's future 2.5 s over randomly drawn training
frames. The two lane-change anchors are then calibrated: the averaged
lane-change frames mostly have not crossed into the adjacent lane yet, so their
lateral motion is overwritten by one 3.5 m lane offset in the first step.

    python -m safedrive_vla.tools.build_action_anchors \\
        --data-root database/simlingo --data-scale 0.2 \\
        --output safedrive_vla/assets/action_anchors.npz

The released table is ``safedrive_vla/assets/action_anchors.npz``.
"""

from __future__ import annotations

import argparse
import math
import os

import numpy as np

from safedrive_vla.constants import COMMAND_TO_META_COMMAND, META_COMMAND_TO_INDEX, META_COMMANDS
from safedrive_vla.data.datamodule import LMDRIVE_TEMPLATES
from safedrive_vla.data.simlingo_base import BaseDataset

LANE_WIDTH_M = 3.5


def body_frame_deltas(waypoints: np.ndarray, num_steps: int):
    """4 Hz ego-frame waypoints -> ``[num_steps, 3]`` body-frame deltas at 2 Hz."""
    if waypoints.shape[0] < 2 * num_steps:
        return None
    pos = np.zeros((num_steps + 1, 2), dtype=np.float32)
    pos[1:] = waypoints[1: 2 * num_steps: 2]
    if float(np.linalg.norm(pos[-1] - pos[0])) < 0.05:  # parked
        return None
    headings = np.zeros(num_steps + 1, dtype=np.float32)
    for k in range(1, num_steps + 1):
        d = pos[k] - pos[k - 1]
        headings[k] = math.atan2(d[1], d[0]) if float(d @ d) > 1e-6 else headings[k - 1]
    deltas = np.zeros((num_steps, 3), dtype=np.float32)
    for k in range(num_steps):
        c, s = math.cos(-headings[k]), math.sin(-headings[k])
        d = pos[k + 1] - pos[k]
        dyaw = (headings[k + 1] - headings[k] + math.pi) % (2.0 * math.pi) - math.pi
        deltas[k] = (c * d[0] - s * d[1], s * d[0] + c * d[1], dyaw)
    return deltas


def calibrate_lane_changes(anchors: np.ndarray) -> np.ndarray:
    """Lane change: one lane width of lateral motion in the first step (+y is right)."""
    out = anchors.copy()
    for name, side in (("lane_change_left", -1.0), ("lane_change_right", 1.0)):
        m = META_COMMAND_TO_INDEX[name]
        out[m, :, 1:] = 0.0
        out[m, 0, 1] = side * LANE_WIDTH_M
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="database/simlingo")
    parser.add_argument("--data-scale", type=float, default=0.2)
    parser.add_argument("--output", required=True)
    parser.add_argument("--num-steps", type=int, default=5)
    parser.add_argument("--max-samples", type=int, default=20000)
    parser.add_argument("--min-speed", type=float, default=0.5, help="skip frames slower than this (m/s)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    ds = BaseDataset(data_root=args.data_root, split="train", data_scale=args.data_scale,
                     lmdrive_templates=LMDRIVE_TEMPLATES)
    order = np.random.default_rng(args.seed).permutation(len(ds))[: args.max_samples]

    sums = np.zeros((len(META_COMMANDS), args.num_steps, 3), dtype=np.float64)
    counts = np.zeros(len(META_COMMANDS), dtype=np.int64)
    for idx in order:
        measurements, current, _ = ds.load_measurements(int(idx))
        meta = COMMAND_TO_META_COMMAND.get(int(current.get("command", 0)))
        if meta is None or float(current["speed"]) < args.min_speed:
            continue
        waypoints = np.asarray(ds.load_trajectory(measurements, current)["waypoints"], dtype=np.float32)
        deltas = body_frame_deltas(waypoints, args.num_steps)
        if deltas is not None:
            sums[META_COMMAND_TO_INDEX[meta]] += deltas
            counts[META_COMMAND_TO_INDEX[meta]] += 1

    anchors = (sums / np.maximum(counts, 1)[:, None, None]).astype(np.float32)
    anchors = calibrate_lane_changes(anchors)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    np.savez(args.output, anchors=anchors, rollout_steps=np.int32(args.num_steps), dt_wm=np.float32(0.5),
             meta_names=np.asarray(META_COMMANDS), counts=counts)
    print(f"wrote {args.output}; samples per meta-command: {dict(zip(META_COMMANDS, counts.tolist()))}")


if __name__ == "__main__":
    main()
