"""Route scanning and sample loading for the SimLingo CARLA dataset.

Adapted from SimLingo (https://github.com/RenzKa/simlingo, Apache-2.0),
``simlingo_training/dataloader/dataset_base.py``, which in turn builds on
carla_garage (MIT). Only the parts used by SafeDriveVLA are kept: the driving
and action-dreaming samples of the PDM-Lite data, a single front-camera frame,
and the deterministic balanced ``data_scale`` subsampling.
"""

from __future__ import annotations

import glob
import gzip
import hashlib
import math
import os
import pickle as pkl
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import ujson
from imgaug import augmenters as ia
from torch.utils.data import Dataset
from tqdm import tqdm

# Sample layout: one current frame and 10 future frames at 4 Hz (2.5 s).
PRED_LEN = 11
SKIP_FIRST_N_FRAMES = 10
NUM_ROUTE_POINTS = 20
# Fraction of routes used for training when all towns are mixed (driving samples).
TRAIN_SPLIT_FRACTION = 0.99
# Image augmentation, following SimLingo.
IMG_AUGMENTATION_PROB = 0.5
IMG_SHIFT_AUGMENTATION_PROB = 0.5

# Bucket names that differ from the keys stored in ``buckets_paths.pkl``.
_BUCKET_ALIASES = {
    "acceleration_negative_5": ["acceleration_-5"],
    "acceleration_negative_1": ["acceleration_-1"],
    "acceleration_positive_1": ["acceleration_5"],
    "acceleration_positive_5": ["acceleration_20"],
    "lateral_control_1_2": ["lateral_control_1", "lateral_control_2"],
    "lateral_control_higher_5": ["lateral_control_5", "lateral_control_1000000"],
}
# Prefix of the paths stored in the released ``buckets_paths.pkl``.
_BUCKET_PATH_PREFIX = "database/simlingo_v2_2025_01_10"

# Route-planner command -> SimLingo navigation text.
COMMAND_TEXT = {
    1: "go left at the next intersection",
    2: "go right at the next intersection",
    3: "go straight at the next intersection",
    4: "follow the road",
    5: "do a lane change to the left",
    6: "do a lane change to the right",
}
# Route-planner command -> indices into the LMDrive instruction templates.
_LMDRIVE_TEMPLATE_IDS = {
    1: [0, 2, 4, 7],
    2: [1, 3, 5, 8],
    3: [6, 9],
    4: [38, 40, 42, 43, 44, 45],
    5: [34, 36],
    6: [35, 37],
}


def _stable_hash(value: str) -> str:
    return hashlib.sha1(value.encode("utf-8")).hexdigest()


def image_augmenter(prob: float = 0.2) -> ia.Sequential:
    return ia.Sequential(
        [
            ia.Sometimes(prob, ia.GaussianBlur((0, 1.0))),
            ia.Sometimes(prob, ia.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5)),
            ia.Sometimes(prob, ia.Dropout((0.01, 0.1), per_channel=0.5)),
            ia.Sometimes(prob, ia.Multiply((1 / 1.2, 1.2), per_channel=0.5)),
            ia.Sometimes(prob, ia.LinearContrast((1 / 1.2, 1.2), per_channel=0.5)),
            ia.Sometimes(prob, ia.Grayscale((0.0, 0.5))),
            ia.Sometimes(prob, ia.ElasticTransformation(alpha=(0.5, 1.5), sigma=0.25)),
        ],
        random_order=True,
    )


class BaseDataset(Dataset):
    """Indexes the frames of the SimLingo dataset and loads their labels.

    Args:
        data_root: the SimLingo ``database/simlingo`` folder (contains ``data/``,
            ``dreamer/`` and ``buckets_paths.pkl``).
        split: ``train`` or ``val``.
        bucket_name: SimLingo scenario bucket (``all`` for every frame).
        data_scale: fraction of frames kept, balanced over (route group, town)
            and deterministic given the data layout.
        action_dreaming: index SimLingo's action-dreaming frames (alternative
            instructions stored under ``dreamer/``) instead of driving frames.
        lmdrive_templates: JSON file with the LMDrive instruction paraphrases.
    """

    def __init__(
        self,
        data_root: str,
        split: str,
        bucket_name: str = "all",
        data_scale: float = 1.0,
        action_dreaming: bool = False,
        lmdrive_templates: str = "",
    ):
        self.data_root = os.path.abspath(data_root)
        self.split = split
        self.bucket_name = bucket_name
        self.data_scale = float(data_scale)
        if not 0.0 < self.data_scale <= 1.0:
            raise ValueError(f"data_scale must be in (0, 1], got {data_scale}")
        self.tfs = image_augmenter(prob=IMG_AUGMENTATION_PROB)
        with open(lmdrive_templates, "r") as f:
            self.command_templates = ujson.load(f)

        bucket_files = None
        if bucket_name != "all":
            bucket_files = self._load_bucket(bucket_name)

        route_dirs = glob.glob(f"{self.data_root}/data/simlingo/*/*/*/Town*")
        route_dirs = sorted(
            route_dirs,
            key=lambda d: _stable_hash(f"route_split:{os.path.relpath(d, self.data_root)}"),
        )
        if action_dreaming:
            # Official split: Town12 + old towns for training, Town13 for validation.
            if split == "train":
                route_dirs = [d for d in route_dirs if "routes_training" in d]
            else:
                route_dirs = [d for d in route_dirs if "routes_validation" in d]
                route_dirs = route_dirs[: int(0.02 * len(route_dirs))]
        else:
            n_train = int(TRAIN_SPLIT_FRACTION * len(route_dirs))
            route_dirs = route_dirs[:n_train] if split == "train" else route_dirs[n_train:]

        images, measurements, sample_start, dreams = [], [], [], []
        n_skipped_routes = 0
        for route_dir in tqdm(route_dirs, file=sys.stdout, desc=f"index {split}/{bucket_name}"):
            if action_dreaming and not os.path.exists(self._to_dreamer_path(route_dir)):
                continue
            if not self._route_is_clean(route_dir):
                n_skipped_routes += 1
                continue

            num_seq = len(os.listdir(f"{route_dir}/rgb"))
            for seq in range(SKIP_FIRST_N_FRAMES, num_seq - PRED_LEN - 2):
                measurement_file = f"{route_dir}/measurements/{seq:04}.json.gz"
                if action_dreaming:
                    dream_file = self._to_dreamer_path(measurement_file.replace("/measurements/", "/dreamer/"))
                    if not os.path.exists(dream_file):
                        continue
                if bucket_files is not None:
                    route_files = bucket_files.get(f"{route_dir}/measurements")
                    if route_files is None or f"{seq:04}.json.gz" not in route_files:
                        continue
                images.append(f"{route_dir}/rgb/{seq:04}.jpg")
                measurements.append(f"{route_dir}/measurements")
                sample_start.append(seq)
                if action_dreaming:
                    dreams.append(dream_file)

        keep = self._balanced_subset(measurements, sample_start, action_dreaming)
        # Store paths as numpy byte strings: Python lists of strings leak memory
        # in multi-worker data loaders (pytorch/pytorch#13246).
        self.images = np.array([images[i] for i in keep]).astype(np.bytes_)
        self.measurements = np.array([measurements[i] for i in keep]).astype(np.bytes_)
        self.sample_start = np.array([sample_start[i] for i in keep])
        self.dreams = np.array([dreams[i] for i in keep]).astype(np.bytes_) if action_dreaming else None
        print(
            f"[{split}/{bucket_name}] {len(self.images)} samples "
            f"({len(route_dirs)} routes, {n_skipped_routes} skipped as imperfect)"
        )

    def __len__(self) -> int:
        return self.images.shape[0]

    # ------------------------------------------------------------------
    # indexing helpers
    # ------------------------------------------------------------------
    def _to_dreamer_path(self, path: str) -> str:
        """``<root>/data/simlingo/...`` -> ``<root>/dreamer/simlingo/...``."""
        rel = os.path.relpath(path, self.data_root)
        return os.path.join(self.data_root, "dreamer", os.path.relpath(rel, "data"))

    def _load_bucket(self, bucket_name: str) -> Dict[str, set]:
        with open(f"{self.data_root}/buckets_paths.pkl", "rb") as f:
            bucket_dict = pkl.load(f)
        keys = _BUCKET_ALIASES.get(bucket_name, [bucket_name])
        if any(k not in bucket_dict for k in keys):
            raise ValueError(f"Bucket {bucket_name} not found in buckets_paths.pkl")
        files: Dict[str, set] = defaultdict(set)
        for key in keys:
            for path in bucket_dict[key]:
                path = Path(path.replace(_BUCKET_PATH_PREFIX, self.data_root))
                files[str(path.parent)].add(path.name)
        return files

    @staticmethod
    def _route_is_clean(route_dir: str) -> bool:
        """Keep expert routes without infractions (minimum-speed infractions and
        up to 6% of missing route completion are tolerated)."""
        results_file = f"{route_dir}/results.json.gz"
        if not os.path.isfile(results_file):
            return False
        try:
            with gzip.open(results_file, "rt") as f:
                results = ujson.load(f)
        except Exception:
            return False
        if results["scores"]["score_composed"] < 100.0:
            infractions = results["infractions"]
            only_min_speed = results["num_infractions"] == (
                len(infractions["min_speed_infractions"]) + len(infractions["outside_route_lanes"])
            )
            if not (results["scores"]["score_route"] > 94.0 and only_min_speed):
                return False
        return True

    def _group_key(self, measurement_path: str, action_dreaming: bool) -> Tuple[str, ...]:
        parts = Path(measurement_path).parent.parent.parts
        town = next((p for p in reversed(parts) if p.startswith("Town")), "unknown_town").split("_", 1)[0]
        simlingo_idx = [i for i, p in enumerate(parts) if p == "simlingo"]
        route_group = "unknown_group"
        if simlingo_idx and simlingo_idx[-1] + 1 < len(parts):
            route_group = parts[simlingo_idx[-1] + 1]
        kind = "dreamer" if action_dreaming else "driving"
        return (self.split, kind, self.bucket_name, route_group, town)

    def _balanced_subset(self, measurements: List[str], sample_start: List[int], action_dreaming: bool) -> List[int]:
        """Indices of a ``data_scale`` subset, balanced over (route group, town)."""
        if self.data_scale >= 1.0 or not measurements:
            return list(range(len(measurements)))

        groups: Dict[Tuple[str, ...], List[int]] = defaultdict(list)
        keys = {}
        for i, (meas_dir, start) in enumerate(zip(measurements, sample_start)):
            path = f"{meas_dir}/{start:04}.json.gz"
            keys[i] = os.path.relpath(path, self.data_root)
            groups[self._group_key(path, action_dreaming)].append(i)

        total = sum(len(v) for v in groups.values())
        target = max(1, min(total, int(round(total * self.data_scale))))
        names = sorted(groups, key=lambda g: _stable_hash(repr(g)))
        desired = {g: len(groups[g]) * self.data_scale for g in names}

        quotas = {g: 0 for g in names}
        if target >= len(names):
            for g in names:
                quotas[g] = min(len(groups[g]), max(1, math.floor(desired[g])))
        else:
            ranked = sorted(names, key=lambda g: (-desired[g], _stable_hash(repr(g))))
            for g in ranked[:target]:
                quotas[g] = 1

        current = sum(quotas.values())
        while current < target:
            candidates = [g for g in names if quotas[g] < len(groups[g])]
            if not candidates:
                break
            g = min(candidates, key=lambda c: (-(desired[c] - quotas[c]), _stable_hash(repr(c))))
            quotas[g] += 1
            current += 1
        min_quota = 1 if target >= len(names) else 0
        while current > target:
            candidates = [g for g in names if quotas[g] > min_quota]
            if not candidates:
                break
            g = min(candidates, key=lambda c: (-(quotas[c] - desired[c]), _stable_hash(repr(c))))
            quotas[g] -= 1
            current -= 1

        selected = []
        for g in names:
            ranked = sorted(groups[g], key=lambda i: (_stable_hash(keys[i]), keys[i]))
            selected.extend(ranked[: quotas[g]])
        return sorted(selected)

    # ------------------------------------------------------------------
    # per-sample loading
    # ------------------------------------------------------------------
    def load_measurements(self, index: int) -> Tuple[List[dict], dict, str]:
        """Current + future measurements, the current one, and its path."""
        meas_dir = str(self.measurements[index], encoding="utf-8")
        start = int(self.sample_start[index])
        loaded = []
        for i in range(PRED_LEN + 1):
            path = f"{meas_dir}/{start + i:04}.json.gz"
            try:
                with gzip.open(path, "rt") as f:
                    loaded.append(ujson.load(f))
            except FileNotFoundError:
                loaded.append(loaded[-1])
        return loaded, loaded[0], f"{meas_dir}/{start:04}.json.gz"

    @staticmethod
    def get_waypoints(measurements: List[dict], y_augmentation: float = 0.0, yaw_augmentation: float = 0.0) -> List[np.ndarray]:
        """Ego positions of ``measurements`` in the frame of the first one."""
        origin_matrix = np.array(measurements[0]["ego_matrix"])[:3]
        origin_translation = origin_matrix[:, 3:4]
        origin_rotation = origin_matrix[:, :3]

        aug_yaw_rad = np.deg2rad(yaw_augmentation)
        rotation = np.array([[np.cos(aug_yaw_rad), -np.sin(aug_yaw_rad)], [np.sin(aug_yaw_rad), np.cos(aug_yaw_rad)]])
        translation = np.array([[0.0], [y_augmentation]])

        waypoints = []
        for m in measurements:
            waypoint = np.array(m["ego_matrix"])[:3, 3:4]
            ego = origin_rotation.T @ (waypoint - origin_translation)
            waypoints.append(np.squeeze(rotation.T @ (np.expand_dims(ego[:2, 0], axis=1) - translation)))
        return waypoints

    @staticmethod
    def augment_route(route: np.ndarray, y_augmentation: float = 0.0, yaw_augmentation: float = 0.0) -> np.ndarray:
        aug_yaw_rad = np.deg2rad(yaw_augmentation)
        rotation = np.array([[np.cos(aug_yaw_rad), -np.sin(aug_yaw_rad)], [np.sin(aug_yaw_rad), np.cos(aug_yaw_rad)]])
        translation = np.array([[0.0, y_augmentation]])
        return (rotation.T @ (route - translation).T).T

    @staticmethod
    def augment_target_point(target_point: np.ndarray, y_augmentation: float = 0.0, yaw_augmentation: float = 0.0) -> np.ndarray:
        aug_yaw_rad = np.deg2rad(yaw_augmentation)
        rotation = np.array([[np.cos(aug_yaw_rad), -np.sin(aug_yaw_rad)], [np.sin(aug_yaw_rad), np.cos(aug_yaw_rad)]])
        translation = np.array([[0.0], [y_augmentation]])
        return np.squeeze(rotation.T @ (np.expand_dims(target_point, axis=1) - translation))

    @staticmethod
    def equal_spacing_route(points: np.ndarray) -> np.ndarray:
        """Resample a route at 1 m spacing (20 points)."""
        route = np.concatenate((np.zeros_like(points[:1]), points))
        shift = np.roll(route, 1, axis=0)
        shift[0] = shift[1]
        dists = np.cumsum(np.linalg.norm(route - shift, axis=1))
        dists += np.arange(0, len(dists)) * 1e-4  # strictly increasing
        x = np.arange(0, NUM_ROUTE_POINTS, 1)
        return np.array([np.interp(x, dists, route[:, 0]), np.interp(x, dists, route[:, 1])]).T

    def load_trajectory(self, measurements: List[dict], current: dict, y_aug: float = 0.0, yaw_aug: float = 0.0) -> Dict[str, np.ndarray]:
        """Future waypoints (4 Hz) and the 1 m-spaced route ahead (the path)."""
        waypoints = self.get_waypoints(measurements, y_augmentation=y_aug, yaw_augmentation=yaw_aug)
        waypoints_org = self.get_waypoints(measurements, y_augmentation=0, yaw_augmentation=0)
        route = np.array(current["route"])
        return {
            "waypoints": np.array(waypoints[1:-1]),
            "waypoints_org": np.array(waypoints_org[1:-1]),
            "path": self.equal_spacing_route(self.augment_route(route, y_augmentation=y_aug, yaw_augmentation=yaw_aug)),
            "path_org": self.equal_spacing_route(self.augment_route(route, y_augmentation=0, yaw_augmentation=0)),
        }

    def load_image(self, index: int, shifted_camera: bool = False) -> np.ndarray:
        """Front-camera frame as ``[1, C, H, W]`` uint8, bonnet cropped."""
        path = str(self.images[index], encoding="utf-8")
        if shifted_camera:
            path = path.replace("/rgb/", "/rgb_augmented/")
        image = cv2.cvtColor(cv2.imread(path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        image = self.tfs(image=image)
        # Remove the bonnet (bottom 4.8/16 of the image); required by the
        # shifted-camera augmentation.
        image = image[: int(image.shape[0] - (image.shape[0] * 4.8) // 16), :, :]
        return np.transpose(image[None], (0, 3, 1, 2))

    def navigation_prompts(self, current: dict, target_point: np.ndarray) -> List[str]:
        """The three navigation-signal phrasings of one frame: target waypoints,
        route-planner command, and an LMDrive natural-language paraphrase."""
        dist_to_command = int(np.linalg.norm(target_point))
        command = COMMAND_TEXT[current["command"]]
        next_command = COMMAND_TEXT[current["next_command"]]
        next_command = f" then {next_command}" if command != next_command else ""
        if current["command"] == 4:
            command_prompt = f"Command: {command}{next_command}."
        else:
            command_prompt = f"Command: {command} in {dist_to_command} meter{next_command}."

        template_id = random.choice(_LMDRIVE_TEMPLATE_IDS[current["command"]])
        lmdrive = random.choice(self.command_templates[str(template_id)]).replace("[x]", str(dist_to_command))
        return ["Target waypoint: <TARGET_POINT><TARGET_POINT>.", command_prompt, f"Command: {lmdrive}."]
