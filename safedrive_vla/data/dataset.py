"""SafeDriveVLA training samples.

Each sample pairs the front-camera frame and a navigation signal with the
assistant target ``<MODE><ACTIONS> a_0 ... a_9``: the driving mode of the
expert trajectory (Sec. 4.1) followed by its discrete action tokens (Sec. 4.3).
When the world model is enabled, the sample also carries the inputs of
navigation-conditioned dreaming (Sec. 4.2): the current two-frame tubelet, the
ego state, and the action sequence the frozen world model rolls out under.

Two sample types are mixed 50/50 during training:

* ``DrivingDataset``: PDM-Lite expert frames. The navigation signal is drawn
  uniformly from the target waypoints, the route-planner command, and a
  natural-language paraphrase of the command.
* ``ActionDreamingDataset``: SimLingo's action-dreaming frames, where the
  prompt carries an alternative instruction. Unsafe instructions are labeled
  ``<FALLBACK>`` and supervised with the safe expert trajectory; safe ones keep
  the trajectory that executes the instruction.
"""

from __future__ import annotations

import gzip
import math
import random
from typing import Any, Dict, List, NamedTuple, Optional, Tuple

import cv2
import numpy as np
import ujson
from PIL import Image

from safedrive_vla.action_tokenizer import ActionTokenizer
from safedrive_vla.constants import (
    ACTIONS_TOKEN,
    COMMAND_TO_META_COMMAND,
    DRIVING_MODE_TO_INDEX,
    META_COMMAND_TO_INDEX,
    MODE_TOKENS,
    TASK_PROMPT,
)
from safedrive_vla.data.driving_mode import classify_driving_mode, classify_speed_instruction
from safedrive_vla.data.simlingo_base import IMG_SHIFT_AUGMENTATION_PROB, BaseDataset

MEASUREMENT_DT = 0.25  # PDM-Lite frames are stored at 4 Hz.
WM_DT = 0.5            # world-model step (2 Hz)

# Leading stationary steps of a stopped vehicle are encoded with the zero-action token.
ZERO_ACTION_SPEED = 0.1
ZERO_ACTION_POS = 0.05
ZERO_ACTION_YAW = 0.01

# Ego-extrapolation action input: hold still below this speed (m/s).
EXTRAPOLATION_MIN_SPEED = 0.5

# Action-dreaming categories whose instruction asks for a speed.
SPEED_DREAM_CATEGORIES = frozenset({"target_speed", "faster", "faster_factor", "slower", "slower_factor", "stop"})


class DrivingSample(NamedTuple):
    conversation: List[Dict[str, Any]]
    image: np.ndarray                    # [1, 3, H, W] uint8 front camera
    waypoints: np.ndarray                # [10, 2] future ego positions at 4 Hz
    path: np.ndarray                     # [20, 2] path ahead at 1 m spacing
    target_points: np.ndarray            # [2, 2] next two route target points
    speed: float
    action_tokens: List[int]
    mode_label: int = -1                 # index into DRIVING_MODES, -1 = no mode token
    meta_command: int = -1               # index into META_COMMANDS, -1 = unknown
    wm_tubelet: Optional[np.ndarray] = None   # [2, 3, H, W] uint8, frames t-0.25 s and t
    wm_state: Optional[np.ndarray] = None     # [4] speed, a_long, a_lat, yaw rate
    wm_actions: Optional[np.ndarray] = None   # [K, 3] (dx, dy, dyaw) per 0.5 s step
    measurement_path: str = ""


def _wrap_pi(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def _read_measurement(path: str) -> Optional[dict]:
    try:
        with gzip.open(path, "rt") as f:
            return ujson.load(f)
    except (FileNotFoundError, OSError):
        return None


def load_wm_tubelet(rgb_dir: str, frame: int, size: Tuple[int, int]) -> np.ndarray:
    """Frames ``frame-1`` and ``frame`` as a ``[2, 3, H, W]`` uint8 tubelet."""
    frames = []
    for fid in (max(frame - 1, 0), frame):
        try:
            img = Image.open(f"{rgb_dir}/{fid:04}.jpg").convert("RGB")
        except (FileNotFoundError, OSError):
            frames.append(np.zeros((size[0], size[1], 3), dtype=np.uint8))
            continue
        if img.size != (size[1], size[0]):
            img = img.resize((size[1], size[0]), Image.BILINEAR)
        frames.append(np.asarray(img, dtype=np.uint8))
    return np.transpose(np.stack(frames, axis=0), (0, 3, 1, 2)).astype(np.uint8)


def load_wm_state(measurements_dir: str, frame: int) -> np.ndarray:
    """Ego state ``[speed, a_long, a_lat, yaw_rate]`` at ``frame``."""
    cur = _read_measurement(f"{measurements_dir}/{frame:04}.json.gz")
    if cur is None:
        return np.zeros(4, dtype=np.float32)
    prev = _read_measurement(f"{measurements_dir}/{max(frame - 1, 0):04}.json.gz") or cur
    speed = float(cur.get("speed", 0.0))
    theta = float(cur.get("theta", 0.0))
    a_long = (speed - float(prev.get("speed", speed))) / MEASUREMENT_DT
    yaw_rate = _wrap_pi(theta - float(prev.get("theta", theta))) / MEASUREMENT_DT
    return np.asarray([speed, a_long, speed * yaw_rate, yaw_rate], dtype=np.float32)


def extrapolate_ego_motion(speeds: List[float], thetas: List[float], num_steps: int) -> np.ndarray:
    """Ego-extrapolation action input (Fig. 3c): repeat the mean speed and yaw
    rate of the past 2.5 s for ``num_steps`` world-model steps."""
    out = np.zeros((num_steps, 3), dtype=np.float32)
    if len(speeds) < 2 or speeds[-1] < EXTRAPOLATION_MIN_SPEED:
        return out
    mean_speed = float(np.mean(speeds))
    if mean_speed < EXTRAPOLATION_MIN_SPEED:
        return out
    dtheta = np.array([_wrap_pi(thetas[i + 1] - thetas[i]) for i in range(len(thetas) - 1)], dtype=np.float64)
    yaw_rate = float(dtheta.mean() / MEASUREMENT_DT)
    out[:, 0] = mean_speed * WM_DT
    out[:, 2] = _wrap_pi(yaw_rate * WM_DT)
    return out


def _past_speeds_and_headings(measurements_dir: str, frame: int, window_s: float = 2.5) -> Tuple[List[float], List[float]]:
    """Speeds and headings of the frames in the past ``window_s`` (current included)."""
    if _read_measurement(f"{measurements_dir}/{frame:04}.json.gz") is None:
        return [], []
    speeds: List[float] = []
    thetas: List[float] = []
    n_back = max(1, int(round(window_s / MEASUREMENT_DT)))
    for offset in range(n_back, -1, -1):
        m = _read_measurement(f"{measurements_dir}/{max(0, frame - offset):04}.json.gz")
        if m is None:
            if speeds:  # repeat the last valid frame
                speeds.append(speeds[-1])
                thetas.append(thetas[-1])
            continue
        speeds.append(float(m.get("speed", 0.0)))
        thetas.append(float(m.get("theta", 0.0)))
    return speeds, thetas


def load_action_anchors(path: str, num_steps: int) -> np.ndarray:
    """Meta-command action anchors ``[num_meta_commands, K, 3]`` (App. C.4)."""
    with np.load(path, allow_pickle=False) as data:
        anchors = np.asarray(data["anchors"], dtype=np.float32)
    if anchors.ndim != 3 or anchors.shape[1:] != (num_steps, 3):
        raise ValueError(f"Expected action anchors of shape [M, {num_steps}, 3], got {anchors.shape}")
    return anchors


class _SafeDriveSampleMixin:
    """Action tokens, driving-mode tokens and world-model inputs."""

    def _init_safedrive(
        self,
        codebook_path: str,
        use_mode_token: bool,
        world_model: Optional[Dict[str, Any]],
    ) -> None:
        self.action_tokenizer = ActionTokenizer.from_file(codebook_path)
        norms = np.linalg.norm(self.action_tokenizer.codebook, axis=-1)
        self.zero_action_token = int(np.argmin(norms)) if norms.min() <= 1e-3 else -1
        self.use_mode_token = bool(use_mode_token)

        wm = dict(world_model or {})
        self.wm_enabled = bool(wm.get("enabled", False))
        self.wm_image_size = (int(wm.get("img_size", 256)), int(wm.get("img_size", 256)))
        self.wm_rollout_steps = int(wm.get("rollout_steps", 5))
        self.wm_action_input = str(wm.get("action_input", "action_anchor"))
        if self.wm_action_input not in ("action_anchor", "ego_extrapolation"):
            raise ValueError(f"Unknown world_model.action_input: {self.wm_action_input}")
        self.action_anchors = None
        if self.wm_enabled and self.wm_action_input == "action_anchor":
            self.action_anchors = load_action_anchors(wm["action_anchor_path"], self.wm_rollout_steps)

    def _snap_zero_prefix(self, indices: np.ndarray, waypoints: np.ndarray, speed: float) -> np.ndarray:
        """Encode the leading stationary steps of a stopped vehicle with the zero action."""
        if self.zero_action_token < 0 or speed >= ZERO_ACTION_SPEED:
            return indices
        out = indices.copy()
        prev = np.zeros(2, dtype=np.float32)
        prev_yaw = 0.0
        for t in range(int(waypoints.shape[0])):
            gx = float(waypoints[t, 0]) - float(prev[0])
            gy = float(waypoints[t, 1]) - float(prev[1])
            c, s = math.cos(prev_yaw), math.sin(prev_yaw)
            dx_local = c * gx + s * gy
            dy_local = -s * gx + c * gy
            disp2 = dx_local * dx_local + dy_local * dy_local
            if disp2 > ZERO_ACTION_POS * ZERO_ACTION_POS:
                break
            if disp2 > 1e-6:
                new_yaw = math.atan2(gy, gx)
                if abs((new_yaw - prev_yaw + math.pi) % (2.0 * math.pi) - math.pi) > ZERO_ACTION_YAW:
                    break
                prev_yaw = new_yaw
            out[t] = self.zero_action_token
            prev = waypoints[t].astype(np.float32)
        return out

    def _wm_inputs(self, index: int, meta_command: int):
        if not self.wm_enabled:
            return None, None, None
        meas_dir = str(self.measurements[index], encoding="utf-8")
        frame = int(self.sample_start[index])
        tubelet = load_wm_tubelet(meas_dir.replace("/measurements", "/rgb"), frame, self.wm_image_size)
        state = load_wm_state(meas_dir, frame)
        if self.wm_action_input == "action_anchor" and meta_command >= 0:
            actions = self.action_anchors[meta_command].copy()
        else:
            speeds, thetas = _past_speeds_and_headings(meas_dir, frame)
            actions = extrapolate_ego_motion(speeds, thetas, self.wm_rollout_steps)
        return tubelet, state, actions

    def _build_sample(
        self,
        index: int,
        prompt: str,
        image: np.ndarray,
        waypoints: np.ndarray,
        path: np.ndarray,
        target_points: np.ndarray,
        speed: float,
        measurement_path: str,
        meta_command: int,
        mode: Optional[str],
    ) -> DrivingSample:
        waypoints = np.asarray(waypoints, dtype=np.float32)
        tokens = self.action_tokenizer.match_trajectory(waypoints)
        tokens = self._snap_zero_prefix(tokens, waypoints, speed)

        mode_label = DRIVING_MODE_TO_INDEX[mode] if (mode is not None and self.use_mode_token) else -1
        mode_token = MODE_TOKENS[mode_label] if mode_label >= 0 else ""
        answer = f"{mode_token}{ACTIONS_TOKEN}{self.action_tokenizer.encode_to_string(tokens)}"
        conversation = [
            {"role": "user", "content": [{"type": "text", "text": prompt}, {"type": "image"}]},
            {"role": "assistant", "content": [{"type": "text", "text": answer}]},
        ]
        tubelet, state, actions = self._wm_inputs(index, meta_command)
        return DrivingSample(
            conversation=conversation,
            image=image,
            waypoints=waypoints,
            path=np.asarray(path, dtype=np.float32),
            target_points=target_points,
            speed=float(speed),
            action_tokens=[int(i) for i in tokens.tolist()],
            mode_label=mode_label,
            meta_command=meta_command,
            wm_tubelet=tubelet,
            wm_state=state,
            wm_actions=actions,
            measurement_path=measurement_path,
        )


class DrivingDataset(_SafeDriveSampleMixin, BaseDataset):
    """PDM-Lite expert frames with driving-mode attribution."""

    def __init__(
        self,
        codebook_path: str,
        use_mode_token: bool = True,
        world_model: Optional[Dict[str, Any]] = None,
        **base_kwargs,
    ):
        BaseDataset.__init__(self, action_dreaming=False, **base_kwargs)
        self._init_safedrive(codebook_path, use_mode_token, world_model)

    def __getitem__(self, index: int) -> DrivingSample:
        cv2.setNumThreads(0)
        measurements, current, measurement_path = self.load_measurements(index)

        # Shifted-camera augmentation (SimLingo): a laterally shifted / rotated
        # camera with the labels transformed accordingly.
        shifted = random.random() <= IMG_SHIFT_AUGMENTATION_PROB
        yaw_aug = current["augmentation_rotation"] if shifted else 0.0
        y_aug = current["augmentation_translation"] if shifted else 0.0

        traj = self.load_trajectory(measurements, current, y_aug, yaw_aug)
        target_point = self.augment_target_point(np.array(current["target_point"]), y_aug, yaw_aug)
        next_target_point = self.augment_target_point(np.array(current["target_point_next"]), y_aug, yaw_aug)
        navigation = random.choice(self.navigation_prompts(current, target_point))
        speed = current["speed"]
        prompt = f"Current speed: {round(speed, 1)} m/s. {navigation} {TASK_PROMPT}".replace("..", ".")
        image = self.load_image(index, shifted_camera=shifted)

        meta_name = COMMAND_TO_META_COMMAND.get(int(current.get("command", 0)))
        meta_command, mode = -1, None
        if meta_name is not None:
            meta_command = META_COMMAND_TO_INDEX[meta_name]
            speed_limit = float(current.get("speed_limit", 0.0))
            mode = classify_driving_mode(traj["waypoints"], meta_name, float(speed), speed_limit)

        return self._build_sample(
            index,
            prompt=prompt,
            image=image,
            waypoints=traj["waypoints"],
            path=traj["path"],
            target_points=np.array([target_point, next_target_point]),
            speed=speed,
            measurement_path=measurement_path,
            meta_command=meta_command,
            mode=mode,
        )


class ActionDreamingDataset(_SafeDriveSampleMixin, BaseDataset):
    """SimLingo action-dreaming frames: alternative (possibly unsafe) instructions."""

    def __init__(
        self,
        codebook_path: str,
        use_mode_token: bool = True,
        world_model: Optional[Dict[str, Any]] = None,
        **base_kwargs,
    ):
        BaseDataset.__init__(self, action_dreaming=True, **base_kwargs)
        self._init_safedrive(codebook_path, use_mode_token, world_model)

    def __getitem__(self, index: int) -> DrivingSample:
        cv2.setNumThreads(0)
        measurements, current, measurement_path = self.load_measurements(index)
        traj = self.load_trajectory(measurements, current)
        speed = current["speed"]
        target_point = np.array(current["target_point"])
        next_target_point = np.array(current["target_point_next"])

        with gzip.open(str(self.dreams[index], encoding="utf-8"), "rt") as f:
            alternatives = ujson.load(f)
        options = [opt for key, opts in alternatives.items() if "factor" not in key for opt in opts]
        option = random.choice(options)
        instruction = random.choice(option["dreamer_instruction"])

        navigation_prompts = self.navigation_prompts(current, target_point)
        if random.random() < 0.8:
            prompt = f"Current speed: {round(speed, 1)} m/s. {random.choice(navigation_prompts)} {instruction}"
        else:
            prompt = f"Current speed: {round(speed, 1)} m/s. {instruction}"
        prompt = prompt.replace("..", ".").replace("  ", " ").replace("!.", "!").replace("?.", "?")
        image = self.load_image(index)

        safe = option.get("safe_to_execute", True) is not False
        if safe:
            waypoints = traj["waypoints"] if option["waypoints"] == "org" else np.array(option["waypoints"])
            path = traj["path"] if option["route"] == "org" else np.array(option["route"])
        else:
            # Refuse the unsafe instruction: keep the safe expert trajectory.
            waypoints, path = traj["waypoints"], traj["path"]

        meta_command, mode = self._dream_label(option, current, waypoints, safe)
        return self._build_sample(
            index,
            prompt=prompt,
            image=image,
            waypoints=waypoints,
            path=path,
            target_points=np.array([target_point, next_target_point]),
            speed=speed,
            measurement_path=measurement_path,
            meta_command=meta_command,
            mode=mode,
        )

    @staticmethod
    def _dream_label(option: dict, current: dict, waypoints: np.ndarray, safe: bool) -> Tuple[int, Optional[str]]:
        """Meta-command and driving mode of an action-dreaming sample.

        Lane-change instructions set the meta-command to a lane change whose
        direction is the sign of the final lateral offset of the supervised
        trajectory (the expert's for a refused, unsafe instruction), as in
        the paper's training run; speed instructions keep the frame's route
        command. Unsafe instructions are always ``fallback``.
        """
        base_meta = COMMAND_TO_META_COMMAND.get(int(current.get("command", 0)))
        if base_meta is None:
            return -1, None
        category = str(option.get("mode", ""))
        meta_name = base_meta
        if category == "lane_change":
            wp = np.asarray(waypoints, dtype=np.float32).reshape(-1, 2)
            end_y = float(wp[-1, 1]) if wp.size else 0.0
            meta_name = "lane_change_right" if end_y > 0 else "lane_change_left"
        meta_command = META_COMMAND_TO_INDEX[meta_name]
        if not safe:
            return meta_command, "fallback"

        current_speed = float(current["speed"])
        wp = np.asarray(waypoints, dtype=np.float32).reshape(-1, 2)
        if category == "lane_change":
            mode = classify_driving_mode(wp, meta_name, current_speed, float(current.get("speed_limit", 0.0)))
        elif category in SPEED_DREAM_CATEGORIES:
            info = option.get("info", {}) or {}
            target_speed = float(info.get("target_speed", info.get("final_speed", current_speed)))
            mode = classify_speed_instruction(wp, current_speed, target_speed)
        else:
            mode = "cautious"
        return meta_command, mode
