"""SafeDriveVLA closed-loop CARLA agent (Bench2Drive).

Every tick the agent
  1. builds the prompt ``Current speed: v m/s. <navigation signal> Predict the actions.``
     where the navigation signal is the next two target waypoints (default) or
     the route-planner command (``SAFEDRIVE_NAV_SIGNAL=command``);
  2. dreams K latents with the frozen world model under the action anchor of
     the current meta-command, and splices the projected world tokens into the
     prompt (Sec. 4.2);
  3. greedily decodes ``<MODE><ACTIONS> a_0 ... a_9`` (Sec. 4.3);
  4. steers along the path-head prediction (lateral PID) and tracks the speed of
     the action-token trajectory (longitudinal PID).

The action anchor is indexed with the route-planner command of the previous
target point, the same command that labels the training frames: the PDM-Lite
expert records ``command`` one target point late.

Sensor setup, state estimation and controllers are adapted from the SimLingo
agent (https://github.com/RenzKa/simlingo, Apache-2.0).
"""

from __future__ import annotations

import json
import math
import os
import time
from collections import deque
from pathlib import Path

import carla
import cv2
import hydra
import numpy as np
import torch
from filterpy.kalman import MerweScaledSigmaPoints
from filterpy.kalman import UnscentedKalmanFilter as UKF
from leaderboard.autoagents import autonomous_agent
from omegaconf import OmegaConf
from PIL import Image, ImageDraw, ImageFont
from scipy.interpolate import PchipInterpolator
from scipy.optimize import fsolve

from safedrive_vla.action_tokenizer import ActionTokenizer
from safedrive_vla.constants import (
    COMMAND_TO_META_COMMAND,
    META_COMMAND_TO_INDEX,
    TASK_PROMPT,
    world_block_text,
)
from safedrive_vla.data.dataset import extrapolate_ego_motion, load_action_anchors
from safedrive_vla.data.simlingo_base import COMMAND_TEXT
from safedrive_vla.models.backbone import load_processor
from team_code import simlingo_utils as utils
from team_code.config import AgentConfig
from team_code.nav_planner import LateralPIDController, RoutePlanner

REPO_ROOT = Path(__file__).resolve().parents[1]
SAVE_VIZ = os.environ.get("SAFEDRIVE_SAVE_VIZ", "0") == "1"

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True


def get_entry_point():
    return "SafeDriveAgent"


def _wrap_pi(angle: float) -> float:
    return float((angle + math.pi) % (2.0 * math.pi) - math.pi)


def _repo_path(path):
    return path if path is None or os.path.isabs(path) else str(REPO_ROOT / path)


def command_prompt(command: int, next_command: int, dist_to_command: int) -> str:
    """Route-planner command rendered as in training (``navigation_prompts``)."""
    text, next_text = COMMAND_TEXT[int(command)], COMMAND_TEXT[int(next_command)]
    next_text = f" then {next_text}" if text != next_text else ""
    if int(command) == 4:
        return f"Command: {text}{next_text}."
    return f"Command: {text} in {dist_to_command} meter{next_text}."


def load_policy(checkpoint: str, device: torch.device):
    """Load a checkpoint (DeepSpeed directory or file) and its config: the
    ``config.yaml`` next to it (released models) or the ``.hydra/config.yaml``
    of its training run directory."""
    checkpoint = Path(checkpoint)
    config = checkpoint.parent / "config.yaml"
    if not config.exists():
        config = checkpoint.parent.parent / ".hydra" / "config.yaml"
    cfg = OmegaConf.load(config)
    if cfg.model.get("world_model") is not None:
        for key in ("encoder_ckpt", "predictor_ckpt", "action_anchor_path"):
            cfg.model.world_model[key] = _repo_path(cfg.model.world_model[key])
    cfg.model.action_token.codebook_path = _repo_path(cfg.model.action_token.codebook_path)

    processor, _ = load_processor(cfg.model.variant, int(cfg.model.action_token.num_tokens))
    default_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    model = hydra.utils.instantiate(cfg.model, processor=processor, gradient_checkpointing=False, _recursive_=False)
    torch.set_default_dtype(default_dtype)

    if checkpoint.is_dir():  # DeepSpeed: only the trainable parameters are stored
        tag = (checkpoint / "latest").read_text().strip() if (checkpoint / "latest").exists() else "checkpoint"
        state = torch.load(checkpoint / tag / "mp_rank_00_model_states.pt", map_location="cpu", weights_only=False)["module"]
    else:
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        state = state.get("state_dict", state)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if unexpected:
        raise RuntimeError(f"Unexpected keys in {checkpoint}: {unexpected[:5]}")
    trainable_missing = [k for k in missing if k.startswith(("target_point_encoder", "path_head", "world_state_projector"))]
    if trainable_missing:
        raise RuntimeError(f"Checkpoint {checkpoint} lacks trained weights: {trainable_missing[:5]}")
    return model.to(device).eval(), processor, cfg


class SafeDriveAgent(autonomous_agent.AutonomousAgent):
    def setup(self, path_to_conf_file):
        torch.cuda.empty_cache()
        self.track = autonomous_agent.Track.SENSORS
        checkpoint, _, save_name = path_to_conf_file.partition("+")
        self.config = AgentConfig()
        self.device = torch.device("cuda")
        self.step = -1
        self.initialized = False
        self.nav_signal = os.environ.get("SAFEDRIVE_NAV_SIGNAL", "target_point")
        if self.nav_signal not in ("target_point", "command"):
            raise ValueError(f"SAFEDRIVE_NAV_SIGNAL must be target_point or command, got {self.nav_signal}")
        # Free-form language instruction; set by the language-benchmark agents.
        self.instruction = None

        self.model, self.processor, cfg = load_policy(checkpoint, self.device)
        self.tokenizer = self.processor.tokenizer
        self.action_tokenizer = ActionTokenizer.from_file(cfg.model.action_token.codebook_path)
        self.horizon = int(cfg.model.action_token.horizon)
        self.frame_subsample = int(round(self.config.carla_fps / float(cfg.model.action_token.hz)))  # 4 Hz

        # ---- world-model dreaming ------------------------------------
        wm = cfg.model.get("world_model")
        self.wm_enabled = wm is not None
        self.action_anchors = None
        if self.wm_enabled:
            self.wm_size = int(wm.img_size)
            self.wm_steps = int(wm.rollout_steps)
            self.wm_action_input = str(wm.action_input)
            if self.wm_action_input == "action_anchor":
                self.action_anchors = load_action_anchors(wm.action_anchor_path, self.wm_steps)
        # 2.5 s of 4 Hz speed / heading history and the last two 4 Hz frames.
        self.wm_speeds = deque(maxlen=11)
        self.wm_thetas = deque(maxlen=11)
        self.wm_frames = deque(maxlen=2)

        # ---- controllers and state estimation ------------------------
        c = self.config
        self.speed_controller = utils.PIDController(k_p=c.speed_kp, k_i=c.speed_ki, k_d=c.speed_kd, n=c.speed_n)
        self.turn_controller = LateralPIDController()
        self.points = MerweScaledSigmaPoints(n=4, alpha=0.00001, beta=2, kappa=0, subtract=utils.residual_state_x)
        self.ukf = UKF(dim_x=4, dim_z=4, fx=utils.bicycle_model_forward, hx=utils.measurement_function_hx,
                       dt=c.carla_frame_rate, points=self.points, x_mean_fn=utils.state_mean,
                       z_mean_fn=utils.measurement_mean, residual_x=utils.residual_state_x,
                       residual_z=utils.residual_measurement_h)
        self.ukf.P = np.diag([0.5, 0.5, 0.000001, 0.000001])
        self.ukf.R = np.diag([0.5, 0.5, 0.000000000000001, 0.000000000000001])
        self.ukf.Q = np.diag([0.0001, 0.0001, 0.001, 0.001])
        self.filter_initialized = False
        self.stuck_detector = 0
        self.force_move = 0

        # Route-planner commands of the current and the previous target point.
        self.commands = deque([4, 4], maxlen=2)
        self.next_commands = deque([4, 4], maxlen=2)
        self.target_point_prev = [1e5, 1e5, 1e5]
        self.control = carla.VehicleControl(steer=0.0, throttle=0.0, brake=1.0)

        # ---- outputs: per-tick ego state (metric_info.json) and optional images
        route_name = Path(os.environ.get("ROUTES", "route")).stem
        save_root = Path(os.environ.get("SAVE_PATH", "eval_results"))
        self.save_dir = save_root / f"{route_name}_{time.strftime('%Y_%m_%d_%H_%M_%S')}"
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.metric_info = {}
        if SAVE_VIZ:
            (self.save_dir / "images").mkdir(exist_ok=True)

    def _init(self):
        # GPS reference of the CARLA map, from the first route point.
        locx, locy = self._global_plan_world_coord[0][0].location.x, self._global_plan_world_coord[0][0].location.y
        lon, lat = self._global_plan[0][0]["lon"], self._global_plan[0][0]["lat"]
        earth_radius_equa = 6378137.0

        def equations(variables):
            x, y = variables
            eq1 = lon * math.cos(x * math.pi / 180.0) - (locx * x * 180.0) / (math.pi * earth_radius_equa) \
                - math.cos(x * math.pi / 180.0) * y
            eq2 = math.log(math.tan((lat + 90.0) * math.pi / 360.0)) * earth_radius_equa * math.cos(x * math.pi / 180.0) \
                + locy - math.cos(x * math.pi / 180.0) * earth_radius_equa * math.log(math.tan((90.0 + x) * math.pi / 360.0))
            return [eq1, eq2]

        lat_ref, lon_ref = fsolve(equations, [0.0, 0.0])
        self._route_planner = RoutePlanner(self.config.route_planner_min_distance,
                                           self.config.route_planner_max_distance, lat_ref, lon_ref)
        self._route_planner.set_route(self._global_plan)
        self.initialized = True

    def sensors(self):
        c = self.config
        return [
            {"type": "sensor.camera.rgb", "x": c.camera_pos[0], "y": c.camera_pos[1], "z": c.camera_pos[2],
             "roll": c.camera_rot[0], "pitch": c.camera_rot[1], "yaw": c.camera_rot[2],
             "width": c.camera_width, "height": c.camera_height, "fov": c.camera_fov, "id": "rgb_0"},
            {"type": "sensor.other.imu", "x": 0.0, "y": 0.0, "z": 0.0, "roll": 0.0, "pitch": 0.0, "yaw": 0.0,
             "sensor_tick": c.carla_frame_rate, "id": "imu"},
            {"type": "sensor.other.gnss", "x": 0.0, "y": 0.0, "z": 0.0, "roll": 0.0, "pitch": 0.0, "yaw": 0.0,
             "sensor_tick": 0.01, "id": "gps"},
            {"type": "sensor.speedometer", "reading_frequency": c.carla_fps, "id": "speed"},
        ]

    # ------------------------------------------------------------------
    # hooks of the language-benchmark agents
    # ------------------------------------------------------------------
    def on_route_command(self, current_command: int) -> None:
        """Called every tick with the route planner's current command."""

    def anchor_meta_command(self) -> int:
        """Meta-command that indexes the action anchor (-1: none)."""
        meta = COMMAND_TO_META_COMMAND.get(int(self.commands[-2]))
        return META_COMMAND_TO_INDEX[meta] if meta is not None else -1

    # ------------------------------------------------------------------
    @torch.inference_mode()
    def tick(self, input_data):
        camera = input_data["rgb_0"][1][:, :, :3]
        self.camera_for_viz = camera.copy()
        # The model saw JPEG-compressed frames during training.
        _, jpg = cv2.imencode(".jpg", camera)
        frame = cv2.cvtColor(cv2.imdecode(jpg, cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB)
        vla_image = frame[: int(frame.shape[0] - (frame.shape[0] * 4.8) // 16)]  # remove the bonnet

        gps = self._route_planner.convert_gps_to_carla(input_data["gps"][1])
        compass = utils.preprocess_compass(input_data["imu"][1][-1])
        speed = input_data["speed"][1]["speed"]
        if not self.filter_initialized:
            self.ukf.x = np.array([gps[0], gps[1], utils.normalize_angle(compass), speed])
            self.filter_initialized = True
        self.ukf.predict(steer=self.control.steer, throttle=self.control.throttle, brake=self.control.brake)
        self.ukf.update(np.array([gps[0], gps[1], utils.normalize_angle(compass), speed]))
        ego_xy = self.ukf.x[0:2]

        route = self._route_planner.run_step(np.append(ego_xy, gps[2]))
        target_point, far_command = route[min(1, len(route) - 1)]
        next_target_point, next_far_command = route[min(2, len(route) - 1)]
        self.on_route_command(int(getattr(route[0][1], "value", route[0][1])))
        if (target_point != self.target_point_prev).all():
            self.target_point_prev = target_point
            self.commands.append(far_command.value)
            self.next_commands.append(next_far_command.value)
        ego_target_point = utils.inverse_conversion_2d(target_point[:2], ego_xy, compass)
        ego_next_target_point = utils.inverse_conversion_2d(next_target_point[:2], ego_xy, compass)
        target_points = np.array([ego_target_point, ego_next_target_point], dtype=np.float32)

        if self.instruction is not None:
            navigation = f"Command: {self.instruction}."
        elif self.nav_signal == "command":
            navigation = command_prompt(self.commands[-2], self.next_commands[-2], int(np.linalg.norm(ego_target_point)))
        else:
            navigation = "Target waypoint: <TARGET_POINT><TARGET_POINT>."
        prompt = f"Current speed: {round(float(speed), 1)} m/s. {navigation} {TASK_PROMPT}"

        wm_inputs = None
        if self.wm_enabled:
            wm_inputs = self._world_model_inputs(frame, speed, compass)
            prompt += world_block_text(self.wm_steps)
        self.prompt = prompt

        conversation = [{"role": "user", "content": [{"type": "text", "text": prompt}, {"type": "image"}]}]
        text = self.processor.apply_chat_template([conversation], add_generation_prompt=True, tokenize=False)
        inputs = self.processor(text=text, images=[Image.fromarray(vla_image)], padding=True, return_tensors="pt")
        return {
            "inputs": {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()},
            "speed": speed,
            "target_points": target_points,
            "wm_inputs": wm_inputs,
        }

    def _world_model_inputs(self, frame, speed: float, compass: float):
        """Two-frame tubelet, ego state and action sequence of the frozen world
        model, sampled at the 4 Hz cadence of the training data."""
        resized = cv2.resize(frame, (self.wm_size, self.wm_size), interpolation=cv2.INTER_LINEAR)
        if not self.wm_speeds:
            self.wm_speeds.extend([float(speed)] * self.wm_speeds.maxlen)
            self.wm_thetas.extend([float(compass)] * self.wm_thetas.maxlen)
            self.wm_frames.extend([resized, resized])
        elif self.step % self.frame_subsample == 0:
            self.wm_speeds.append(float(speed))
            self.wm_thetas.append(float(compass))
            self.wm_frames.append(resized)

        tubelet = np.transpose(np.stack(list(self.wm_frames)), (0, 3, 1, 2)).astype(np.uint8)
        dt = 1.0 / 4.0
        speed_cur, speed_prev = self.wm_speeds[-1], self.wm_speeds[-2]
        yaw_rate = _wrap_pi(self.wm_thetas[-1] - self.wm_thetas[-2]) / dt
        state = np.array([speed_cur, (speed_cur - speed_prev) / dt, speed_cur * yaw_rate, yaw_rate], dtype=np.float32)

        meta_command = self.anchor_meta_command() if self.wm_action_input == "action_anchor" else -1
        if meta_command >= 0:
            actions = self.action_anchors[meta_command].copy()
        else:
            actions = extrapolate_ego_motion(list(self.wm_speeds), list(self.wm_thetas), self.wm_steps)
        return tubelet, state, actions

    # ------------------------------------------------------------------
    @torch.no_grad()
    def run_step(self, input_data, timestamp, sensors=None):
        self.step += 1
        if not self.initialized:
            self._init()
            self.control = carla.VehicleControl(steer=0.0, throttle=0.0, brake=1.0)
            self.tick(input_data)
            return self.control

        tick = self.tick(input_data)
        if self.step < self.config.initial_frames_delay:
            self.control = carla.VehicleControl(0.0, 0.0, 1.0)
            self._log_metric_info()
            return self.control

        inputs = tick["inputs"]
        target_points = torch.from_numpy(tick["target_points"]).to(self.device).unsqueeze(0)
        wm = [None, None, None]
        if tick["wm_inputs"] is not None:
            wm = [torch.from_numpy(x).to(self.device).unsqueeze(0) for x in tick["wm_inputs"]]
        self.model.set_conditioning(target_points=target_points, wm_tubelet=wm[0], wm_state=wm[1], wm_actions=wm[2])
        try:
            generated = self.model.vlm.generate(**inputs, do_sample=False, max_new_tokens=self.config.max_new_tokens,
                                                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id)
            path = self._predict_path(generated, inputs)
        finally:
            self.model.clear_conditioning()

        prompt_len = inputs["input_ids"].shape[1]
        indices = [t - self.model.action_start_id for t in generated[0, prompt_len:].tolist()
                   if self.model.action_start_id <= t < self.model.action_end_id][: self.horizon]
        indices += [0] * (self.horizon - len(indices))
        trajectory = self.action_tokenizer.rollout_xy(indices)
        steer_points = path if path is not None else trajectory
        velocity = torch.tensor([tick["speed"]], dtype=torch.float32)
        steer, throttle, brake = self.control_pid(steer_points, velocity[0].numpy(), trajectory)

        if SAVE_VIZ and self.step % 5 == 0:
            answer = self.tokenizer.decode(generated[0, prompt_len:], skip_special_tokens=False)
            self._save_image(tick["target_points"], trajectory, path, answer)

        if velocity < 0.1:
            self.stuck_detector += 1
        else:
            self.stuck_detector = 0
        if self.stuck_detector > self.config.stuck_threshold:
            self.force_move = self.config.creep_duration
        if self.force_move > 0:
            throttle = max(self.config.creep_throttle, throttle)
            brake = False
            self.force_move -= 1

        self.control = carla.VehicleControl(steer=float(steer), throttle=float(throttle), brake=float(brake))
        self._log_metric_info()
        return self.control

    def _predict_path(self, generated: torch.Tensor, inputs: dict):
        """Path head on the hidden state of ``<ACTIONS>`` (teacher-forced pass
        over prompt + generated answer); ``None`` if no ``<ACTIONS>`` was emitted."""
        if not bool((generated == self.model.actions_token_id).any()):
            return None
        _, hidden = self.model.forward_vlm(generated, torch.ones_like(generated), inputs["pixel_values"])
        path, _ = self.model.predict_path(generated, hidden.to(self.model.path_head.mlp[0].weight.dtype))
        return path[0].float().cpu().numpy()

    def control_pid(self, path: np.ndarray, speed: np.ndarray, trajectory: np.ndarray):
        """Lateral PID along ``path``; longitudinal PID towards the speed of the
        action-token ``trajectory`` between 0.5 s and 1.0 s."""
        half, full = 1, 3  # waypoint indices of 0.5 s and 1.0 s at 4 Hz
        desired_speed = np.linalg.norm(trajectory[half] - trajectory[full]) * 2.0
        brake = desired_speed < self.config.brake_speed or (speed / max(float(desired_speed), 1e-4)) > self.config.brake_ratio
        delta = np.clip(desired_speed - speed, 0.0, self.config.clip_delta)
        throttle = np.clip(self.speed_controller.step(delta), 0.0, self.config.clip_throttle)
        throttle = throttle if not brake else 0.0
        steer = self.turn_controller.step(self.interpolate_waypoints(path), speed)
        return round(float(np.clip(steer, -1.0, 1.0)), 3), throttle, brake

    @staticmethod
    def interpolate_waypoints(waypoints: np.ndarray) -> np.ndarray:
        """Densify a path (starting at the ego origin) to 0.1 m spacing."""
        waypoints = np.concatenate((np.zeros_like(waypoints[:1]), waypoints))
        shift = np.roll(waypoints, 1, axis=0)
        shift[0] = shift[1]
        dists = np.cumsum(np.linalg.norm(waypoints - shift, axis=1)) + np.arange(len(waypoints)) * 1e-4
        points = PchipInterpolator(dists, waypoints, axis=0)(np.arange(0.1, dists[-1], 0.1))
        return points if points.shape[0] > 0 else waypoints[None, -1]

    def _log_metric_info(self) -> None:
        self.metric_info[self.step] = self.get_metric_info()
        # write-then-rename, so a killed process never leaves a truncated file
        tmp = self.save_dir / "metric_info.json.tmp"
        with open(tmp, "w") as f:
            json.dump(self.metric_info, f, indent=4)
        os.replace(tmp, self.save_dir / "metric_info.json")

    def _save_image(self, target_points, trajectory, path, answer: str) -> None:
        image = Image.fromarray(cv2.cvtColor(self.camera_for_viz, cv2.COLOR_BGR2RGB))
        w, h = image.size
        k = utils.get_camera_intrinsics(w, h, self.config.camera_fov)
        draw = ImageDraw.Draw(image)
        for points, color, r in ((target_points, (0, 0, 255), 4), (trajectory, (0, 255, 0), 3), (path, (255, 0, 0), 3)):
            if points is None:
                continue
            for p in utils.project_points(points, k):
                draw.ellipse((p[0] - r, p[1] - r, p[0] + r, p[1] + r), fill=color)
        canvas = Image.new("RGB", (w, h + 120))
        canvas.paste(image, (0, 0))
        draw = ImageDraw.Draw(canvas)
        try:
            font = ImageFont.truetype("DejaVuSans.ttf", 16)
        except OSError:
            font = ImageFont.load_default()
        draw.text((10, h + 10), f"Prompt: {self.prompt.split(' Lookahead')[0]}", font=font, fill=(255, 255, 255))
        draw.text((10, h + 40), f"Answer: {answer[:160]}", font=font, fill=(255, 255, 255))
        canvas.save(self.save_dir / "images" / f"{self.step:05d}.png")

    def destroy(self, results=None):
        if hasattr(self, "model"):
            del self.model
        torch.cuda.empty_cache()
