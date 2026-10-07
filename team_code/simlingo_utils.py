"""Controller, state-estimation and visualization helpers of the CARLA agent.

Adapted from SimLingo (https://github.com/RenzKa/simlingo, Apache-2.0),
``team_code/transfuser_utils.py`` and ``team_code/simlingo_utils.py``, which
build on carla_garage (MIT).
"""

from __future__ import annotations

import math
from collections import deque

import cv2
import numpy as np


def normalize_angle(x: float) -> float:
    x = x % (2 * np.pi)  # [0, 2 pi)
    if x > np.pi:  # [-pi, pi)
        x -= 2 * np.pi
    return x


def preprocess_compass(compass: float) -> float:
    """IMU compass (rad) -> yaw in the CARLA coordinate system."""
    if math.isnan(compass):  # simulation bug
        compass = 0.0
    return normalize_angle(compass - np.deg2rad(90.0))


def inverse_conversion_2d(point, translation, yaw):
    """World point -> frame with origin ``translation`` and heading ``yaw``."""
    rotation = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
    return rotation.T @ (point - translation)


class PIDController:
    def __init__(self, k_p=1.0, k_i=0.0, k_d=0.0, n=20):
        self.k_p = k_p
        self.k_i = k_i
        self.k_d = k_d
        self.window = deque([0 for _ in range(n)], maxlen=n)

    def step(self, error):
        self.window.append(error)
        if len(self.window) >= 2:
            integral = np.mean(self.window)
            derivative = self.window[-1] - self.window[-2]
        else:
            integral = 0.0
            derivative = 0.0
        return self.k_p * error + self.k_i * integral + self.k_d * derivative


# ---------------------------------------------------------------------------
# Unscented Kalman filter on GPS / compass / speed
# ---------------------------------------------------------------------------

def bicycle_model_forward(x, dt, steer, throttle, brake):
    """Kinematic bicycle model of the UKF motion update."""
    front_wb = -0.090769015
    rear_wb = 1.4178275
    steer_gain = 0.36848336
    brake_accel = -4.952399
    throt_accel = 0.5633837

    locs_0, locs_1, yaw, speed = x[0], x[1], x[2], x[3]
    accel = brake_accel if brake else throt_accel * throttle
    wheel = steer_gain * steer
    beta = math.atan(rear_wb / (front_wb + rear_wb) * math.tan(wheel))
    next_locs_0 = locs_0.item() + speed * math.cos(yaw + beta) * dt
    next_locs_1 = locs_1.item() + speed * math.sin(yaw + beta) * dt
    next_yaws = yaw + speed / rear_wb * math.sin(beta) * dt
    next_speed = speed + accel * dt
    next_speed = next_speed * (next_speed > 0.0)
    return np.array([next_locs_0, next_locs_1, next_yaws, next_speed])


def measurement_function_hx(vehicle_state):
    return vehicle_state


def state_mean(state, wm):
    """Average sigma points, handling the heading wrap-around."""
    x = np.zeros(4)
    sum_sin = np.sum(np.dot(np.sin(state[:, 2]), wm))
    sum_cos = np.sum(np.dot(np.cos(state[:, 2]), wm))
    x[0] = np.sum(np.dot(state[:, 0], wm))
    x[1] = np.sum(np.dot(state[:, 1], wm))
    x[2] = math.atan2(sum_sin, sum_cos)
    x[3] = np.sum(np.dot(state[:, 3], wm))
    return x


def measurement_mean(state, wm):
    return state_mean(state, wm)


def residual_state_x(a, b):
    y = a - b
    y[2] = normalize_angle(y[2])
    return y


def residual_measurement_h(a, b):
    y = a - b
    y[2] = normalize_angle(y[2])
    return y


# ---------------------------------------------------------------------------
# camera projection (visualization)
# ---------------------------------------------------------------------------

def get_camera_intrinsics(w, h, fov) -> np.ndarray:
    focal = w / (2.0 * np.tan(fov * np.pi / 360.0))
    k = np.identity(3)
    k[0, 0] = k[1, 1] = focal
    k[0, 2] = w / 2.0
    k[1, 2] = h / 2.0
    return k.astype(np.float32)


def project_points(points_2d, k):
    """Ego-frame ground points ``(x forward, y right)`` -> image pixels."""
    rvec = np.zeros((3, 1), np.float32)
    tvec = np.array([[0.0, 2.0, 1.5]], np.float32)
    out = []
    for point in points_2d:
        pos_3d = np.array([point[1], 0, point[0] + tvec[0][2]])
        pixels, _ = cv2.projectPoints(pos_3d, rvec=rvec, tvec=tvec, cameraMatrix=k, distCoeffs=np.zeros((5, 1), np.float32))
        out.append(pixels[0][0])
    return out
