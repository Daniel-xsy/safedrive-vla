"""Driving-mode attribution (Sec. 4.1, App. C.2).

Every expert frame is relabeled with a driving mode, ``strict``, ``cautious``
or ``fallback``, from the 2.5 s of future waypoints the expert actually drove
(10 waypoints at 4 Hz, ego frame: +x forward, +y right), the current speed and
the road speed limit. The rule depends on the meta-command of the frame.
"""

from __future__ import annotations

import numpy as np

HORIZON_S = 2.5
WAYPOINT_DT = 0.25

# Turn and lane-change progress thresholds.
PROGRESS_STRICT = 0.7
PROGRESS_FALLBACK = 0.1
# Comfort-bounded acceleration of the kinematic-maximum arc used for turns.
MAX_ACCEL = 2.0
# Lateral offset (m) that counts as having entered the adjacent lane.
LANE_CHANGE_LATERAL_M = 0.3
# Cruise (follow_road / go_straight_intersection) thresholds.
CRUISE_SPEED = 6.0       # m/s, at or above: cruising speed
STOPPED_SPEED = 1.0      # m/s, below: stopped
ACCEL_STRICT = -0.5      # m/s^2, at or above: holding speed or accelerating
ACCEL_FALLBACK = -2.5    # m/s^2, below: hard braking
LATERAL_DEPARTURE_M = 1.5

TURN_META_COMMANDS = ("turn_left", "turn_right")
LANE_CHANGE_META_COMMANDS = ("lane_change_left", "lane_change_right")


def _segment_speeds(waypoints: np.ndarray) -> np.ndarray:
    pts = np.vstack([np.zeros((1, 2), dtype=np.float32), waypoints])
    return np.linalg.norm(np.diff(pts, axis=0), axis=1) / WAYPOINT_DT


def classify_cruise(waypoints: np.ndarray, current_speed: float) -> str:
    """follow_road / go_straight_intersection.

    strict:   cruising, holding speed or accelerating, no lateral departure.
    fallback: stopped and staying stopped, or hard braking without lateral motion.
    cautious: everything else (slow cruise, mild braking, lateral departure).
    """
    wp = np.asarray(waypoints).reshape(-1, 2)
    if wp.shape[0] < 2:
        return "fallback"
    speeds = _segment_speeds(wp)
    avg_accel = (float(speeds[-2:].mean()) - float(current_speed)) / HORIZON_S
    lateral_departure = float(np.abs(wp[:, 1]).max()) > LATERAL_DEPARTURE_M

    if current_speed >= CRUISE_SPEED and avg_accel >= ACCEL_STRICT and not lateral_departure:
        return "strict"
    stays_stopped = current_speed < STOPPED_SPEED and float(speeds.mean()) < STOPPED_SPEED
    if stays_stopped or (avg_accel < ACCEL_FALLBACK and not lateral_departure):
        return "fallback"
    return "cautious"


def _progress_to_mode(progress: float) -> str:
    if progress >= PROGRESS_STRICT:
        return "strict"
    if progress < PROGRESS_FALLBACK:
        return "fallback"
    return "cautious"


def classify_turn(waypoints: np.ndarray, current_speed: float, speed_limit: float) -> str:
    """turn_left / turn_right: arc length over the kinematic-maximum distance."""
    pts = np.vstack([np.zeros((1, 2), dtype=np.float32), waypoints])
    arc = float(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum()) if waypoints.shape[0] else 0.0
    max_arc = current_speed * HORIZON_S + 0.5 * MAX_ACCEL * HORIZON_S ** 2
    if speed_limit > 0:
        max_arc = min(max_arc, speed_limit * HORIZON_S)
    return _progress_to_mode(float(np.clip(arc / max(1e-3, max_arc), 0.0, 1.0)))


def classify_lane_change(waypoints: np.ndarray, current_speed: float, side: str) -> str:
    """lane_change_left / lane_change_right: how early the trajectory crosses
    ``LANE_CHANGE_LATERAL_M`` towards the commanded side."""
    sign = -1.0 if side == "left" else 1.0
    commit_m = None
    for k in range(waypoints.shape[0]):
        if sign * float(waypoints[k, 1]) > LANE_CHANGE_LATERAL_M:
            commit_m = max(0.0, float(waypoints[k, 0]))
            break
    if commit_m is None:
        return "fallback"
    max_commit_m = max(5.0, float(current_speed) * 2.0 - 1.0)
    return _progress_to_mode(float(np.clip(1.0 - commit_m / max(max_commit_m, 1e-3), 0.0, 1.0)))


def classify_driving_mode(
    waypoints: np.ndarray, meta_command: str, current_speed: float, speed_limit: float
) -> str:
    """Driving mode of an expert (or safe action-dreaming) trajectory."""
    wp = np.asarray(waypoints, dtype=np.float32).reshape(-1, 2)
    if meta_command in TURN_META_COMMANDS:
        return classify_turn(wp, float(current_speed), float(speed_limit))
    if meta_command in LANE_CHANGE_META_COMMANDS:
        return classify_lane_change(wp, float(current_speed), "left" if meta_command.endswith("left") else "right")
    return classify_cruise(wp, float(current_speed))


def classify_speed_instruction(waypoints: np.ndarray, current_speed: float, target_speed: float) -> str:
    """Driving mode of a trajectory that follows a speed instruction, measured
    as the fraction of the requested speed change that was achieved."""
    wp = np.asarray(waypoints).reshape(-1, 2)
    if wp.shape[0] < 2:
        return "fallback"
    final_speed = float(_segment_speeds(wp)[-2:].mean())
    intended = float(target_speed) - float(current_speed)
    achieved = final_speed - float(current_speed)
    if abs(intended) < 0.5:
        return "strict" if abs(achieved) < 0.5 else "cautious"
    progress = achieved / intended
    if progress >= PROGRESS_STRICT:
        return "strict"
    if progress >= PROGRESS_FALLBACK:
        return "cautious"
    return "fallback"
