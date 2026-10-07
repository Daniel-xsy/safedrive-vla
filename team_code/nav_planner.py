"""Route planner and lateral controller of the CARLA agent.

Adapted from SimLingo (https://github.com/RenzKa/simlingo, Apache-2.0),
``team_code/nav_planner.py``, which builds on carla_garage (MIT).
"""

from __future__ import annotations

import math
from collections import deque

import numpy as np


class LateralPIDController:
    """Steers towards a look-ahead point of the predicted path; the look-ahead
    distance grows with the speed."""

    def __init__(self, k_p=3.118357247806046, k_d=1.3782508892109167, k_i=0.6406067986034124,
                 speed_scale=0.9755321901954155, speed_offset=1.9152884533402488, n=6):
        self.k_p = k_p
        self.k_d = k_d
        self.k_i = k_i
        self.speed_scale = speed_scale
        self.speed_offset = speed_offset
        self.n = n
        self._window = []

    def step(self, route_np: np.ndarray, current_speed: float) -> float:
        """``route_np``: path points at 0.1 m spacing (ego frame); speed in m/s."""
        current_speed = current_speed * 3.6
        n_lookahead = int(min(np.clip(self.speed_scale * current_speed + self.speed_offset, 24, 105), route_np.shape[0] - 1))
        n_lookahead = min(n_lookahead, len(route_np) - 1)
        desired_heading_vec = route_np[n_lookahead]

        heading_error = np.arctan2(desired_heading_vec[1], desired_heading_vec[0]) % (2 * np.pi)
        heading_error = heading_error if heading_error < np.pi else heading_error - 2 * np.pi
        # The scaling is kept from the implementation the gains were tuned on.
        heading_error = heading_error * 180.0 / np.pi / 90.0

        self._window.append(heading_error)
        self._window = self._window[-self.n:]
        derivative = 0.0 if len(self._window) == 1 else self._window[-1] - self._window[-2]
        integral = np.mean(self._window)
        return np.clip(self.k_p * heading_error + self.k_d * derivative + self.k_i * integral, -1.0, 1.0).item()


class RoutePlanner:
    """Pops the route points the ego has passed and returns the remaining route."""

    def __init__(self, min_distance: float, max_distance: float, lat_ref: float = 0.0, lon_ref: float = 0.0):
        self.route = deque()
        self.route_distances = deque()
        self.lat_ref = lat_ref
        self.lon_ref = lon_ref
        self.min_distance = min_distance
        self.max_distance = max_distance

    def convert_gps_to_carla(self, gps) -> np.ndarray:
        """GNSS ``(lat, lon, z)`` -> CARLA world coordinates."""
        earth_radius_equa = 6378137.0  # CARLA leaderboard GPS simulation constant
        lat, lon, _ = gps
        scale = math.cos(self.lat_ref * math.pi / 180.0)
        my = math.log(math.tan((lat + 90) * math.pi / 360.0)) * (earth_radius_equa * scale)
        mx = (lon * (math.pi * earth_radius_equa * scale)) / 180.0
        y = scale * earth_radius_equa * math.log(math.tan((90.0 + self.lat_ref) * math.pi / 360.0)) - my
        x = mx - scale * self.lon_ref * math.pi * earth_radius_equa / 180.0
        return np.array([x, y, gps[2]])

    def set_route(self, global_plan_gps) -> None:
        self.route.clear()
        for pos, cmd in global_plan_gps:
            self.route.append((self.convert_gps_to_carla(np.array([pos["lat"], pos["lon"], pos["z"]])), cmd))
        self.route_distances.append(0.0)
        for i in range(1, len(self.route)):
            diff = self.route[i][0] - self.route[i - 1][0]
            self.route_distances.append((diff[0] ** 2 + diff[1] ** 2) ** 0.5)

    def run_step(self, gps: np.ndarray):
        if len(self.route) <= 2:
            return self.route
        to_pop = 0
        farthest_in_range = -np.inf
        cumulative_distance = 0.0
        for i in range(1, len(self.route)):
            if cumulative_distance > self.max_distance:
                break
            cumulative_distance += self.route_distances[i]
            diff = self.route[i][0] - gps
            distance = (diff[0] ** 2 + diff[1] ** 2) ** 0.5
            if farthest_in_range < distance <= self.min_distance:
                farthest_in_range = distance
                to_pop = i
        for _ in range(to_pop):
            if len(self.route) > 2:
                self.route.popleft()
                self.route_distances.popleft()
        return self.route
