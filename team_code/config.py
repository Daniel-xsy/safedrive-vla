"""CARLA agent settings: sensors and the PID controllers, adapted from the
SimLingo agent configuration (https://github.com/RenzKa/simlingo, Apache-2.0)."""


class AgentConfig:
    def __init__(self):
        # generation budget for <MODE><ACTIONS> + 10 action tokens
        self.max_new_tokens = 128

        # CARLA timing
        self.carla_frame_rate = 1.0 / 20.0
        self.carla_fps = 20
        self.initial_frames_delay = 2.0 / self.carla_frame_rate  # brake for the first 2 s
        self.stuck_threshold = 800
        self.creep_duration = 15
        self.creep_throttle = 0.4

        # longitudinal controller
        self.brake_speed = 0.4
        self.brake_ratio = 1.1
        self.clip_delta = 1.0
        self.clip_throttle = 1.0
        self.speed_kp = 1.75
        self.speed_ki = 1.0
        self.speed_kd = 2.0
        self.speed_n = 20

        # front camera
        self.camera_pos = [-1.5, 0.0, 2.0]
        self.camera_rot = [0.0, 0.0, 0.0]
        self.camera_width = 1024
        self.camera_height = 512
        self.camera_fov = 110

        # route planner
        self.route_planner_min_distance = 7.5
        self.route_planner_max_distance = 50.0
