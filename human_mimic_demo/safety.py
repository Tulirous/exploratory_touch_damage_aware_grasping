from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np


class DemoState(str, Enum):
    WAITING_CALIBRATION = "waiting_calibration"
    DISARMED = "disarmed"
    ARMED = "armed"
    TRACKING_LOST = "tracking_lost"
    ESTOP = "estop"


@dataclass
class RelativeArmMapper:
    rotation: np.ndarray
    scale: np.ndarray
    max_delta_m: np.ndarray
    workspace_min_m: np.ndarray
    workspace_max_m: np.ndarray
    max_step_m: float
    human_origin: np.ndarray | None = None
    robot_origin_tcp: np.ndarray | None = None
    previous_target: np.ndarray | None = None

    @classmethod
    def from_config(cls, config: dict) -> "RelativeArmMapper":
        rotation = np.asarray(config["camera_to_robot_rotation"], dtype=np.float64)
        if rotation.shape != (3, 3):
            raise ValueError("camera_to_robot_rotation must be 3x3")
        if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-3):
            raise ValueError("camera_to_robot_rotation must be orthonormal")
        return cls(
            rotation=rotation,
            scale=np.asarray(config.get("translation_scale", [0.7] * 3), dtype=np.float64),
            max_delta_m=np.asarray(config.get("max_delta_m", [0.20, 0.20, 0.15]), dtype=np.float64),
            workspace_min_m=np.asarray(config["workspace_min_m"], dtype=np.float64),
            workspace_max_m=np.asarray(config["workspace_max_m"], dtype=np.float64),
            max_step_m=float(config.get("max_tcp_step_m", 0.01)),
        )

    def calibrate(self, human_wrist: np.ndarray, robot_tcp: np.ndarray) -> None:
        self.human_origin = np.asarray(human_wrist, dtype=np.float64).copy()
        self.robot_origin_tcp = np.asarray(robot_tcp, dtype=np.float64).copy()
        self.previous_target = self.robot_origin_tcp.copy()

    @property
    def calibrated(self) -> bool:
        return self.human_origin is not None and self.robot_origin_tcp is not None

    def target(self, human_wrist: np.ndarray) -> np.ndarray:
        if not self.calibrated:
            raise RuntimeError("Arm mapper has not been calibrated")
        human_delta = np.asarray(human_wrist) - self.human_origin
        robot_delta = self.rotation @ (human_delta * self.scale)
        robot_delta = np.clip(robot_delta, -self.max_delta_m, self.max_delta_m)
        target = self.robot_origin_tcp.copy()
        target[:3] = np.clip(
            self.robot_origin_tcp[:3] + robot_delta,
            self.workspace_min_m,
            self.workspace_max_m,
        )
        if self.previous_target is not None:
            step = target[:3] - self.previous_target[:3]
            norm = float(np.linalg.norm(step))
            if norm > self.max_step_m:
                target[:3] = self.previous_target[:3] + step * (self.max_step_m / norm)
        self.previous_target = target.copy()
        return target


class ForceGuard:
    def __init__(self, config: dict, open_position: np.ndarray) -> None:
        self.enabled = bool(config.get("enabled", False))
        self.normal_soft_limit = float(config.get("normal_soft_limit", 180.0))
        self.retreat_step = float(config.get("retreat_command_step", 5.0))
        self.open_position = open_position.astype(np.float64)

    def apply(self, command: np.ndarray, force: object) -> tuple[np.ndarray, bool]:
        if not self.enabled or force is None:
            return command, False
        try:
            normal = np.asarray(force[0], dtype=np.float64)
        except (TypeError, ValueError, IndexError):
            return command, False
        if normal.size == 0 or np.nanmax(normal) < self.normal_soft_limit:
            return command, False
        direction = np.sign(self.open_position - command)
        relaxed = command.astype(np.float64) + direction * self.retreat_step
        return np.rint(relaxed).astype(np.int64), True
