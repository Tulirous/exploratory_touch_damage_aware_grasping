from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class ArmController(ABC):
    @abstractmethod
    def current_tcp(self) -> np.ndarray:
        pass

    @abstractmethod
    def command_tcp(self, pose: np.ndarray) -> None:
        pass

    def hold(self) -> None:
        pass

    def close(self) -> None:
        pass


class DryRunArm(ArmController):
    def __init__(self, initial_tcp: list[float] | None = None) -> None:
        self.pose = np.asarray(initial_tcp or [0.40, -0.20, 0.35, 0.0, 3.14, 0.0], dtype=np.float64)

    def current_tcp(self) -> np.ndarray:
        return self.pose.copy()

    def command_tcp(self, pose: np.ndarray) -> None:
        self.pose = np.asarray(pose, dtype=np.float64).copy()


class URRTDEArm(ArmController):
    """Translation-only UR5 servo; TCP orientation is fixed at calibration."""

    def __init__(self, config: dict) -> None:
        try:
            from rtde_control import RTDEControlInterface
            from rtde_receive import RTDEReceiveInterface
        except ImportError as exc:
            raise RuntimeError("Hardware mode requires the ur_rtde Python package") from exc
        host = str(config["host"])
        self.control = RTDEControlInterface(host)
        self.receive = RTDEReceiveInterface(host)
        self.speed = float(config.get("servo_speed", 0.15))
        self.acceleration = float(config.get("servo_acceleration", 0.3))
        self.period = float(config.get("servo_period_s", 0.04))
        self.lookahead = float(config.get("servo_lookahead_s", 0.1))
        self.gain = float(config.get("servo_gain", 200.0))

    def current_tcp(self) -> np.ndarray:
        return np.asarray(self.receive.getActualTCPPose(), dtype=np.float64)

    def command_tcp(self, pose: np.ndarray) -> None:
        target = np.asarray(pose, dtype=np.float64).tolist()
        ok = self.control.servoL(
            target,
            self.speed,
            self.acceleration,
            self.period,
            self.lookahead,
            self.gain,
        )
        if ok is False:
            raise RuntimeError("ur_rtde servoL rejected the target pose")

    def hold(self) -> None:
        self.control.servoStop()

    def close(self) -> None:
        try:
            self.control.servoStop()
        finally:
            self.control.stopScript()
