from __future__ import annotations

from abc import ABC, abstractmethod
import sys
from typing import Any

import numpy as np


class HandController(ABC):
    @abstractmethod
    def command(self, position: np.ndarray) -> None:
        pass

    def state(self) -> list[float] | None:
        return None

    def force(self) -> Any:
        return None

    def close(self) -> None:
        pass


class DryRunHand(HandController):
    def __init__(self) -> None:
        self.position = np.full(6, 255, dtype=np.int64)

    def command(self, position: np.ndarray) -> None:
        self.position = np.asarray(position, dtype=np.int64).copy()

    def state(self) -> list[float]:
        return self.position.astype(float).tolist()


class LinkerHandO6(HandController):
    def __init__(self, config: dict) -> None:
        sdk_path = str(config.get("sdk_path", "")).strip()
        if sdk_path and sdk_path not in sys.path:
            sys.path.insert(0, sdk_path)
        try:
            from LinkerHand.linker_hand_api import LinkerHandApi
        except ImportError as exc:
            raise RuntimeError(
                "Linker Hand SDK was not found. Set hand.sdk_path to the cloned SDK root."
            ) from exc
        can_channel = str(config.get("can_channel", "can0"))
        modbus = str(config.get("modbus", "None"))
        self.api = LinkerHandApi(
            hand_type="right",
            hand_joint="O6",
            can=can_channel,
            modbus=modbus,
        )
        speed = [int(value) for value in config.get("speed", [60] * 6)]
        torque = [int(value) for value in config.get("torque", [50] * 6)]
        self.api.set_speed(speed=speed)
        self.api.set_torque(torque=torque)
        self.force_enabled = bool(config.get("force_sensor_enabled", False))

    def command(self, position: np.ndarray) -> None:
        values = [int(value) for value in np.asarray(position).tolist()]
        if len(values) != 6 or any(value < 0 or value > 255 for value in values):
            raise ValueError("O6 command must contain six values in [0, 255]")
        self.api.finger_move(pose=values)

    def state(self) -> list[float]:
        return self.api.get_state()

    def force(self) -> Any:
        return self.api.get_force() if self.force_enabled else None
