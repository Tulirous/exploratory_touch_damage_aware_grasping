from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from human_mimic_demo.geometry import angle_between, clamp01, limit_step, low_pass, safe_norm
from human_mimic_demo.messages import HandFeatures


FINGER_CHAINS = {
    "thumb": (1, 2, 3, 4),
    "index": (5, 6, 7, 8),
    "middle": (9, 10, 11, 12),
    "ring": (13, 14, 15, 16),
    "pinky": (17, 18, 19, 20),
}


@dataclass
class FeatureCalibration:
    open_flex: np.ndarray
    closed_flex: np.ndarray

    @classmethod
    def defaults(cls) -> "FeatureCalibration":
        return cls(
            open_flex=np.zeros(5, dtype=np.float64),
            closed_flex=np.ones(5, dtype=np.float64),
        )


class O6Retargeter:
    """Geometry-preserving projection from 21 human joints to O6 commands."""

    def __init__(self, config: dict) -> None:
        self.max_bend_radians = float(config.get("max_bend_radians", 3.5))
        self.thumb_abduction_range = np.asarray(
            config.get("thumb_abduction_distance_ratio", [0.45, 1.70]), dtype=np.float64
        )
        self.open_position = np.asarray(config["open_position"], dtype=np.float64)
        self.closed_position = np.asarray(config["closed_position"], dtype=np.float64)
        self.safe_min = np.asarray(config.get("safe_min", [0] * 6), dtype=np.float64)
        self.safe_max = np.asarray(config.get("safe_max", [255] * 6), dtype=np.float64)
        self.alpha = float(config.get("low_pass_alpha", 0.3))
        self.max_step = float(config.get("max_command_step", 8.0))
        if any(array.shape != (6,) for array in (self.open_position, self.closed_position, self.safe_min, self.safe_max)):
            raise ValueError("Every O6 position/range field must contain exactly six values")
        if np.any(self.safe_min > self.safe_max):
            raise ValueError("O6 safe_min must not exceed safe_max")
        self.calibration = FeatureCalibration.defaults()
        self._previous_command: np.ndarray | None = None

    def _finger_bend(self, joints: np.ndarray, chain: tuple[int, int, int, int]) -> float:
        wrist, mcp, pip, dip, tip = joints[0], *(joints[index] for index in chain)
        bend = (
            angle_between(mcp - wrist, pip - mcp)
            + angle_between(pip - mcp, dip - pip)
            + angle_between(dip - pip, tip - dip)
        )
        return clamp01(bend / self.max_bend_radians)

    def extract_raw_features(self, joints: np.ndarray) -> np.ndarray:
        joints = np.asarray(joints, dtype=np.float64)
        if joints.shape != (21, 3) or not np.isfinite(joints).all():
            raise ValueError("hand_joints must be a finite [21, 3] array")
        flex = np.asarray(
            [self._finger_bend(joints, FINGER_CHAINS[name]) for name in FINGER_CHAINS],
            dtype=np.float64,
        )
        palm_width = safe_norm(joints[5] - joints[17])
        thumb_distance_ratio = safe_norm(joints[4] - joints[5]) / palm_width
        thumb_abduction = clamp01(
            (thumb_distance_ratio - self.thumb_abduction_range[0])
            / max(self.thumb_abduction_range[1] - self.thumb_abduction_range[0], 1e-6)
        )
        # thumb, index, middle, ring, pinky flex followed by thumb abduction
        return np.concatenate([flex, [thumb_abduction]])

    def normalize_features(self, raw: np.ndarray) -> HandFeatures:
        denominator = np.maximum(
            self.calibration.closed_flex - self.calibration.open_flex, 1e-4
        )
        flex = np.clip((raw[:5] - self.calibration.open_flex) / denominator, 0.0, 1.0)
        return HandFeatures(
            thumb_flex=float(flex[0]),
            thumb_abduction=float(raw[5]),
            index_flex=float(flex[1]),
            middle_flex=float(flex[2]),
            ring_flex=float(flex[3]),
            pinky_flex=float(flex[4]),
        )

    def record_open(self, joints: np.ndarray) -> None:
        self.calibration.open_flex = self.extract_raw_features(joints)[:5]

    def record_closed(self, joints: np.ndarray) -> None:
        closed = self.extract_raw_features(joints)[:5]
        self.calibration.closed_flex = np.maximum(
            closed, self.calibration.open_flex + 0.05
        )

    def retarget(self, joints: np.ndarray) -> tuple[np.ndarray, HandFeatures]:
        features = self.normalize_features(self.extract_raw_features(joints))
        synergy = features.as_array()
        # Flexion grows from open to closed. Thumb abduction has the opposite
        # semantic: 1 means spread away from the palm and therefore uses the
        # configured open/abducted position.
        synergy[1] = 1.0 - synergy[1]
        desired = self.open_position + synergy * (self.closed_position - self.open_position)
        desired = np.clip(desired, self.safe_min, self.safe_max)
        filtered = low_pass(self._previous_command, desired, self.alpha)
        limited = limit_step(self._previous_command, filtered, self.max_step)
        command = np.rint(np.clip(limited, self.safe_min, self.safe_max)).astype(np.int64)
        self._previous_command = command.astype(np.float64)
        return command, features

    def reset_filter(self) -> None:
        self._previous_command = None
