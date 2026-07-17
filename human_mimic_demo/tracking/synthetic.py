from __future__ import annotations

import time

import numpy as np

from human_mimic_demo.messages import TrackingSample
from human_mimic_demo.tracking.base import HandTracker


class SyntheticTracker(HandTracker):
    """Dependency-free signal source used to verify the complete control pipeline."""

    def __init__(self) -> None:
        self.started = time.monotonic()

    @staticmethod
    def _hand_landmarks(curl: float) -> np.ndarray:
        points = np.zeros((21, 3), dtype=np.float64)
        points[0] = [0.0, 0.0, 0.0]
        bases = {
            1: (-0.035, 0.025),
            5: (-0.025, 0.050),
            9: (0.000, 0.055),
            13: (0.022, 0.052),
            17: (0.040, 0.045),
        }
        for start, (x, y) in bases.items():
            segment = 0.022 if start != 1 else 0.019
            points[start] = [x, y, 0.0]
            direction = np.asarray([0.0, 1.0 - 0.65 * curl, 0.75 * curl])
            direction /= np.linalg.norm(direction)
            for offset in range(1, 4):
                points[start + offset] = points[start + offset - 1] + segment * direction
                direction = np.asarray(
                    [0.0, direction[1] - 0.25 * curl, direction[2] + 0.15 * curl]
                )
                direction /= np.linalg.norm(direction)
        return points

    def read(self) -> TrackingSample:
        elapsed = time.monotonic() - self.started
        curl = 0.5 + 0.5 * np.sin(elapsed * 1.2)
        wrist = np.asarray(
            [0.08 * np.sin(elapsed * 0.6), 0.05 * np.sin(elapsed * 0.4), 0.65],
            dtype=np.float64,
        )
        return TrackingSample(
            timestamp=time.time(),
            wrist_xyz_m=wrist,
            hand_joints=self._hand_landmarks(float(curl)),
            confidence=1.0,
            source="synthetic",
        )
