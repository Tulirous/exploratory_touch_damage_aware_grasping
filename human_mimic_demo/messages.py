from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np


@dataclass
class TrackingSample:
    """One timestamped human-hand observation in the D435 camera frame."""

    timestamp: float
    wrist_xyz_m: np.ndarray
    hand_joints: np.ndarray
    confidence: float
    frame_bgr: Optional[Any] = None
    source: str = "unknown"

    def valid(self) -> bool:
        return (
            self.wrist_xyz_m.shape == (3,)
            and self.hand_joints.shape == (21, 3)
            and np.isfinite(self.wrist_xyz_m).all()
            and np.isfinite(self.hand_joints).all()
            and 0.0 <= self.confidence <= 1.0
        )


@dataclass
class HandFeatures:
    """Six O6-compatible human hand synergies in [0, 1]."""

    thumb_flex: float
    thumb_abduction: float
    index_flex: float
    middle_flex: float
    ring_flex: float
    pinky_flex: float

    def as_array(self) -> np.ndarray:
        return np.asarray(
            [
                self.thumb_flex,
                self.thumb_abduction,
                self.index_flex,
                self.middle_flex,
                self.ring_flex,
                self.pinky_flex,
            ],
            dtype=np.float64,
        )
