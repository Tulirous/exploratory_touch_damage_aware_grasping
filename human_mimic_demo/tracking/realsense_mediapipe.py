from __future__ import annotations

import time
from typing import Optional

import numpy as np

from human_mimic_demo.messages import TrackingSample
from human_mimic_demo.tracking.base import HandTracker


class RealSenseMediaPipeTracker(HandTracker):
    """D435 RGB-D wrist translation plus markerless right-hand landmarks.

    MediaPipe is an immediately runnable landmark backend, not a MANO parameter
    estimator. Its 21-point output is converted to the same geometry contract used
    by a future MANO/SMPL-X backend.
    """

    def __init__(self, config: dict) -> None:
        try:
            import cv2
            import mediapipe as mp
            import pyrealsense2 as rs
        except ImportError as exc:
            raise RuntimeError(
                "RealSense mode requires OpenCV, MediaPipe and pyrealsense2. "
                "See human_mimic_demo/requirements-camera.txt."
            ) from exc

        self.cv2 = cv2
        self.mp = mp
        self.rs = rs
        self.width = int(config.get("width", 640))
        self.height = int(config.get("height", 480))
        self.fps = int(config.get("fps", 30))
        self.min_confidence = float(config.get("min_confidence", 0.65))
        self.depth_radius = int(config.get("depth_median_radius_px", 3))

        self.pipeline = rs.pipeline()
        stream_config = rs.config()
        stream_config.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps)
        stream_config.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, self.fps)
        self.profile = self.pipeline.start(stream_config)
        self.align = rs.align(rs.stream.color)
        self.holistic = mp.solutions.holistic.Holistic(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            refine_face_landmarks=False,
            min_detection_confidence=self.min_confidence,
            min_tracking_confidence=self.min_confidence,
        )

    def _median_depth(self, depth_frame: object, u: int, v: int) -> float:
        values = []
        for y in range(max(0, v - self.depth_radius), min(self.height, v + self.depth_radius + 1)):
            for x in range(max(0, u - self.depth_radius), min(self.width, u + self.depth_radius + 1)):
                depth = float(depth_frame.get_distance(x, y))
                if depth > 0.0:
                    values.append(depth)
        return float(np.median(values)) if values else 0.0

    def read(self) -> Optional[TrackingSample]:
        frames = self.pipeline.wait_for_frames(timeout_ms=1000)
        frames = self.align.process(frames)
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()
        if not color_frame or not depth_frame:
            return None

        frame_bgr = np.asanyarray(color_frame.get_data())
        rgb = self.cv2.cvtColor(frame_bgr, self.cv2.COLOR_BGR2RGB)
        result = self.holistic.process(rgb)
        hand_landmarks = result.right_hand_landmarks
        if hand_landmarks is None:
            return None
        image_landmarks = hand_landmarks.landmark
        wrist_u = int(np.clip(image_landmarks[0].x * self.width, 0, self.width - 1))
        wrist_v = int(np.clip(image_landmarks[0].y * self.height, 0, self.height - 1))
        wrist_depth = self._median_depth(depth_frame, wrist_u, wrist_v)
        if wrist_depth <= 0.0:
            return None

        intrinsics = color_frame.profile.as_video_stream_profile().intrinsics
        wrist_xyz = np.asarray(
            self.rs.rs2_deproject_pixel_to_point(intrinsics, [wrist_u, wrist_v], wrist_depth),
            dtype=np.float64,
        )

        # Convert normalized image landmarks to an aspect-ratio-correct local
        # geometry. Absolute wrist translation comes independently from D435 depth.
        joints = np.asarray(
            [[p.x * self.width, p.y * self.height, p.z * self.width] for p in image_landmarks],
            dtype=np.float64,
        )
        joints -= joints[0]

        self.mp.solutions.drawing_utils.draw_landmarks(
            frame_bgr,
            hand_landmarks,
            self.mp.solutions.hands.HAND_CONNECTIONS,
        )
        return TrackingSample(
            timestamp=time.time(),
            wrist_xyz_m=wrist_xyz,
            hand_joints=joints,
            confidence=self.min_confidence,
            frame_bgr=frame_bgr,
            source="realsense_mediapipe",
        )

    def close(self) -> None:
        self.holistic.close()
        self.pipeline.stop()
