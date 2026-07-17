from __future__ import annotations

from pathlib import Path
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
        self.input_is_mirrored = bool(config.get("input_is_mirrored", False))
        model_path = Path(
            config.get(
                "hand_landmarker_model_path",
                "human_mimic_demo/assets/hand_landmarker.task",
            )
        )
        if not model_path.is_file():
            raise RuntimeError(
                f"MediaPipe Hand Landmarker model not found: {model_path}. "
                "Run: python -m human_mimic_demo.scripts.download_mediapipe_models"
            )

        self.pipeline = rs.pipeline()
        stream_config = rs.config()
        stream_config.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps)
        stream_config.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, self.fps)
        self.profile = self.pipeline.start(stream_config)
        self.align = rs.align(rs.stream.color)
        options = mp.tasks.vision.HandLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path)),
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=self.min_confidence,
            min_hand_presence_confidence=self.min_confidence,
            min_tracking_confidence=self.min_confidence,
        )
        self.hand_landmarker = mp.tasks.vision.HandLandmarker.create_from_options(options)
        self._last_timestamp_ms = 0

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
        mp_image = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=rgb)
        timestamp_ms = max(int(time.monotonic() * 1000), self._last_timestamp_ms + 1)
        self._last_timestamp_ms = timestamp_ms
        result = self.hand_landmarker.detect_for_video(mp_image, timestamp_ms)
        if not result.hand_landmarks or not result.handedness:
            return None

        # MediaPipe handedness assumes selfie-mirrored input. D435 frames are not
        # mirrored by default, so physical right is reported as Left.
        expected_label = "Right" if self.input_is_mirrored else "Left"
        selected_index = None
        selected_score = 0.0
        for index, categories in enumerate(result.handedness):
            if not categories:
                continue
            category = categories[0]
            label = str(category.category_name)
            score = float(category.score)
            if label == expected_label and score >= self.min_confidence:
                selected_index = index
                selected_score = score
                break
        if selected_index is None:
            return None

        image_landmarks = result.hand_landmarks[selected_index]
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
        if result.hand_world_landmarks and len(result.hand_world_landmarks) > selected_index:
            world_landmarks = result.hand_world_landmarks[selected_index]
            joints = np.asarray(
                [[p.x, p.y, p.z] for p in world_landmarks], dtype=np.float64
            )
        else:
            joints = np.asarray(
                [[p.x * self.width, p.y * self.height, p.z * self.width] for p in image_landmarks],
                dtype=np.float64,
            )
        joints -= joints[0]

        self._draw_hand(frame_bgr, image_landmarks)
        return TrackingSample(
            timestamp=time.time(),
            wrist_xyz_m=wrist_xyz,
            hand_joints=joints,
            confidence=selected_score,
            frame_bgr=frame_bgr,
            source="realsense_mediapipe",
        )

    def close(self) -> None:
        self.hand_landmarker.close()
        self.pipeline.stop()

    def _draw_hand(self, frame_bgr: np.ndarray, landmarks: list) -> None:
        connections = (
            (0, 1), (1, 2), (2, 3), (3, 4),
            (0, 5), (5, 6), (6, 7), (7, 8),
            (5, 9), (9, 10), (10, 11), (11, 12),
            (9, 13), (13, 14), (14, 15), (15, 16),
            (13, 17), (0, 17), (17, 18), (18, 19), (19, 20),
        )
        pixels = [
            (
                int(np.clip(point.x * self.width, 0, self.width - 1)),
                int(np.clip(point.y * self.height, 0, self.height - 1)),
            )
            for point in landmarks
        ]
        for start, end in connections:
            self.cv2.line(frame_bgr, pixels[start], pixels[end], (0, 200, 0), 2)
        for pixel in pixels:
            self.cv2.circle(frame_bgr, pixel, 3, (0, 255, 255), -1)
