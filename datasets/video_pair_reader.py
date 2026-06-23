from __future__ import annotations

from pathlib import Path

import numpy as np


def read_video_frame(video_path: str | Path, frame_index: int = 0) -> np.ndarray:
    """Read one RGB frame from a video file.

    The server-side Fast-WAM environment already depends on imageio for rollout
    videos, so this keeps the project free of an extra OpenCV requirement.
    """

    path = Path(video_path)
    if not path.exists():
        raise FileNotFoundError(f"Video file not found: {path}")
    if frame_index < 0:
        raise ValueError(f"frame_index must be non-negative, got {frame_index}")

    try:
        import imageio.v3 as iio
    except ImportError as exc:
        raise ImportError(
            "imageio is required for video frame reading. Install it in the "
            "server environment or run inside the Fast-WAM environment."
        ) from exc

    for idx, frame in enumerate(iio.imiter(path)):
        if idx == frame_index:
            arr = np.asarray(frame)
            if arr.ndim == 2:
                arr = np.repeat(arr[..., None], 3, axis=-1)
            if arr.shape[-1] == 4:
                arr = arr[..., :3]
            if arr.shape[-1] != 3:
                raise ValueError(f"Expected RGB/RGBA frame from {path}, got shape {arr.shape}")
            return np.ascontiguousarray(arr.astype(np.uint8, copy=False))

    raise IndexError(f"Video {path} has fewer than {frame_index + 1} frames")


def read_base_wrist_frames(
    base_video_path: str | Path,
    wrist_video_path: str | Path,
    frame_index: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Read synchronized base/wrist RGB frames by index."""

    return (
        read_video_frame(base_video_path, frame_index=frame_index),
        read_video_frame(wrist_video_path, frame_index=frame_index),
    )
