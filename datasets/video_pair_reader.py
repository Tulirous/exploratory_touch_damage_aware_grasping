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


def read_video_frames(
    video_path: str | Path,
    frame_indices: list[int] | set[int] | tuple[int, ...],
) -> dict[int, np.ndarray]:
    """Decode multiple RGB frames in one sequential pass through a video."""

    path = Path(video_path)
    if not path.exists():
        raise FileNotFoundError(f"Video file not found: {path}")
    requested = sorted(set(int(index) for index in frame_indices))
    if not requested:
        return {}
    if requested[0] < 0:
        raise ValueError(f"frame indices must be non-negative, got {requested[0]}")

    try:
        import imageio.v3 as iio
    except ImportError as exc:
        raise ImportError(
            "imageio is required for video frame reading. Install it in the "
            "server environment or run inside the Fast-WAM environment."
        ) from exc

    requested_set = set(requested)
    last_requested = requested[-1]
    frames: dict[int, np.ndarray] = {}
    for index, frame in enumerate(iio.imiter(path)):
        if index in requested_set:
            arr = np.asarray(frame)
            if arr.ndim == 2:
                arr = np.repeat(arr[..., None], 3, axis=-1)
            if arr.shape[-1] == 4:
                arr = arr[..., :3]
            if arr.shape[-1] != 3:
                raise ValueError(f"Expected RGB/RGBA frame from {path}, got shape {arr.shape}")
            frames[index] = np.ascontiguousarray(arr.astype(np.uint8, copy=False))
        if index >= last_requested:
            break

    missing = [index for index in requested if index not in frames]
    if missing:
        raise IndexError(f"Video {path} is missing requested frame indices: {missing[:10]}")
    return frames


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
