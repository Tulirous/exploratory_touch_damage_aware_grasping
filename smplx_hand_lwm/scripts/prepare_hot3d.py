from __future__ import annotations

import argparse
import io
import json
import math
import random
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


HAND_SIDES = ("left", "right")


@dataclass
class TrackInfo:
    clip_id: str
    handedness: str
    sequence_path: Path
    valid: np.ndarray
    frame_count: int


def axis_angle_to_matrix(axis_angle: np.ndarray) -> np.ndarray:
    """Convert one 3D axis-angle vector to a 3x3 rotation matrix."""

    vector = np.asarray(axis_angle, dtype=np.float64)
    angle = float(np.linalg.norm(vector))
    if angle < 1e-8:
        skew = _skew(vector)
        return np.eye(3, dtype=np.float64) + skew + 0.5 * (skew @ skew)
    axis = vector / angle
    skew = _skew(axis)
    return (
        np.eye(3, dtype=np.float64)
        + math.sin(angle) * skew
        + (1.0 - math.cos(angle)) * (skew @ skew)
    )


def matrix_to_rotation_6d(matrix: np.ndarray) -> np.ndarray:
    """Use the first two rows, matching the PyTorch3D 6D convention."""

    return np.asarray(matrix, dtype=np.float64)[:2, :].reshape(6)


def _skew(vector: np.ndarray) -> np.ndarray:
    x, y, z = vector
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def _read_json(tar: tarfile.TarFile, member_name: str) -> Any:
    member = tar.extractfile(member_name)
    if member is None:
        raise ValueError(f"cannot extract {member_name}")
    return json.load(io.TextIOWrapper(member, encoding="utf-8"))


def _frame_ids(tar: tarfile.TarFile) -> list[str]:
    return sorted(
        name.removesuffix(".hands.json")
        for name in tar.getnames()
        if name.endswith(".hands.json")
    )


def _is_visible(hand: dict[str, Any], threshold: float) -> bool:
    if threshold <= 0:
        return True
    scores = list(hand.get("visibilities_modeled", {}).values())
    return bool(scores) and max(float(score) for score in scores) >= threshold


def convert_clip(
    tar_path: Path,
    output_dir: Path,
    coordinate_frame: str,
    visibility_threshold: float,
    hand_sides: tuple[str, ...] = HAND_SIDES,
) -> list[TrackInfo]:
    clip_id = tar_path.stem.removeprefix("clip-")
    tracks: list[TrackInfo] = []
    with tarfile.open(tar_path, "r") as tar:
        frame_ids = _frame_ids(tar)
        if not frame_ids:
            raise ValueError(f"{tar_path}: no hands.json frames")
        shape = (
            _read_json(tar, "__hand_shapes.json__")
            if "__hand_shapes.json__" in tar.getnames()
            else {}
        )
        mano_beta = np.asarray(shape.get("mano", []), dtype=np.float32)
        annotations = [
            _read_json(tar, f"{frame_id}.hands.json") for frame_id in frame_ids
        ]

    for handedness in hand_sides:
        wrist = np.zeros((len(frame_ids), 6), dtype=np.float64)
        mano_pca = np.zeros((len(frame_ids), 15), dtype=np.float64)
        valid = np.zeros(len(frame_ids), dtype=bool)
        for index, annotation in enumerate(annotations):
            hand = annotation.get(handedness)
            if (
                not hand
                or "mano_pose" not in hand
                or not _is_visible(hand, visibility_threshold)
            ):
                continue
            pose = hand["mano_pose"]
            current_wrist = np.asarray(
                pose.get("wrist_xform", []), dtype=np.float64
            )
            current_pca = np.asarray(pose.get("thetas", []), dtype=np.float64)
            if current_wrist.shape != (6,) or current_pca.shape != (15,):
                continue
            if (
                not np.isfinite(current_wrist).all()
                or not np.isfinite(current_pca).all()
            ):
                continue
            wrist[index] = current_wrist
            mano_pca[index] = current_pca
            valid[index] = True
        if not valid.any():
            continue

        reference_index = int(np.flatnonzero(valid)[0])
        reference_rotation = axis_angle_to_matrix(wrist[reference_index, :3])
        reference_translation = wrist[reference_index, 3:]
        hand_state = np.zeros((len(frame_ids), 24), dtype=np.float32)
        for index in np.flatnonzero(valid):
            rotation = axis_angle_to_matrix(wrist[index, :3])
            translation = wrist[index, 3:]
            if coordinate_frame == "clip_initial_wrist":
                rotation = reference_rotation.T @ rotation
                translation = reference_rotation.T @ (
                    translation - reference_translation
                )
            hand_state[index, :3] = translation.astype(np.float32)
            hand_state[index, 3:9] = matrix_to_rotation_6d(rotation).astype(
                np.float32
            )
            hand_state[index, 9:] = mano_pca[index].astype(np.float32)

        output_path = output_dir / f"clip-{clip_id}_{handedness}.npz"
        np.savez_compressed(
            output_path,
            hand_state=hand_state,
            valid=valid,
            frame_ids=np.asarray(frame_ids),
            mano_beta=mano_beta,
            mano_pca=mano_pca.astype(np.float32),
            wrist_xform=wrist.astype(np.float32),
            coordinate_frame=np.asarray(coordinate_frame),
        )
        tracks.append(
            TrackInfo(
                clip_id=clip_id,
                handedness=handedness,
                sequence_path=output_path.resolve(),
                valid=valid,
                frame_count=len(frame_ids),
            )
        )
    return tracks


def build_windows(
    tracks: list[TrackInfo],
    context_length: int,
    future_length: int,
    stride: int,
) -> list[dict[str, Any]]:
    window_length = context_length + future_length
    rows: list[dict[str, Any]] = []
    for track in tracks:
        for start in range(0, track.frame_count - window_length + 1, stride):
            stop = start + window_length
            if not track.valid[start:stop].all():
                continue
            rows.append(
                {
                    "episode_id": f"hot3d_{track.clip_id}_{track.handedness}",
                    "sequence_path": str(track.sequence_path),
                    "handedness": track.handedness,
                    "start_index": start,
                    "source_dataset": "HOT3D-Clips",
                    "clip_id": track.clip_id,
                }
            )
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert HOT3D-Clips MANO-PCA annotations into Stage-1 hand tracks."
    )
    parser.add_argument("--clips-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--train-manifest",
        type=Path,
        default=Path("data/manifests/hot3d_hand_train.jsonl"),
    )
    parser.add_argument(
        "--val-manifest",
        type=Path,
        default=Path("data/manifests/hot3d_hand_val.jsonl"),
    )
    parser.add_argument("--context-length", type=int, default=4)
    parser.add_argument("--future-length", type=int, default=12)
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-clips", type=int, default=None)
    parser.add_argument(
        "--coordinate-frame",
        choices=("world", "clip_initial_wrist"),
        default="clip_initial_wrist",
    )
    parser.add_argument("--visibility-threshold", type=float, default=0.1)
    parser.add_argument(
        "--handedness",
        choices=("left", "right", "both"),
        default="right",
        help="Use one side for the pilot; both keeps separate left/right tracks.",
    )
    args = parser.parse_args()

    if not 0.0 < args.val_fraction < 1.0:
        raise ValueError("--val-fraction must be between 0 and 1")
    tar_paths = sorted(args.clips_dir.glob("clip-*.tar"))
    if args.max_clips is not None:
        tar_paths = tar_paths[: args.max_clips]
    if len(tar_paths) < 2:
        raise ValueError("at least two clips are required for a train/val split")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    tracks_by_clip: dict[str, list[TrackInfo]] = {}
    failures: list[dict[str, str]] = []
    hand_sides = HAND_SIDES if args.handedness == "both" else (args.handedness,)
    for tar_path in tar_paths:
        clip_id = tar_path.stem.removeprefix("clip-")
        try:
            tracks_by_clip[clip_id] = convert_clip(
                tar_path,
                args.output_dir,
                args.coordinate_frame,
                args.visibility_threshold,
                hand_sides,
            )
        except (OSError, tarfile.TarError, ValueError, json.JSONDecodeError) as error:
            failures.append({"clip": str(tar_path), "error": str(error)})

    clip_ids = sorted(clip_id for clip_id, tracks in tracks_by_clip.items() if tracks)
    if len(clip_ids) < 2:
        raise RuntimeError("fewer than two clips produced valid hand tracks")
    random.Random(args.seed).shuffle(clip_ids)
    val_count = max(1, round(len(clip_ids) * args.val_fraction))
    val_ids = set(clip_ids[:val_count])
    train_ids = set(clip_ids[val_count:])
    train_tracks = [track for key in sorted(train_ids) for track in tracks_by_clip[key]]
    val_tracks = [track for key in sorted(val_ids) for track in tracks_by_clip[key]]
    train_rows = build_windows(
        train_tracks, args.context_length, args.future_length, args.stride
    )
    val_rows = build_windows(
        val_tracks, args.context_length, args.future_length, args.stride
    )
    if not train_rows or not val_rows:
        raise RuntimeError("train or validation split has no fully valid windows")
    write_jsonl(args.train_manifest, train_rows)
    write_jsonl(args.val_manifest, val_rows)

    report = {
        "input_clips": len(tar_paths),
        "converted_clips": len(clip_ids),
        "failed_clips": failures,
        "train_clip_ids": sorted(train_ids),
        "val_clip_ids": sorted(val_ids),
        "train_tracks": len(train_tracks),
        "val_tracks": len(val_tracks),
        "train_windows": len(train_rows),
        "val_windows": len(val_rows),
        "state_dim": 24,
        "coordinate_frame": args.coordinate_frame,
        "handedness": args.handedness,
    }
    report_path = args.output_dir / "preparation_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"report: {report_path}")


if __name__ == "__main__":
    main()
