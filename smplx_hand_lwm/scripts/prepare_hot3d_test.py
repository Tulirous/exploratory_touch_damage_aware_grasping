from __future__ import annotations

import argparse
import json
import tarfile
from pathlib import Path
from typing import Any

from smplx_hand_lwm.scripts.prepare_hot3d import (
    HAND_SIDES,
    TrackInfo,
    build_windows,
    convert_clip,
    write_jsonl,
)


def clip_ids_from_manifest(path: Path) -> set[str]:
    clip_ids: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if "clip_id" not in row:
                raise ValueError(f"{path}: manifest row has no clip_id")
            clip_ids.add(str(row["clip_id"]))
    return clip_ids


def write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Create a test-only HOT3D hand manifest and reject any clip "
            "overlap with existing train/validation manifests."
        )
    )
    parser.add_argument("--clips-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--test-manifest", type=Path, required=True)
    parser.add_argument("--clip-start", type=int, required=True)
    parser.add_argument("--clip-end", type=int, required=True)
    parser.add_argument(
        "--target-valid-clips",
        type=int,
        default=None,
        help=(
            "Select the first N clip IDs in the requested range that each "
            "produce at least one fully valid window."
        ),
    )
    parser.add_argument(
        "--exclude-manifest",
        type=Path,
        action="append",
        default=[],
        help="Train/validation manifest whose clip IDs must not enter test.",
    )
    parser.add_argument("--context-length", type=int, default=4)
    parser.add_argument("--future-length", type=int, default=12)
    parser.add_argument("--stride", type=int, default=4)
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
    )
    args = parser.parse_args()

    if args.clip_start < 0 or args.clip_end < args.clip_start:
        raise ValueError("clip range must satisfy 0 <= start <= end")
    if args.target_valid_clips is not None and args.target_valid_clips <= 0:
        raise ValueError("--target-valid-clips must be positive")
    requested_ids = {
        f"{index:06d}" for index in range(args.clip_start, args.clip_end + 1)
    }

    excluded_ids: set[str] = set()
    for manifest in args.exclude_manifest:
        excluded_ids.update(clip_ids_from_manifest(manifest))
    overlap = requested_ids.intersection(excluded_ids)
    if overlap:
        raise ValueError(
            "test clips overlap excluded train/validation manifests: "
            f"{sorted(overlap)}"
        )

    tar_paths = [
        args.clips_dir / f"clip-{clip_id}.tar"
        for clip_id in sorted(requested_ids)
    ]
    missing = [str(path) for path in tar_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "missing requested test archives: " + ", ".join(missing)
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    hand_sides = HAND_SIDES if args.handedness == "both" else (args.handedness,)
    tracks_by_clip: dict[str, list[TrackInfo]] = {}
    converted_ids: list[str] = []
    failures: list[dict[str, str]] = []
    for index, tar_path in enumerate(tar_paths, start=1):
        clip_id = tar_path.stem.removeprefix("clip-")
        print(
            f"converting test clip {index}/{len(tar_paths)}: {clip_id}",
            flush=True,
        )
        try:
            clip_tracks = convert_clip(
                tar_path,
                args.output_dir,
                args.coordinate_frame,
                args.visibility_threshold,
                hand_sides,
            )
            if not clip_tracks:
                raise ValueError("no requested hand track passed validation")
            tracks_by_clip[clip_id] = clip_tracks
            converted_ids.append(clip_id)
        except (OSError, tarfile.TarError, ValueError, json.JSONDecodeError) as error:
            failures.append({"clip_id": clip_id, "error": str(error)})

    rows_by_clip: dict[str, list[dict[str, Any]]] = {}
    clips_without_windows: list[str] = []
    for clip_id in sorted(converted_ids):
        clip_rows = build_windows(
            tracks_by_clip[clip_id],
            args.context_length,
            args.future_length,
            args.stride,
        )
        if clip_rows:
            rows_by_clip[clip_id] = clip_rows
        else:
            clips_without_windows.append(clip_id)

    eligible_ids = sorted(rows_by_clip)
    if args.target_valid_clips is not None:
        selected_ids = eligible_ids[: args.target_valid_clips]
        selection_rule = "first_n_clips_with_valid_windows"
    else:
        selected_ids = eligible_ids
        selection_rule = "all_requested_clips_strict"
    selected_tracks = [
        track for clip_id in selected_ids for track in tracks_by_clip[clip_id]
    ]
    rows = [row for clip_id in selected_ids for row in rows_by_clip[clip_id]]
    for row in rows:
        row["split"] = "test"

    report = {
        "split": "test",
        "requested_clip_ids": sorted(requested_ids),
        "requested_clips": len(requested_ids),
        "converted_clip_ids": sorted(converted_ids),
        "converted_clips": len(converted_ids),
        "failed_clips": failures,
        "clips_without_valid_windows": clips_without_windows,
        "selection_rule": selection_rule,
        "target_valid_clips": args.target_valid_clips,
        "selected_clip_ids": selected_ids,
        "selected_clips": len(selected_ids),
        "excluded_manifests": [str(path) for path in args.exclude_manifest],
        "excluded_clip_count": len(excluded_ids),
        "overlap_with_excluded": [],
        "test_tracks": len(selected_tracks),
        "test_windows": len(rows),
        "state_dim": 24,
        "coordinate_frame": args.coordinate_frame,
        "handedness": args.handedness,
    }
    report_path = args.output_dir / "test_preparation_report.json"
    write_report(report_path, report)
    if args.target_valid_clips is not None and len(selected_ids) < args.target_valid_clips:
        raise RuntimeError(
            f"only {len(selected_ids)} clips produced valid windows; requested "
            f"{args.target_valid_clips}; see {report_path}"
        )
    if args.target_valid_clips is None and (failures or clips_without_windows):
        raise RuntimeError(
            "not every requested test clip produced valid windows; see "
            f"{report_path}"
        )
    if not rows:
        raise RuntimeError("test split has no fully valid windows")
    write_jsonl(args.test_manifest, rows)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"report: {report_path}")


if __name__ == "__main__":
    main()
