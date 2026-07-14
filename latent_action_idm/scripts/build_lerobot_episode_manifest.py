from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from latent_action_idm.utils import write_jsonl


def load_info(root: Path) -> dict:
    with (root / "meta/info.json").open("r", encoding="utf-8") as f:
        return json.load(f)


def episode_metadata(root: Path) -> pd.DataFrame:
    files = sorted((root / "meta/episodes").glob("chunk-*/file-*.parquet"))
    if not files:
        raise FileNotFoundError(f"No episode metadata found under {root / 'meta/episodes'}")
    return pd.concat((pd.read_parquet(path) for path in files), ignore_index=True)


def format_dataset_path(root: Path, template: str, **values: int | str) -> Path:
    return root / template.format(**values)


def data_path(root: Path, info: dict, chunk_index: int, file_index: int) -> Path:
    return format_dataset_path(
        root,
        info["data_path"],
        chunk_index=chunk_index,
        file_index=file_index,
    )


def video_path(
    root: Path,
    info: dict,
    video_key: str,
    chunk_index: int,
    file_index: int,
) -> Path:
    return format_dataset_path(
        root,
        info["video_path"],
        video_key=video_key,
        chunk_index=chunk_index,
        file_index=file_index,
    )


def video_fps(info: dict, video_key: str) -> float:
    try:
        return float(info["features"][video_key]["info"]["video.fps"])
    except KeyError as exc:
        raise KeyError(f"Could not find video FPS for {video_key!r} in meta/info.json") from exc


def metadata_column(video_key: str, field: str) -> str:
    return f"videos/{video_key}/{field}"


def required_metadata_columns(base_video_key: str, wrist_video_key: str) -> set[str]:
    columns = {
        "episode_index",
        "length",
        "data/chunk_index",
        "data/file_index",
        "dataset_from_index",
        "dataset_to_index",
    }
    for key in (base_video_key, wrist_video_key):
        columns.update(
            {
                metadata_column(key, "chunk_index"),
                metadata_column(key, "file_index"),
                metadata_column(key, "from_timestamp"),
                metadata_column(key, "to_timestamp"),
            }
        )
    return columns


def filter_available_metadata(
    metadata: pd.DataFrame,
    root: Path,
    info: dict,
    base_video_key: str,
    wrist_video_key: str,
) -> pd.DataFrame:
    available = []
    path_cache: dict[Path, bool] = {}

    def is_available(path: Path) -> bool:
        if path not in path_cache:
            path_cache[path] = path.exists() and path.stat().st_size > 0
        return path_cache[path]

    for record in metadata.to_dict(orient="records"):
        paths = [
            data_path(
                root,
                info,
                int(record["data/chunk_index"]),
                int(record["data/file_index"]),
            )
        ]
        for video_key in (base_video_key, wrist_video_key):
            paths.append(
                video_path(
                    root,
                    info,
                    video_key,
                    int(record[metadata_column(video_key, "chunk_index")]),
                    int(record[metadata_column(video_key, "file_index")]),
                )
            )
        available.append(all(is_available(path) for path in paths))

    filtered = metadata.loc[available].copy()
    if filtered.empty:
        raise ValueError("No episodes have all required data and video files available locally")
    return filtered


def export_episode_rows(
    root: Path,
    state_dir: Path,
    state_key: str,
    base_video_key: str,
    wrist_video_key: str,
    max_episodes: int | None = None,
    seed: int = 42,
    available_only: bool = False,
    include_episode_indices: set[int] | None = None,
) -> list[dict]:
    info = load_info(root)
    metadata = episode_metadata(root)
    missing = required_metadata_columns(base_video_key, wrist_video_key).difference(metadata.columns)
    if missing:
        raise KeyError(f"Episode metadata is missing columns: {sorted(missing)}")
    if metadata["episode_index"].duplicated().any():
        duplicates = metadata.loc[metadata["episode_index"].duplicated(), "episode_index"].head().tolist()
        raise ValueError(f"Official episode metadata contains duplicate episode_index values: {duplicates}")
    if available_only:
        original_count = len(metadata)
        metadata = filter_available_metadata(
            metadata,
            root,
            info,
            base_video_key,
            wrist_video_key,
        )
        print(f"available episodes: {len(metadata)}/{original_count}")
    include_episode_indices = include_episode_indices or set()
    available_indices = set(metadata["episode_index"].astype(int).tolist())
    missing_includes = include_episode_indices.difference(available_indices)
    if missing_includes:
        raise ValueError(
            "Included episode indices are unavailable after filtering: "
            f"{sorted(missing_includes)[:10]}"
        )
    if max_episodes is not None:
        if max_episodes <= 0:
            raise ValueError("max_episodes must be positive")
        if max_episodes < len(include_episode_indices):
            raise ValueError(
                f"max_episodes={max_episodes} is smaller than the "
                f"{len(include_episode_indices)} required included episodes"
            )
        if max_episodes < len(metadata):
            included = metadata[metadata["episode_index"].isin(include_episode_indices)]
            candidates = metadata[~metadata["episode_index"].isin(include_episode_indices)]
            additional_count = max_episodes - len(included)
            additional = candidates.sample(n=additional_count, random_state=seed)
            metadata = pd.concat([included, additional], ignore_index=True).sort_values("episode_index")

    state_dir.mkdir(parents=True, exist_ok=True)
    base_fps = video_fps(info, base_video_key)
    wrist_fps = video_fps(info, wrist_video_key)
    rows: list[dict] = []

    grouped_metadata: dict[tuple[int, int], list[dict]] = defaultdict(list)
    for record in metadata.to_dict(orient="records"):
        key = (int(record["data/chunk_index"]), int(record["data/file_index"]))
        grouped_metadata[key].append(record)

    for (data_chunk_index, data_file_index), records in sorted(grouped_metadata.items()):
        parquet_path = data_path(root, info, data_chunk_index, data_file_index)
        if not parquet_path.exists():
            raise FileNotFoundError(parquet_path)
        df = pd.read_parquet(parquet_path)
        required_data = {state_key, "episode_index", "frame_index"}
        missing_data = required_data.difference(df.columns)
        if missing_data:
            raise KeyError(f"{parquet_path} missing columns: {sorted(missing_data)}")

        episode_groups = {
            int(episode_index): group.sort_values("frame_index")
            for episode_index, group in df.groupby("episode_index", sort=False)
        }
        for record in sorted(records, key=lambda item: int(item["episode_index"])):
            episode_index = int(record["episode_index"])
            if episode_index not in episode_groups:
                raise ValueError(
                    f"Episode {episode_index} is assigned to {parquet_path} by metadata "
                    "but is absent from that data shard."
                )
            group = episode_groups[episode_index]
            length = int(record["length"])
            if len(group) != length:
                raise ValueError(
                    f"Episode {episode_index} length mismatch: metadata={length}, "
                    f"data={len(group)} in {parquet_path}"
                )

            frame_indices = group["frame_index"].to_numpy()
            if not np.array_equal(frame_indices, np.arange(length)):
                raise ValueError(f"Episode {episode_index} frame_index is not contiguous from zero")

            states = np.stack(group[state_key].to_numpy()).astype(np.float32)
            episode_id = f"episode_{episode_index:06d}"
            state_path = state_dir / f"{episode_id}_state.npy"
            np.save(state_path, states)

            def view_fields(video_key: str, fps: float) -> tuple[Path, int, int, int, float, float]:
                chunk = int(record[metadata_column(video_key, "chunk_index")])
                file_index = int(record[metadata_column(video_key, "file_index")])
                from_timestamp = float(record[metadata_column(video_key, "from_timestamp")])
                to_timestamp = float(record[metadata_column(video_key, "to_timestamp")])
                path = video_path(root, info, video_key, chunk, file_index)
                if not path.exists() or path.stat().st_size == 0:
                    raise FileNotFoundError(path)
                frame_offset = int(round(from_timestamp * fps))
                timestamp_frames = int(round((to_timestamp - from_timestamp) * fps))
                if abs(timestamp_frames - length) > 1:
                    raise ValueError(
                        f"Episode {episode_index} {video_key} duration maps to "
                        f"{timestamp_frames} frames, expected {length}"
                    )
                return path, frame_offset, chunk, file_index, from_timestamp, to_timestamp

            base = view_fields(base_video_key, base_fps)
            wrist = view_fields(wrist_video_key, wrist_fps)
            rows.append(
                {
                    "episode_id": episode_id,
                    "episode_index": episode_index,
                    "num_frames": length,
                    "data_chunk_index": data_chunk_index,
                    "data_file_index": data_file_index,
                    "dataset_from_index": int(record["dataset_from_index"]),
                    "dataset_to_index": int(record["dataset_to_index"]),
                    "base_video_key": base_video_key,
                    "wrist_video_key": wrist_video_key,
                    "base_video_path": str(base[0]),
                    "wrist_video_path": str(wrist[0]),
                    "base_frame_offset": base[1],
                    "wrist_frame_offset": wrist[1],
                    "base_video_chunk_index": base[2],
                    "base_video_file_index": base[3],
                    "wrist_video_chunk_index": wrist[2],
                    "wrist_video_file_index": wrist[3],
                    "base_from_timestamp": base[4],
                    "base_to_timestamp": base[5],
                    "wrist_from_timestamp": wrist[4],
                    "wrist_to_timestamp": wrist[5],
                    "robot_state_path": str(state_path),
                }
            )

    return sorted(rows, key=lambda row: row["episode_index"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lerobot-root", required=True)
    parser.add_argument("--out", default="data/manifests/episodes.jsonl")
    parser.add_argument("--state-dir", default="data/processed/lerobot_states")
    parser.add_argument("--state-key", default="observation.state")
    parser.add_argument("--base-video-key", default="observation.images.base")
    parser.add_argument("--wrist-video-key", default="observation.images.wrist")
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=None,
        help="Reproducibly sample at most this many episodes from official metadata.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--available-only",
        action="store_true",
        help="Only use episodes whose data shard and both video shards exist and are non-empty.",
    )
    parser.add_argument(
        "--include-manifest",
        default=None,
        help="Require all episode_index values from this JSONL manifest in the output sample.",
    )
    args = parser.parse_args()

    include_episode_indices: set[int] = set()
    if args.include_manifest:
        with Path(args.include_manifest).open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    include_episode_indices.add(int(json.loads(line)["episode_index"]))
        print(f"required included episodes: {len(include_episode_indices)}")

    root = Path(args.lerobot_root)
    rows = export_episode_rows(
        root,
        Path(args.state_dir),
        args.state_key,
        args.base_video_key,
        args.wrist_video_key,
        args.max_episodes,
        args.seed,
        args.available_only,
        include_episode_indices,
    )
    write_jsonl(args.out, rows)
    print(f"wrote {len(rows)} episode rows -> {args.out}")


if __name__ == "__main__":
    main()
