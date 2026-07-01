from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from latent_action_idm.utils import write_jsonl


def load_info(root: Path) -> dict:
    with (root / "meta/info.json").open("r", encoding="utf-8") as f:
        return json.load(f)


def video_path(root: Path, info: dict, video_key: str, file_index: int) -> Path:
    chunk_index = file_index // int(info.get("chunks_size", 1000))
    rel = info["video_path"].format(
        video_key=video_key,
        chunk_index=chunk_index,
        file_index=file_index,
    )
    return root / rel


def data_files(root: Path) -> list[Path]:
    return sorted((root / "data").glob("chunk-*/file-*.parquet"))


def parse_file_index(path: Path) -> int:
    return int(path.stem.split("-")[-1])


def export_episode_rows(root: Path, state_dir: Path, state_key: str) -> list[dict]:
    info = load_info(root)
    rows = []
    state_dir.mkdir(parents=True, exist_ok=True)

    for parquet_path in data_files(root):
        file_index = parse_file_index(parquet_path)
        df = pd.read_parquet(parquet_path)
        required = {state_key, "episode_index", "frame_index"}
        missing = required.difference(df.columns)
        if missing:
            raise KeyError(f"{parquet_path} missing columns: {sorted(missing)}")

        base_video = video_path(root, info, "observation.images.base", file_index)
        wrist_video = video_path(root, info, "observation.images.wrist", file_index)
        if not base_video.exists():
            raise FileNotFoundError(base_video)
        if not wrist_video.exists():
            raise FileNotFoundError(wrist_video)

        for episode_index, group in df.groupby("episode_index", sort=True):
            group = group.sort_values("frame_index")
            file_row_indices = group.index.to_numpy()
            video_frame_offset = int(file_row_indices.min())
            expected = np.arange(video_frame_offset, video_frame_offset + len(group))
            if not np.array_equal(file_row_indices, expected):
                raise ValueError(
                    f"Episode {episode_index} rows in {parquet_path} are not contiguous; "
                    "cannot map frame_index to chunk video frame offset safely."
                )

            states = np.stack(group[state_key].to_numpy()).astype(np.float32)
            episode_id = f"episode_{int(episode_index):06d}"
            state_path = state_dir / f"{episode_id}_state.npy"
            np.save(state_path, states)

            rows.append(
                {
                    "episode_id": episode_id,
                    "episode_index": int(episode_index),
                    "file_index": file_index,
                    "num_frames": int(len(group)),
                    "base_video_path": str(base_video),
                    "wrist_video_path": str(wrist_video),
                    "base_frame_offset": video_frame_offset,
                    "wrist_frame_offset": video_frame_offset,
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
    args = parser.parse_args()

    root = Path(args.lerobot_root)
    rows = export_episode_rows(root, Path(args.state_dir), args.state_key)
    write_jsonl(args.out, rows)
    print(f"wrote {len(rows)} episode rows -> {args.out}")


if __name__ == "__main__":
    main()

