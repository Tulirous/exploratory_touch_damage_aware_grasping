"""Build FastWAM-Fragile JSONL manifests from GELLO/LeRobot metadata.

This script uses explicit episode metadata instead of guessing the internal
LeRobot directory layout. LeRobot layouts can change between versions, while
this project's training code only needs stable file paths.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def attach_visual_latent_path(row: dict, latent_dir: Path) -> dict:
    episode_id = row["episode_id"]
    row = dict(row)
    row.setdefault("visual_latent_path", str(latent_dir / f"{episode_id}_visual.npy"))
    return row


def split_rows(rows: list[dict], val_ratio: float) -> tuple[list[dict], list[dict]]:
    if not 0.0 <= val_ratio < 1.0:
        raise ValueError("--val-ratio must be in [0, 1).")
    val_count = int(round(len(rows) * val_ratio))
    if val_count == 0:
        return rows, []
    return rows[:-val_count], rows[-val_count:]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", required=True, help="Episode metadata JSONL.")
    parser.add_argument("--latent-dir", default="data/latents")
    parser.add_argument("--train-out", default="data/manifests/train.jsonl")
    parser.add_argument("--val-out", default="data/manifests/val.jsonl")
    parser.add_argument("--val-ratio", type=float, default=0.2)
    args = parser.parse_args()

    rows = read_jsonl(Path(args.episodes))
    rows = [attach_visual_latent_path(row, Path(args.latent_dir)) for row in rows]
    train_rows, val_rows = split_rows(rows, args.val_ratio)
    write_jsonl(Path(args.train_out), train_rows)
    write_jsonl(Path(args.val_out), val_rows)

    print(f"wrote {len(train_rows)} train rows -> {args.train_out}")
    print(f"wrote {len(val_rows)} val rows -> {args.val_out}")


if __name__ == "__main__":
    main()

