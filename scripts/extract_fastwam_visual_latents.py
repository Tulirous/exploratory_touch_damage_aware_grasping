"""Extract Fast-WAM/Wan visual latents for this project's JSONL manifests.

Run this on the GPU server where the official FastWAM repository is installed.
The script does not modify FastWAM; it imports it via --fastwam-root.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets.video_pair_reader import read_base_wrist_frames
from models.visual_backbone_adapters import FastWAMWanLatentAdapter


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, help="JSONL manifest with base/wrist video paths.")
    parser.add_argument("--fastwam-root", required=True, help="Path to official FastWAM repository, e.g. ~/fastwam.")
    parser.add_argument("--ckpt", required=True, help="FastWAM released checkpoint path.")
    parser.add_argument("--task-config", default="libero_uncond_2cam224_1e-4")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--mixed-precision", default="bf16", choices=["bf16", "fp16", "fp32", "no"])
    parser.add_argument("--frame-index", type=int, default=0)
    parser.add_argument("--pool", default="mean", choices=["mean", "first"])
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    rows = read_jsonl(Path(args.manifest))
    adapter = FastWAMWanLatentAdapter(
        fastwam_root=args.fastwam_root,
        ckpt_path=args.ckpt,
        task_config=args.task_config,
        device=args.device,
        mixed_precision=args.mixed_precision,
        pool=args.pool,
    )

    for row in rows:
        output = Path(row["visual_latent_path"])
        if output.exists() and not args.overwrite:
            print(f"skip existing {row['episode_id']} -> {output}")
            continue

        base_frame, wrist_frame = read_base_wrist_frames(
            row["base_video_path"],
            row["wrist_video_path"],
            frame_index=args.frame_index,
        )
        prompt = row.get("task_instruction")
        latent = adapter.encode_frames(base_frame, wrist_frame, prompt=prompt)
        if latent.shape != (3072,):
            raise ValueError(f"Expected visual latent shape (3072,), got {latent.shape}")

        output.parent.mkdir(parents=True, exist_ok=True)
        np.save(output, latent.astype(np.float32, copy=False))
        print(f"saved {row['episode_id']} -> {output}")


if __name__ == "__main__":
    main()
