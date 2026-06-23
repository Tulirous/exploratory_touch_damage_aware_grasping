"""Extract visual latents for FastWAM-Fragile manifests.

Production path:
1. Read GELLO/LeRobot episode manifests.
2. Load a frozen visual/video backbone such as Fast-WAM/Wan, WALL, OpenPI, or
   a temporary ViT baseline.
3. Encode base/wrist videos or selected frames.
4. Save one visual_latent.npy file per episode.

Use ``--backbone dummy`` only to test the downstream training pipeline shape.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def extract_dummy_latent(dim: int, seed: int, episode_id: str) -> np.ndarray:
    rng = np.random.default_rng(seed + abs(hash(episode_id)) % 1_000_000)
    return rng.standard_normal(dim, dtype=np.float32)


def extract_backbone_latent(row: dict, backbone: str, dim: int, seed: int) -> np.ndarray:
    if backbone == "dummy":
        return extract_dummy_latent(dim, seed, row["episode_id"])
    raise NotImplementedError(
        f"Backbone '{backbone}' is not implemented yet. "
        "Implement Fast-WAM/Wan, WALL, OpenPI, or ViT loading here."
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--backbone", default="dummy")
    parser.add_argument("--latent-dim", type=int, default=3072)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rows = read_jsonl(Path(args.manifest))
    for row in rows:
        output = Path(row["visual_latent_path"])
        output.parent.mkdir(parents=True, exist_ok=True)
        latent = extract_backbone_latent(row, args.backbone, args.latent_dim, args.seed)
        np.save(output, latent)
        print(f"saved {row['episode_id']} -> {output}")


if __name__ == "__main__":
    main()

