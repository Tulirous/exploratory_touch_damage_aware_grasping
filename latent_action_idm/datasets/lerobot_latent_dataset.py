from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from latent_action_idm.utils import read_jsonl


def load_visual_stats(path: str | Path | None) -> tuple[np.ndarray, np.ndarray] | None:
    if path is None:
        return None
    data = np.load(path)
    return data["mean"].astype(np.float32), data["std"].astype(np.float32)


def compute_visual_stats(
    manifest_path: str | Path,
    output_path: str | Path,
    eps: float = 1e-6,
) -> None:
    rows = read_jsonl(manifest_path)
    if not rows:
        raise ValueError(f"No rows found in {manifest_path}")

    count = 0
    total = None
    total_sq = None
    for row in rows:
        data = np.load(row["latent_path"])
        visual = np.concatenate([data["visual_t"], data["visual_future"]], axis=0).astype(np.float64)
        if total is None:
            total = np.zeros((visual.shape[-1],), dtype=np.float64)
            total_sq = np.zeros((visual.shape[-1],), dtype=np.float64)
        total += visual.sum(axis=0)
        total_sq += np.square(visual).sum(axis=0)
        count += visual.shape[0]

    if total is None or total_sq is None or count == 0:
        raise ValueError(f"Unable to compute stats from {manifest_path}")
    mean = total / count
    var = np.maximum(total_sq / count - np.square(mean), eps * eps)
    std = np.sqrt(var)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, mean=mean.astype(np.float32), std=std.astype(np.float32), count=count)


class LatentIDMDataset(Dataset):
    """Window-level dataset for latent-action IDM training."""

    def __init__(self, manifest_path: str | Path, visual_stats_path: str | Path | None = None) -> None:
        self.manifest_path = Path(manifest_path)
        self.rows = read_jsonl(self.manifest_path)
        if not self.rows:
            raise ValueError(f"No rows found in {self.manifest_path}")
        self.visual_stats = load_visual_stats(visual_stats_path)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        data = np.load(row["latent_path"])

        visual_t = data["visual_t"].astype(np.float32)
        visual_future = data["visual_future"].astype(np.float32)
        if self.visual_stats is not None:
            mean, std = self.visual_stats
            visual_t = (visual_t - mean) / std
            visual_future = (visual_future - mean) / std

        return {
            "episode_id": row["episode_id"],
            "sample_id": row["sample_id"],
            "t_index": int(row["t_index"]),
            "future_index": int(row["future_index"]),
            "visual_t": torch.from_numpy(visual_t).float(),
            "visual_future": torch.from_numpy(visual_future).float(),
            "state_t": torch.from_numpy(data["state_t"]).float(),
            "state_future": torch.from_numpy(data["state_future"]).float(),
        }
