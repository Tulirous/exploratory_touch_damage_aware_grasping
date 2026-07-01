from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from latent_action_idm.utils import read_jsonl


class LatentIDMDataset(Dataset):
    """Window-level dataset for latent-action IDM training."""

    def __init__(self, manifest_path: str | Path) -> None:
        self.manifest_path = Path(manifest_path)
        self.rows = read_jsonl(self.manifest_path)
        if not self.rows:
            raise ValueError(f"No rows found in {self.manifest_path}")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        data = np.load(row["latent_path"])

        return {
            "episode_id": row["episode_id"],
            "sample_id": row["sample_id"],
            "t_index": int(row["t_index"]),
            "future_index": int(row["future_index"]),
            "visual_t": torch.from_numpy(data["visual_t"]).float(),
            "visual_future": torch.from_numpy(data["visual_future"]).float(),
            "state_t": torch.from_numpy(data["state_t"]).float(),
            "state_future": torch.from_numpy(data["state_future"]).float(),
        }

