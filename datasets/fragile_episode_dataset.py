from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset


class FragileEpisodeDataset(Dataset):
    """JSONL based dataset for FastWAM-Fragile outcome prediction."""

    def __init__(
        self,
        manifest_path: str | Path,
        binary_labels: list[str],
        regression_labels: list[str],
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.binary_labels = binary_labels
        self.regression_labels = regression_labels
        with self.manifest_path.open("r", encoding="utf-8") as f:
            self.items = [json.loads(line) for line in f if line.strip()]

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = self.items[index]
        labels = item.get("labels", {})

        visual_latent = self._load_array(item["visual_latent_path"])
        tactile = self._concat_touch_inputs(item)
        candidate_actions = self._load_array(item["candidate_action_path"])

        binary = torch.tensor(
            [float(labels.get(name, 0.0)) for name in self.binary_labels],
            dtype=torch.float32,
        )
        regression = torch.tensor(
            [float(labels.get(name, 0.0)) for name in self.regression_labels],
            dtype=torch.float32,
        )

        return {
            "episode_id": item["episode_id"],
            "task_instruction": item.get("task_instruction", ""),
            "visual_latent": torch.from_numpy(visual_latent).float(),
            "tactile": torch.from_numpy(tactile).float(),
            "candidate_actions": torch.from_numpy(candidate_actions).float(),
            "binary_labels": binary,
            "regression_labels": regression,
        }

    def _load_array(self, path: str) -> np.ndarray:
        return np.load(path)

    def _concat_touch_inputs(self, item: dict[str, Any]) -> np.ndarray:
        arrays = [
            self._load_array(item["tactile_path"]),
            self._load_array(item["force_path"]),
            self._load_array(item["gripper_path"]),
        ]
        min_len = min(arr.shape[0] for arr in arrays)
        arrays = [arr[:min_len] for arr in arrays]
        return np.concatenate(arrays, axis=-1)
