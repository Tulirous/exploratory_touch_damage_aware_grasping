from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class HandSequenceDataset(Dataset):
    """Load fixed windows from offline SMPL-X/MANO hand-track NPZ files."""

    def __init__(
        self,
        manifest_path: str | Path,
        context_length: int,
        future_length: int,
        state_dim: int = 24,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.context_length = context_length
        self.future_length = future_length
        self.state_dim = state_dim
        with self.manifest_path.open("r", encoding="utf-8") as handle:
            self.rows = [json.loads(line) for line in handle if line.strip()]
        if not self.rows:
            raise ValueError(f"empty hand manifest: {self.manifest_path}")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        row = self.rows[index]
        sequence_path = self._resolve_path(row["sequence_path"])
        start = int(row["start_index"])
        split = start + self.context_length
        stop = split + self.future_length

        with np.load(sequence_path, allow_pickle=False) as data:
            hand_state = np.asarray(data["hand_state"], dtype=np.float32)
            if hand_state.ndim != 2 or hand_state.shape[1] != self.state_dim:
                raise ValueError(
                    f"{sequence_path}: expected hand_state [T, {self.state_dim}], "
                    f"got {hand_state.shape}"
                )
            if stop > hand_state.shape[0]:
                raise IndexError(
                    f"{sequence_path}: window stop {stop} exceeds length {hand_state.shape[0]}"
                )
            item: dict[str, torch.Tensor | str] = {
                "episode_id": str(row["episode_id"]),
                "handedness": str(row.get("handedness", "right")),
                "hand_context": torch.from_numpy(hand_state[start:split].copy()),
                "hand_future": torch.from_numpy(hand_state[split:stop].copy()),
            }
            if "joints_3d" in data:
                joints = np.asarray(data["joints_3d"], dtype=np.float32)
                item["joints_future"] = torch.from_numpy(joints[split:stop].copy())
            if "contact" in data:
                contact = np.asarray(data["contact"], dtype=np.float32)
                item["contact_future"] = torch.from_numpy(contact[split:stop].copy())
        return item

    def _resolve_path(self, value: str) -> Path:
        path = Path(value)
        if path.is_absolute():
            return path
        project_relative = Path.cwd() / path
        if project_relative.exists():
            return project_relative
        return self.manifest_path.parent / path
