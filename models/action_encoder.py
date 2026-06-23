from __future__ import annotations

import torch
from torch import nn


class ActionChunkEncoder(nn.Module):
    """Encode candidate action chunks into action latents."""

    def __init__(self, action_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.step_encoder = nn.Sequential(
            nn.Linear(action_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.output_proj = nn.Linear(hidden_dim, output_dim)

    def forward(self, action_chunks: torch.Tensor) -> torch.Tensor:
        # action_chunks: [B, K, H, A] or [B, H, A]
        has_candidates = action_chunks.ndim == 4
        if not has_candidates:
            action_chunks = action_chunks.unsqueeze(1)

        x = self.step_encoder(action_chunks)
        x = x.mean(dim=2)
        x = self.output_proj(x)

        if not has_candidates:
            x = x.squeeze(1)
        return x

