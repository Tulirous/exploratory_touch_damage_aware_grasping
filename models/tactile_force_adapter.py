from __future__ import annotations

import torch
from torch import nn


class TactileForceAdapter(nn.Module):
    """Encode pressure, force and gripper time series into a contact latent."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=8,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.output_proj = nn.Linear(hidden_dim, output_dim)

    def forward(self, tactile_seq: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(tactile_seq)
        x = self.encoder(x)
        x = x.mean(dim=1)
        return self.output_proj(x)

