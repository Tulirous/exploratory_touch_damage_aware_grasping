from __future__ import annotations

import torch
import torch.nn as nn


class HandSequenceEmbedding(nn.Module):
    """Embed a fixed-length sequence of structured hand-state vectors."""

    def __init__(
        self,
        state_dim: int,
        hidden_dim: int,
        max_sequence_length: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.max_sequence_length = max_sequence_length
        self.input_proj = nn.Linear(state_dim, hidden_dim)
        self.position = nn.Parameter(torch.zeros(1, max_sequence_length, hidden_dim))
        self.dropout = nn.Dropout(dropout)
        nn.init.normal_(self.position, std=0.02)

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        ensure_hand_sequence(sequence, self.state_dim)
        length = sequence.shape[1]
        if length > self.max_sequence_length:
            raise ValueError(
                f"sequence length {length} exceeds max_sequence_length="
                f"{self.max_sequence_length}"
            )
        return self.dropout(self.input_proj(sequence) + self.position[:, :length])


def ensure_hand_sequence(sequence: torch.Tensor, state_dim: int | None = None) -> None:
    if sequence.ndim != 3:
        raise ValueError(f"expected hand sequence [B, T, D], got {tuple(sequence.shape)}")
    if state_dim is not None and sequence.shape[-1] != state_dim:
        raise ValueError(
            f"expected hand state dimension {state_dim}, got {sequence.shape[-1]}"
        )


def sample_gaussian(
    mean: torch.Tensor,
    log_variance: torch.Tensor,
    sample: bool,
) -> torch.Tensor:
    if not sample:
        return mean
    std = torch.exp(0.5 * log_variance.clamp(min=-10.0, max=10.0))
    return mean + torch.randn_like(std) * std
