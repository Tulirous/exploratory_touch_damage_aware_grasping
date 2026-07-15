from __future__ import annotations

import torch
import torch.nn as nn

from .common import HandSequenceEmbedding, ensure_hand_sequence, sample_gaussian


class HandLatentActionPrior(nn.Module):
    """Current-only prior p(z_h | H_context)."""

    def __init__(
        self,
        state_dim: int = 24,
        latent_action_dim: int = 64,
        hidden_dim: int = 128,
        num_layers: int = 2,
        num_heads: int = 4,
        ffn_dim: int = 512,
        dropout: float = 0.1,
        max_context_length: int = 4,
    ) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.context_embed = HandSequenceEmbedding(
            state_dim, hidden_dim, max_context_length, dropout
        )
        self.prior_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(hidden_dim),
        )
        self.distribution = nn.Linear(hidden_dim, latent_action_dim * 2)
        nn.init.normal_(self.prior_token, std=0.02)

    def forward(
        self,
        hand_context: torch.Tensor,
        sample: bool | None = None,
    ) -> dict[str, torch.Tensor]:
        ensure_hand_sequence(hand_context, self.state_dim)
        batch_size = hand_context.shape[0]
        tokens = torch.cat(
            [
                self.prior_token.expand(batch_size, -1, -1),
                self.context_embed(hand_context),
            ],
            dim=1,
        )
        encoded = self.encoder(tokens)
        mean, log_variance = self.distribution(encoded[:, 0]).chunk(2, dim=-1)
        should_sample = self.training if sample is None else sample
        latent_action = sample_gaussian(mean, log_variance, should_sample)
        return {
            "latent_action": latent_action,
            "latent_mean": mean,
            "latent_log_variance": log_variance,
            "prior_token": encoded[:, 0],
        }
