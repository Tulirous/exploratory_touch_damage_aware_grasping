from __future__ import annotations

import torch
import torch.nn as nn

from .common import HandSequenceEmbedding, ensure_hand_sequence, sample_gaussian


class HandInverseDynamics(nn.Module):
    """Posterior q(z_h | H_context, H_future) for structured hand motion."""

    def __init__(
        self,
        state_dim: int = 24,
        latent_action_dim: int = 64,
        hidden_dim: int = 256,
        num_layers: int = 4,
        num_heads: int = 8,
        ffn_dim: int = 1024,
        dropout: float = 0.1,
        max_context_length: int = 8,
        max_future_length: int = 16,
        wrist_aware_auxiliary_head: bool = False,
        window_local_wrist_translation: bool = False,
    ) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.window_local_wrist_translation = window_local_wrist_translation
        self.context_embed = HandSequenceEmbedding(
            state_dim, hidden_dim, max_context_length, dropout
        )
        self.future_embed = HandSequenceEmbedding(
            state_dim, hidden_dim, max_future_length, dropout
        )
        self.posterior_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.context_type = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.future_type = nn.Parameter(torch.zeros(1, 1, hidden_dim))

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
        self.posterior = nn.Linear(hidden_dim, latent_action_dim * 2)
        self.wrist_correction_head = (
            nn.Linear(latent_action_dim, max_future_length * 3)
            if wrist_aware_auxiliary_head
            else None
        )
        self.max_future_length = max_future_length
        nn.init.normal_(self.posterior_token, std=0.02)
        nn.init.normal_(self.context_type, std=0.02)
        nn.init.normal_(self.future_type, std=0.02)

    def forward(
        self,
        hand_context: torch.Tensor,
        hand_future: torch.Tensor,
        sample: bool | None = None,
    ) -> dict[str, torch.Tensor]:
        ensure_hand_sequence(hand_context, self.state_dim)
        ensure_hand_sequence(hand_future, self.state_dim)
        if self.window_local_wrist_translation:
            hand_context, hand_future = self._canonicalize_wrist_translation(
                hand_context,
                hand_future,
            )
        batch_size = hand_context.shape[0]
        tokens = torch.cat(
            [
                self.posterior_token.expand(batch_size, -1, -1),
                self.context_embed(hand_context) + self.context_type,
                self.future_embed(hand_future) + self.future_type,
            ],
            dim=1,
        )
        encoded = self.encoder(tokens)
        mean, log_variance = self.posterior(encoded[:, 0]).chunk(2, dim=-1)
        should_sample = self.training if sample is None else sample
        latent_action = sample_gaussian(mean, log_variance, should_sample)
        outputs = {
            "latent_action": latent_action,
            "latent_mean": mean,
            "latent_log_variance": log_variance,
            "posterior_token": encoded[:, 0],
        }
        if self.wrist_correction_head is not None:
            outputs["predicted_wrist_cv_correction"] = self.wrist_correction_head(
                mean
            ).view(batch_size, self.max_future_length, 3)
        return outputs

    @staticmethod
    def _canonicalize_wrist_translation(
        hand_context: torch.Tensor,
        hand_future: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Express wrist positions relative to the last context position."""

        wrist_origin = hand_context[:, -1:, :3]
        canonical_context = torch.cat(
            [hand_context[..., :3] - wrist_origin, hand_context[..., 3:]],
            dim=-1,
        )
        canonical_future = torch.cat(
            [hand_future[..., :3] - wrist_origin, hand_future[..., 3:]],
            dim=-1,
        )
        return canonical_context, canonical_future
