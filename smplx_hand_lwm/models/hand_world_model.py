from __future__ import annotations

import torch
import torch.nn as nn

from .common import HandSequenceEmbedding, ensure_hand_sequence


class HandWorldModelDecoder(nn.Module):
    """Forward model p(H_future | H_context, z_h).

    The predicted structured state replaces LaWAM's predicted visual feature.
    Joint positions and fingertip contacts are auxiliary training heads; they do
    not require an SMPL-X layer inside the model.
    """

    def __init__(
        self,
        state_dim: int = 24,
        latent_action_dim: int = 64,
        hidden_dim: int = 256,
        future_length: int = 12,
        num_layers: int = 4,
        num_heads: int = 8,
        ffn_dim: int = 1024,
        dropout: float = 0.1,
        max_context_length: int = 8,
        num_hand_joints: int = 21,
        num_contact_points: int = 5,
        residual_prediction: bool = True,
        wrist_constant_velocity_anchor: bool = False,
    ) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.future_length = future_length
        self.num_hand_joints = num_hand_joints
        self.num_contact_points = num_contact_points
        self.residual_prediction = residual_prediction
        self.wrist_constant_velocity_anchor = wrist_constant_velocity_anchor

        self.context_embed = HandSequenceEmbedding(
            state_dim, hidden_dim, max_context_length, dropout
        )
        self.future_queries = nn.Parameter(
            torch.zeros(1, future_length, hidden_dim)
        )
        self.latent_proj = nn.Sequential(
            nn.LayerNorm(latent_action_dim),
            nn.Linear(latent_action_dim, hidden_dim),
        )
        layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(
            layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(hidden_dim),
        )
        self.state_head = nn.Linear(hidden_dim, state_dim)
        self.joint_head = nn.Linear(hidden_dim, num_hand_joints * 3)
        self.contact_head = nn.Linear(hidden_dim, num_contact_points)
        nn.init.normal_(self.future_queries, std=0.02)

    def forward(
        self,
        hand_context: torch.Tensor,
        latent_action: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        ensure_hand_sequence(hand_context, self.state_dim)
        batch_size = hand_context.shape[0]
        memory = self.context_embed(hand_context)
        queries = self.future_queries.expand(batch_size, -1, -1)
        queries = queries + self.latent_proj(latent_action).unsqueeze(1)
        decoded = self.decoder(tgt=queries, memory=memory)

        predicted_state = self.state_head(decoded)
        if self.residual_prediction:
            base = hand_context[:, -1:, :].expand(
                -1, self.future_length, -1
            ).clone()
            if self.wrist_constant_velocity_anchor and hand_context.shape[1] >= 2:
                wrist_velocity = (
                    hand_context[:, -1, :3] - hand_context[:, -2, :3]
                )
                steps = torch.arange(
                    1,
                    self.future_length + 1,
                    device=hand_context.device,
                    dtype=hand_context.dtype,
                ).view(1, self.future_length, 1)
                base[..., :3] = (
                    hand_context[:, -1:, :3]
                    + steps * wrist_velocity[:, None, :]
                )
            predicted_state = predicted_state + base
        predicted_joints = self.joint_head(decoded).view(
            batch_size,
            self.future_length,
            self.num_hand_joints,
            3,
        )
        predicted_contact_logits = self.contact_head(decoded)
        return {
            "predicted_hand_future": predicted_state,
            "predicted_joints_future": predicted_joints,
            "predicted_contact_logits": predicted_contact_logits,
            "future_tokens": decoded,
        }
