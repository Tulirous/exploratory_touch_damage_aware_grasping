from __future__ import annotations

import torch
import torch.nn as nn

from .hand_world_model import HandWorldModelDecoder
from .inverse_dynamics import HandInverseDynamics


class Stage1HandLWM(nn.Module):
    """Stage-1 latent action model with SMPL-X/MANO hand states as modality."""

    def __init__(
        self,
        state_dim: int = 24,
        latent_action_dim: int = 64,
        hidden_dim: int = 256,
        context_length: int = 4,
        future_length: int = 12,
        encoder_layers: int = 4,
        decoder_layers: int = 4,
        num_heads: int = 8,
        ffn_dim: int = 1024,
        dropout: float = 0.1,
        num_hand_joints: int = 21,
        num_contact_points: int = 5,
        residual_prediction: bool = True,
        wrist_constant_velocity_anchor: bool = False,
    ) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.context_length = context_length
        self.future_length = future_length
        self.inverse_dynamics = HandInverseDynamics(
            state_dim=state_dim,
            latent_action_dim=latent_action_dim,
            hidden_dim=hidden_dim,
            num_layers=encoder_layers,
            num_heads=num_heads,
            ffn_dim=ffn_dim,
            dropout=dropout,
            max_context_length=context_length,
            max_future_length=future_length,
        )
        self.hand_world_model = HandWorldModelDecoder(
            state_dim=state_dim,
            latent_action_dim=latent_action_dim,
            hidden_dim=hidden_dim,
            future_length=future_length,
            num_layers=decoder_layers,
            num_heads=num_heads,
            ffn_dim=ffn_dim,
            dropout=dropout,
            max_context_length=context_length,
            num_hand_joints=num_hand_joints,
            num_contact_points=num_contact_points,
            residual_prediction=residual_prediction,
            wrist_constant_velocity_anchor=wrist_constant_velocity_anchor,
        )

    def forward(
        self,
        hand_context: torch.Tensor,
        hand_future: torch.Tensor,
        sample: bool | None = None,
    ) -> dict[str, torch.Tensor]:
        posterior = self.encode_latent_action(
            hand_context,
            hand_future,
            sample=sample,
        )
        prediction = self.decode_future(
            hand_context,
            posterior["latent_action"],
        )
        return {**posterior, **prediction}

    def encode_latent_action(
        self,
        hand_context: torch.Tensor,
        hand_future: torch.Tensor,
        sample: bool | None = None,
    ) -> dict[str, torch.Tensor]:
        return self.inverse_dynamics(hand_context, hand_future, sample=sample)

    def decode_future(
        self,
        hand_context: torch.Tensor,
        latent_action: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        return self.hand_world_model(hand_context, latent_action)
