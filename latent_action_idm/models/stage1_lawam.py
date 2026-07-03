from __future__ import annotations

import torch
import torch.nn as nn

from .common import MLP
from .inverse_dynamics import InverseDynamicsTransformer
from .latent_world_model import LatentWorldModelDecoder


class Stage1LaWAM(nn.Module):
    """Composable Stage-1 LaWAM model.

    Modules are intentionally exposed as public attributes:

    - inverse_dynamics: q(z | u_t, u_T, state_t)
    - latent_world_model: p(u_T | u_t, z)
    - state_predictor: auxiliary state consistency head
    """

    def __init__(
        self,
        visual_token_dim: int,
        state_dim: int,
        latent_action_dim: int = 128,
        hidden_dim: int = 768,
        encoder_layers: int = 8,
        decoder_layers: int = 8,
        num_heads: int = 12,
        ffn_dim: int = 3072,
        dropout: float = 0.1,
        max_visual_tokens: int = 512,
    ) -> None:
        super().__init__()
        self.visual_token_dim = visual_token_dim
        self.state_dim = state_dim
        self.latent_action_dim = latent_action_dim
        self.hidden_dim = hidden_dim

        self.inverse_dynamics = InverseDynamicsTransformer(
            visual_token_dim=visual_token_dim,
            state_dim=state_dim,
            latent_action_dim=latent_action_dim,
            hidden_dim=hidden_dim,
            num_layers=encoder_layers,
            num_heads=num_heads,
            ffn_dim=ffn_dim,
            dropout=dropout,
            max_visual_tokens=max_visual_tokens,
        )
        self.latent_world_model = LatentWorldModelDecoder(
            visual_token_dim=visual_token_dim,
            latent_action_dim=latent_action_dim,
            hidden_dim=hidden_dim,
            num_layers=decoder_layers,
            num_heads=num_heads,
            ffn_dim=ffn_dim,
            dropout=dropout,
            max_visual_tokens=max_visual_tokens,
        )
        self.state_predictor = MLP(
            state_dim + latent_action_dim,
            hidden_dim,
            state_dim,
            dropout=dropout,
        )

    def forward(
        self,
        visual_t: torch.Tensor,
        visual_future: torch.Tensor,
        state_t: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        posterior = self.encode_latent_action(visual_t, visual_future, state_t)
        latent_action = posterior["latent_action"]
        predicted_visual_future = self.decode_future(visual_t, latent_action)
        predicted_state_future = self.predict_state_future(state_t, latent_action)
        return {
            **posterior,
            "predicted_state_future": predicted_state_future,
            "predicted_visual_future": predicted_visual_future,
        }

    def encode_latent_action(
        self,
        visual_t: torch.Tensor,
        visual_future: torch.Tensor,
        state_t: torch.Tensor,
        sample: bool | None = None,
    ) -> dict[str, torch.Tensor]:
        return self.inverse_dynamics(visual_t, visual_future, state_t, sample=sample)

    def decode_future(self, visual_t: torch.Tensor, latent_action: torch.Tensor) -> torch.Tensor:
        return self.latent_world_model(visual_t, latent_action)

    def predict_state_future(self, state_t: torch.Tensor, latent_action: torch.Tensor) -> torch.Tensor:
        state_delta = self.state_predictor(torch.cat([state_t, latent_action], dim=-1))
        return state_t + state_delta


# Backward-compatible name used by existing training scripts.
LatentActionIDM = Stage1LaWAM

