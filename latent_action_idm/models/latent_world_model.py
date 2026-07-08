from __future__ import annotations

import torch
import torch.nn as nn

from .common import AdaLNBlock, VisualTokenProjector, ensure_visual_tokens


class LatentWorldModelDecoder(nn.Module):
    """LaWM decoder p(u_future | u_current, z)."""

    def __init__(
        self,
        visual_token_dim: int,
        latent_action_dim: int = 128,
        hidden_dim: int = 768,
        num_layers: int = 8,
        num_heads: int = 12,
        ffn_dim: int = 3072,
        dropout: float = 0.1,
        max_visual_tokens: int = 512,
        num_views: int = 0,
        residual_prediction: bool = False,
    ) -> None:
        super().__init__()
        self.visual_token_dim = visual_token_dim
        self.latent_action_dim = latent_action_dim
        self.hidden_dim = hidden_dim
        self.residual_prediction = residual_prediction
        self.visual_projector = VisualTokenProjector(
            visual_token_dim=visual_token_dim,
            hidden_dim=hidden_dim,
            max_visual_tokens=max_visual_tokens,
            num_views=num_views,
        )
        self.current_type = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.latent_to_hidden = nn.Sequential(
            nn.LayerNorm(latent_action_dim),
            nn.Linear(latent_action_dim, hidden_dim),
        )
        self.blocks = nn.ModuleList(
            [
                AdaLNBlock(
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    ffn_dim=ffn_dim,
                    cond_dim=hidden_dim,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, visual_token_dim)
        nn.init.normal_(self.current_type, std=0.02)

    def forward(self, visual_t: torch.Tensor, latent_action: torch.Tensor) -> torch.Tensor:
        visual_t = ensure_visual_tokens(visual_t)
        tokens = self.visual_projector(visual_t) + self.current_type
        cond = self.latent_to_hidden(latent_action)
        for block in self.blocks:
            tokens = block(tokens, cond)
        delta_or_future = self.output_proj(self.norm(tokens))
        if self.residual_prediction:
            return visual_t + delta_or_future
        return delta_or_future
