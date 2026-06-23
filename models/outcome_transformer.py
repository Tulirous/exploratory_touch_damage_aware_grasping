from __future__ import annotations

import torch
from torch import nn


class OutcomeTransformer(nn.Module):
    """DiT-style token fusion for action-conditioned outcome prediction."""

    def __init__(
        self,
        visual_dim: int,
        text_dim: int | None,
        tactile_dim: int,
        action_dim: int,
        hidden_dim: int,
        num_layers: int,
        num_heads: int,
        num_binary: int,
        num_regression: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.visual_proj = nn.Linear(visual_dim, hidden_dim)
        self.text_proj = nn.Linear(text_dim, hidden_dim) if text_dim is not None else None
        self.tactile_proj = nn.Linear(tactile_dim, hidden_dim)
        self.action_proj = nn.Linear(action_dim, hidden_dim)
        self.cls = nn.Parameter(torch.zeros(1, 1, hidden_dim))

        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.fusion = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.binary_head = nn.Linear(hidden_dim, num_binary)
        self.regression_head = nn.Linear(hidden_dim, num_regression)

    def forward(
        self,
        visual_latent: torch.Tensor,
        text_latent: torch.Tensor | None,
        tactile_latent: torch.Tensor,
        action_latent: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        # visual_latent: [B, D] or [B, N, D]
        if visual_latent.ndim == 2:
            visual_tokens = self.visual_proj(visual_latent).unsqueeze(1)
        else:
            visual_tokens = self.visual_proj(visual_latent)

        tactile_token = self.tactile_proj(tactile_latent).unsqueeze(1)
        text_token = None
        if text_latent is not None:
            if self.text_proj is None:
                raise ValueError("text_latent was provided but OutcomeTransformer was built without text_dim")
            text_token = self.text_proj(text_latent).unsqueeze(1)

        has_candidates = action_latent.ndim == 3
        if has_candidates:
            outputs = []
            for k in range(action_latent.shape[1]):
                outputs.append(
                    self._forward_one_action(
                        visual_tokens,
                        text_token,
                        tactile_token,
                        action_latent[:, k],
                    )
                )
            return {
                "binary_logits": torch.stack([o["binary_logits"] for o in outputs], dim=1),
                "regression": torch.stack([o["regression"] for o in outputs], dim=1),
            }

        return self._forward_one_action(visual_tokens, text_token, tactile_token, action_latent)

    def _forward_one_action(
        self,
        visual_tokens: torch.Tensor,
        text_token: torch.Tensor | None,
        tactile_token: torch.Tensor,
        action_latent: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        batch = visual_tokens.shape[0]
        cls = self.cls.expand(batch, -1, -1)
        action_token = self.action_proj(action_latent).unsqueeze(1)
        pieces = [cls, visual_tokens]
        if text_token is not None:
            pieces.append(text_token)
        pieces.extend([tactile_token, action_token])
        tokens = torch.cat(pieces, dim=1)
        fused = self.fusion(tokens)[:, 0]
        return {
            "binary_logits": self.binary_head(fused),
            "regression": self.regression_head(fused),
        }
