from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import ensure_visual_tokens


class FutureTokenEvaluator(nn.Module):
    """Learnable evaluator for predicted future visual tokens.

    This module is intentionally label-ready. It can be trained later with
    task-success, goal-progress, slip, or damage labels. For now, analysis
    scripts use metric-based evaluation, while this module defines the network
    that will eventually score candidate futures during planning.
    """

    def __init__(
        self,
        visual_token_dim: int = 768,
        state_dim: int = 7,
        latent_action_dim: int = 128,
        hidden_dim: int = 512,
        num_layers: int = 4,
        num_heads: int = 8,
        ffn_dim: int = 2048,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.visual_token_dim = visual_token_dim
        self.state_dim = state_dim
        self.latent_action_dim = latent_action_dim
        self.hidden_dim = hidden_dim

        self.token_proj = nn.Linear(visual_token_dim, hidden_dim)
        self.delta_proj = nn.Linear(visual_token_dim, hidden_dim)
        self.state_proj = nn.Linear(state_dim, hidden_dim)
        self.latent_proj = nn.Linear(latent_action_dim, hidden_dim)
        self.cls = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.current_type = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.future_type = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.delta_type = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.state_type = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.latent_type = nn.Parameter(torch.zeros(1, 1, hidden_dim))

        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers, norm=nn.LayerNorm(hidden_dim))
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.goal_progress = nn.Linear(hidden_dim, 1)
        self.success_logit = nn.Linear(hidden_dim, 1)
        self.risk_logit = nn.Linear(hidden_dim, 1)
        self.value = nn.Linear(hidden_dim, 1)
        self._init_parameters()

    def forward(
        self,
        visual_t: torch.Tensor,
        predicted_visual_future: torch.Tensor,
        state_t: torch.Tensor | None = None,
        latent_action: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        visual_t = ensure_visual_tokens(visual_t)
        predicted_visual_future = ensure_visual_tokens(predicted_visual_future)
        if visual_t.shape != predicted_visual_future.shape:
            raise ValueError(
                f"visual_t and predicted_visual_future must have same shape, got "
                f"{tuple(visual_t.shape)} and {tuple(predicted_visual_future.shape)}"
            )

        batch_size = visual_t.shape[0]
        current = self.token_proj(visual_t) + self.current_type
        future = self.token_proj(predicted_visual_future) + self.future_type
        delta = self.delta_proj(predicted_visual_future - visual_t) + self.delta_type
        tokens = [self.cls.expand(batch_size, -1, -1), current, future, delta]

        if state_t is not None:
            tokens.append(self.state_proj(state_t).unsqueeze(1) + self.state_type)
        if latent_action is not None:
            tokens.append(self.latent_proj(latent_action).unsqueeze(1) + self.latent_type)

        encoded = self.encoder(torch.cat(tokens, dim=1))
        pooled = self.head(encoded[:, 0])
        return {
            "goal_progress": self.goal_progress(pooled).squeeze(-1),
            "success_logit": self.success_logit(pooled).squeeze(-1),
            "risk_logit": self.risk_logit(pooled).squeeze(-1),
            "value": self.value(pooled).squeeze(-1),
        }

    def _init_parameters(self) -> None:
        nn.init.normal_(self.cls, std=0.02)
        nn.init.normal_(self.current_type, std=0.02)
        nn.init.normal_(self.future_type, std=0.02)
        nn.init.normal_(self.delta_type, std=0.02)
        nn.init.normal_(self.state_type, std=0.02)
        nn.init.normal_(self.latent_type, std=0.02)


def metric_future_scores(
    visual_t: torch.Tensor,
    predicted_visual_future: torch.Tensor,
    visual_future: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Non-learned future-token metrics usable before labels exist."""

    visual_t = ensure_visual_tokens(visual_t)
    predicted_visual_future = ensure_visual_tokens(predicted_visual_future)
    visual_future = ensure_visual_tokens(visual_future)
    pred_mse = (predicted_visual_future - visual_future).pow(2).mean(dim=(1, 2))
    identity_mse = (visual_t - visual_future).pow(2).mean(dim=(1, 2))
    pred_delta = predicted_visual_future - visual_t
    true_delta = visual_future - visual_t
    delta_cosine = F.cosine_similarity(
        pred_delta.flatten(1),
        true_delta.flatten(1),
        dim=-1,
    )
    improvement = 1.0 - pred_mse / identity_mse.clamp_min(1e-12)
    return {
        "future_mse": pred_mse,
        "identity_mse": identity_mse,
        "future_improvement_vs_identity": improvement,
        "transition_delta_cosine": delta_cosine,
    }

