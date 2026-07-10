from __future__ import annotations

import torch
import torch.nn as nn

from .common import VisualTokenProjector, ensure_visual_tokens


class InverseDynamicsTransformer(nn.Module):
    """Posterior q(z | current visual tokens, future visual tokens, robot state)."""

    def __init__(
        self,
        visual_token_dim: int,
        state_dim: int,
        latent_action_dim: int = 128,
        hidden_dim: int = 768,
        num_layers: int = 8,
        num_heads: int = 12,
        ffn_dim: int = 3072,
        dropout: float = 0.1,
        max_visual_tokens: int = 512,
        use_state_condition: bool = True,
        num_views: int = 0,
        cross_view_fusion_layers: int = 0,
        use_residual_branch: bool = False,
    ) -> None:
        super().__init__()
        self.visual_token_dim = visual_token_dim
        self.state_dim = state_dim
        self.latent_action_dim = latent_action_dim
        self.hidden_dim = hidden_dim
        self.use_state_condition = use_state_condition
        self.use_residual_branch = use_residual_branch

        self.visual_projector = VisualTokenProjector(
            visual_token_dim=visual_token_dim,
            hidden_dim=hidden_dim,
            max_visual_tokens=max_visual_tokens,
            num_views=num_views,
            cross_view_layers=cross_view_fusion_layers,
            num_heads=num_heads,
            ffn_dim=ffn_dim,
            dropout=dropout,
        )
        self.state_token = nn.Linear(state_dim, hidden_dim)
        self.inverse_cls = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.current_type = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.future_type = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.residual_type = nn.Parameter(torch.zeros(1, 1, hidden_dim)) if use_residual_branch else None
        self.state_type = nn.Parameter(torch.zeros(1, 1, hidden_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(hidden_dim),
        )
        self.posterior = nn.Linear(hidden_dim, latent_action_dim * 2)
        self._init_parameters()

    def forward(
        self,
        visual_t: torch.Tensor,
        visual_future: torch.Tensor,
        state_t: torch.Tensor,
        sample: bool | None = None,
    ) -> dict[str, torch.Tensor]:
        visual_t = ensure_visual_tokens(visual_t)
        visual_future = ensure_visual_tokens(visual_future)
        if visual_t.shape != visual_future.shape:
            raise ValueError(
                f"visual_t and visual_future must have same shape, got "
                f"{tuple(visual_t.shape)} and {tuple(visual_future.shape)}"
            )
        tokens = self.build_tokens(visual_t, visual_future, state_t)
        encoded = self.encoder(tokens)
        mu, logvar = self.posterior(encoded[:, 0]).chunk(2, dim=-1)
        latent_action = self.sample_latent(mu, logvar, sample=sample)
        return {
            "latent_action": latent_action,
            "latent_mu": mu,
            "latent_logvar": logvar,
            "posterior_cls": encoded[:, 0],
        }

    def build_tokens(
        self,
        visual_t: torch.Tensor,
        visual_future: torch.Tensor,
        state_t: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = visual_t.shape[0]
        current = self.visual_projector(visual_t) + self.current_type
        future = self.visual_projector(visual_future) + self.future_type
        cls = self.inverse_cls.expand(batch_size, -1, -1)
        segments = [cls]
        if self.use_state_condition:
            state = self.state_token(state_t).unsqueeze(1) + self.state_type
            segments.append(state)
        segments.extend([current, future])
        if self.use_residual_branch:
            if self.residual_type is None:
                raise RuntimeError("residual_type is not initialized")
            residual = self.visual_projector(visual_future - visual_t) + self.residual_type
            segments.append(residual)
        return torch.cat(segments, dim=1)

    def sample_latent(
        self,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        sample: bool | None = None,
    ) -> torch.Tensor:
        should_sample = self.training if sample is None else sample
        if should_sample:
            std = torch.exp(0.5 * logvar.clamp(min=-10.0, max=10.0))
            return mu + torch.randn_like(std) * std
        return mu

    def _init_parameters(self) -> None:
        nn.init.normal_(self.inverse_cls, std=0.02)
        nn.init.normal_(self.current_type, std=0.02)
        nn.init.normal_(self.future_type, std=0.02)
        if self.residual_type is not None:
            nn.init.normal_(self.residual_type, std=0.02)
        nn.init.normal_(self.state_type, std=0.02)
