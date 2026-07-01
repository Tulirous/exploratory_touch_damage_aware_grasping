from __future__ import annotations

import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class AdaLNBlock(nn.Module):
    """Transformer block with latent-action adaptive LayerNorm conditioning."""

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        ffn_dim: int,
        cond_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.attn = nn.MultiheadAttention(
            hidden_dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, hidden_dim),
        )
        self.dropout = nn.Dropout(dropout)
        self.cond = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, hidden_dim * 4),
        )

    def forward(self, tokens: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        shift1, scale1, shift2, scale2 = self.cond(cond).chunk(4, dim=-1)
        h = self._modulate(self.norm1(tokens), shift1, scale1)
        attn_out, _ = self.attn(h, h, h, need_weights=False)
        tokens = tokens + self.dropout(attn_out)
        h = self._modulate(self.norm2(tokens), shift2, scale2)
        tokens = tokens + self.dropout(self.ffn(h))
        return tokens

    def _modulate(self, x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class LaWAMStage1IDM(nn.Module):
    """Patch-token inverse dynamics model and LaWM decoder.

    This follows the Stage-1 LaWAM interface:

    - frozen visual encoder produces current and horizon patch features
    - an inverse-dynamics posterior infers latent action z from (u_t, u_T)
    - a LaWM decoder predicts horizon patch features from (u_t, z)
    - an auxiliary predictor maps (state_t, z) to state_T

    The implementation keeps the exact patch-token interface needed for LaWAM
    reproduction while keeping layer counts configurable for small-data debug
    and full-scale reproduction runs.
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
        self.max_visual_tokens = max_visual_tokens

        self.input_proj = nn.Linear(visual_token_dim, hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, visual_token_dim)
        self.state_token = nn.Linear(state_dim, hidden_dim)
        self.inverse_cls = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.current_type = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.future_type = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.state_type = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, max_visual_tokens, hidden_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.inverse_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=encoder_layers,
            norm=nn.LayerNorm(hidden_dim),
        )
        self.posterior = nn.Linear(hidden_dim, latent_action_dim * 2)

        self.latent_to_hidden = nn.Sequential(
            nn.LayerNorm(latent_action_dim),
            nn.Linear(latent_action_dim, hidden_dim),
        )
        self.decoder_blocks = nn.ModuleList(
            [
                AdaLNBlock(
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    ffn_dim=ffn_dim,
                    cond_dim=hidden_dim,
                    dropout=dropout,
                )
                for _ in range(decoder_layers)
            ]
        )
        self.decoder_norm = nn.LayerNorm(hidden_dim)
        self.state_predictor = MLP(
            state_dim + latent_action_dim,
            hidden_dim,
            state_dim,
            dropout=dropout,
        )
        self._init_parameters()

    def forward(
        self,
        visual_t: torch.Tensor,
        visual_future: torch.Tensor,
        state_t: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        visual_t = self._ensure_tokens(visual_t)
        visual_future = self._ensure_tokens(visual_future)
        if visual_t.shape != visual_future.shape:
            raise ValueError(
                f"visual_t and visual_future must have same shape, got "
                f"{tuple(visual_t.shape)} and {tuple(visual_future.shape)}"
            )

        posterior_tokens = self._build_inverse_tokens(visual_t, visual_future, state_t)
        encoded = self.inverse_encoder(posterior_tokens)
        mu, logvar = self.posterior(encoded[:, 0]).chunk(2, dim=-1)
        latent_action = self._sample_latent(mu, logvar)

        predicted_visual_future = self.decode_future(visual_t, latent_action)
        state_delta = self.state_predictor(torch.cat([state_t, latent_action], dim=-1))
        predicted_state_future = state_t + state_delta

        return {
            "latent_action": latent_action,
            "latent_mu": mu,
            "latent_logvar": logvar,
            "predicted_state_future": predicted_state_future,
            "predicted_visual_future": predicted_visual_future,
        }

    def decode_future(self, visual_t: torch.Tensor, latent_action: torch.Tensor) -> torch.Tensor:
        visual_t = self._ensure_tokens(visual_t)
        tokens = self._project_visual_tokens(visual_t) + self.current_type
        cond = self.latent_to_hidden(latent_action)
        for block in self.decoder_blocks:
            tokens = block(tokens, cond)
        return self.output_proj(self.decoder_norm(tokens))

    def _build_inverse_tokens(
        self,
        visual_t: torch.Tensor,
        visual_future: torch.Tensor,
        state_t: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = visual_t.shape[0]
        current = self._project_visual_tokens(visual_t) + self.current_type
        future = self._project_visual_tokens(visual_future) + self.future_type
        state = self.state_token(state_t).unsqueeze(1) + self.state_type
        cls = self.inverse_cls.expand(batch_size, -1, -1)
        return torch.cat([cls, state, current, future], dim=1)

    def _project_visual_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        num_tokens = tokens.shape[1]
        if num_tokens > self.max_visual_tokens:
            raise ValueError(
                f"num visual tokens {num_tokens} exceeds max_visual_tokens={self.max_visual_tokens}"
            )
        return self.input_proj(tokens) + self.pos_embed[:, :num_tokens]

    def _ensure_tokens(self, tensor: torch.Tensor) -> torch.Tensor:
        if tensor.ndim == 2:
            raise ValueError(
                "Expected DINO patch tokens with shape [B, N, D]. "
                "Regenerate latents with dino.feature_mode='patch_tokens'."
            )
        if tensor.ndim != 3:
            raise ValueError(f"Expected visual tokens [B, N, D], got {tuple(tensor.shape)}")
        return tensor

    def _sample_latent(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        if self.training:
            std = torch.exp(0.5 * logvar.clamp(min=-10.0, max=10.0))
            return mu + torch.randn_like(std) * std
        return mu

    def _init_parameters(self) -> None:
        nn.init.normal_(self.inverse_cls, std=0.02)
        nn.init.normal_(self.current_type, std=0.02)
        nn.init.normal_(self.future_type, std=0.02)
        nn.init.normal_(self.state_type, std=0.02)
        nn.init.normal_(self.pos_embed, std=0.02)


# Backward-compatible import name used by train_idm.py.
LatentActionIDM = LaWAMStage1IDM

