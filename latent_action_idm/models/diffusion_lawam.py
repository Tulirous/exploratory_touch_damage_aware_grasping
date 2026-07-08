from __future__ import annotations

import math

import torch
import torch.nn as nn

from .common import VisualTokenProjector, modulate, ensure_visual_tokens


class SinusoidalTimestepEmbedding(nn.Module):
    def __init__(self, hidden_dim: int, max_period: int = 10000) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.max_period = max_period
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        half = self.hidden_dim // 2
        freqs = torch.exp(
            -math.log(self.max_period)
            * torch.arange(half, dtype=torch.float32, device=timesteps.device)
            / max(half, 1)
        )
        args = timesteps.float()[:, None] * freqs[None]
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if self.hidden_dim % 2:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
        return self.mlp(emb)


class DiTCrossBlock(nn.Module):
    """DiT block over noisy future tokens, cross-attending to current tokens."""

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        ffn_dim: int,
        cond_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.norm_self = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.self_attn = nn.MultiheadAttention(
            hidden_dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm_cross = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.cross_attn = nn.MultiheadAttention(
            hidden_dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm_ffn = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, hidden_dim),
        )
        self.dropout = nn.Dropout(dropout)
        self.cond = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, hidden_dim * 6),
        )

    def forward(
        self,
        noisy_tokens: torch.Tensor,
        current_tokens: torch.Tensor,
        cond: torch.Tensor,
    ) -> torch.Tensor:
        shift1, scale1, shift2, scale2, shift3, scale3 = self.cond(cond).chunk(6, dim=-1)

        h = modulate(self.norm_self(noisy_tokens), shift1, scale1)
        attn_out, _ = self.self_attn(h, h, h, need_weights=False)
        noisy_tokens = noisy_tokens + self.dropout(attn_out)

        h = modulate(self.norm_cross(noisy_tokens), shift2, scale2)
        cross_out, _ = self.cross_attn(h, current_tokens, current_tokens, need_weights=False)
        noisy_tokens = noisy_tokens + self.dropout(cross_out)

        h = modulate(self.norm_ffn(noisy_tokens), shift3, scale3)
        noisy_tokens = noisy_tokens + self.dropout(self.ffn(h))
        return noisy_tokens


class DiffusionLatentWorldModel(nn.Module):
    """DiT-LaWM denoiser for future DINO patch tokens.

    It models p(u_future | u_current, z) with a DDPM-style epsilon prediction
    objective in DINO latent space.
    """

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
        num_diffusion_steps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 2e-2,
        num_views: int = 0,
    ) -> None:
        super().__init__()
        self.visual_token_dim = visual_token_dim
        self.latent_action_dim = latent_action_dim
        self.hidden_dim = hidden_dim
        self.num_diffusion_steps = num_diffusion_steps

        self.current_projector = VisualTokenProjector(
            visual_token_dim=visual_token_dim,
            hidden_dim=hidden_dim,
            max_visual_tokens=max_visual_tokens,
            num_views=num_views,
        )
        self.noisy_projector = VisualTokenProjector(
            visual_token_dim=visual_token_dim,
            hidden_dim=hidden_dim,
            max_visual_tokens=max_visual_tokens,
            num_views=num_views,
        )
        self.current_type = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.noisy_type = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.time_embedding = SinusoidalTimestepEmbedding(hidden_dim)
        self.latent_embedding = nn.Sequential(
            nn.LayerNorm(latent_action_dim),
            nn.Linear(latent_action_dim, hidden_dim),
        )
        self.blocks = nn.ModuleList(
            [
                DiTCrossBlock(
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

        betas = torch.linspace(beta_start, beta_end, num_diffusion_steps, dtype=torch.float32)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        self.register_buffer("betas", betas, persistent=False)
        self.register_buffer("alpha_bars", alpha_bars, persistent=False)
        nn.init.normal_(self.current_type, std=0.02)
        nn.init.normal_(self.noisy_type, std=0.02)

    def forward(
        self,
        noisy_future: torch.Tensor,
        visual_t: torch.Tensor,
        latent_action: torch.Tensor,
        diffusion_timesteps: torch.Tensor,
    ) -> torch.Tensor:
        noisy_future = ensure_visual_tokens(noisy_future)
        visual_t = ensure_visual_tokens(visual_t)
        noisy_tokens = self.noisy_projector(noisy_future) + self.noisy_type
        current_tokens = self.current_projector(visual_t) + self.current_type
        cond = self.latent_embedding(latent_action) + self.time_embedding(diffusion_timesteps)
        for block in self.blocks:
            noisy_tokens = block(noisy_tokens, current_tokens, cond)
        return self.output_proj(self.norm(noisy_tokens))

    def sample_timesteps(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.randint(0, self.num_diffusion_steps, (batch_size,), device=device)

    def q_sample(
        self,
        clean_future: torch.Tensor,
        timesteps: torch.Tensor,
        noise: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        clean_future = ensure_visual_tokens(clean_future)
        if noise is None:
            noise = torch.randn_like(clean_future)
        alpha_bar = self.alpha_bars[timesteps].view(-1, 1, 1).to(clean_future.device)
        noisy_future = alpha_bar.sqrt() * clean_future + (1 - alpha_bar).sqrt() * noise
        return noisy_future, noise

    def predict_clean_from_noise(
        self,
        noisy_future: torch.Tensor,
        timesteps: torch.Tensor,
        predicted_noise: torch.Tensor,
    ) -> torch.Tensor:
        alpha_bar = self.alpha_bars[timesteps].view(-1, 1, 1).to(noisy_future.device)
        return (noisy_future - (1 - alpha_bar).sqrt() * predicted_noise) / alpha_bar.sqrt().clamp_min(1e-6)
