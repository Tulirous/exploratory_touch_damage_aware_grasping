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
        prediction_type: str = "epsilon",
        timestep_sampling: str = "uniform",
        train_timestep_min: int = 0,
        train_timestep_max: int | None = None,
        logit_normal_mean: float = 0.0,
        logit_normal_std: float = 1.0,
    ) -> None:
        super().__init__()
        if prediction_type not in {"epsilon", "x0", "v"}:
            raise ValueError(f"Unsupported prediction_type={prediction_type}")
        if timestep_sampling not in {"uniform", "logit_normal"}:
            raise ValueError(f"Unsupported timestep_sampling={timestep_sampling}")
        self.visual_token_dim = visual_token_dim
        self.latent_action_dim = latent_action_dim
        self.hidden_dim = hidden_dim
        self.num_diffusion_steps = num_diffusion_steps
        self.prediction_type = prediction_type
        self.timestep_sampling = timestep_sampling
        self.train_timestep_min = int(train_timestep_min)
        self.train_timestep_max = int(train_timestep_max) if train_timestep_max is not None else num_diffusion_steps - 1
        self.logit_normal_mean = float(logit_normal_mean)
        self.logit_normal_std = float(logit_normal_std)

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
        low = max(self.train_timestep_min, 0)
        high = min(self.train_timestep_max, self.num_diffusion_steps - 1)
        if high < low:
            raise ValueError(f"Invalid timestep range: [{low}, {high}]")
        if self.timestep_sampling == "uniform":
            return torch.randint(low, high + 1, (batch_size,), device=device)
        samples = torch.randn(batch_size, device=device) * self.logit_normal_std + self.logit_normal_mean
        scaled = torch.sigmoid(samples)
        timesteps = low + torch.floor(scaled * (high - low + 1)).long()
        return timesteps.clamp(min=low, max=high)

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

    def training_target(
        self,
        clean_future: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        if self.prediction_type == "epsilon":
            return noise
        if self.prediction_type == "x0":
            return clean_future
        alpha_bar = self.alpha_bars[timesteps].view(-1, 1, 1).to(clean_future.device)
        return alpha_bar.sqrt() * noise - (1 - alpha_bar).sqrt() * clean_future

    def predict_clean_from_model_output(
        self,
        noisy_future: torch.Tensor,
        timesteps: torch.Tensor,
        model_output: torch.Tensor,
    ) -> torch.Tensor:
        alpha_bar = self.alpha_bars[timesteps].view(-1, 1, 1).to(noisy_future.device)
        if self.prediction_type == "epsilon":
            return self.predict_clean_from_noise(noisy_future, timesteps, model_output)
        if self.prediction_type == "x0":
            return model_output
        return alpha_bar.sqrt() * noisy_future - (1 - alpha_bar).sqrt() * model_output

    def predict_noise_from_model_output(
        self,
        noisy_future: torch.Tensor,
        timesteps: torch.Tensor,
        model_output: torch.Tensor,
    ) -> torch.Tensor:
        alpha_bar = self.alpha_bars[timesteps].view(-1, 1, 1).to(noisy_future.device)
        if self.prediction_type == "epsilon":
            return model_output
        if self.prediction_type == "x0":
            return (noisy_future - alpha_bar.sqrt() * model_output) / (1 - alpha_bar).sqrt().clamp_min(1e-6)
        return (1 - alpha_bar).sqrt() * noisy_future + alpha_bar.sqrt() * model_output

    @torch.no_grad()
    def sample(
        self,
        visual_t: torch.Tensor,
        latent_action: torch.Tensor,
        shape: tuple[int, int, int],
        num_steps: int = 50,
        sampler: str = "ddim",
        eta: float = 0.0,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        if sampler not in {"ddim", "ddpm"}:
            raise ValueError(f"Unsupported sampler={sampler}")
        device = visual_t.device
        x = torch.randn(shape, device=device, generator=generator)
        step_ids = torch.linspace(self.num_diffusion_steps - 1, 0, num_steps, device=device).long().unique_consecutive()
        if step_ids[-1].item() != 0:
            step_ids = torch.cat([step_ids, torch.zeros(1, dtype=torch.long, device=device)])

        for index, timestep in enumerate(step_ids):
            t = torch.full((shape[0],), int(timestep.item()), dtype=torch.long, device=device)
            model_output = self(
                noisy_future=x,
                visual_t=visual_t,
                latent_action=latent_action,
                diffusion_timesteps=t,
            )
            x0 = self.predict_clean_from_model_output(x, t, model_output)
            eps = self.predict_noise_from_model_output(x, t, model_output)
            if timestep.item() == 0:
                x = x0
                continue

            prev_timestep = step_ids[index + 1] if index + 1 < len(step_ids) else torch.tensor(0, device=device)
            alpha_t = self.alpha_bars[timestep].to(device)
            alpha_prev = self.alpha_bars[prev_timestep].to(device)
            if sampler == "ddim":
                sigma = eta * torch.sqrt((1 - alpha_prev) / (1 - alpha_t)) * torch.sqrt(1 - alpha_t / alpha_prev)
                direction_scale = torch.sqrt((1 - alpha_prev - sigma.square()).clamp_min(0.0))
                x = alpha_prev.sqrt() * x0 + direction_scale * eps
                if eta > 0:
                    x = x + sigma * torch.randn(shape, device=device, generator=generator)
                continue

            beta_t = self.betas[timestep].to(device)
            alpha_step = 1.0 - beta_t
            posterior_var = beta_t * (1 - alpha_prev) / (1 - alpha_t)
            coef1 = beta_t * alpha_prev.sqrt() / (1 - alpha_t)
            coef2 = (1 - alpha_prev) * alpha_step.sqrt() / (1 - alpha_t)
            x = coef1 * x0 + coef2 * x
            x = x + posterior_var.sqrt() * torch.randn(shape, device=device, generator=generator)
        return x
