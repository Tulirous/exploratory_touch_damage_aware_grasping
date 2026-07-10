from __future__ import annotations

import torch
import torch.nn as nn

from .common import MLP
from .diffusion_lawam import DiffusionLatentWorldModel
from .inverse_dynamics import InverseDynamicsTransformer


class Stage1DiTLaWAM(nn.Module):
    """Stage-1 LaWAM with a diffusion Transformer latent world model."""

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
        num_diffusion_steps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 2e-2,
        use_state_in_idm: bool = True,
        num_views: int = 0,
        prediction_type: str = "epsilon",
        timestep_sampling: str = "uniform",
        train_timestep_min: int = 0,
        train_timestep_max: int | None = None,
        logit_normal_mean: float = 0.0,
        logit_normal_std: float = 1.0,
    ) -> None:
        super().__init__()
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
            use_state_condition=use_state_in_idm,
            num_views=num_views,
        )
        self.latent_world_model = DiffusionLatentWorldModel(
            visual_token_dim=visual_token_dim,
            latent_action_dim=latent_action_dim,
            hidden_dim=hidden_dim,
            num_layers=decoder_layers,
            num_heads=num_heads,
            ffn_dim=ffn_dim,
            dropout=dropout,
            max_visual_tokens=max_visual_tokens,
            num_diffusion_steps=num_diffusion_steps,
            beta_start=beta_start,
            beta_end=beta_end,
            num_views=num_views,
            prediction_type=prediction_type,
            timestep_sampling=timestep_sampling,
            train_timestep_min=train_timestep_min,
            train_timestep_max=train_timestep_max,
            logit_normal_mean=logit_normal_mean,
            logit_normal_std=logit_normal_std,
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
        diffusion_timesteps: torch.Tensor | None = None,
        noise: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        posterior = self.inverse_dynamics(visual_t, visual_future, state_t)
        latent_action = posterior["latent_action"]
        if diffusion_timesteps is None:
            diffusion_timesteps = self.latent_world_model.sample_timesteps(
                visual_t.shape[0],
                visual_t.device,
            )
        noisy_future, noise = self.latent_world_model.q_sample(visual_future, diffusion_timesteps, noise=noise)
        model_prediction = self.latent_world_model(
            noisy_future=noisy_future,
            visual_t=visual_t,
            latent_action=latent_action,
            diffusion_timesteps=diffusion_timesteps,
        )
        diffusion_target = self.latent_world_model.training_target(
            visual_future,
            noise,
            diffusion_timesteps,
        )
        predicted_noise = self.latent_world_model.predict_noise_from_model_output(
            noisy_future,
            diffusion_timesteps,
            model_prediction,
        )
        predicted_visual_future = self.latent_world_model.predict_clean_from_model_output(
            noisy_future,
            diffusion_timesteps,
            model_prediction,
        )
        predicted_state_future = self.predict_state_future(state_t, latent_action)
        return {
            **posterior,
            "diffusion_timesteps": diffusion_timesteps,
            "alpha_bars": self.latent_world_model.alpha_bars,
            "noise": noise,
            "noisy_visual_future": noisy_future,
            "model_prediction": model_prediction,
            "diffusion_target": diffusion_target,
            "predicted_noise": predicted_noise,
            "predicted_visual_future": predicted_visual_future,
            "predicted_state_future": predicted_state_future,
        }

    def predict_state_future(self, state_t: torch.Tensor, latent_action: torch.Tensor) -> torch.Tensor:
        state_delta = self.state_predictor(torch.cat([state_t, latent_action], dim=-1))
        return state_t + state_delta
