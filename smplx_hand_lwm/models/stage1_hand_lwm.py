from __future__ import annotations

import torch
import torch.nn as nn

from .hand_world_model import (
    AdaLNCrossAttentionHandWorldModelDecoder,
    HandWorldModelDecoder,
    LaWMStyleHandWorldModelDecoder,
)
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
        wrist_aware_auxiliary_head: bool = False,
        window_local_wrist_translation: bool = False,
        hmwm_decoder_type: str = "transformer_decoder",
    ) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.context_length = context_length
        self.future_length = future_length
        self.inverse_dynamics_frozen = False
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
            wrist_aware_auxiliary_head=wrist_aware_auxiliary_head,
            window_local_wrist_translation=window_local_wrist_translation,
        )
        decoder_types = {
            "transformer_decoder": HandWorldModelDecoder,
            "lawam_adaln_zero": LaWMStyleHandWorldModelDecoder,
            "lawam_adaln_zero_context": LaWMStyleHandWorldModelDecoder,
            "adaln_zero_cross_attention": (
                AdaLNCrossAttentionHandWorldModelDecoder
            ),
        }
        if hmwm_decoder_type not in decoder_types:
            raise ValueError(
                f"unsupported hmwm_decoder_type={hmwm_decoder_type!r}; "
                f"expected one of {sorted(decoder_types)}"
            )
        decoder_class = decoder_types[hmwm_decoder_type]
        self.hmwm_decoder_type = hmwm_decoder_type
        decoder_specific_kwargs: dict[str, object]
        if hmwm_decoder_type in {
            "transformer_decoder",
            "adaln_zero_cross_attention",
        }:
            decoder_specific_kwargs = {"max_context_length": context_length}
        else:
            decoder_specific_kwargs = {
                "include_context_tokens": (
                    hmwm_decoder_type == "lawam_adaln_zero_context"
                ),
                "max_context_length": context_length,
            }
        self.hand_world_model = decoder_class(
            state_dim=state_dim,
            latent_action_dim=latent_action_dim,
            hidden_dim=hidden_dim,
            future_length=future_length,
            num_layers=decoder_layers,
            num_heads=num_heads,
            ffn_dim=ffn_dim,
            dropout=dropout,
            num_hand_joints=num_hand_joints,
            num_contact_points=num_contact_points,
            residual_prediction=residual_prediction,
            wrist_constant_velocity_anchor=wrist_constant_velocity_anchor,
            **decoder_specific_kwargs,
        )

    def forward(
        self,
        hand_context: torch.Tensor,
        hand_future: torch.Tensor,
        sample: bool | None = None,
    ) -> dict[str, torch.Tensor]:
        if self.inverse_dynamics_frozen:
            # A frozen teacher must be invariant to both posterior sampling and
            # encoder dropout. This makes the same window produce exactly the
            # same LA while the decoder is retrained from scratch.
            self.inverse_dynamics.eval()
            with torch.no_grad():
                posterior = self.encode_latent_action(
                    hand_context,
                    hand_future,
                    sample=False,
                )
        else:
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

    def freeze_inverse_dynamics_(self) -> Stage1HandLWM:
        """Freeze the Hand-IDM and use its deterministic posterior mean."""

        self.inverse_dynamics_frozen = True
        self.inverse_dynamics.requires_grad_(False)
        self.inverse_dynamics.eval()
        return self

    def train(self, mode: bool = True) -> Stage1HandLWM:
        super().train(mode)
        if self.inverse_dynamics_frozen:
            self.inverse_dynamics.eval()
        return self
