from __future__ import annotations

import math

import torch
import torch.nn as nn

from .common import HandSequenceEmbedding, ensure_hand_sequence


class HandWorldModelDecoder(nn.Module):
    """Forward model p(H_future | H_context, z_h).

    The predicted structured state replaces LaWAM's predicted visual feature.
    Joint positions and fingertip contacts are auxiliary training heads; they do
    not require an SMPL-X layer inside the model.
    """

    def __init__(
        self,
        state_dim: int = 24,
        latent_action_dim: int = 64,
        hidden_dim: int = 256,
        future_length: int = 12,
        num_layers: int = 4,
        num_heads: int = 8,
        ffn_dim: int = 1024,
        dropout: float = 0.1,
        max_context_length: int = 8,
        num_hand_joints: int = 21,
        num_contact_points: int = 5,
        residual_prediction: bool = True,
        wrist_constant_velocity_anchor: bool = False,
    ) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.future_length = future_length
        self.num_hand_joints = num_hand_joints
        self.num_contact_points = num_contact_points
        self.residual_prediction = residual_prediction
        self.wrist_constant_velocity_anchor = wrist_constant_velocity_anchor

        self.context_embed = HandSequenceEmbedding(
            state_dim, hidden_dim, max_context_length, dropout
        )
        self.future_queries = nn.Parameter(
            torch.zeros(1, future_length, hidden_dim)
        )
        self.latent_proj = nn.Sequential(
            nn.LayerNorm(latent_action_dim),
            nn.Linear(latent_action_dim, hidden_dim),
        )
        layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(
            layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(hidden_dim),
        )
        self.state_head = nn.Linear(hidden_dim, state_dim)
        self.joint_head = nn.Linear(hidden_dim, num_hand_joints * 3)
        self.contact_head = nn.Linear(hidden_dim, num_contact_points)
        nn.init.normal_(self.future_queries, std=0.02)

    def forward(
        self,
        hand_context: torch.Tensor,
        latent_action: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        ensure_hand_sequence(hand_context, self.state_dim)
        batch_size = hand_context.shape[0]
        memory = self.context_embed(hand_context)
        queries = self.future_queries.expand(batch_size, -1, -1)
        queries = queries + self.latent_proj(latent_action).unsqueeze(1)
        decoded = self.decoder(tgt=queries, memory=memory)

        predicted_state = self.state_head(decoded)
        if self.residual_prediction:
            base = hand_context[:, -1:, :].expand(
                -1, self.future_length, -1
            ).clone()
            if self.wrist_constant_velocity_anchor and hand_context.shape[1] >= 2:
                wrist_velocity = (
                    hand_context[:, -1, :3] - hand_context[:, -2, :3]
                )
                steps = torch.arange(
                    1,
                    self.future_length + 1,
                    device=hand_context.device,
                    dtype=hand_context.dtype,
                ).view(1, self.future_length, 1)
                base[..., :3] = (
                    hand_context[:, -1:, :3]
                    + steps * wrist_velocity[:, None, :]
                )
            predicted_state = predicted_state + base
        predicted_joints = self.joint_head(decoded).view(
            batch_size,
            self.future_length,
            self.num_hand_joints,
            3,
        )
        predicted_contact_logits = self.contact_head(decoded)
        return {
            "predicted_hand_future": predicted_state,
            "predicted_joints_future": predicted_joints,
            "predicted_contact_logits": predicted_contact_logits,
            "future_tokens": decoded,
        }


def _modulate(
    normalized: torch.Tensor,
    shift: torch.Tensor,
    scale: torch.Tensor,
) -> torch.Tensor:
    """Apply DiT-style adaptive layer-normalization parameters."""

    return normalized * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class HandAdaLNZeroBlock(nn.Module):
    """LaWM/DiT-style self-attention block conditioned by latent action.

    The conditioning projection follows LaWAM's six-way AdaLN-Zero design:
    shift, scale, and residual gate for both self-attention and the FFN.  The
    projection is zero initialized, so each block starts as an identity map.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        ffn_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.norm_attention = nn.LayerNorm(
            hidden_dim, elementwise_affine=False, eps=1e-6
        )
        self.attention = nn.MultiheadAttention(
            hidden_dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm_ffn = nn.LayerNorm(
            hidden_dim, elementwise_affine=False, eps=1e-6
        )
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, hidden_dim),
            nn.Dropout(dropout),
        )
        self.adaln_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, 6 * hidden_dim),
        )
        nn.init.zeros_(self.adaln_modulation[-1].weight)
        nn.init.zeros_(self.adaln_modulation[-1].bias)

    def forward(
        self,
        tokens: torch.Tensor,
        condition: torch.Tensor,
    ) -> torch.Tensor:
        (
            shift_attention,
            scale_attention,
            gate_attention,
            shift_ffn,
            scale_ffn,
            gate_ffn,
        ) = self.adaln_modulation(condition).chunk(6, dim=-1)

        normalized = _modulate(
            self.norm_attention(tokens),
            shift_attention,
            scale_attention,
        )
        attention_output, _ = self.attention(
            normalized,
            normalized,
            normalized,
            need_weights=False,
        )
        tokens = tokens + gate_attention.unsqueeze(1) * attention_output

        normalized = _modulate(
            self.norm_ffn(tokens),
            shift_ffn,
            scale_ffn,
        )
        return tokens + gate_ffn.unsqueeze(1) * self.ffn(normalized)


class LaWMStyleHandWorldModelDecoder(nn.Module):
    """LaWAM-style AdaLN-Zero decoder for structured hand trajectories.

    LaWAM applies action-conditioned self-attention to current visual tokens.
    Here the modality-preserving analogue is a deterministic future anchor:
    the last hand pose, optionally extrapolated with constant wrist velocity.
    Its horizon tokens receive fixed 1D positional encodings and are modulated
    by the teacher latent action in every block. HMWM-LaWM-v1 additionally
    prepends the complete context sequence with fixed negative-time positions;
    only the future tokens are decoded. The output remains a residual correction
    to the same anchor, preserving the existing Stage-1 target and losses.
    """

    def __init__(
        self,
        state_dim: int = 24,
        latent_action_dim: int = 64,
        hidden_dim: int = 256,
        future_length: int = 12,
        num_layers: int = 4,
        num_heads: int = 8,
        ffn_dim: int = 1024,
        dropout: float = 0.1,
        num_hand_joints: int = 21,
        num_contact_points: int = 5,
        residual_prediction: bool = True,
        wrist_constant_velocity_anchor: bool = False,
        include_context_tokens: bool = False,
        max_context_length: int = 8,
    ) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.future_length = future_length
        self.num_hand_joints = num_hand_joints
        self.num_contact_points = num_contact_points
        self.residual_prediction = residual_prediction
        self.wrist_constant_velocity_anchor = wrist_constant_velocity_anchor
        self.include_context_tokens = include_context_tokens
        self.max_context_length = max_context_length

        self.input_projection = nn.Linear(state_dim, hidden_dim)
        self.register_buffer(
            "horizon_position",
            self._fixed_1d_position(future_length, hidden_dim),
            persistent=True,
        )
        if include_context_tokens:
            context_steps = torch.arange(
                -max_context_length, 0, dtype=torch.float32
            )
            self.register_buffer(
                "context_position",
                self._fixed_1d_position_from_steps(context_steps, hidden_dim),
                persistent=True,
            )
        else:
            # A None buffer is absent from state_dict, preserving strict loading
            # compatibility with HMWM-LaWM-v0 checkpoints.
            self.register_buffer("context_position", None, persistent=True)
        self.input_dropout = nn.Dropout(dropout)
        self.latent_projection = nn.Sequential(
            nn.LayerNorm(latent_action_dim),
            nn.Linear(latent_action_dim, hidden_dim),
        )
        self.blocks = nn.ModuleList(
            [
                HandAdaLNZeroBlock(
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    ffn_dim=ffn_dim,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.state_head = nn.Linear(hidden_dim, state_dim)
        self.joint_head = nn.Linear(hidden_dim, num_hand_joints * 3)
        self.contact_head = nn.Linear(hidden_dim, num_contact_points)

    @staticmethod
    def _fixed_1d_position(length: int, hidden_dim: int) -> torch.Tensor:
        steps = torch.arange(length, dtype=torch.float32)
        return LaWMStyleHandWorldModelDecoder._fixed_1d_position_from_steps(
            steps, hidden_dim
        )

    @staticmethod
    def _fixed_1d_position_from_steps(
        steps: torch.Tensor,
        hidden_dim: int,
    ) -> torch.Tensor:
        position = steps.unsqueeze(1)
        even_dimension = torch.arange(0, hidden_dim, 2, dtype=torch.float32)
        frequency = torch.exp(-math.log(10000.0) * even_dimension / hidden_dim)
        encoding = torch.zeros(
            1, steps.numel(), hidden_dim, dtype=torch.float32
        )
        encoding[0, :, 0::2] = torch.sin(position * frequency)
        odd_width = encoding[0, :, 1::2].shape[-1]
        encoding[0, :, 1::2] = torch.cos(position * frequency[:odd_width])
        return encoding

    def _future_anchor(self, hand_context: torch.Tensor) -> torch.Tensor:
        anchor = hand_context[:, -1:, :].expand(
            -1, self.future_length, -1
        ).clone()
        if self.wrist_constant_velocity_anchor and hand_context.shape[1] >= 2:
            wrist_velocity = hand_context[:, -1, :3] - hand_context[:, -2, :3]
            steps = torch.arange(
                1,
                self.future_length + 1,
                device=hand_context.device,
                dtype=hand_context.dtype,
            ).view(1, self.future_length, 1)
            anchor[..., :3] = (
                hand_context[:, -1:, :3]
                + steps * wrist_velocity[:, None, :]
            )
        return anchor

    def forward(
        self,
        hand_context: torch.Tensor,
        latent_action: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        ensure_hand_sequence(hand_context, self.state_dim)
        batch_size = hand_context.shape[0]
        anchor = self._future_anchor(hand_context)
        future_tokens = self.input_projection(anchor)
        future_tokens = (
            future_tokens + self.horizon_position.to(dtype=future_tokens.dtype)
        )
        if self.include_context_tokens:
            context_length = hand_context.shape[1]
            if context_length > self.max_context_length:
                raise ValueError(
                    f"context length {context_length} exceeds "
                    f"max_context_length={self.max_context_length}"
                )
            context_tokens = self.input_projection(hand_context)
            context_tokens = context_tokens + self.context_position[
                :, -context_length:
            ].to(dtype=context_tokens.dtype)
            tokens = torch.cat([context_tokens, future_tokens], dim=1)
        else:
            tokens = future_tokens
        tokens = self.input_dropout(tokens)
        condition = self.latent_projection(latent_action)
        for block in self.blocks:
            tokens = block(tokens, condition)
        decoded = self.output_norm(tokens[:, -self.future_length :])

        predicted_state = self.state_head(decoded)
        if self.residual_prediction:
            predicted_state = predicted_state + anchor
        predicted_joints = self.joint_head(decoded).view(
            batch_size,
            self.future_length,
            self.num_hand_joints,
            3,
        )
        predicted_contact_logits = self.contact_head(decoded)
        return {
            "predicted_hand_future": predicted_state,
            "predicted_joints_future": predicted_joints,
            "predicted_contact_logits": predicted_contact_logits,
            "future_tokens": decoded,
        }
