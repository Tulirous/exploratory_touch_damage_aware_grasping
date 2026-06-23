from __future__ import annotations

import torch
from torch import nn

from .action_encoder import ActionChunkEncoder
from .outcome_transformer import OutcomeTransformer
from .tactile_force_adapter import TactileForceAdapter
from .text_instruction_encoder import TextInstructionEncoder


class FastWAMFragile(nn.Module):
    """Fast-WAM-style outcome model for fragile grasping and handover safety."""

    def __init__(
        self,
        visual_latent_dim: int,
        use_text: bool,
        text_vocab_size: int,
        text_max_tokens: int,
        text_latent_dim: int,
        tactile_input_dim: int,
        action_dim: int,
        hidden_dim: int,
        tactile_latent_dim: int,
        action_latent_dim: int,
        fusion_layers: int,
        fusion_heads: int,
        num_binary: int,
        num_regression: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.text_encoder = (
            TextInstructionEncoder(
                vocab_size=text_vocab_size,
                max_tokens=text_max_tokens,
                hidden_dim=hidden_dim,
                output_dim=text_latent_dim,
                dropout=dropout,
            )
            if use_text
            else None
        )
        self.tactile_adapter = TactileForceAdapter(
            input_dim=tactile_input_dim,
            hidden_dim=hidden_dim,
            output_dim=tactile_latent_dim,
        )
        self.action_encoder = ActionChunkEncoder(
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            output_dim=action_latent_dim,
        )
        self.outcome = OutcomeTransformer(
            visual_dim=visual_latent_dim,
            text_dim=text_latent_dim if use_text else None,
            tactile_dim=tactile_latent_dim,
            action_dim=action_latent_dim,
            hidden_dim=hidden_dim,
            num_layers=fusion_layers,
            num_heads=fusion_heads,
            num_binary=num_binary,
            num_regression=num_regression,
            dropout=dropout,
        )

    def forward(
        self,
        visual_latent: torch.Tensor,
        task_instruction: list[str] | tuple[str, ...] | None,
        tactile_seq: torch.Tensor,
        candidate_actions: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        text_latent = None
        if self.text_encoder is not None:
            if task_instruction is None:
                raise ValueError("task_instruction is required when text conditioning is enabled")
            text_latent = self.text_encoder(task_instruction)
        tactile_latent = self.tactile_adapter(tactile_seq)
        action_latent = self.action_encoder(candidate_actions)
        return self.outcome(visual_latent, text_latent, tactile_latent, action_latent)
