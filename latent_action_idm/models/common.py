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


class VisualTokenProjector(nn.Module):
    """Project frozen DINO patch tokens into the Stage-1 hidden space."""

    def __init__(
        self,
        visual_token_dim: int,
        hidden_dim: int,
        max_visual_tokens: int,
    ) -> None:
        super().__init__()
        self.visual_token_dim = visual_token_dim
        self.hidden_dim = hidden_dim
        self.max_visual_tokens = max_visual_tokens
        self.input_proj = nn.Linear(visual_token_dim, hidden_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, max_visual_tokens, hidden_dim))
        nn.init.normal_(self.pos_embed, std=0.02)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        tokens = ensure_visual_tokens(tokens)
        num_tokens = tokens.shape[1]
        if num_tokens > self.max_visual_tokens:
            raise ValueError(
                f"num visual tokens {num_tokens} exceeds max_visual_tokens={self.max_visual_tokens}"
            )
        return self.input_proj(tokens) + self.pos_embed[:, :num_tokens]


class AdaLNBlock(nn.Module):
    """Self-attention block conditioned by latent action via AdaLN."""

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
        h = modulate(self.norm1(tokens), shift1, scale1)
        attn_out, _ = self.attn(h, h, h, need_weights=False)
        tokens = tokens + self.dropout(attn_out)
        h = modulate(self.norm2(tokens), shift2, scale2)
        tokens = tokens + self.dropout(self.ffn(h))
        return tokens


def ensure_visual_tokens(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.ndim == 2:
        raise ValueError(
            "Expected DINO patch tokens with shape [B, N, D]. "
            "Regenerate latents with dino.feature_mode='patch_tokens'."
        )
    if tensor.ndim != 3:
        raise ValueError(f"Expected visual tokens [B, N, D], got {tuple(tensor.shape)}")
    return tensor


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

