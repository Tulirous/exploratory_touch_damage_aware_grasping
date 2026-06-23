from __future__ import annotations

import hashlib
import re

import torch
from torch import nn


_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]")


class TextInstructionEncoder(nn.Module):
    """Trainable hashing text encoder for task instructions.

    This keeps the first-version model dependency-light. Later, this module can
    be replaced by frozen Fast-WAM/T5/OpenPI text embeddings without changing
    the outcome fusion interface.
    """

    def __init__(
        self,
        vocab_size: int,
        max_tokens: int,
        hidden_dim: int,
        output_dim: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if vocab_size <= 1:
            raise ValueError("vocab_size must be greater than 1")
        self.vocab_size = vocab_size
        self.max_tokens = max_tokens
        self.embedding = nn.Embedding(vocab_size, hidden_dim, padding_idx=0)
        self.norm = nn.LayerNorm(hidden_dim)
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, instructions: list[str] | tuple[str, ...]) -> torch.Tensor:
        if not isinstance(instructions, (list, tuple)):
            raise TypeError(f"instructions must be a list/tuple of strings, got {type(instructions)}")
        device = self.embedding.weight.device
        token_ids = torch.tensor(
            [self._encode_one(text) for text in instructions],
            dtype=torch.long,
            device=device,
        )
        mask = token_ids.ne(0).unsqueeze(-1)
        emb = self.embedding(token_ids)
        denom = mask.sum(dim=1).clamp_min(1)
        pooled = (emb * mask).sum(dim=1) / denom
        return self.proj(self.norm(pooled))

    def _encode_one(self, text: str) -> list[int]:
        tokens = _TOKEN_PATTERN.findall(str(text).lower())[: self.max_tokens]
        ids = [self._hash_token(token) for token in tokens]
        if len(ids) < self.max_tokens:
            ids.extend([0] * (self.max_tokens - len(ids)))
        return ids

    def _hash_token(self, token: str) -> int:
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        value = int.from_bytes(digest, byteorder="little", signed=False)
        return value % (self.vocab_size - 1) + 1
