from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


Backend = Literal["transformers", "timm", "torchhub"]
FeatureMode = Literal["cls", "mean_patch", "patch_tokens"]
Precision = Literal["fp32", "fp16", "bf16"]


@dataclass(frozen=True)
class DINOConfig:
    backend: Backend = "transformers"
    model_name: str = "facebook/dinov3-vitb16-pretrain-lvd1689m"
    hub_repo: str = "facebookresearch/dinov3"
    image_size: int = 224
    feature_mode: FeatureMode = "patch_tokens"
    num_prefix_tokens: int = 1
    precision: Precision = "fp16"


class DINOFeatureExtractor(nn.Module):
    """Frozen DINO image feature extractor.

    The extractor accepts uint8 RGB images with shape [B, H, W, 3] and returns a
    either a global vector or the spatial patch-token feature map. LaWAM-style
    training should use feature_mode='patch_tokens'.
    """

    def __init__(self, cfg: DINOConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.backend = cfg.backend
        self.processor = None

        if cfg.backend == "transformers":
            try:
                from transformers import AutoModel
            except ImportError as exc:
                raise ImportError(
                    "transformers is required for backend='transformers'. "
                    "Install transformers or set dino.backend to 'timm'."
                ) from exc
            self.model = AutoModel.from_pretrained(cfg.model_name)
        elif cfg.backend == "timm":
            try:
                import timm
            except ImportError as exc:
                raise ImportError(
                    "timm is required for backend='timm'. Install timm or use "
                    "backend='transformers'."
                ) from exc
            self.model = timm.create_model(cfg.model_name, pretrained=True, num_classes=0)
        elif cfg.backend == "torchhub":
            self.model = torch.hub.load(cfg.hub_repo, cfg.model_name)
        else:
            raise ValueError(f"Unsupported DINO backend: {cfg.backend}")

        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad_(False)

    @torch.no_grad()
    def forward(self, images: np.ndarray | torch.Tensor) -> torch.Tensor:
        if self.backend == "transformers":
            return self._forward_transformers(images)
        tensor = self._preprocess_torch(images)
        if next(self.model.parameters()).device != tensor.device:
            self.model.to(tensor.device)
        with torch.autocast(
            device_type=tensor.device.type,
            dtype=self._autocast_dtype(),
            enabled=tensor.device.type == "cuda" and self.cfg.precision != "fp32",
        ):
            features = self.model(tensor)
        if isinstance(features, (tuple, list)):
            features = features[0]
        if features.ndim == 3:
            features = self._pool_tokens(features)
        return features.float()

    def _forward_transformers(self, images: np.ndarray | torch.Tensor) -> torch.Tensor:
        device = next(self.model.parameters()).device
        pixel_values = self._preprocess_torch(images).to(device)
        with torch.autocast(
            device_type=device.type,
            dtype=self._autocast_dtype(),
            enabled=device.type == "cuda" and self.cfg.precision != "fp32",
        ):
            outputs = self.model(pixel_values=pixel_values)
        tokens = outputs.last_hidden_state
        return self._pool_tokens(tokens).float()

    def _pool_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        if self.cfg.feature_mode == "cls":
            return tokens[:, 0]
        patch_tokens = tokens[:, self.cfg.num_prefix_tokens :]
        if self.cfg.feature_mode == "mean_patch":
            return patch_tokens.mean(dim=1)
        if self.cfg.feature_mode == "patch_tokens":
            return patch_tokens
        raise ValueError(f"Unsupported feature mode: {self.cfg.feature_mode}")

    def _preprocess_torch(self, images: np.ndarray | torch.Tensor) -> torch.Tensor:
        if isinstance(images, np.ndarray):
            tensor = torch.from_numpy(images)
        else:
            tensor = images
        if tensor.ndim == 3:
            tensor = tensor.unsqueeze(0)
        if tensor.shape[-1] != 3:
            raise ValueError(f"Expected RGB images with last dim 3, got {tuple(tensor.shape)}")
        tensor = tensor.float().permute(0, 3, 1, 2) / 255.0
        tensor = F.interpolate(
            tensor,
            size=(self.cfg.image_size, self.cfg.image_size),
            mode="bilinear",
            align_corners=False,
        )
        mean = tensor.new_tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = tensor.new_tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        tensor = (tensor - mean) / std
        return tensor.to(next(self.model.parameters()).device)

    def _to_numpy_list(self, images: np.ndarray | torch.Tensor) -> list[np.ndarray]:
        if torch.is_tensor(images):
            images = images.detach().cpu().numpy()
        if images.ndim == 3:
            images = images[None]
        return [img.astype(np.uint8, copy=False) for img in images]

    def _autocast_dtype(self) -> torch.dtype:
        if self.cfg.precision == "fp16":
            return torch.float16
        if self.cfg.precision == "bf16":
            return torch.bfloat16
        return torch.float32


@torch.no_grad()
def encode_two_views(
    extractor: DINOFeatureExtractor,
    base_rgb: np.ndarray,
    wrist_rgb: np.ndarray,
    view_fusion: str = "concat",
) -> torch.Tensor:
    images = np.stack([base_rgb, wrist_rgb], axis=0)
    features = extractor(images)
    if view_fusion == "concat":
        if features.ndim == 3:
            return torch.cat([features[0], features[1]], dim=0)
        return torch.cat([features[0], features[1]], dim=-1)
    if view_fusion == "mean":
        return features.mean(dim=0)
    raise ValueError(f"Unsupported view_fusion: {view_fusion}")
