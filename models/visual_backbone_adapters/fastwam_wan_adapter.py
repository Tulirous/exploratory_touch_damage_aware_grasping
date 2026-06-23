from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image


def _center_crop_resize(image: np.ndarray, width: int, height: int) -> np.ndarray:
    pil_image = Image.fromarray(image)
    src_w, src_h = pil_image.size
    scale = max(width / src_w, height / src_h)
    resized = pil_image.resize((round(src_w * scale), round(src_h * scale)), resample=Image.BILINEAR)
    rw, rh = resized.size
    left = max((rw - width) // 2, 0)
    top = max((rh - height) // 2, 0)
    cropped = resized.crop((left, top, left + width, top + height))
    return np.asarray(cropped, dtype=np.uint8)


class FastWAMWanLatentAdapter:
    """Frozen Fast-WAM/Wan visual-token adapter.

    This adapter is intended to run on the server where the official FastWAM
    repository is available. It does not vendor or modify FastWAM code.
    """

    def __init__(
        self,
        fastwam_root: str | Path,
        ckpt_path: str | Path,
        task_config: str = "libero_uncond_2cam224_1e-4",
        device: str = "cuda",
        mixed_precision: str = "bf16",
        prompt: str = "gently grasp the fragile object without damaging it",
        base_size: tuple[int, int] = (224, 224),
        wrist_size: tuple[int, int] = (224, 224),
        concat: str = "horizontal",
        pool: str = "mean",
        tiled: bool = False,
    ) -> None:
        self.fastwam_root = Path(fastwam_root).expanduser().resolve()
        self.ckpt_path = Path(ckpt_path).expanduser().resolve()
        self.task_config = task_config
        self.device = device
        self.prompt = prompt
        self.base_size = base_size
        self.wrist_size = wrist_size
        self.concat = concat
        self.pool = pool
        self.tiled = tiled

        if not self.fastwam_root.exists():
            raise FileNotFoundError(f"FastWAM root not found: {self.fastwam_root}")
        if not self.ckpt_path.exists():
            raise FileNotFoundError(f"FastWAM checkpoint not found: {self.ckpt_path}")
        if str(self.fastwam_root) not in sys.path:
            sys.path.insert(0, str(self.fastwam_root))

        self.model_dtype = self._resolve_dtype(mixed_precision)
        self.model = self._load_model()

    @staticmethod
    def _resolve_dtype(mixed_precision: str) -> torch.dtype:
        key = mixed_precision.strip().lower()
        if key == "bf16":
            return torch.bfloat16
        if key == "fp16":
            return torch.float16
        if key in {"no", "fp32", "float32"}:
            return torch.float32
        raise ValueError(f"Unsupported mixed_precision: {mixed_precision}")

    def _load_model(self) -> torch.nn.Module:
        from hydra import compose, initialize_config_dir
        from hydra.utils import instantiate
        from omegaconf import OmegaConf

        OmegaConf.register_new_resolver("eval", eval, replace=True)
        OmegaConf.register_new_resolver("max", lambda x: max(x), replace=True)
        OmegaConf.register_new_resolver("split", lambda s, idx: s.split("/")[int(idx)], replace=True)

        config_dir = str(self.fastwam_root / "configs")
        with initialize_config_dir(version_base="1.3", config_dir=config_dir):
            cfg = compose(
                config_name="sim_libero",
                overrides=[
                    f"task={self.task_config}",
                    f"ckpt={self.ckpt_path}",
                ],
            )

        model = instantiate(cfg.model, model_dtype=self.model_dtype, device=self.device)
        model.load_checkpoint(str(self.ckpt_path))
        return model.to(self.device).eval()

    def preprocess_pair(self, base_frame: np.ndarray, wrist_frame: np.ndarray) -> torch.Tensor:
        base_w, base_h = self.base_size
        wrist_w, wrist_h = self.wrist_size
        base = _center_crop_resize(base_frame, width=base_w, height=base_h)
        wrist = _center_crop_resize(wrist_frame, width=wrist_w, height=wrist_h)

        if self.concat == "horizontal":
            rgb = np.concatenate([base, wrist], axis=1)
        elif self.concat == "vertical":
            rgb = np.concatenate([base, wrist], axis=0)
        else:
            raise ValueError(f"Unsupported concat mode: {self.concat}")

        x = torch.as_tensor(rgb).permute(2, 0, 1).unsqueeze(0)
        x = x.to(device=self.device, dtype=self.model_dtype)
        return x * (2.0 / 255.0) - 1.0

    @torch.no_grad()
    def encode_frames(self, base_frame: np.ndarray, wrist_frame: np.ndarray, prompt: Optional[str] = None) -> np.ndarray:
        input_image = self.preprocess_pair(base_frame, wrist_frame)
        model = self.model

        first_frame_latents = model._encode_input_image_latents_tensor(
            input_image=input_image,
            tiled=self.tiled,
        )
        context, context_mask = model.encode_prompt(self.prompt if prompt is None else prompt)
        timestep_video = torch.zeros(
            (first_frame_latents.shape[0],),
            dtype=first_frame_latents.dtype,
            device=model.device,
        )
        fuse_flag = bool(getattr(model.video_expert, "fuse_vae_embedding_in_latents", False))
        video_pre = model.video_expert.pre_dit(
            x=first_frame_latents,
            timestep=timestep_video,
            context=context,
            context_mask=context_mask,
            action=None,
            fuse_vae_embedding_in_latents=fuse_flag,
        )
        tokens = video_pre["tokens"]
        if self.pool == "mean":
            latent = tokens.mean(dim=1)
        elif self.pool == "first":
            latent = tokens[:, 0]
        else:
            raise ValueError(f"Unsupported pool mode: {self.pool}")
        return latent.squeeze(0).detach().to(device="cpu", dtype=torch.float32).numpy()
