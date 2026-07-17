from __future__ import annotations

import numpy as np


class ManoForwardAdapter:
    """Optional MANO/SMPL-X forward adapter producing the 21-point contract.

    MANO model files are licensed separately and are intentionally not distributed
    with this repository. The returned joint ordering is normalized to MediaPipe's
    wrist/thumb/index/middle/ring/pinky convention.
    """

    # Typical SMPL-X MANO output has wrist + 15 articulated joints + 5 fingertips.
    # Callers may override this mapping for a different MANO implementation.
    DEFAULT_TO_21 = [0, 13, 14, 15, 16, 1, 2, 3, 17, 4, 5, 6, 18, 10, 11, 12, 19, 7, 8, 9, 20]

    def __init__(self, model_path: str, is_right: bool = True, joint_mapping: list[int] | None = None) -> None:
        try:
            import smplx
            import torch
        except ImportError as exc:
            raise RuntimeError("MANO replay requires the smplx and torch packages") from exc
        self.torch = torch
        self.mapping = joint_mapping or self.DEFAULT_TO_21
        self.layer = smplx.create(
            model_path,
            model_type="mano",
            is_rhand=is_right,
            use_pca=False,
            flat_hand_mean=False,
        )

    def forward(self, hand_pose: np.ndarray, betas: np.ndarray | None = None) -> np.ndarray:
        pose = np.asarray(hand_pose, dtype=np.float32).reshape(1, 45)
        beta = np.zeros((1, 10), dtype=np.float32) if betas is None else np.asarray(betas, dtype=np.float32).reshape(1, 10)
        with self.torch.no_grad():
            output = self.layer(
                hand_pose=self.torch.from_numpy(pose),
                betas=self.torch.from_numpy(beta),
                global_orient=self.torch.zeros((1, 3)),
                transl=self.torch.zeros((1, 3)),
            )
        joints = output.joints[0].detach().cpu().numpy()
        if max(self.mapping) >= len(joints):
            raise ValueError("Configured MANO joint mapping exceeds model output")
        return joints[self.mapping].astype(np.float64)
