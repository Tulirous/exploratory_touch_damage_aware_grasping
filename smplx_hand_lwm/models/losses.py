from __future__ import annotations

import torch
import torch.nn.functional as F


def compute_stage1_loss(
    outputs: dict[str, torch.Tensor],
    hand_future: torch.Tensor,
    hand_context: torch.Tensor | None = None,
    joints_future: torch.Tensor | None = None,
    contact_future: torch.Tensor | None = None,
    weights: dict[str, float] | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute losses without requiring proprietary SMPL-X model assets."""

    weights = weights or {}
    predicted = outputs["predicted_hand_future"]
    if predicted.shape != hand_future.shape:
        raise ValueError(
            f"predicted and target hand sequences must match, got "
            f"{tuple(predicted.shape)} and {tuple(hand_future.shape)}"
        )

    losses: dict[str, torch.Tensor] = {}
    losses["state"] = F.smooth_l1_loss(predicted, hand_future)
    losses["wrist_translation"] = F.smooth_l1_loss(
        predicted[..., :3], hand_future[..., :3]
    )
    losses["wrist_rotation"] = F.smooth_l1_loss(
        predicted[..., 3:9], hand_future[..., 3:9]
    )
    losses["mano_pose"] = (
        F.smooth_l1_loss(predicted[..., 9:24], hand_future[..., 9:24])
        if predicted.shape[-1] == 24
        else predicted.new_zeros(())
    )
    losses["velocity"] = _temporal_l1(predicted, hand_future, order=1)
    losses["acceleration"] = _temporal_l1(predicted, hand_future, order=2)
    losses["kl"] = -0.5 * torch.mean(
        1
        + outputs["latent_log_variance"]
        - outputs["latent_mean"].pow(2)
        - outputs["latent_log_variance"].exp()
    )
    losses["latent_l2"] = outputs["latent_action"].pow(2).mean()

    losses["la_wrist_cv_correction"] = predicted.new_zeros(())
    if "predicted_wrist_cv_correction" in outputs:
        if hand_context is None:
            raise ValueError(
                "hand_context is required by the wrist-aware LA auxiliary head"
            )
        target_correction = _wrist_cv_correction_target(
            hand_context,
            hand_future,
        )
        auxiliary_prediction = outputs["predicted_wrist_cv_correction"]
        if auxiliary_prediction.shape != target_correction.shape:
            raise ValueError(
                "predicted wrist correction and target must match, got "
                f"{tuple(auxiliary_prediction.shape)} and "
                f"{tuple(target_correction.shape)}"
            )
        losses["la_wrist_cv_correction"] = F.smooth_l1_loss(
            auxiliary_prediction,
            target_correction,
        )

    zero = predicted.new_zeros(())
    losses["joints"] = zero
    if joints_future is not None:
        losses["joints"] = F.smooth_l1_loss(
            outputs["predicted_joints_future"], joints_future
        )
    losses["contact"] = zero
    if contact_future is not None:
        losses["contact"] = F.binary_cross_entropy_with_logits(
            outputs["predicted_contact_logits"],
            contact_future.float(),
        )

    default_weights = {
        "state": 1.0,
        "wrist_translation": 1.0,
        "wrist_rotation": 0.5,
        "mano_pose": 0.5,
        "velocity": 0.2,
        "acceleration": 0.05,
        "joints": 1.0,
        "contact": 0.2,
        "kl": 1e-4,
        "latent_l2": 1e-5,
        "la_wrist_cv_correction": 0.0,
    }
    default_weights.update(weights)
    total = sum(default_weights[name] * value for name, value in losses.items())
    return total, {"total": total, **losses}


def _wrist_cv_correction_target(
    hand_context: torch.Tensor,
    hand_future: torch.Tensor,
) -> torch.Tensor:
    """Future wrist residual relative to a constant-velocity trajectory."""

    if hand_context.ndim != 3 or hand_context.shape[1] < 2:
        raise ValueError(
            "wrist CV correction requires hand_context shaped [B, T>=2, D]"
        )
    if hand_context.shape[0] != hand_future.shape[0]:
        raise ValueError("hand_context and hand_future batch sizes must match")
    wrist_velocity = hand_context[:, -1, :3] - hand_context[:, -2, :3]
    steps = torch.arange(
        1,
        hand_future.shape[1] + 1,
        device=hand_future.device,
        dtype=hand_future.dtype,
    )
    cv_trajectory = (
        hand_context[:, -1:, :3]
        + steps[None, :, None] * wrist_velocity[:, None, :]
    )
    return hand_future[..., :3] - cv_trajectory


def _temporal_l1(
    predicted: torch.Tensor,
    target: torch.Tensor,
    order: int,
) -> torch.Tensor:
    if predicted.shape[1] <= order:
        return predicted.new_zeros(())
    pred_delta = predicted
    target_delta = target
    for _ in range(order):
        pred_delta = pred_delta[:, 1:] - pred_delta[:, :-1]
        target_delta = target_delta[:, 1:] - target_delta[:, :-1]
    return F.smooth_l1_loss(pred_delta, target_delta)
