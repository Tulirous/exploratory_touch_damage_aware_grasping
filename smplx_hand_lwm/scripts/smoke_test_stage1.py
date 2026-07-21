from __future__ import annotations

import argparse

import torch

from smplx_hand_lwm.models import (
    HandAdaLNZeroBlock,
    Stage1HandLWM,
    compute_stage1_loss,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--context-length", type=int, default=4)
    parser.add_argument("--future-length", type=int, default=12)
    parser.add_argument("--state-dim", type=int, default=24)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    device = torch.device(
        args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu"
    )
    context = torch.randn(
        args.batch_size, args.context_length, args.state_dim, device=device
    )
    future = torch.randn(
        args.batch_size, args.future_length, args.state_dim, device=device
    )
    joints = torch.randn(
        args.batch_size, args.future_length, 21, 3, device=device
    )
    contact = torch.randint(
        0, 2, (args.batch_size, args.future_length, 5), device=device
    ).float()

    for decoder_type in ("transformer_decoder", "lawam_adaln_zero"):
        model = Stage1HandLWM(
            state_dim=args.state_dim,
            latent_action_dim=16,
            hidden_dim=64,
            context_length=args.context_length,
            future_length=args.future_length,
            encoder_layers=1,
            decoder_layers=1,
            num_heads=4,
            ffn_dim=128,
            wrist_constant_velocity_anchor=True,
            wrist_aware_auxiliary_head=True,
            window_local_wrist_translation=True,
            hmwm_decoder_type=decoder_type,
        ).to(device)
        outputs = model(context, future, sample=False)
        loss, metrics = compute_stage1_loss(
            outputs,
            future,
            hand_context=context,
            joints_future=joints,
            contact_future=contact,
            weights={"la_wrist_cv_correction": 5.0},
        )
        loss.backward()

        assert outputs["latent_action"].shape == (args.batch_size, 16)
        assert outputs["predicted_hand_future"].shape == future.shape
        assert outputs["predicted_wrist_cv_correction"].shape == (
            args.batch_size,
            args.future_length,
            3,
        )
        assert outputs["predicted_joints_future"].shape == joints.shape
        assert model.inverse_dynamics.wrist_correction_head.weight.grad is not None
        assert torch.isfinite(metrics["la_wrist_cv_correction"])
        assert torch.isfinite(loss)

        model.eval()
        translation_shift = torch.randn(
            args.batch_size, 1, 3, device=device
        )
        shifted_context = context.clone()
        shifted_future = future.clone()
        shifted_context[..., :3] += translation_shift
        shifted_future[..., :3] += translation_shift
        with torch.no_grad():
            original_latent = model.encode_latent_action(
                context, future, sample=False
            )["latent_mean"]
            shifted_latent = model.encode_latent_action(
                shifted_context, shifted_future, sample=False
            )["latent_mean"]
        assert torch.allclose(
            original_latent, shifted_latent, atol=1e-6, rtol=1e-6
        )

        if decoder_type == "lawam_adaln_zero":
            blocks = [
                module
                for module in model.hand_world_model.modules()
                if isinstance(module, HandAdaLNZeroBlock)
            ]
            assert blocks
            for block in blocks:
                projection = block.adaln_modulation[-1]
                assert torch.count_nonzero(projection.weight.detach()) == 0
                assert torch.count_nonzero(projection.bias.detach()) == 0
                assert projection.weight.grad is not None
                assert torch.isfinite(projection.weight.grad).all()
                assert torch.count_nonzero(projection.weight.grad) > 0

        print(decoder_type, "latent_action", tuple(outputs["latent_action"].shape))
        print(
            decoder_type,
            "predicted_hand_future",
            tuple(outputs["predicted_hand_future"].shape),
        )
        print(decoder_type, "loss", float(metrics["total"].detach()))
        print(decoder_type, "window_translation_invariance", "ok")
        print(decoder_type, "backward", "ok")


if __name__ == "__main__":
    main()
