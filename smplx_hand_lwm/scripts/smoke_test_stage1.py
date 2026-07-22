from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import torch

from smplx_hand_lwm.models import (
    HandAdaLNCrossAttentionBlock,
    HandAdaLNZeroBlock,
    Stage1HandLWM,
    compute_stage1_loss,
)
from smplx_hand_lwm.train_stage1 import load_pretrained_inverse_dynamics


def build_smoke_model(
    args: argparse.Namespace,
    device: torch.device,
    decoder_type: str,
) -> Stage1HandLWM:
    return Stage1HandLWM(
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

    for decoder_type in (
        "transformer_decoder",
        "lawam_adaln_zero",
        "lawam_adaln_zero_context",
        "adaln_zero_cross_attention",
    ):
        model = build_smoke_model(args, device, decoder_type)
        input_token_lengths: list[int] = []
        hook = None
        if decoder_type.startswith("lawam_adaln_zero"):
            hook = model.hand_world_model.blocks[0].register_forward_pre_hook(
                lambda _module, inputs: input_token_lengths.append(
                    inputs[0].shape[1]
                )
            )
        elif decoder_type == "adaln_zero_cross_attention":
            hook = model.hand_world_model.blocks[0].register_forward_pre_hook(
                lambda _module, inputs: input_token_lengths.extend(
                    [inputs[0].shape[1], inputs[1].shape[1]]
                )
            )
        outputs = model(context, future, sample=False)
        if hook is not None:
            hook.remove()
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

        if decoder_type.startswith("lawam_adaln_zero"):
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
            expected_tokens = args.future_length
            if decoder_type == "lawam_adaln_zero_context":
                expected_tokens += args.context_length
                assert model.hand_world_model.include_context_tokens
            else:
                assert not model.hand_world_model.include_context_tokens
            assert input_token_lengths == [expected_tokens]
        elif decoder_type == "adaln_zero_cross_attention":
            blocks = [
                module
                for module in model.hand_world_model.modules()
                if isinstance(module, HandAdaLNCrossAttentionBlock)
            ]
            assert blocks
            for block in blocks:
                projection = block.adaln_modulation[-1]
                assert torch.count_nonzero(projection.weight.detach()) == 0
                assert torch.count_nonzero(projection.bias.detach()) == 0
                assert projection.weight.grad is not None
                assert torch.isfinite(projection.weight.grad).all()
                assert torch.count_nonzero(projection.weight.grad) > 0
            assert input_token_lengths == [
                args.future_length,
                args.context_length,
            ]
            with torch.no_grad():
                prediction_a = model.decode_future(
                    context,
                    torch.randn(args.batch_size, 16, device=device),
                )["predicted_hand_future"]
                prediction_b = model.decode_future(
                    context,
                    torch.randn(args.batch_size, 16, device=device),
                )["predicted_hand_future"]
            assert torch.allclose(
                prediction_a, prediction_b, atol=1e-6, rtol=1e-6
            )

        print(decoder_type, "latent_action", tuple(outputs["latent_action"].shape))
        print(
            decoder_type,
            "predicted_hand_future",
            tuple(outputs["predicted_hand_future"].shape),
        )
        print(decoder_type, "loss", float(metrics["total"].detach()))
        print(decoder_type, "window_translation_invariance", "ok")
        print(decoder_type, "backward", "ok")

    source = build_smoke_model(
        args,
        device,
        "adaln_zero_cross_attention",
    )
    target = build_smoke_model(args, device, "transformer_decoder")
    decoder_before_load = {
        name: value.detach().clone()
        for name, value in target.hand_world_model.state_dict().items()
    }
    with tempfile.TemporaryDirectory() as temporary_directory:
        checkpoint_path = Path(temporary_directory) / "source.pt"
        torch.save(
            {"model": source.state_dict(), "epoch": 7},
            checkpoint_path,
        )
        checkpoint = load_pretrained_inverse_dynamics(
            target,
            checkpoint_path,
        )
    assert checkpoint["epoch"] == 7
    for name, value in source.inverse_dynamics.state_dict().items():
        assert torch.equal(value, target.inverse_dynamics.state_dict()[name])
    for name, value in target.hand_world_model.state_dict().items():
        assert torch.equal(value, decoder_before_load[name])

    target.freeze_inverse_dynamics_().train()
    assert not target.inverse_dynamics.training
    assert all(
        not parameter.requires_grad
        for parameter in target.inverse_dynamics.parameters()
    )
    frozen_outputs_a = target(context, future)
    frozen_outputs_b = target(context, future)
    assert torch.equal(
        frozen_outputs_a["latent_action"],
        frozen_outputs_a["latent_mean"],
    )
    assert torch.equal(
        frozen_outputs_a["latent_action"],
        frozen_outputs_b["latent_action"],
    )
    frozen_loss, _ = compute_stage1_loss(
        frozen_outputs_a,
        future,
        hand_context=context,
        weights={"la_wrist_cv_correction": 5.0},
    )
    frozen_loss.backward()
    assert all(
        parameter.grad is None
        for parameter in target.inverse_dynamics.parameters()
    )
    assert any(
        parameter.grad is not None
        for parameter in target.hand_world_model.parameters()
    )
    print("frozen_v2_idm", "load_only_inverse_dynamics", "ok")
    print("frozen_v2_idm", "deterministic_teacher_latent", "ok")
    print("frozen_v2_idm", "decoder_only_backward", "ok")


if __name__ == "__main__":
    main()
