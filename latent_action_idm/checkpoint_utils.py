from __future__ import annotations

import torch


def remap_legacy_stage1_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Map pre-refactor LaWAMStage1IDM parameter names to modular names."""

    mapped = {}
    for key, value in state_dict.items():
        new_key = key
        replacements = [
            ("input_proj.", "inverse_dynamics.visual_projector.input_proj."),
            ("pos_embed", "inverse_dynamics.visual_projector.pos_embed"),
            ("state_token.", "inverse_dynamics.state_token."),
            ("inverse_cls", "inverse_dynamics.inverse_cls"),
            ("future_type", "inverse_dynamics.future_type"),
            ("state_type", "inverse_dynamics.state_type"),
            ("inverse_encoder.", "inverse_dynamics.encoder."),
            ("posterior.", "inverse_dynamics.posterior."),
            ("latent_to_hidden.", "latent_world_model.latent_to_hidden."),
            ("decoder_blocks.", "latent_world_model.blocks."),
            ("decoder_norm.", "latent_world_model.norm."),
            ("output_proj.", "latent_world_model.output_proj."),
            ("state_predictor.", "state_predictor."),
        ]
        if key.startswith("current_type"):
            mapped["inverse_dynamics.current_type"] = value
            mapped["latent_world_model.current_type"] = value.clone()
            continue
        for old, new in replacements:
            if key.startswith(old):
                new_key = key.replace(old, new, 1)
                break
        mapped[new_key] = value

    # Reuse the inverse visual projection for the LaWM decoder. Future training
    # will learn separate projectors, but this keeps old checkpoints analyzable.
    for key, value in list(mapped.items()):
        prefix = "inverse_dynamics.visual_projector."
        if key.startswith(prefix):
            mapped[key.replace(prefix, "latent_world_model.visual_projector.", 1)] = value.clone()
    return mapped

