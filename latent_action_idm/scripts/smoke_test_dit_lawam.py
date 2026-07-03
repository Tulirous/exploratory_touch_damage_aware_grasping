from __future__ import annotations

import argparse

import torch

from latent_action_idm.models import DiffusionLatentWorldModel, Stage1DiTLaWAM


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-tokens", type=int, default=392)
    parser.add_argument("--token-dim", type=int, default=768)
    parser.add_argument("--state-dim", type=int, default=7)
    parser.add_argument("--latent-dim", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--diffusion-steps", type=int, default=1000)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    visual_t = torch.randn(args.batch_size, args.num_tokens, args.token_dim, device=device)
    visual_future = torch.randn(args.batch_size, args.num_tokens, args.token_dim, device=device)
    latent_action = torch.randn(args.batch_size, args.latent_dim, device=device)
    state_t = torch.randn(args.batch_size, args.state_dim, device=device)

    dit = DiffusionLatentWorldModel(
        visual_token_dim=args.token_dim,
        latent_action_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.layers,
        num_heads=args.heads,
        ffn_dim=args.hidden_dim * 4,
        max_visual_tokens=512,
        num_diffusion_steps=args.diffusion_steps,
    ).to(device)
    timesteps = dit.sample_timesteps(args.batch_size, device)
    noisy_future, noise = dit.q_sample(visual_future, timesteps)
    pred_noise = dit(noisy_future, visual_t, latent_action, timesteps)
    pred_clean = dit.predict_clean_from_noise(noisy_future, timesteps, pred_noise)
    print("DiT noisy_future:", tuple(noisy_future.shape))
    print("DiT noise target:", tuple(noise.shape))
    print("DiT predicted_noise:", tuple(pred_noise.shape))
    print("DiT predicted_clean:", tuple(pred_clean.shape))

    stage1 = Stage1DiTLaWAM(
        visual_token_dim=args.token_dim,
        state_dim=args.state_dim,
        latent_action_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
        encoder_layers=args.layers,
        decoder_layers=args.layers,
        num_heads=args.heads,
        ffn_dim=args.hidden_dim * 4,
        max_visual_tokens=512,
        num_diffusion_steps=args.diffusion_steps,
    ).to(device)
    outputs = stage1(visual_t, visual_future, state_t)
    print("Stage1DiT latent_mu:", tuple(outputs["latent_mu"].shape))
    print("Stage1DiT predicted_noise:", tuple(outputs["predicted_noise"].shape))
    print("Stage1DiT predicted_visual_future:", tuple(outputs["predicted_visual_future"].shape))
    print("Stage1DiT predicted_state_future:", tuple(outputs["predicted_state_future"].shape))


if __name__ == "__main__":
    main()

