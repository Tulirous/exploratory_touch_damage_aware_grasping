from __future__ import annotations

import argparse

import torch

from latent_action_idm.models import FutureTokenEvaluator, metric_future_scores


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
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    visual_t = torch.randn(args.batch_size, args.num_tokens, args.token_dim, device=device)
    visual_future = torch.randn(args.batch_size, args.num_tokens, args.token_dim, device=device)
    predicted_future = visual_future + 0.1 * torch.randn_like(visual_future)
    state_t = torch.randn(args.batch_size, args.state_dim, device=device)
    latent_action = torch.randn(args.batch_size, args.latent_dim, device=device)

    evaluator = FutureTokenEvaluator(
        visual_token_dim=args.token_dim,
        state_dim=args.state_dim,
        latent_action_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.layers,
        num_heads=args.heads,
        ffn_dim=args.hidden_dim * 4,
    ).to(device)
    outputs = evaluator(visual_t, predicted_future, state_t=state_t, latent_action=latent_action)
    for key, value in outputs.items():
        print(key, tuple(value.shape))

    scores = metric_future_scores(visual_t, predicted_future, visual_future)
    for key, value in scores.items():
        print(key, tuple(value.shape))


if __name__ == "__main__":
    main()

