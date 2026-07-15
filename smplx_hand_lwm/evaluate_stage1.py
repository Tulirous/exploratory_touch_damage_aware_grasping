from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from smplx_hand_lwm.datasets import HandSequenceDataset
from smplx_hand_lwm.train_stage1 import build_model


class Metrics:
    def __init__(self) -> None:
        self.totals: dict[str, float] = {}
        self.counts: dict[str, int] = {}

    def update(
        self,
        prefix: str,
        predicted: torch.Tensor,
        target: torch.Tensor,
    ) -> None:
        batch_size = target.shape[0]
        wrist_error = torch.linalg.vector_norm(
            predicted[..., :3] - target[..., :3], dim=-1
        )
        values = {
            f"{prefix}/state_mae": (predicted - target).abs().mean(),
            f"{prefix}/wrist_ade_mm": wrist_error.mean() * 1000.0,
            f"{prefix}/wrist_fde_mm": wrist_error[:, -1].mean() * 1000.0,
            f"{prefix}/rotation6d_mae": (
                predicted[..., 3:9] - target[..., 3:9]
            ).abs().mean(),
            f"{prefix}/mano_pca_mae": (
                predicted[..., 9:24] - target[..., 9:24]
            ).abs().mean(),
        }
        for name, value in values.items():
            self.totals[name] = self.totals.get(name, 0.0) + float(value) * batch_size
            self.counts[name] = self.counts.get(name, 0) + batch_size

    def compute(self) -> dict[str, float]:
        return {
            name: value / max(self.counts[name], 1)
            for name, value in sorted(self.totals.items())
        }


def last_pose(context: torch.Tensor, horizon: int) -> torch.Tensor:
    return context[:, -1:, :].expand(-1, horizon, -1)


def constant_velocity(context: torch.Tensor, horizon: int) -> torch.Tensor:
    velocity = context[:, -1, :] - context[:, -2, :]
    steps = torch.arange(
        1, horizon + 1, device=context.device, dtype=context.dtype
    ).view(1, horizon, 1)
    return context[:, -1:, :] + steps * velocity[:, None, :]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", default=None)
    parser.add_argument("--shuffle-seed", type=int, default=1234)
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = checkpoint["config"]
    data_cfg = cfg["data"]
    manifest = args.manifest or data_cfg["val_manifest"]
    dataset = HandSequenceDataset(
        manifest,
        context_length=int(data_cfg["context_length"]),
        future_length=int(data_cfg["future_length"]),
        state_dim=int(data_cfg["hand_state_dim"]),
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    device = torch.device(
        args.device
        if args.device == "cpu" or torch.cuda.is_available()
        else "cpu"
    )
    model = build_model(cfg).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    metrics = Metrics()
    with torch.inference_mode():
        for batch in loader:
            context = batch["hand_context"].to(device)
            future = batch["hand_future"].to(device)
            posterior = model.encode_latent_action(context, future, sample=False)
            reconstruction = model.decode_future(
                context, posterior["latent_action"]
            )["predicted_hand_future"]
            metrics.update("posterior", reconstruction, future)
            metrics.update("last_pose", last_pose(context, future.shape[1]), future)
            metrics.update(
                "constant_velocity",
                constant_velocity(context, future.shape[1]),
                future,
            )

        shuffle_generator = torch.Generator().manual_seed(args.shuffle_seed)
        shuffle_loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True,
            generator=shuffle_generator,
            num_workers=args.num_workers,
        )
        for batch in shuffle_loader:
            context = batch["hand_context"].to(device)
            future = batch["hand_future"].to(device)
            if context.shape[0] <= 1:
                continue
            posterior = model.encode_latent_action(context, future, sample=False)
            permutation = torch.randperm(
                context.shape[0], generator=shuffle_generator
            ).to(device)
            if torch.equal(
                permutation,
                torch.arange(context.shape[0], device=device),
            ):
                permutation = permutation.roll(1)
            shuffled_prediction = model.decode_future(
                context, posterior["latent_action"][permutation]
            )["predicted_hand_future"]
            metrics.update("shuffle_z", shuffled_prediction, future)

    result = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "manifest": str(Path(manifest).resolve()),
        "samples": len(dataset),
        "metrics": metrics.compute(),
        "interpretation": (
            "posterior uses the ground-truth future through the IDM; this evaluates "
            "Stage-1 reconstruction and latent usage, not future-only inference"
        ),
    }
    rendered = json.dumps(result, indent=2, ensure_ascii=False)
    print(rendered)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
