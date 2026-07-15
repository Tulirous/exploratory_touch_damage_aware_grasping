from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from smplx_hand_lwm.datasets import HandSequenceDataset
from smplx_hand_lwm.evaluate_stage1 import Metrics, constant_velocity, last_pose
from smplx_hand_lwm.train_stage1 import build_model
from smplx_hand_lwm.train_stage2_prior import build_prior, load_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prior-checkpoint", required=True)
    parser.add_argument("--stage1-checkpoint", default=None)
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    prior_checkpoint = load_checkpoint(args.prior_checkpoint)
    cfg = prior_checkpoint["config"]
    stage1_path = args.stage1_checkpoint or prior_checkpoint["stage1_checkpoint"]
    stage1_checkpoint = load_checkpoint(stage1_path)
    stage1_cfg = stage1_checkpoint["config"]
    data_cfg = stage1_cfg["data"]
    manifest = args.manifest or cfg["data"]["val_manifest"]
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
    stage1 = build_model(stage1_cfg).to(device)
    stage1.load_state_dict(stage1_checkpoint["model"])
    stage1.eval()
    prior = build_prior(cfg, stage1_cfg).to(device)
    prior.load_state_dict(prior_checkpoint["prior"])
    prior.eval()

    metrics = Metrics()
    latent_error_sum = 0.0
    latent_count = 0
    with torch.inference_mode():
        for batch in loader:
            context = batch["hand_context"].to(device)
            future = batch["hand_future"].to(device)
            prior_outputs = prior(context, sample=False)
            prior_future = stage1.decode_future(
                context, prior_outputs["latent_mean"]
            )["predicted_hand_future"]
            teacher_outputs = stage1.encode_latent_action(
                context, future, sample=False
            )
            teacher_future = stage1.decode_future(
                context, teacher_outputs["latent_mean"]
            )["predicted_hand_future"]
            metrics.update("current_only_prior", prior_future, future)
            metrics.update("posterior_teacher", teacher_future, future)
            metrics.update("last_pose", last_pose(context, future.shape[1]), future)
            metrics.update(
                "constant_velocity",
                constant_velocity(context, future.shape[1]),
                future,
            )
            latent_error_sum += float(
                (prior_outputs["latent_mean"] - teacher_outputs["latent_mean"])
                .abs()
                .mean()
            ) * context.shape[0]
            latent_count += context.shape[0]

    result = {
        "prior_checkpoint": str(Path(args.prior_checkpoint).resolve()),
        "stage1_checkpoint": str(Path(stage1_path).resolve()),
        "manifest": str(Path(manifest).resolve()),
        "samples": len(dataset),
        "latent_mean_mae": latent_error_sum / max(latent_count, 1),
        "metrics": metrics.compute(),
        "interpretation": (
            "current_only_prior uses H_context only; posterior_teacher sees the "
            "ground-truth future and is an upper-bound reconstruction reference"
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
