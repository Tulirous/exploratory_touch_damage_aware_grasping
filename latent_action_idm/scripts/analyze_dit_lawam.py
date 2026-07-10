from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from latent_action_idm.datasets import LatentIDMDataset
from latent_action_idm.models import metric_future_scores
from latent_action_idm.train_dit_lawam import build_model
from latent_action_idm.train_idm import move_batch, prepare_visual_stats


def load_checkpoint(path: Path, device: torch.device) -> tuple[dict[str, Any], torch.nn.Module]:
    ckpt = torch.load(path, map_location=device)
    cfg = ckpt["config"]
    model = build_model(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return cfg, model


def fixed_timesteps(batch_size: int, timestep: int, device: torch.device) -> torch.Tensor:
    return torch.full((batch_size,), int(timestep), dtype=torch.long, device=device)


@torch.no_grad()
def collect_outputs(
    model: torch.nn.Module,
    dataset: LatentIDMDataset,
    batch_size: int,
    device: torch.device,
    timestep: int,
    seed: int,
) -> dict[str, Any]:
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    rows = []
    arrays: dict[str, list[np.ndarray]] = {
        "latent_mu": [],
        "state_delta": [],
        "pred_state_delta": [],
        "predicted_visual_future": [],
        "patch_mse": [],
        "base_patch_mse": [],
        "wrist_patch_mse": [],
    }
    totals = {
        "future_mse_sum": 0.0,
        "identity_future_mse_sum": 0.0,
        "noise_mse_sum": 0.0,
        "state_mse_sum": 0.0,
        "identity_state_mse_sum": 0.0,
        "delta_cosine_sum": 0.0,
        "improvement_sum": 0.0,
        "count": 0,
    }

    for batch in loader:
        batch = move_batch(batch, device)
        count = int(batch["visual_t"].shape[0])
        t = fixed_timesteps(count, timestep, device)
        noise = torch.randn(
            batch["visual_future"].shape,
            dtype=batch["visual_future"].dtype,
            device=device,
            generator=generator,
        )
        outputs = model(batch["visual_t"], batch["visual_future"], batch["state_t"], diffusion_timesteps=t, noise=noise)
        scores = metric_future_scores(
            batch["visual_t"],
            outputs["predicted_visual_future"],
            batch["visual_future"],
        )
        noise_mse = F.mse_loss(outputs["predicted_noise"], outputs["noise"], reduction="sum")
        state_mse = F.mse_loss(outputs["predicted_state_future"], batch["state_future"], reduction="sum")
        identity_state_mse = F.mse_loss(batch["state_t"], batch["state_future"], reduction="sum")

        token_dims = batch["visual_future"].shape[1] * batch["visual_future"].shape[2]
        totals["future_mse_sum"] += float(scores["future_mse"].sum().cpu()) * token_dims
        totals["identity_future_mse_sum"] += float(scores["identity_mse"].sum().cpu()) * token_dims
        totals["noise_mse_sum"] += float(noise_mse.cpu())
        totals["state_mse_sum"] += float(state_mse.cpu())
        totals["identity_state_mse_sum"] += float(identity_state_mse.cpu())
        totals["delta_cosine_sum"] += float(scores["transition_delta_cosine"].sum().cpu())
        totals["improvement_sum"] += float(scores["future_improvement_vs_identity"].sum().cpu())
        totals["count"] += count

        patch_mse = (outputs["predicted_visual_future"] - batch["visual_future"]).pow(2).mean(dim=-1)
        state_delta = batch["state_future"] - batch["state_t"]
        pred_state_delta = outputs["predicted_state_future"] - batch["state_t"]
        arrays["latent_mu"].append(outputs["latent_mu"].cpu().numpy())
        arrays["state_delta"].append(state_delta.cpu().numpy())
        arrays["pred_state_delta"].append(pred_state_delta.cpu().numpy())
        arrays["predicted_visual_future"].append(outputs["predicted_visual_future"].cpu().numpy())
        arrays["patch_mse"].append(patch_mse.cpu().numpy())
        arrays["base_patch_mse"].append(patch_mse[:, :196].cpu().numpy())
        arrays["wrist_patch_mse"].append(patch_mse[:, 196:392].cpu().numpy())

        for i in range(len(batch["episode_id"])):
            rows.append(
                {
                    "episode_id": batch["episode_id"][i],
                    "sample_id": batch["sample_id"][i],
                    "t_index": int(batch["t_index"][i]),
                    "future_index": int(batch["future_index"][i]),
                    "future_mse": float(scores["future_mse"][i].cpu()),
                    "identity_mse": float(scores["identity_mse"][i].cpu()),
                    "future_improvement_vs_identity": float(scores["future_improvement_vs_identity"][i].cpu()),
                    "transition_delta_cosine": float(scores["transition_delta_cosine"][i].cpu()),
                }
            )

    merged = {key: np.concatenate(value, axis=0) for key, value in arrays.items()}
    sample = dataset[0]
    token_count = int(np.prod(sample["visual_t"].shape))
    state_dim = int(sample["state_t"].numel())
    count = int(totals["count"])
    metrics = {
        "eval_diffusion_timestep": float(timestep),
        "future_mse_per_token": totals["future_mse_sum"] / max(count * token_count, 1),
        "identity_future_mse_per_token": totals["identity_future_mse_sum"] / max(count * token_count, 1),
        "noise_mse_per_token": totals["noise_mse_sum"] / max(count * token_count, 1),
        "state_mse_per_dim": totals["state_mse_sum"] / max(count * state_dim, 1),
        "identity_state_mse_per_dim": totals["identity_state_mse_sum"] / max(count * state_dim, 1),
        "transition_delta_cosine": totals["delta_cosine_sum"] / max(count, 1),
        "sample_mean_future_improvement_vs_identity": totals["improvement_sum"] / max(count, 1),
    }
    metrics["future_improvement_vs_identity"] = 1.0 - (
        metrics["future_mse_per_token"] / max(metrics["identity_future_mse_per_token"], 1e-12)
    )
    metrics["state_improvement_vs_identity"] = 1.0 - (
        metrics["state_mse_per_dim"] / max(metrics["identity_state_mse_per_dim"], 1e-12)
    )
    return {"rows": rows, "arrays": merged, "metrics": metrics}


@torch.no_grad()
def collect_sample_outputs(
    model: torch.nn.Module,
    dataset: LatentIDMDataset,
    batch_size: int,
    device: torch.device,
    sampler: str,
    sample_steps: int,
    eta: float,
    seed: int,
) -> dict[str, Any]:
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    rows = []
    arrays: dict[str, list[np.ndarray]] = {
        "latent_mu": [],
        "state_delta": [],
        "pred_state_delta": [],
        "predicted_visual_future": [],
        "patch_mse": [],
        "base_patch_mse": [],
        "wrist_patch_mse": [],
    }
    totals = {
        "future_mse_sum": 0.0,
        "identity_future_mse_sum": 0.0,
        "state_mse_sum": 0.0,
        "identity_state_mse_sum": 0.0,
        "delta_cosine_sum": 0.0,
        "improvement_sum": 0.0,
        "count": 0,
    }

    for batch in loader:
        batch = move_batch(batch, device)
        count = int(batch["visual_t"].shape[0])
        posterior = model.inverse_dynamics(batch["visual_t"], batch["visual_future"], batch["state_t"], sample=False)
        latent_action = posterior["latent_action"]
        predicted_future = model.latent_world_model.sample(
            visual_t=batch["visual_t"],
            latent_action=latent_action,
            shape=tuple(batch["visual_future"].shape),
            num_steps=sample_steps,
            sampler=sampler,
            eta=eta,
            generator=generator,
        )
        predicted_state_future = model.predict_state_future(batch["state_t"], latent_action)
        scores = metric_future_scores(batch["visual_t"], predicted_future, batch["visual_future"])
        state_mse = F.mse_loss(predicted_state_future, batch["state_future"], reduction="sum")
        identity_state_mse = F.mse_loss(batch["state_t"], batch["state_future"], reduction="sum")

        token_dims = batch["visual_future"].shape[1] * batch["visual_future"].shape[2]
        totals["future_mse_sum"] += float(scores["future_mse"].sum().cpu()) * token_dims
        totals["identity_future_mse_sum"] += float(scores["identity_mse"].sum().cpu()) * token_dims
        totals["state_mse_sum"] += float(state_mse.cpu())
        totals["identity_state_mse_sum"] += float(identity_state_mse.cpu())
        totals["delta_cosine_sum"] += float(scores["transition_delta_cosine"].sum().cpu())
        totals["improvement_sum"] += float(scores["future_improvement_vs_identity"].sum().cpu())
        totals["count"] += count

        patch_mse = (predicted_future - batch["visual_future"]).pow(2).mean(dim=-1)
        state_delta = batch["state_future"] - batch["state_t"]
        pred_state_delta = predicted_state_future - batch["state_t"]
        arrays["latent_mu"].append(posterior["latent_mu"].cpu().numpy())
        arrays["state_delta"].append(state_delta.cpu().numpy())
        arrays["pred_state_delta"].append(pred_state_delta.cpu().numpy())
        arrays["predicted_visual_future"].append(predicted_future.cpu().numpy())
        arrays["patch_mse"].append(patch_mse.cpu().numpy())
        arrays["base_patch_mse"].append(patch_mse[:, :196].cpu().numpy())
        arrays["wrist_patch_mse"].append(patch_mse[:, 196:392].cpu().numpy())

        for i in range(len(batch["episode_id"])):
            rows.append(
                {
                    "episode_id": batch["episode_id"][i],
                    "sample_id": batch["sample_id"][i],
                    "t_index": int(batch["t_index"][i]),
                    "future_index": int(batch["future_index"][i]),
                    "future_mse": float(scores["future_mse"][i].cpu()),
                    "identity_mse": float(scores["identity_mse"][i].cpu()),
                    "future_improvement_vs_identity": float(scores["future_improvement_vs_identity"][i].cpu()),
                    "transition_delta_cosine": float(scores["transition_delta_cosine"][i].cpu()),
                }
            )

    merged = {key: np.concatenate(value, axis=0) for key, value in arrays.items()}
    sample = dataset[0]
    token_count = int(np.prod(sample["visual_t"].shape))
    state_dim = int(sample["state_t"].numel())
    count = int(totals["count"])
    metrics = {
        "sampler": sampler,
        "sample_steps": float(sample_steps),
        "sample_eta": float(eta),
        "future_mse_per_token": totals["future_mse_sum"] / max(count * token_count, 1),
        "identity_future_mse_per_token": totals["identity_future_mse_sum"] / max(count * token_count, 1),
        "state_mse_per_dim": totals["state_mse_sum"] / max(count * state_dim, 1),
        "identity_state_mse_per_dim": totals["identity_state_mse_sum"] / max(count * state_dim, 1),
        "transition_delta_cosine": totals["delta_cosine_sum"] / max(count, 1),
        "sample_mean_future_improvement_vs_identity": totals["improvement_sum"] / max(count, 1),
    }
    metrics["future_improvement_vs_identity"] = 1.0 - (
        metrics["future_mse_per_token"] / max(metrics["identity_future_mse_per_token"], 1e-12)
    )
    metrics["state_improvement_vs_identity"] = 1.0 - (
        metrics["state_mse_per_dim"] / max(metrics["identity_state_mse_per_dim"], 1e-12)
    )
    return {"rows": rows, "arrays": merged, "metrics": metrics}


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_patch_maps(prefix: Path, arrays: dict[str, np.ndarray]) -> None:
    base_mean = arrays["base_patch_mse"].mean(axis=0).reshape(14, 14)
    wrist_mean = arrays["wrist_patch_mse"].mean(axis=0).reshape(14, 14)
    np.savetxt(prefix.with_name(prefix.name + "_base_patch_mse_14x14.csv"), base_mean, delimiter=",")
    np.savetxt(prefix.with_name(prefix.name + "_wrist_patch_mse_14x14.csv"), wrist_mean, delimiter=",")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/dit_lawam/best.pt")
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--split-name", default="val")
    parser.add_argument("--output-dir", default="outputs/dit_lawam")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--timestep", type=int, default=500)
    parser.add_argument("--sampler", choices=["single", "ddim", "ddpm"], default="single")
    parser.add_argument("--sample-steps", type=int, default=50)
    parser.add_argument("--eta", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    cfg, model = load_checkpoint(Path(args.checkpoint), device)
    manifest = args.manifest or cfg["data"]["val_manifest"]
    visual_stats_path = prepare_visual_stats(cfg)
    dataset = LatentIDMDataset(manifest, visual_stats_path=visual_stats_path)
    if args.sampler == "single":
        result = collect_outputs(model, dataset, args.batch_size, device, args.timestep, args.seed)
    else:
        result = collect_sample_outputs(
            model,
            dataset,
            args.batch_size,
            device,
            args.sampler,
            args.sample_steps,
            args.eta,
            args.seed,
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = output_dir / args.split_name
    np.savez_compressed(prefix.with_suffix(".npz"), **result["arrays"])
    write_csv(prefix.with_suffix(".csv"), result["rows"])
    write_patch_maps(prefix, result["arrays"])
    with prefix.with_suffix(".metrics.txt").open("w", encoding="utf-8") as f:
        for key, value in sorted(result["metrics"].items()):
            line = f"{key}: {value:.8f}" if isinstance(value, float) else f"{key}: {value}"
            print(line)
            f.write(line + "\n")


if __name__ == "__main__":
    main()
