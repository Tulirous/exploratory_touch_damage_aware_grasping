from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from latent_action_idm.checkpoint_utils import remap_legacy_stage1_state_dict
from latent_action_idm.datasets import LatentIDMDataset
from latent_action_idm.models import Stage1DiTLaWAM
from latent_action_idm.train_dit_lawam import build_model, diffusion_loss_weights
from latent_action_idm.train_idm import format_duration, move_batch, prepare_visual_stats
from latent_action_idm.utils import load_config, seed_everything


def extract_module_state(
    state_dict: dict[str, torch.Tensor],
    prefix: str,
) -> dict[str, torch.Tensor]:
    prefix_with_dot = f"{prefix}."
    return {
        key.removeprefix(prefix_with_dot): value
        for key, value in state_dict.items()
        if key.startswith(prefix_with_dot)
    }


def load_idm_teacher(model: Stage1DiTLaWAM, checkpoint_path: Path, device: torch.device) -> dict[str, Any]:
    ckpt = torch.load(checkpoint_path, map_location=device)
    state_dict = ckpt["model"]
    try:
        inverse_state = extract_module_state(state_dict, "inverse_dynamics")
        state_predictor_state = extract_module_state(state_dict, "state_predictor")
        model.inverse_dynamics.load_state_dict(inverse_state)
        model.state_predictor.load_state_dict(state_predictor_state)
    except RuntimeError:
        remapped = remap_legacy_stage1_state_dict(state_dict)
        inverse_state = extract_module_state(remapped, "inverse_dynamics")
        state_predictor_state = extract_module_state(remapped, "state_predictor")
        model.inverse_dynamics.load_state_dict(inverse_state)
        model.state_predictor.load_state_dict(state_predictor_state)

    for param in model.inverse_dynamics.parameters():
        param.requires_grad_(False)
    for param in model.state_predictor.parameters():
        param.requires_grad_(False)
    model.inverse_dynamics.eval()
    model.state_predictor.eval()
    return ckpt


def compute_loss(outputs: dict[str, torch.Tensor], batch: dict, cfg: dict) -> tuple[torch.Tensor, dict]:
    per_sample_denoise = F.mse_loss(outputs["model_prediction"], outputs["diffusion_target"], reduction="none")
    per_sample_denoise = per_sample_denoise.flatten(1).mean(dim=1)
    weights_per_sample = diffusion_loss_weights(outputs, cfg)
    loss_denoise = (per_sample_denoise * weights_per_sample).mean()
    loss_noise = F.mse_loss(outputs["predicted_noise"], outputs["noise"])
    loss_state = F.mse_loss(outputs["predicted_state_future"], batch["state_future"])
    loss_kl = -0.5 * torch.mean(
        1 + outputs["latent_logvar"] - outputs["latent_mu"].pow(2) - outputs["latent_logvar"].exp()
    )
    loss_action_smooth = outputs["latent_action"].pow(2).mean()

    weights = cfg["training"]
    total = (
        float(weights.get("loss_noise", 1.0)) * loss_denoise
        + float(weights.get("loss_state", 0.0)) * loss_state
        + float(weights.get("loss_kl", 0.0)) * loss_kl
        + float(weights.get("loss_action_smooth", 0.0)) * loss_action_smooth
    )
    metrics = {
        "loss": float(total.detach().cpu()),
        "noise": float(loss_denoise.detach().cpu()),
        "epsilon": float(loss_noise.detach().cpu()),
        "state": float(loss_state.detach().cpu()),
        "kl": float(loss_kl.detach().cpu()),
        "action_l2": float(loss_action_smooth.detach().cpu()),
        "loss_weight": float(weights_per_sample.mean().detach().cpu()),
    }
    return total, metrics


def set_frozen_modules_eval(model: Stage1DiTLaWAM) -> None:
    model.inverse_dynamics.eval()
    model.state_predictor.eval()


def run_epoch(
    model: Stage1DiTLaWAM,
    loader: DataLoader,
    cfg: dict,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)
    set_frozen_modules_eval(model)
    totals = {"loss": 0.0, "noise": 0.0, "state": 0.0, "kl": 0.0, "action_l2": 0.0}

    for batch in loader:
        batch = move_batch(batch, device)
        with torch.set_grad_enabled(is_train):
            outputs = model(batch["visual_t"], batch["visual_future"], batch["state_t"])
            loss, metrics = compute_loss(outputs, batch, cfg)
            if is_train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
        for key in totals:
            totals[key] += metrics[key]

    return {key: value / max(len(loader), 1) for key, value in totals.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="latent_action_idm/configs/dit_lawam_frozen_idm_droid100.yaml")
    parser.add_argument("--idm-checkpoint", default=None)
    parser.add_argument("--resume", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed_everything(int(cfg["project"].get("seed", 42)))
    requested_device = cfg["training"].get("device", "cuda")
    device = torch.device(requested_device if torch.cuda.is_available() or requested_device == "cpu" else "cpu")

    visual_stats_path = prepare_visual_stats(cfg)
    train_set = LatentIDMDataset(cfg["data"]["latent_manifest"], visual_stats_path=visual_stats_path)
    val_manifest = Path(cfg["data"]["val_manifest"])
    val_set = (
        LatentIDMDataset(val_manifest, visual_stats_path=visual_stats_path)
        if val_manifest.exists() and val_manifest.stat().st_size
        else None
    )

    train_loader = DataLoader(
        train_set,
        batch_size=int(cfg["training"]["batch_size"]),
        shuffle=True,
        num_workers=int(cfg["data"].get("num_workers", 0)),
    )
    val_loader = (
        DataLoader(
            val_set,
            batch_size=int(cfg["training"]["batch_size"]),
            shuffle=False,
            num_workers=int(cfg["data"].get("num_workers", 0)),
        )
        if val_set is not None
        else None
    )

    model = build_model(cfg).to(device)
    idm_checkpoint = Path(args.idm_checkpoint or cfg["training"]["idm_checkpoint"])
    teacher_ckpt = load_idm_teacher(model, idm_checkpoint, device)

    trainable_params = [param for param in model.parameters() if param.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=float(cfg["training"]["lr"]),
        weight_decay=float(cfg["training"]["weight_decay"]),
    )
    start_epoch = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        set_frozen_modules_eval(model)
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = int(ckpt.get("epoch", -1)) + 1

    checkpoint_dir = Path(cfg["training"]["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_val = float("inf")
    total_epochs = int(cfg["training"]["epochs"])
    train_start = time.perf_counter()
    epoch_durations: list[float] = []

    frozen_param_count = sum(param.numel() for param in model.inverse_dynamics.parameters())
    frozen_param_count += sum(param.numel() for param in model.state_predictor.parameters())
    trainable_param_count = sum(param.numel() for param in trainable_params)
    print(f"loaded frozen IDM teacher: {idm_checkpoint}")
    print(f"teacher epoch: {teacher_ckpt.get('epoch', 'unknown')}")
    print(f"frozen parameters: {frozen_param_count}")
    print(f"trainable parameters: {trainable_param_count}")

    for epoch in range(start_epoch, total_epochs):
        epoch_start = time.perf_counter()
        train_metrics = run_epoch(model, train_loader, cfg, device, optimizer)
        if val_loader is not None:
            with torch.no_grad():
                val_metrics = run_epoch(model, val_loader, cfg, device)
        else:
            val_metrics = train_metrics
        epoch_seconds = time.perf_counter() - epoch_start
        epoch_durations.append(epoch_seconds)
        recent_mean = sum(epoch_durations[-5:]) / min(len(epoch_durations), 5)
        remaining_epochs = max(total_epochs - epoch - 1, 0)
        eta_seconds = recent_mean * remaining_epochs

        print(
            "epoch={:03d} train_loss={:.6f} train_noise={:.6f} train_state={:.6f} "
            "val_loss={:.6f} val_noise={:.6f} val_state={:.6f} epoch_time={} eta={}".format(
                epoch,
                train_metrics["loss"],
                train_metrics["noise"],
                train_metrics["state"],
                val_metrics["loss"],
                val_metrics["noise"],
                val_metrics["state"],
                format_duration(epoch_seconds),
                format_duration(eta_seconds),
            )
        )

        state = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": cfg,
            "idm_checkpoint": str(idm_checkpoint),
            "teacher_epoch": teacher_ckpt.get("epoch"),
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
        }
        torch.save(state, checkpoint_dir / "latest.pt")
        if val_metrics["loss"] < best_val:
            best_val = val_metrics["loss"]
            torch.save(state, checkpoint_dir / "best.pt")
    print(f"training complete total_time={format_duration(time.perf_counter() - train_start)}")


if __name__ == "__main__":
    main()
