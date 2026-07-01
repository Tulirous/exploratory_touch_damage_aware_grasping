from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from latent_action_idm.datasets import LatentIDMDataset
from latent_action_idm.models import LatentActionIDM
from latent_action_idm.utils import load_config, seed_everything


def move_batch(batch: dict, device: torch.device) -> dict:
    moved = {}
    for key, value in batch.items():
        moved[key] = value.to(device) if torch.is_tensor(value) else value
    return moved


def build_model(cfg: dict) -> LatentActionIDM:
    return LatentActionIDM(
        visual_token_dim=int(cfg["model"]["visual_token_dim"]),
        state_dim=int(cfg["model"]["state_dim"]),
        latent_action_dim=int(cfg["model"]["latent_action_dim"]),
        hidden_dim=int(cfg["model"]["hidden_dim"]),
        encoder_layers=int(cfg["model"].get("encoder_layers", 8)),
        decoder_layers=int(cfg["model"].get("decoder_layers", 8)),
        num_heads=int(cfg["model"].get("num_heads", 12)),
        ffn_dim=int(cfg["model"].get("ffn_dim", 3072)),
        dropout=float(cfg["model"].get("dropout", 0.1)),
        max_visual_tokens=int(cfg["model"].get("max_visual_tokens", 512)),
    )


def compute_loss(outputs: dict[str, torch.Tensor], batch: dict, cfg: dict) -> tuple[torch.Tensor, dict]:
    loss_state = F.mse_loss(outputs["predicted_state_future"], batch["state_future"])
    loss_future_latent = F.mse_loss(outputs["predicted_visual_future"], batch["visual_future"])
    loss_kl = -0.5 * torch.mean(
        1 + outputs["latent_logvar"] - outputs["latent_mu"].pow(2) - outputs["latent_logvar"].exp()
    )
    loss_action_smooth = outputs["latent_action"].pow(2).mean()

    weights = cfg["training"]
    total = (
        float(weights.get("loss_future_latent", 1.0)) * loss_future_latent
        + float(weights.get("loss_state", 0.1)) * loss_state
        + float(weights.get("loss_kl", 0.0001)) * loss_kl
        + float(weights.get("loss_action_smooth", 0.001)) * loss_action_smooth
    )
    metrics = {
        "loss": float(total.detach().cpu()),
        "state": float(loss_state.detach().cpu()),
        "future_latent": float(loss_future_latent.detach().cpu()),
        "kl": float(loss_kl.detach().cpu()),
        "action_l2": float(loss_action_smooth.detach().cpu()),
    }
    return total, metrics


def run_epoch(
    model: LatentActionIDM,
    loader: DataLoader,
    cfg: dict,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)
    totals = {"loss": 0.0, "state": 0.0, "future_latent": 0.0, "kl": 0.0, "action_l2": 0.0}

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
    parser.add_argument("--config", default="latent_action_idm/configs/dino_idm.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed_everything(int(cfg["project"].get("seed", 42)))
    requested_device = cfg["training"].get("device", "cuda")
    device = torch.device(requested_device if torch.cuda.is_available() or requested_device == "cpu" else "cpu")

    train_set = LatentIDMDataset(cfg["data"]["latent_manifest"])
    val_manifest = Path(cfg["data"]["val_manifest"])
    val_set = LatentIDMDataset(val_manifest) if val_manifest.exists() and val_manifest.stat().st_size else None

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
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["training"]["lr"]),
        weight_decay=float(cfg["training"]["weight_decay"]),
    )

    checkpoint_dir = Path(cfg["training"]["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_val = float("inf")

    for epoch in range(int(cfg["training"]["epochs"])):
        train_metrics = run_epoch(model, train_loader, cfg, device, optimizer)
        if val_loader is not None:
            with torch.no_grad():
                val_metrics = run_epoch(model, val_loader, cfg, device)
        else:
            val_metrics = train_metrics

        print(
            "epoch={:03d} train_loss={:.6f} train_state={:.6f} "
            "val_loss={:.6f} val_state={:.6f}".format(
                epoch,
                train_metrics["loss"],
                train_metrics["state"],
                val_metrics["loss"],
                val_metrics["state"],
            )
        )

        state = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": cfg,
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
        }
        torch.save(state, checkpoint_dir / "latest.pt")
        if val_metrics["loss"] < best_val:
            best_val = val_metrics["loss"]
            torch.save(state, checkpoint_dir / "best.pt")


if __name__ == "__main__":
    main()
