from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from latent_action_idm.checkpoint_utils import remap_legacy_stage1_state_dict
from latent_action_idm.datasets import LatentIDMDataset, compute_visual_stats
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
        use_state_in_idm=bool(cfg["model"].get("use_state_in_idm", True)),
        num_views=int(cfg["model"].get("num_views", 0)),
        residual_future_prediction=bool(cfg["model"].get("residual_future_prediction", False)),
    )


def prepare_visual_stats(cfg: dict) -> str | None:
    normalizer_cfg = cfg["data"].get("visual_normalization", {})
    if not normalizer_cfg or not bool(normalizer_cfg.get("enabled", False)):
        return None
    stats_path = normalizer_cfg.get("stats_path")
    if not stats_path:
        raise ValueError("data.visual_normalization.stats_path is required when normalization is enabled")
    stats_path = str(stats_path)
    path = Path(stats_path)
    if not path.exists() or bool(normalizer_cfg.get("overwrite", False)):
        print(f"computing visual normalization stats -> {stats_path}")
        compute_visual_stats(
            cfg["data"]["latent_manifest"],
            stats_path,
            eps=float(normalizer_cfg.get("eps", 1e-6)),
        )
    return stats_path


def format_duration(seconds: float) -> str:
    seconds = max(int(round(seconds)), 0)
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours:d}h{minutes:02d}m{seconds:02d}s"
    if minutes:
        return f"{minutes:d}m{seconds:02d}s"
    return f"{seconds:d}s"


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
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
) -> dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)
    totals = {"loss": 0.0, "state": 0.0, "future_latent": 0.0, "kl": 0.0, "action_l2": 0.0}
    grad_clip_norm = cfg["training"].get("grad_clip_norm")

    for batch in loader:
        batch = move_batch(batch, device)
        with torch.set_grad_enabled(is_train):
            outputs = model(batch["visual_t"], batch["visual_future"], batch["state_t"])
            loss, metrics = compute_loss(outputs, batch, cfg)
            if is_train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if grad_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip_norm))
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
        for key in totals:
            totals[key] += metrics[key]

    return {key: value / max(len(loader), 1) for key, value in totals.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="latent_action_idm/configs/dino_idm.yaml")
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
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["training"]["lr"]),
        weight_decay=float(cfg["training"]["weight_decay"]),
    )
    scheduler = None
    if cfg["training"].get("lr_schedule") == "cosine":
        total_steps = max(len(train_loader) * int(cfg["training"]["epochs"]), 1)
        min_lr = float(cfg["training"].get("min_lr", 0.0))
        base_lr = float(cfg["training"]["lr"])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=total_steps,
            eta_min=min_lr,
        )
        if min_lr > base_lr:
            raise ValueError(f"min_lr must be <= lr, got min_lr={min_lr} lr={base_lr}")
    start_epoch = 0

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        try:
            model.load_state_dict(ckpt["model"])
        except RuntimeError:
            model.load_state_dict(remap_legacy_stage1_state_dict(ckpt["model"]))
        if "optimizer" in ckpt:
            try:
                optimizer.load_state_dict(ckpt["optimizer"])
            except ValueError:
                print("warning: optimizer state is incompatible with the refactored model; starting optimizer fresh")
        if scheduler is not None and "scheduler" in ckpt:
            try:
                scheduler.load_state_dict(ckpt["scheduler"])
            except ValueError:
                print("warning: scheduler state is incompatible; starting scheduler fresh")
        start_epoch = int(ckpt.get("epoch", -1)) + 1

    checkpoint_dir = Path(cfg["training"]["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_val = float("inf")
    total_epochs = int(cfg["training"]["epochs"])
    train_start = time.perf_counter()
    epoch_durations: list[float] = []

    for epoch in range(start_epoch, total_epochs):
        epoch_start = time.perf_counter()
        train_metrics = run_epoch(model, train_loader, cfg, device, optimizer, scheduler)
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
            "epoch={:03d} train_loss={:.6f} train_state={:.6f} "
            "val_loss={:.6f} val_state={:.6f} epoch_time={} eta={}".format(
                epoch,
                train_metrics["loss"],
                train_metrics["state"],
                val_metrics["loss"],
                val_metrics["state"],
                format_duration(epoch_seconds),
                format_duration(eta_seconds),
            )
        )

        state = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "config": cfg,
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
