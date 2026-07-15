from __future__ import annotations

import argparse
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from smplx_hand_lwm.datasets import HandSequenceDataset
from smplx_hand_lwm.models import HandLatentActionPrior, Stage1HandLWM
from smplx_hand_lwm.train_stage1 import (
    build_model,
    format_duration,
    load_config,
)


def load_checkpoint(path: str | Path) -> dict:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def build_prior(cfg: dict, stage1_cfg: dict) -> HandLatentActionPrior:
    data = stage1_cfg["data"]
    teacher_model = stage1_cfg["model"]
    prior_cfg = cfg["prior"]
    return HandLatentActionPrior(
        state_dim=int(data["hand_state_dim"]),
        latent_action_dim=int(teacher_model["latent_action_dim"]),
        hidden_dim=int(prior_cfg["hidden_dim"]),
        num_layers=int(prior_cfg["num_layers"]),
        num_heads=int(prior_cfg["num_heads"]),
        ffn_dim=int(prior_cfg["ffn_dim"]),
        dropout=float(prior_cfg["dropout"]),
        max_context_length=int(data["context_length"]),
    )


def temporal_loss(
    predicted: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    if predicted.shape[1] < 2:
        return predicted.new_zeros(())
    return F.smooth_l1_loss(
        predicted[:, 1:] - predicted[:, :-1],
        target[:, 1:] - target[:, :-1],
    )


def compute_prior_loss(
    prior_outputs: dict[str, torch.Tensor],
    teacher_outputs: dict[str, torch.Tensor],
    predicted_future: torch.Tensor,
    target_future: torch.Tensor,
    weights: dict[str, float],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    losses = {
        "latent_mean": F.smooth_l1_loss(
            prior_outputs["latent_mean"], teacher_outputs["latent_mean"]
        ),
        "latent_log_variance": F.smooth_l1_loss(
            prior_outputs["latent_log_variance"].clamp(-10.0, 10.0),
            teacher_outputs["latent_log_variance"].clamp(-10.0, 10.0),
        ),
        "rollout_state": F.smooth_l1_loss(predicted_future, target_future),
        "rollout_wrist_translation": F.smooth_l1_loss(
            predicted_future[..., :3], target_future[..., :3]
        ),
        "rollout_wrist_rotation": F.smooth_l1_loss(
            predicted_future[..., 3:9], target_future[..., 3:9]
        ),
        "rollout_mano_pose": F.smooth_l1_loss(
            predicted_future[..., 9:24], target_future[..., 9:24]
        ),
        "rollout_velocity": temporal_loss(predicted_future, target_future),
    }
    total = sum(float(weights[name]) * value for name, value in losses.items())
    return total, {"total": total, **losses}


def run_epoch(
    prior: HandLatentActionPrior,
    teacher: Stage1HandLWM,
    loader: DataLoader,
    device: torch.device,
    weights: dict[str, float],
    optimizer: torch.optim.Optimizer | None,
    amp_dtype: torch.dtype | None,
    grad_clip_norm: float | None,
) -> dict[str, float]:
    training = optimizer is not None
    prior.train(training)
    teacher.eval()
    totals: dict[str, float] = {}
    sample_count = 0
    for batch in loader:
        context = batch["hand_context"].to(device, non_blocking=True)
        future = batch["hand_future"].to(device, non_blocking=True)
        batch_size = context.shape[0]
        with torch.set_grad_enabled(training):
            with torch.autocast(
                device_type=device.type,
                dtype=amp_dtype,
                enabled=amp_dtype is not None and device.type == "cuda",
            ):
                with torch.no_grad():
                    teacher_outputs = teacher.encode_latent_action(
                        context, future, sample=False
                    )
                prior_outputs = prior(context, sample=False)
                predicted_future = teacher.decode_future(
                    context, prior_outputs["latent_mean"]
                )["predicted_hand_future"]
                loss, metrics = compute_prior_loss(
                    prior_outputs,
                    teacher_outputs,
                    predicted_future,
                    future,
                    weights,
                )
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if grad_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(
                        prior.parameters(), grad_clip_norm
                    )
                optimizer.step()
        for name, value in metrics.items():
            totals[name] = totals.get(name, 0.0) + float(value.detach()) * batch_size
        sample_count += batch_size
    return {name: value / max(sample_count, 1) for name, value in totals.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="smplx_hand_lwm/configs/stage2_hand_prior.yaml",
    )
    parser.add_argument("--stage1-checkpoint", default=None)
    parser.add_argument("--train-manifest", default=None)
    parser.add_argument("--val-manifest", default=None)
    parser.add_argument("--checkpoint-dir", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.stage1_checkpoint:
        cfg["teacher"]["stage1_checkpoint"] = args.stage1_checkpoint
    if args.train_manifest:
        cfg["data"]["train_manifest"] = args.train_manifest
    if args.val_manifest:
        cfg["data"]["val_manifest"] = args.val_manifest
    if args.checkpoint_dir:
        cfg["training"]["checkpoint_dir"] = args.checkpoint_dir
    if args.epochs is not None:
        cfg["training"]["epochs"] = args.epochs
    if args.batch_size is not None:
        cfg["training"]["batch_size"] = args.batch_size
    if args.num_workers is not None:
        cfg["data"]["num_workers"] = args.num_workers

    seed = int(cfg["project"].get("seed", 43))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    requested_device = str(cfg["training"].get("device", "cuda"))
    device = torch.device(
        requested_device
        if requested_device == "cpu" or torch.cuda.is_available()
        else "cpu"
    )
    precision = str(cfg["training"].get("precision", "bf16")).lower()
    amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(precision)
    if precision not in {"bf16", "fp16", "fp32", "none"}:
        raise ValueError(f"unsupported precision: {precision}")

    stage1_path = cfg["teacher"]["stage1_checkpoint"]
    stage1_checkpoint = load_checkpoint(stage1_path)
    stage1_cfg = stage1_checkpoint["config"]
    teacher = build_model(stage1_cfg).to(device)
    teacher.load_state_dict(stage1_checkpoint["model"])
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    prior = build_prior(cfg, stage1_cfg).to(device)

    data_cfg = stage1_cfg["data"]
    dataset_args = {
        "context_length": int(data_cfg["context_length"]),
        "future_length": int(data_cfg["future_length"]),
        "state_dim": int(data_cfg["hand_state_dim"]),
    }
    train_set = HandSequenceDataset(cfg["data"]["train_manifest"], **dataset_args)
    val_set = HandSequenceDataset(cfg["data"]["val_manifest"], **dataset_args)
    loader_args = {
        "batch_size": int(cfg["training"]["batch_size"]),
        "num_workers": int(cfg["data"].get("num_workers", 0)),
        "pin_memory": device.type == "cuda",
    }
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, **loader_args)
    optimizer = torch.optim.AdamW(
        prior.parameters(),
        lr=float(cfg["training"]["lr"]),
        weight_decay=float(cfg["training"]["weight_decay"]),
    )
    checkpoint_dir = Path(cfg["training"]["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    total_epochs = int(cfg["training"]["epochs"])
    best_val = float("inf")
    started = time.perf_counter()
    for epoch in range(1, total_epochs + 1):
        epoch_started = time.perf_counter()
        train_metrics = run_epoch(
            prior,
            teacher,
            train_loader,
            device,
            cfg["training"]["loss_weights"],
            optimizer,
            amp_dtype,
            float(cfg["training"].get("grad_clip_norm", 1.0)),
        )
        val_metrics = run_epoch(
            prior,
            teacher,
            val_loader,
            device,
            cfg["training"]["loss_weights"],
            None,
            amp_dtype,
            None,
        )
        if val_metrics["total"] < best_val:
            best_val = val_metrics["total"]
            torch.save(
                {
                    "prior": prior.state_dict(),
                    "config": cfg,
                    "stage1_checkpoint": str(Path(stage1_path).resolve()),
                    "stage1_config": stage1_cfg,
                    "epoch": epoch,
                    "val_metrics": val_metrics,
                },
                checkpoint_dir / "best.pt",
            )
        epoch_seconds = time.perf_counter() - epoch_started
        elapsed = time.perf_counter() - started
        eta = elapsed / epoch * (total_epochs - epoch)
        print(
            f"epoch={epoch:03d}/{total_epochs:03d} "
            f"train={train_metrics['total']:.6f} "
            f"val={val_metrics['total']:.6f} "
            f"latent={val_metrics['latent_mean']:.6f} "
            f"rollout={val_metrics['rollout_state']:.6f} "
            f"best_val={best_val:.6f} "
            f"epoch_time={format_duration(epoch_seconds)} "
            f"elapsed={format_duration(elapsed)} eta={format_duration(eta)}",
            flush=True,
        )


if __name__ == "__main__":
    main()
