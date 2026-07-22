from __future__ import annotations

import argparse
import random
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from smplx_hand_lwm.datasets import HandSequenceDataset
from smplx_hand_lwm.models import Stage1HandLWM, compute_stage1_loss


def load_config(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def build_model(cfg: dict) -> Stage1HandLWM:
    data = cfg["data"]
    model = cfg["model"]
    return Stage1HandLWM(
        state_dim=int(data["hand_state_dim"]),
        latent_action_dim=int(model["latent_action_dim"]),
        hidden_dim=int(model["hidden_dim"]),
        context_length=int(data["context_length"]),
        future_length=int(data["future_length"]),
        encoder_layers=int(model["encoder_layers"]),
        decoder_layers=int(model["decoder_layers"]),
        num_heads=int(model["num_heads"]),
        ffn_dim=int(model["ffn_dim"]),
        dropout=float(model["dropout"]),
        num_hand_joints=int(data["num_hand_joints"]),
        num_contact_points=int(data["num_contact_points"]),
        residual_prediction=bool(model.get("residual_prediction", True)),
        wrist_constant_velocity_anchor=bool(
            model.get("wrist_constant_velocity_anchor", False)
        ),
        wrist_aware_auxiliary_head=bool(
            model.get("wrist_aware_auxiliary_head", False)
        ),
        window_local_wrist_translation=bool(
            model.get("window_local_wrist_translation", False)
        ),
        hmwm_decoder_type=str(
            model.get("hmwm_decoder_type", "transformer_decoder")
        ),
    )


def load_pretrained_inverse_dynamics(
    model: Stage1HandLWM,
    checkpoint_path: str | Path,
) -> dict:
    """Load only Hand-IDM weights from a complete Stage-1 checkpoint."""

    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"pretrained Hand-IDM checkpoint does not exist: {checkpoint_path}"
        )
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    state_dict = checkpoint.get("model", checkpoint)
    prefix = "inverse_dynamics."
    inverse_dynamics_state = {
        key.removeprefix(prefix): value
        for key, value in state_dict.items()
        if key.startswith(prefix)
    }
    if not inverse_dynamics_state:
        raise ValueError(
            f"checkpoint contains no {prefix!r} parameters: {checkpoint_path}"
        )
    model.inverse_dynamics.load_state_dict(inverse_dynamics_state, strict=True)
    return checkpoint


def format_duration(seconds: float) -> str:
    seconds = max(0, round(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    return f"{minutes:02d}:{seconds:02d}"


def run_epoch(
    model: Stage1HandLWM,
    loader: DataLoader,
    device: torch.device,
    loss_weights: dict[str, float],
    optimizer: torch.optim.Optimizer | None,
    amp_dtype: torch.dtype | None,
    grad_clip_norm: float | None,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    totals: dict[str, float] = {}
    for batch in loader:
        context = batch["hand_context"].to(device, non_blocking=True)
        future = batch["hand_future"].to(device, non_blocking=True)
        joints = batch.get("joints_future")
        contact = batch.get("contact_future")
        joints = joints.to(device, non_blocking=True) if joints is not None else None
        contact = contact.to(device, non_blocking=True) if contact is not None else None

        with torch.set_grad_enabled(training):
            with torch.autocast(
                device_type=device.type,
                dtype=amp_dtype,
                enabled=amp_dtype is not None and device.type == "cuda",
            ):
                outputs = model(context, future)
                loss, metrics = compute_stage1_loss(
                    outputs,
                    future,
                    hand_context=context,
                    joints_future=joints,
                    contact_future=contact,
                    weights=loss_weights,
                )
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if grad_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                optimizer.step()
        for name, value in metrics.items():
            totals[name] = totals.get(name, 0.0) + float(value.detach().cpu())
    return {name: value / max(len(loader), 1) for name, value in totals.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="smplx_hand_lwm/configs/stage1_hand_lwm.yaml",
    )
    parser.add_argument("--train-manifest", default=None)
    parser.add_argument("--val-manifest", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--checkpoint-dir", default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.train_manifest is not None:
        cfg["data"]["train_manifest"] = args.train_manifest
    if args.val_manifest is not None:
        cfg["data"]["val_manifest"] = args.val_manifest
    if args.epochs is not None:
        cfg["training"]["epochs"] = args.epochs
    if args.batch_size is not None:
        cfg["training"]["batch_size"] = args.batch_size
    if args.num_workers is not None:
        cfg["data"]["num_workers"] = args.num_workers
    if args.checkpoint_dir is not None:
        cfg["training"]["checkpoint_dir"] = args.checkpoint_dir
    seed = int(cfg["project"].get("seed", 42))
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
    amp_dtype = None
    if precision == "bf16":
        amp_dtype = torch.bfloat16
    elif precision == "fp16":
        amp_dtype = torch.float16
    elif precision not in {"fp32", "none"}:
        raise ValueError(f"unsupported precision: {precision}")

    data_cfg = cfg["data"]
    dataset_args = {
        "context_length": int(data_cfg["context_length"]),
        "future_length": int(data_cfg["future_length"]),
        "state_dim": int(data_cfg["hand_state_dim"]),
    }
    train_set = HandSequenceDataset(data_cfg["train_manifest"], **dataset_args)
    val_set = HandSequenceDataset(data_cfg["val_manifest"], **dataset_args)
    loader_args = {
        "batch_size": int(cfg["training"]["batch_size"]),
        "num_workers": int(data_cfg.get("num_workers", 0)),
        "pin_memory": device.type == "cuda",
    }
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, **loader_args)

    model = build_model(cfg)
    pretrained_idm_checkpoint = cfg["training"].get(
        "pretrained_idm_checkpoint"
    )
    freeze_inverse_dynamics = bool(
        cfg["training"].get("freeze_inverse_dynamics", False)
    )
    deterministic_teacher_latent = bool(
        cfg["training"].get("deterministic_teacher_latent", False)
    )
    if freeze_inverse_dynamics and not pretrained_idm_checkpoint:
        raise ValueError(
            "freeze_inverse_dynamics=true requires pretrained_idm_checkpoint"
        )
    if deterministic_teacher_latent and not freeze_inverse_dynamics:
        raise ValueError(
            "deterministic_teacher_latent=true requires "
            "freeze_inverse_dynamics=true"
        )
    if pretrained_idm_checkpoint:
        source = load_pretrained_inverse_dynamics(
            model,
            pretrained_idm_checkpoint,
        )
        source_epoch = source.get("epoch", "unknown")
        print(
            "loaded pretrained Hand-IDM "
            f"checkpoint={pretrained_idm_checkpoint} epoch={source_epoch}",
            flush=True,
        )
    if freeze_inverse_dynamics:
        if not deterministic_teacher_latent:
            raise ValueError(
                "a frozen Hand-IDM comparison must set "
                "deterministic_teacher_latent=true"
            )
        model.freeze_inverse_dynamics_()
        print(
            "frozen Hand-IDM: eval mode, no gradients, posterior mean teacher LA",
            flush=True,
        )
    model = model.to(device)
    trainable_parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    if not trainable_parameters:
        raise ValueError("model has no trainable parameters")
    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=float(cfg["training"]["lr"]),
        weight_decay=float(cfg["training"]["weight_decay"]),
    )
    checkpoint_dir = Path(cfg["training"]["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_val = float("inf")
    total_epochs = int(cfg["training"]["epochs"])
    training_started = time.perf_counter()
    for epoch in range(1, total_epochs + 1):
        epoch_started = time.perf_counter()
        train_metrics = run_epoch(
            model,
            train_loader,
            device,
            cfg["training"]["loss_weights"],
            optimizer,
            amp_dtype,
            float(cfg["training"].get("grad_clip_norm", 1.0)),
        )
        with torch.no_grad():
            val_metrics = run_epoch(
                model,
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
                    "model": model.state_dict(),
                    "config": cfg,
                    "epoch": epoch,
                    "val_metrics": val_metrics,
                },
                checkpoint_dir / "best.pt",
            )
        epoch_seconds = time.perf_counter() - epoch_started
        elapsed_seconds = time.perf_counter() - training_started
        average_epoch_seconds = elapsed_seconds / epoch
        eta_seconds = average_epoch_seconds * (total_epochs - epoch)
        auxiliary_status = ""
        if float(cfg["training"]["loss_weights"].get(
            "la_wrist_cv_correction", 0.0
        )) > 0.0:
            auxiliary_status = (
                f" train_la_wrist={train_metrics['la_wrist_cv_correction']:.6f}"
                f" val_la_wrist={val_metrics['la_wrist_cv_correction']:.6f}"
            )
        print(
            f"epoch={epoch:03d}/{total_epochs:03d} "
            f"train={train_metrics['total']:.6f} "
            f"val={val_metrics['total']:.6f} "
            f"best_val={best_val:.6f} "
            f"{auxiliary_status} "
            f"epoch_time={format_duration(epoch_seconds)} "
            f"elapsed={format_duration(elapsed_seconds)} "
            f"eta={format_duration(eta_seconds)}",
            flush=True,
        )


if __name__ == "__main__":
    main()
