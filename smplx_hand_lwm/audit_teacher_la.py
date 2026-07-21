from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from smplx_hand_lwm.datasets import HandSequenceDataset
from smplx_hand_lwm.evaluate_stage1 import constant_velocity
from smplx_hand_lwm.train_stage1 import build_model


def load_checkpoint(path: str | Path) -> dict:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def extract_teacher_la(
    model: torch.nn.Module,
    manifest: str | Path,
    cfg: dict,
    device: torch.device,
    batch_size: int,
    num_workers: int,
) -> dict[str, np.ndarray]:
    data_cfg = cfg["data"]
    dataset = HandSequenceDataset(
        manifest,
        context_length=int(data_cfg["context_length"]),
        future_length=int(data_cfg["future_length"]),
        state_dim=int(data_cfg["hand_state_dim"]),
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    collected: dict[str, list[np.ndarray]] = {
        "latent_mean": [],
        "latent_log_variance": [],
        "wrist_displacement": [],
        "wrist_cv_correction": [],
        "wrist_rotation_change": [],
        "mano_pose_change": [],
    }
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            context = batch["hand_context"].to(device)
            future = batch["hand_future"].to(device)
            teacher = model.encode_latent_action(context, future, sample=False)
            cv = constant_velocity(context, future.shape[1])
            targets = {
                "latent_mean": teacher["latent_mean"],
                "latent_log_variance": teacher["latent_log_variance"],
                "wrist_displacement": (
                    future[..., :3] - context[:, -1:, :3]
                ),
                "wrist_cv_correction": future[..., :3] - cv[..., :3],
                "wrist_rotation_change": (
                    future[..., 3:9] - context[:, -1:, 3:9]
                ),
                "mano_pose_change": (
                    future[..., 9:24] - context[:, -1:, 9:24]
                ),
            }
            for name, value in targets.items():
                collected[name].append(
                    value.detach().cpu().float().numpy().reshape(context.shape[0], -1)
                )
    return {
        name: np.concatenate(values, axis=0).astype(np.float64)
        for name, values in collected.items()
    }


class RidgeProbe:
    def __init__(self, alpha: float) -> None:
        self.alpha = alpha

    def fit(self, x: np.ndarray, y: np.ndarray) -> "RidgeProbe":
        self.x_mean = x.mean(axis=0)
        self.x_std = x.std(axis=0).clip(min=1e-6)
        self.y_mean = y.mean(axis=0)
        self.y_std = y.std(axis=0).clip(min=1e-6)
        x_norm = (x - self.x_mean) / self.x_std
        y_norm = (y - self.y_mean) / self.y_std
        x_augmented = np.concatenate(
            [x_norm, np.ones((x_norm.shape[0], 1), dtype=x_norm.dtype)], axis=1
        )
        identity = np.eye(x_augmented.shape[1], dtype=x_augmented.dtype)
        identity[-1, -1] = 0.0
        self.weight = np.linalg.solve(
            x_augmented.T @ x_augmented + self.alpha * identity,
            x_augmented.T @ y_norm,
        )
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        x_norm = (x - self.x_mean) / self.x_std
        x_augmented = np.concatenate(
            [x_norm, np.ones((x_norm.shape[0], 1), dtype=x_norm.dtype)], axis=1
        )
        return (x_augmented @ self.weight) * self.y_std + self.y_mean


def r2_score(target: np.ndarray, predicted: np.ndarray) -> float:
    residual = np.sum((target - predicted) ** 2)
    centered = np.sum((target - target.mean(axis=0, keepdims=True)) ** 2)
    return float(1.0 - residual / max(centered, 1e-12))


def target_metrics(
    name: str,
    target: np.ndarray,
    predicted: np.ndarray,
    future_length: int,
) -> dict[str, float]:
    result = {
        "r2": r2_score(target, predicted),
        "mae": float(np.abs(target - predicted).mean()),
    }
    if name in {"wrist_displacement", "wrist_cv_correction"}:
        target_sequence = target.reshape(-1, future_length, 3)
        predicted_sequence = predicted.reshape(-1, future_length, 3)
        error = np.linalg.norm(target_sequence - predicted_sequence, axis=-1)
        result["trajectory_ade_mm"] = float(error.mean() * 1000.0)
        result["trajectory_fde_mm"] = float(error[:, -1].mean() * 1000.0)
    elif name == "wrist_rotation_change":
        result["rotation6d_mae"] = result["mae"]
    elif name == "mano_pose_change":
        result["mano_pca_mae"] = result["mae"]
    return result


def probe_target(
    name: str,
    train: dict[str, np.ndarray],
    val: dict[str, np.ndarray],
    alpha: float,
    future_length: int,
) -> dict[str, Any]:
    probe = RidgeProbe(alpha).fit(train["latent_mean"], train[name])
    train_prediction = probe.predict(train["latent_mean"])
    val_prediction = probe.predict(val["latent_mean"])
    train_mean_baseline = np.broadcast_to(
        train[name].mean(axis=0, keepdims=True), train[name].shape
    )
    val_mean_baseline = np.broadcast_to(
        train[name].mean(axis=0, keepdims=True), val[name].shape
    )
    return {
        "train": target_metrics(
            name, train[name], train_prediction, future_length
        ),
        "val": target_metrics(name, val[name], val_prediction, future_length),
        "train_mean_baseline": target_metrics(
            name, train[name], train_mean_baseline, future_length
        ),
        "val_train_mean_baseline": target_metrics(
            name, val[name], val_mean_baseline, future_length
        ),
    }


def latent_statistics(values: dict[str, np.ndarray]) -> dict[str, Any]:
    mean = values["latent_mean"]
    log_variance = values["latent_log_variance"]
    dimension_std = mean.std(axis=0)
    return {
        "samples": int(mean.shape[0]),
        "dimension": int(mean.shape[1]),
        "mean_absolute_value": float(np.abs(mean).mean()),
        "dimension_std_min": float(dimension_std.min()),
        "dimension_std_median": float(np.median(dimension_std)),
        "dimension_std_max": float(dimension_std.max()),
        "active_dimensions_std_gt_0_05": int(np.sum(dimension_std > 0.05)),
        "active_dimensions_std_gt_0_10": int(np.sum(dimension_std > 0.10)),
        "mean_log_variance": float(log_variance.mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--train-manifest", required=True)
    parser.add_argument("--val-manifest", required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--ridge-alpha", type=float, default=1.0)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    checkpoint = load_checkpoint(args.checkpoint)
    cfg = checkpoint["config"]
    device = torch.device(
        args.device
        if args.device == "cpu" or torch.cuda.is_available()
        else "cpu"
    )
    model = build_model(cfg).to(device)
    model.load_state_dict(checkpoint["model"])
    train = extract_teacher_la(
        model,
        args.train_manifest,
        cfg,
        device,
        args.batch_size,
        args.num_workers,
    )
    val = extract_teacher_la(
        model,
        args.val_manifest,
        cfg,
        device,
        args.batch_size,
        args.num_workers,
    )
    future_length = int(cfg["data"]["future_length"])
    target_names = (
        "wrist_displacement",
        "wrist_cv_correction",
        "wrist_rotation_change",
        "mano_pose_change",
    )
    result = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "ridge_alpha": args.ridge_alpha,
        "interpretation": (
            "The probe is fit on train teacher-LA means only. High train and val "
            "R2 means the motion variable is linearly accessible in teacher LA; "
            "high train but low val indicates clip-specific latent semantics."
        ),
        "latent_statistics": {
            "train": latent_statistics(train),
            "val": latent_statistics(val),
        },
        "probes": {
            name: probe_target(
                name, train, val, args.ridge_alpha, future_length
            )
            for name in target_names
        },
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(result, indent=2, ensure_ascii=False)
    output_path.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
