from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from smplx_hand_lwm.datasets import HandSequenceDataset
from smplx_hand_lwm.evaluate_stage1 import Metrics, constant_velocity, last_pose
from smplx_hand_lwm.train_stage1 import build_model


class IndexedDataset(Dataset):
    def __init__(self, dataset: HandSequenceDataset) -> None:
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = self.dataset[index]
        item["sample_index"] = index
        return item


def load_checkpoint(path: str | Path) -> dict:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def feature_statistics(dataset: HandSequenceDataset) -> dict[str, Any]:
    paths = sorted({str(row["sequence_path"]) for row in dataset.rows})
    states: list[np.ndarray] = []
    for value in paths:
        path = dataset._resolve_path(value)
        with np.load(path, allow_pickle=False) as data:
            state = np.asarray(data["hand_state"], dtype=np.float64)
            valid = np.asarray(
                data["valid"] if "valid" in data else np.ones(state.shape[0]),
                dtype=bool,
            )
            states.append(state[valid])
    merged = np.concatenate(states, axis=0)
    groups = {
        "wrist_translation": merged[:, :3],
        "wrist_rotation6d": merged[:, 3:9],
        "mano_pca": merged[:, 9:24],
    }
    return {
        "unique_tracks": len(paths),
        "unique_valid_frames": int(merged.shape[0]),
        "groups": {
            name: {
                "mean": values.mean(axis=0).tolist(),
                "std": values.std(axis=0).tolist(),
                "min": values.min(axis=0).tolist(),
                "max": values.max(axis=0).tolist(),
                "mean_absolute_value": float(np.abs(values).mean()),
            }
            for name, values in groups.items()
        },
    }


def diagnose_split(
    name: str,
    manifest: str | Path,
    model: torch.nn.Module,
    cfg: dict,
    device: torch.device,
    batch_size: int,
    num_workers: int,
) -> dict[str, Any]:
    data_cfg = cfg["data"]
    dataset = HandSequenceDataset(
        manifest,
        context_length=int(data_cfg["context_length"]),
        future_length=int(data_cfg["future_length"]),
        state_dim=int(data_cfg["hand_state_dim"]),
    )
    loader = DataLoader(
        IndexedDataset(dataset),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    horizon = int(data_cfg["future_length"])
    horizon_totals = {
        key: np.zeros(horizon, dtype=np.float64)
        for key in (
            "posterior_wrist_error_mm",
            "constant_velocity_wrist_error_mm",
            "last_pose_wrist_error_mm",
            "predicted_correction_norm_mm",
            "target_correction_norm_mm",
            "correction_error_mm",
        )
    }
    correction_direction_sum = np.zeros(horizon, dtype=np.float64)
    correction_direction_count = np.zeros(horizon, dtype=np.int64)
    correction_opposite_count = np.zeros(horizon, dtype=np.int64)
    predicted_correction_vector_sum = np.zeros((horizon, 3), dtype=np.float64)
    target_correction_vector_sum = np.zeros((horizon, 3), dtype=np.float64)
    horizon_count = 0
    clip_totals: dict[str, dict[str, float]] = defaultdict(
        lambda: defaultdict(float)
    )
    metrics = Metrics()
    use_cv_anchor = bool(
        cfg["model"].get("wrist_constant_velocity_anchor", False)
    )
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            context = batch["hand_context"].to(device)
            future = batch["hand_future"].to(device)
            posterior = model.encode_latent_action(context, future, sample=False)
            predicted = model.decode_future(
                context, posterior["latent_mean"]
            )["predicted_hand_future"]
            cv = constant_velocity(context, horizon)
            static = last_pose(context, horizon)
            anchor = cv if use_cv_anchor else static
            metrics.update("posterior", predicted, future)
            metrics.update("constant_velocity", cv, future)
            metrics.update("last_pose", static, future)

            posterior_error = torch.linalg.vector_norm(
                predicted[..., :3] - future[..., :3], dim=-1
            )
            cv_error = torch.linalg.vector_norm(
                cv[..., :3] - future[..., :3], dim=-1
            )
            static_error = torch.linalg.vector_norm(
                static[..., :3] - future[..., :3], dim=-1
            )
            predicted_correction = predicted[..., :3] - anchor[..., :3]
            target_correction = future[..., :3] - anchor[..., :3]
            correction_error = predicted_correction - target_correction
            predicted_correction_norm = torch.linalg.vector_norm(
                predicted_correction, dim=-1
            )
            target_correction_norm = torch.linalg.vector_norm(
                target_correction, dim=-1
            )
            direction_valid = (predicted_correction_norm > 1e-3) & (
                target_correction_norm > 1e-3
            )
            direction_cosine = torch.sum(
                predicted_correction * target_correction, dim=-1
            ) / (
                predicted_correction_norm * target_correction_norm
            ).clamp_min(1e-12)
            tensors = {
                "posterior_wrist_error_mm": posterior_error,
                "constant_velocity_wrist_error_mm": cv_error,
                "last_pose_wrist_error_mm": static_error,
                "predicted_correction_norm_mm": predicted_correction_norm,
                "target_correction_norm_mm": target_correction_norm,
                "correction_error_mm": torch.linalg.vector_norm(
                    correction_error, dim=-1
                ),
            }
            for key, value in tensors.items():
                horizon_totals[key] += (
                    value.sum(dim=0).detach().cpu().double().numpy() * 1000.0
                )
            correction_direction_sum += (
                (direction_cosine * direction_valid)
                .sum(dim=0)
                .detach()
                .cpu()
                .double()
                .numpy()
            )
            correction_direction_count += (
                direction_valid.sum(dim=0).detach().cpu().numpy()
            )
            correction_opposite_count += (
                ((direction_cosine < 0) & direction_valid)
                .sum(dim=0)
                .detach()
                .cpu()
                .numpy()
            )
            predicted_correction_vector_sum += (
                predicted_correction.sum(dim=0)
                .detach()
                .cpu()
                .double()
                .numpy()
                * 1000.0
            )
            target_correction_vector_sum += (
                target_correction.sum(dim=0)
                .detach()
                .cpu()
                .double()
                .numpy()
                * 1000.0
            )
            horizon_count += context.shape[0]

            indices = batch["sample_index"].tolist()
            for local_index, sample_index in enumerate(indices):
                row = dataset.rows[int(sample_index)]
                clip_id = str(row.get("clip_id", row["episode_id"]))
                clip = clip_totals[clip_id]
                clip["samples"] += 1
                clip["posterior_error_sum_mm"] += float(
                    posterior_error[local_index].sum()
                ) * 1000.0
                clip["constant_velocity_error_sum_mm"] += float(
                    cv_error[local_index].sum()
                ) * 1000.0
                clip["last_pose_error_sum_mm"] += float(
                    static_error[local_index].sum()
                ) * 1000.0
                clip["posterior_fde_sum_mm"] += float(
                    posterior_error[local_index, -1]
                ) * 1000.0
                clip["constant_velocity_fde_sum_mm"] += float(
                    cv_error[local_index, -1]
                ) * 1000.0
                clip["last_pose_fde_sum_mm"] += float(
                    static_error[local_index, -1]
                ) * 1000.0

    clips = []
    for clip_id, values in clip_totals.items():
        samples = int(values["samples"])
        positions = samples * horizon
        clips.append(
            {
                "clip_id": clip_id,
                "samples": samples,
                "posterior_wrist_ade_mm": values[
                    "posterior_error_sum_mm"
                ] / positions,
                "constant_velocity_wrist_ade_mm": values[
                    "constant_velocity_error_sum_mm"
                ] / positions,
                "last_pose_wrist_ade_mm": values[
                    "last_pose_error_sum_mm"
                ] / positions,
                "posterior_wrist_fde_mm": values[
                    "posterior_fde_sum_mm"
                ] / samples,
                "constant_velocity_wrist_fde_mm": values[
                    "constant_velocity_fde_sum_mm"
                ] / samples,
                "last_pose_wrist_fde_mm": values[
                    "last_pose_fde_sum_mm"
                ] / samples,
            }
        )
    clips.sort(key=lambda item: item["posterior_wrist_ade_mm"], reverse=True)
    horizon_metrics = {
        key: (values / max(horizon_count, 1)).tolist()
        for key, values in horizon_totals.items()
    }
    direction_denominator = np.maximum(correction_direction_count, 1)
    horizon_metrics["correction_cosine_similarity"] = (
        correction_direction_sum / direction_denominator
    ).tolist()
    horizon_metrics["correction_opposite_direction_fraction"] = (
        correction_opposite_count / direction_denominator
    ).tolist()
    horizon_metrics["correction_direction_valid_samples"] = (
        correction_direction_count.tolist()
    )
    horizon_metrics["predicted_correction_mean_vector_mm"] = (
        predicted_correction_vector_sum / max(horizon_count, 1)
    ).tolist()
    horizon_metrics["target_correction_mean_vector_mm"] = (
        target_correction_vector_sum / max(horizon_count, 1)
    ).tolist()
    predicted_mean = float(
        np.mean(horizon_metrics["predicted_correction_norm_mm"])
    )
    target_mean = float(np.mean(horizon_metrics["target_correction_norm_mm"]))
    return {
        "split": name,
        "manifest": str(Path(manifest).resolve()),
        "samples": len(dataset),
        "anchor": "constant_velocity" if use_cv_anchor else "last_pose",
        "overall_metrics": metrics.compute(),
        "horizon_metrics": horizon_metrics,
        "correction_summary": {
            "predicted_correction_mean_mm": predicted_mean,
            "target_correction_mean_mm": target_mean,
            "predicted_to_target_ratio": predicted_mean / max(target_mean, 1e-8),
            "correction_cosine_similarity": float(
                correction_direction_sum.sum()
                / max(correction_direction_count.sum(), 1)
            ),
            "opposite_direction_fraction": float(
                correction_opposite_count.sum()
                / max(correction_direction_count.sum(), 1)
            ),
            "direction_threshold_mm": 1.0,
        },
        "feature_statistics": feature_statistics(dataset),
        "clips_by_posterior_wrist_ade_desc": clips,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--train-manifest", required=True)
    parser.add_argument("--val-manifest", required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda")
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
    train = diagnose_split(
        "train",
        args.train_manifest,
        model,
        cfg,
        device,
        args.batch_size,
        args.num_workers,
    )
    val = diagnose_split(
        "val",
        args.val_manifest,
        model,
        cfg,
        device,
        args.batch_size,
        args.num_workers,
    )
    train_metrics = train["overall_metrics"]
    val_metrics = val["overall_metrics"]
    gap_keys = (
        "posterior/wrist_ade_mm",
        "posterior/wrist_fde_mm",
        "posterior/mano_pca_mae",
        "posterior/rotation6d_mae",
        "posterior/state_mae",
    )
    result = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "train": train,
        "val": val,
        "generalization_gap_val_over_train": {
            key: val_metrics[key] / max(train_metrics[key], 1e-12)
            for key in gap_keys
        },
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(result, indent=2, ensure_ascii=False)
    output_path.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
