from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from latent_action_idm.checkpoint_utils import remap_legacy_stage1_state_dict
from latent_action_idm.datasets import LatentIDMDataset
from latent_action_idm.train_idm import build_model, move_batch
from latent_action_idm.utils import load_config


def ridge_r2(x: np.ndarray, y: np.ndarray, ridge: float = 1e-3) -> float:
    """Fit a closed-form ridge regressor and return in-sample R^2."""

    if x.shape[0] < 2:
        return float("nan")
    x_aug = np.concatenate([x, np.ones((x.shape[0], 1), dtype=x.dtype)], axis=1)
    xtx = x_aug.T @ x_aug
    xtx = xtx + ridge * np.eye(xtx.shape[0], dtype=x.dtype)
    weights = np.linalg.solve(xtx, x_aug.T @ y)
    pred = x_aug @ weights
    ss_res = float(np.square(y - pred).sum())
    ss_tot = float(np.square(y - y.mean(axis=0, keepdims=True)).sum())
    if ss_tot <= 1e-12:
        return float("nan")
    return 1.0 - ss_res / ss_tot


def load_checkpoint(path: Path, device: torch.device) -> tuple[dict[str, Any], torch.nn.Module]:
    ckpt = torch.load(path, map_location=device)
    cfg = ckpt["config"]
    model = build_model(cfg).to(device)
    state_dict = ckpt["model"]
    try:
        model.load_state_dict(state_dict)
    except RuntimeError:
        model.load_state_dict(remap_legacy_stage1_state_dict(state_dict))
    model.eval()
    return cfg, model


def visual_stats_path_from_config(cfg: dict) -> str | None:
    normalizer_cfg = cfg.get("data", {}).get("visual_normalization", {})
    if not normalizer_cfg or not bool(normalizer_cfg.get("enabled", False)):
        return None
    return normalizer_cfg.get("stats_path")




@torch.no_grad()
def collect_outputs(
    model: torch.nn.Module,
    dataset: LatentIDMDataset,
    batch_size: int,
    device: torch.device,
) -> dict[str, Any]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    visual_stats = getattr(dataset, "visual_stats", None)
    stats_tensors = None
    if visual_stats is not None:
        mean, std = visual_stats
        stats_tensors = (
            torch.from_numpy(mean).to(device=device, dtype=torch.float32),
            torch.from_numpy(std).to(device=device, dtype=torch.float32),
        )
    rows = []
    arrays: dict[str, list[np.ndarray]] = {
        "latent_mu": [],
        "latent_logvar": [],
        "state_delta": [],
        "pred_state_delta": [],
        "state_future": [],
        "pred_state_future": [],
    }
    total_future_mse = 0.0
    total_state_mse = 0.0
    total_identity_future_mse = 0.0
    total_identity_state_mse = 0.0
    total_count = 0

    for batch in loader:
        batch = move_batch(batch, device)
        outputs = model(batch["visual_t"], batch["visual_future"], batch["state_t"])
        count = int(batch["visual_t"].shape[0])
        total_count += count

        pred_visual_future = outputs["predicted_visual_future"]
        visual_future = batch["visual_future"]
        visual_t = batch["visual_t"]
        if stats_tensors is not None:
            mean, std = stats_tensors
            pred_visual_future = pred_visual_future * std + mean
            visual_future = visual_future * std + mean
            visual_t = visual_t * std + mean

        future_mse = F.mse_loss(pred_visual_future, visual_future, reduction="sum")
        state_mse = F.mse_loss(outputs["predicted_state_future"], batch["state_future"], reduction="sum")
        identity_future_mse = F.mse_loss(visual_t, visual_future, reduction="sum")
        identity_state_mse = F.mse_loss(batch["state_t"], batch["state_future"], reduction="sum")
        total_future_mse += float(future_mse.cpu())
        total_state_mse += float(state_mse.cpu())
        total_identity_future_mse += float(identity_future_mse.cpu())
        total_identity_state_mse += float(identity_state_mse.cpu())

        state_delta = batch["state_future"] - batch["state_t"]
        pred_state_delta = outputs["predicted_state_future"] - batch["state_t"]
        arrays["latent_mu"].append(outputs["latent_mu"].cpu().numpy())
        arrays["latent_logvar"].append(outputs["latent_logvar"].cpu().numpy())
        arrays["state_delta"].append(state_delta.cpu().numpy())
        arrays["pred_state_delta"].append(pred_state_delta.cpu().numpy())
        arrays["state_future"].append(batch["state_future"].cpu().numpy())
        arrays["pred_state_future"].append(outputs["predicted_state_future"].cpu().numpy())

        batch_size_actual = len(batch["episode_id"])
        for i in range(batch_size_actual):
            rows.append(
                {
                    "episode_id": batch["episode_id"][i],
                    "sample_id": batch["sample_id"][i],
                    "t_index": int(batch["t_index"][i]),
                    "future_index": int(batch["future_index"][i]),
                }
            )

    merged = {key: np.concatenate(value, axis=0) for key, value in arrays.items()}
    token_count = np.prod(dataset[0]["visual_t"].shape)
    state_dim = dataset[0]["state_t"].numel()
    metrics = {
        "future_mse_per_token": total_future_mse / max(total_count * token_count, 1),
        "state_mse_per_dim": total_state_mse / max(total_count * state_dim, 1),
        "identity_future_mse_per_token": total_identity_future_mse / max(total_count * token_count, 1),
        "identity_state_mse_per_dim": total_identity_state_mse / max(total_count * state_dim, 1),
    }
    metrics["future_improvement_vs_identity"] = 1.0 - (
        metrics["future_mse_per_token"] / max(metrics["identity_future_mse_per_token"], 1e-12)
    )
    metrics["state_improvement_vs_identity"] = 1.0 - (
        metrics["state_mse_per_dim"] / max(metrics["identity_state_mse_per_dim"], 1e-12)
    )
    return {"rows": rows, "arrays": merged, "metrics": metrics}


def latent_smoothness(rows: list[dict[str, Any]], latent_mu: np.ndarray) -> float:
    by_episode: dict[str, list[tuple[int, int]]] = {}
    for index, row in enumerate(rows):
        by_episode.setdefault(row["episode_id"], []).append((int(row["t_index"]), index))
    distances = []
    for episode_rows in by_episode.values():
        episode_rows = sorted(episode_rows)
        for (_, left), (_, right) in zip(episode_rows[:-1], episode_rows[1:]):
            distances.append(float(np.linalg.norm(latent_mu[left] - latent_mu[right])))
    return float(np.mean(distances)) if distances else float("nan")


def write_csv(path: Path, rows: list[dict[str, Any]], arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    latent_mu = arrays["latent_mu"]
    state_delta = arrays["state_delta"]
    pred_state_delta = arrays["pred_state_delta"]
    fieldnames = ["episode_id", "sample_id", "t_index", "future_index"]
    fieldnames += [f"z_{i}" for i in range(latent_mu.shape[1])]
    fieldnames += [f"state_delta_{i}" for i in range(state_delta.shape[1])]
    fieldnames += [f"pred_state_delta_{i}" for i in range(pred_state_delta.shape[1])]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i, row in enumerate(rows):
            out = dict(row)
            out.update({f"z_{j}": float(latent_mu[i, j]) for j in range(latent_mu.shape[1])})
            out.update({f"state_delta_{j}": float(state_delta[i, j]) for j in range(state_delta.shape[1])})
            out.update(
                {f"pred_state_delta_{j}": float(pred_state_delta[i, j]) for j in range(pred_state_delta.shape[1])}
            )
            writer.writerow(out)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/latent_action_idm/best.pt")
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--split-name", default="val")
    parser.add_argument("--output-dir", default="outputs/latent_action_idm")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    cfg, model = load_checkpoint(Path(args.checkpoint), device)
    manifest = args.manifest or cfg["data"]["val_manifest"]
    dataset = LatentIDMDataset(manifest, visual_stats_path=visual_stats_path_from_config(cfg))
    result = collect_outputs(model, dataset, args.batch_size, device)
    arrays = result["arrays"]
    metrics = result["metrics"]

    latent_mu = arrays["latent_mu"]
    state_delta = arrays["state_delta"]
    metrics["latent_mu_mean_abs"] = float(np.abs(latent_mu).mean())
    metrics["latent_mu_std_mean"] = float(latent_mu.std(axis=0).mean())
    metrics["latent_mu_std_min"] = float(latent_mu.std(axis=0).min())
    metrics["latent_mu_std_max"] = float(latent_mu.std(axis=0).max())
    metrics["latent_to_state_delta_r2"] = ridge_r2(latent_mu.astype(np.float64), state_delta.astype(np.float64))
    metrics["temporal_latent_l2"] = latent_smoothness(result["rows"], latent_mu)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = output_dir / args.split_name
    np.savez_compressed(prefix.with_suffix(".npz"), **arrays)
    write_csv(prefix.with_suffix(".csv"), result["rows"], arrays)
    with prefix.with_suffix(".metrics.txt").open("w", encoding="utf-8") as f:
        for key, value in sorted(metrics.items()):
            line = f"{key}: {value:.8f}" if isinstance(value, float) else f"{key}: {value}"
            print(line)
            f.write(line + "\n")


if __name__ == "__main__":
    main()
