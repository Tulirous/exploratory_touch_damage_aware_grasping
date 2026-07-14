from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from latent_action_idm.datasets import LatentIDMDataset
from latent_action_idm.scripts.analyze_latent_actions import load_checkpoint, visual_stats_path_from_config
from latent_action_idm.train_idm import move_batch
from latent_action_idm.utils import read_jsonl


def flatten_mean_tokens(tokens: np.ndarray) -> np.ndarray:
    return tokens.mean(axis=0)


def normalize_rows(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    return x / np.linalg.norm(x, axis=1, keepdims=True).clip(min=eps)


def load_npz_visual(row: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    data = np.load(row["latent_path"])
    return data["visual_t"].astype(np.float32), data["visual_future"].astype(np.float32)


def metric_init() -> dict[str, float]:
    return {
        "future_mse_sum": 0.0,
        "identity_mse_sum": 0.0,
        "delta_mse_sum": 0.0,
        "delta_var_sum": 0.0,
        "delta_cosine_sum": 0.0,
        "count": 0.0,
        "numel": 0.0,
    }


def metric_update(metrics: dict[str, float], pred: np.ndarray, visual_t: np.ndarray, visual_future: np.ndarray) -> None:
    pred_delta = pred - visual_t
    true_delta = visual_future - visual_t
    future_err = pred - visual_future
    identity_err = visual_t - visual_future
    metrics["future_mse_sum"] += float(np.square(future_err).sum())
    metrics["identity_mse_sum"] += float(np.square(identity_err).sum())
    metrics["delta_mse_sum"] += float(np.square(pred_delta - true_delta).sum())
    metrics["delta_var_sum"] += float(np.square(true_delta - true_delta.mean()).sum())
    pred_flat = pred_delta.reshape(-1)
    true_flat = true_delta.reshape(-1)
    denom = max(float(np.linalg.norm(pred_flat) * np.linalg.norm(true_flat)), 1e-12)
    metrics["delta_cosine_sum"] += float(np.dot(pred_flat, true_flat) / denom)
    metrics["count"] += 1.0
    metrics["numel"] += float(visual_future.size)


def metric_finalize(metrics: dict[str, float]) -> dict[str, float]:
    future_mse = metrics["future_mse_sum"] / max(metrics["numel"], 1.0)
    identity_mse = metrics["identity_mse_sum"] / max(metrics["numel"], 1.0)
    delta_mse = metrics["delta_mse_sum"] / max(metrics["numel"], 1.0)
    delta_var = metrics["delta_var_sum"] / max(metrics["numel"], 1.0)
    return {
        "future_mse_per_token": future_mse,
        "identity_future_mse_per_token": identity_mse,
        "future_improvement_vs_identity": 1.0 - future_mse / max(identity_mse, 1e-12),
        "delta_mse_per_token": delta_mse,
        "delta_r2": 1.0 - delta_mse / max(delta_var, 1e-12),
        "transition_delta_cosine": metrics["delta_cosine_sum"] / max(metrics["count"], 1.0),
        "num_samples": metrics["count"],
    }


def retrieval_scores(pred_embeddings: np.ndarray, true_future_embeddings: np.ndarray) -> dict[str, float]:
    pred_norm = normalize_rows(pred_embeddings)
    future_norm = normalize_rows(true_future_embeddings)
    sim = pred_norm @ future_norm.T
    ranks = np.argsort(-sim, axis=1)
    targets = np.arange(sim.shape[0])
    top1 = ranks[:, 0] == targets
    top5 = np.any(ranks[:, : min(5, ranks.shape[1])] == targets[:, None], axis=1)
    top10 = np.any(ranks[:, : min(10, ranks.shape[1])] == targets[:, None], axis=1)
    target_rank = np.empty(sim.shape[0], dtype=np.int64)
    for i in range(sim.shape[0]):
        target_rank[i] = int(np.where(ranks[i] == i)[0][0]) + 1
    return {
        "retrieval_top1": float(top1.mean()),
        "retrieval_top5": float(top5.mean()),
        "retrieval_top10": float(top10.mean()),
        "retrieval_median_rank": float(np.median(target_rank)),
        "retrieval_mean_rank": float(target_rank.mean()),
    }


def compute_train_baselines(train_rows: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    mean_delta = None
    train_current_embeddings = []
    for row in train_rows:
        visual_t, visual_future = load_npz_visual(row)
        delta = visual_future - visual_t
        mean_delta = delta.astype(np.float64) if mean_delta is None else mean_delta + delta
        train_current_embeddings.append(flatten_mean_tokens(visual_t))
    if mean_delta is None:
        raise ValueError("No train rows available")
    mean_delta = (mean_delta / len(train_rows)).astype(np.float32)
    return mean_delta, np.stack(train_current_embeddings, axis=0), train_rows


@torch.no_grad()
def collect_model_predictions(
    checkpoint: Path,
    manifest: Path,
    batch_size: int,
    device: torch.device,
) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray]:
    cfg, model = load_checkpoint(checkpoint, device)
    dataset = LatentIDMDataset(manifest, visual_stats_path=visual_stats_path_from_config(cfg))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    stats = getattr(dataset, "visual_stats", None)
    stats_tensors = None
    if stats is not None:
        mean, std = stats
        stats_tensors = (
            torch.from_numpy(mean).to(device=device, dtype=torch.float32),
            torch.from_numpy(std).to(device=device, dtype=torch.float32),
        )

    sample_ids = []
    pred_embeddings = []
    true_future_embeddings = []
    identity_embeddings = []
    metrics = metric_init()

    for batch in loader:
        batch = move_batch(batch, device)
        outputs = model(batch["visual_t"], batch["visual_future"], batch["state_t"])
        pred = outputs["predicted_visual_future"]
        visual_t = batch["visual_t"]
        visual_future = batch["visual_future"]
        if stats_tensors is not None:
            mean, std = stats_tensors
            pred = pred * std + mean
            visual_t = visual_t * std + mean
            visual_future = visual_future * std + mean

        pred_np = pred.cpu().numpy()
        visual_t_np = visual_t.cpu().numpy()
        visual_future_np = visual_future.cpu().numpy()
        for i in range(pred_np.shape[0]):
            metric_update(metrics, pred_np[i], visual_t_np[i], visual_future_np[i])
            sample_ids.append(batch["sample_id"][i])
            pred_embeddings.append(flatten_mean_tokens(pred_np[i]))
            identity_embeddings.append(flatten_mean_tokens(visual_t_np[i]))
            true_future_embeddings.append(flatten_mean_tokens(visual_future_np[i]))

    finalized = metric_finalize(metrics)
    return (
        sample_ids,
        np.stack(pred_embeddings, axis=0),
        np.stack(identity_embeddings, axis=0),
        np.stack(true_future_embeddings, axis=0),
        finalized,
    )


def evaluate_nonparametric_baselines(
    train_rows: list[dict[str, Any]],
    val_rows: list[dict[str, Any]],
) -> dict[str, dict[str, float]]:
    mean_delta, train_current_embeddings, train_rows = compute_train_baselines(train_rows)
    train_current_norm = normalize_rows(train_current_embeddings)

    metrics = {
        "identity": metric_init(),
        "mean_delta": metric_init(),
        "nearest_neighbor": metric_init(),
    }
    embeddings = {
        "identity": [],
        "mean_delta": [],
        "nearest_neighbor": [],
        "true_future": [],
    }

    for row in val_rows:
        visual_t, visual_future = load_npz_visual(row)
        identity_pred = visual_t
        mean_delta_pred = visual_t + mean_delta

        query = normalize_rows(flatten_mean_tokens(visual_t)[None, :])
        nn_index = int(np.argmax(query @ train_current_norm.T))
        _, nn_future = load_npz_visual(train_rows[nn_index])

        preds = {
            "identity": identity_pred,
            "mean_delta": mean_delta_pred,
            "nearest_neighbor": nn_future,
        }
        for name, pred in preds.items():
            metric_update(metrics[name], pred, visual_t, visual_future)
            embeddings[name].append(flatten_mean_tokens(pred))
        embeddings["true_future"].append(flatten_mean_tokens(visual_future))

    true_future_embeddings = np.stack(embeddings["true_future"], axis=0)
    out = {}
    for name, metric in metrics.items():
        result = metric_finalize(metric)
        result.update(retrieval_scores(np.stack(embeddings[name], axis=0), true_future_embeddings))
        out[name] = result
    return out


def write_metrics(path: Path, metrics_by_name: dict[str, dict[str, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for name, metrics in metrics_by_name.items():
            print(f"[{name}]")
            f.write(f"[{name}]\n")
            for key, value in sorted(metrics.items()):
                line = f"{key}: {value:.8f}"
                print(line)
                f.write(line + "\n")
            print()
            f.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--train-manifest", required=True)
    parser.add_argument("--val-manifest", required=True)
    parser.add_argument("--output", default="outputs/idm_teacher_benchmark/metrics.txt")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    train_rows = read_jsonl(args.train_manifest)
    val_rows = read_jsonl(args.val_manifest)

    baseline_metrics = evaluate_nonparametric_baselines(train_rows, val_rows)
    _, pred_emb, identity_emb, true_future_emb, teacher_metrics = collect_model_predictions(
        Path(args.checkpoint),
        Path(args.val_manifest),
        args.batch_size,
        device,
    )
    teacher_metrics.update(retrieval_scores(pred_emb, true_future_emb))
    baseline_metrics["teacher"] = teacher_metrics
    # Also report the model's identity retrieval computed through the same val order.
    baseline_metrics["identity_model_order"] = retrieval_scores(identity_emb, true_future_emb)
    write_metrics(Path(args.output), baseline_metrics)


if __name__ == "__main__":
    main()
