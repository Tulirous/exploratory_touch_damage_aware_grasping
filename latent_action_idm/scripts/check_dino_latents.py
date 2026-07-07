from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import numpy as np

from latent_action_idm.utils import read_jsonl


def flatten_tokens(tokens: np.ndarray) -> np.ndarray:
    if tokens.ndim != 2:
        raise ValueError(f"Expected patch tokens [N, D], got {tokens.shape}")
    return tokens.reshape(-1).astype(np.float64)


def cosine(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom < eps:
        return float("nan")
    return float(np.dot(a, b) / denom)


def ridge_r2(x: np.ndarray, y: np.ndarray, ridge: float = 1e-3) -> float:
    if x.shape[0] < 2:
        return float("nan")
    x_aug = np.concatenate([x, np.ones((x.shape[0], 1), dtype=x.dtype)], axis=1)
    xtx = x_aug.T @ x_aug + ridge * np.eye(x_aug.shape[1], dtype=x.dtype)
    weights = np.linalg.solve(xtx, x_aug.T @ y)
    pred = x_aug @ weights
    ss_res = float(np.square(y - pred).sum())
    ss_tot = float(np.square(y - y.mean(axis=0, keepdims=True)).sum())
    return float("nan") if ss_tot <= 1e-12 else 1.0 - ss_res / ss_tot


def random_project(
    x: np.ndarray,
    output_dim: int,
    seed: int,
) -> np.ndarray:
    """Project very high-dimensional latents before diagnostic ridge fitting."""

    if x.shape[1] <= output_dim:
        return x
    rng = np.random.default_rng(seed)
    projection = rng.normal(
        loc=0.0,
        scale=1.0 / np.sqrt(output_dim),
        size=(x.shape[1], output_dim),
    ).astype(np.float32)
    return (x.astype(np.float32) @ projection).astype(np.float64)


def load_manifest_arrays(manifest: Path, max_samples: int | None = None) -> tuple[list[dict[str, Any]], dict[str, np.ndarray]]:
    rows = read_jsonl(manifest)
    if max_samples is not None:
        rows = rows[:max_samples]
    visual_t = []
    visual_future = []
    state_t = []
    state_future = []
    kept_rows = []
    for row in rows:
        data = np.load(row["latent_path"])
        vt = data["visual_t"].astype(np.float32)
        vf = data["visual_future"].astype(np.float32)
        st = data["state_t"].astype(np.float32)
        sf = data["state_future"].astype(np.float32)
        if vt.shape != vf.shape:
            raise ValueError(f"{row['latent_path']} visual_t and visual_future shapes differ: {vt.shape} vs {vf.shape}")
        visual_t.append(vt)
        visual_future.append(vf)
        state_t.append(st)
        state_future.append(sf)
        kept_rows.append(row)
    return kept_rows, {
        "visual_t": np.stack(visual_t),
        "visual_future": np.stack(visual_future),
        "state_t": np.stack(state_t),
        "state_future": np.stack(state_future),
    }


def summarize_latents(
    arrays: dict[str, np.ndarray],
    ridge_features: int,
    seed: int,
) -> dict[str, float]:
    vt = arrays["visual_t"]
    vf = arrays["visual_future"]
    st = arrays["state_t"]
    sf = arrays["state_future"]
    delta = vf - vt
    state_delta = sf - st
    flat_t = vt.reshape(vt.shape[0], -1).astype(np.float64)
    flat_f = vf.reshape(vf.shape[0], -1).astype(np.float64)
    flat_delta = delta.reshape(delta.shape[0], -1).astype(np.float64)
    state_delta = state_delta.astype(np.float64)

    per_sample_cos = np.array([cosine(flat_t[i], flat_f[i]) for i in range(flat_t.shape[0])])
    per_sample_delta_norm = np.linalg.norm(flat_delta, axis=1)
    token_std = vt.std(axis=0)
    projected_delta = random_project(flat_delta, ridge_features, seed)
    metrics = {
        "num_samples": float(vt.shape[0]),
        "visual_shape_0_tokens": float(vt.shape[1]),
        "visual_shape_1_dim": float(vt.shape[2]),
        "visual_t_finite_ratio": float(np.isfinite(vt).mean()),
        "visual_future_finite_ratio": float(np.isfinite(vf).mean()),
        "visual_t_mean": float(vt.mean()),
        "visual_t_std": float(vt.std()),
        "visual_future_mean": float(vf.mean()),
        "visual_future_std": float(vf.std()),
        "token_std_mean": float(token_std.mean()),
        "token_std_min": float(token_std.min()),
        "token_std_max": float(token_std.max()),
        "current_future_mse": float(np.square(delta).mean()),
        "current_future_cosine_mean": float(np.nanmean(per_sample_cos)),
        "current_future_delta_l2_mean": float(per_sample_delta_norm.mean()),
        "current_future_delta_l2_std": float(per_sample_delta_norm.std()),
        "state_delta_l2_mean": float(np.linalg.norm(state_delta, axis=1).mean()),
        "visual_delta_ridge_features": float(projected_delta.shape[1]),
        "visual_delta_to_state_delta_r2": ridge_r2(projected_delta, state_delta),
    }
    if vt.shape[1] >= 392:
        base_delta = delta[:, :196]
        wrist_delta = delta[:, 196:392]
        metrics["base_current_future_mse"] = float(np.square(base_delta).mean())
        metrics["wrist_current_future_mse"] = float(np.square(wrist_delta).mean())
    return metrics


def nearest_neighbor_report(
    rows: list[dict[str, Any]],
    arrays: dict[str, np.ndarray],
    predicted_npz: Path | None,
    top_k: int,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    if predicted_npz is None:
        return {}, []
    pred_data = np.load(predicted_npz)
    if "predicted_visual_future" not in pred_data:
        raise KeyError(f"{predicted_npz} does not contain predicted_visual_future")
    pred = pred_data["predicted_visual_future"].astype(np.float64)
    real_future = arrays["visual_future"].astype(np.float64)
    if pred.shape != real_future.shape:
        raise ValueError(f"predicted_visual_future shape {pred.shape} != real future shape {real_future.shape}")

    pred_flat = pred.reshape(pred.shape[0], -1)
    real_flat = real_future.reshape(real_future.shape[0], -1)
    pred_norm = pred_flat / np.linalg.norm(pred_flat, axis=1, keepdims=True).clip(min=1e-12)
    real_norm = real_flat / np.linalg.norm(real_flat, axis=1, keepdims=True).clip(min=1e-12)
    sim = pred_norm @ real_norm.T
    nn_idx = np.argsort(-sim, axis=1)[:, :top_k]

    top1 = nn_idx[:, 0]
    top1_same_sample = top1 == np.arange(len(rows))
    top1_same_episode = np.array([rows[i]["episode_id"] == rows[j]["episode_id"] for i, j in enumerate(top1)])
    top1_time_error = np.array([abs(int(rows[i]["future_index"]) - int(rows[j]["future_index"])) for i, j in enumerate(top1)])
    topk_contains_true = np.array([i in nn_idx[i] for i in range(len(rows))])
    metrics = {
        "nn_top1_true_sample_rate": float(top1_same_sample.mean()),
        "nn_topk_true_sample_rate": float(topk_contains_true.mean()),
        "nn_top1_same_episode_rate": float(top1_same_episode.mean()),
        "nn_top1_future_index_abs_error_mean": float(top1_time_error.mean()),
        "nn_top1_cosine_mean": float(sim[np.arange(len(rows)), top1].mean()),
    }
    report_rows = []
    for i, j in enumerate(top1):
        report_rows.append(
            {
                "query_sample_id": rows[i]["sample_id"],
                "query_episode_id": rows[i]["episode_id"],
                "query_future_index": int(rows[i]["future_index"]),
                "nn_sample_id": rows[j]["sample_id"],
                "nn_episode_id": rows[j]["episode_id"],
                "nn_future_index": int(rows[j]["future_index"]),
                "nn_cosine": float(sim[i, j]),
                "is_true_sample": bool(i == j),
                "same_episode": bool(rows[i]["episode_id"] == rows[j]["episode_id"]),
                "future_index_abs_error": int(abs(int(rows[i]["future_index"]) - int(rows[j]["future_index"]))),
            }
        )
    return metrics, report_rows


def write_metrics(path: Path, metrics: dict[str, float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for key, value in sorted(metrics.items()):
            line = f"{key}: {value:.8f}"
            print(line)
            f.write(line + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="data/manifests/idm_val.jsonl")
    parser.add_argument("--predicted-npz", default=None, help="Optional analyze_dit_lawam val.npz with predicted futures.")
    parser.add_argument("--output-dir", default="outputs/dino_latent_check")
    parser.add_argument("--split-name", default="val")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--ridge-features", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rows, arrays = load_manifest_arrays(Path(args.manifest), max_samples=args.max_samples)
    metrics = summarize_latents(arrays, ridge_features=args.ridge_features, seed=args.seed)
    nn_metrics, nn_rows = nearest_neighbor_report(
        rows,
        arrays,
        Path(args.predicted_npz) if args.predicted_npz else None,
        args.top_k,
    )
    metrics.update(nn_metrics)
    output_dir = Path(args.output_dir)
    prefix = output_dir / args.split_name
    write_metrics(prefix.with_suffix(".metrics.txt"), metrics)
    write_csv(prefix.with_name(prefix.name + "_nearest_neighbors.csv"), nn_rows)


if __name__ == "__main__":
    main()
