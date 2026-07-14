from __future__ import annotations

import argparse
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from datasets.video_pair_reader import read_video_frames
from latent_action_idm.models.dino_encoder import DINOConfig, DINOFeatureExtractor
from latent_action_idm.utils import load_config, read_jsonl, seed_everything, write_jsonl


def resolve_path(path: str | Path, root: Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else root / path


def load_robot_states(row: dict, root: Path, state_dim: int) -> np.ndarray:
    path = row.get("robot_state_path") or row.get("state_path")
    if path is None:
        raise KeyError(
            "Each episode row needs robot_state_path or state_path pointing to "
            "a .npy array with shape [T, state_dim]."
        )
    states = np.load(resolve_path(path, root))
    if states.ndim != 2:
        raise ValueError(f"Expected robot states [T, C], got {states.shape} for {path}")
    if states.shape[1] < state_dim:
        raise ValueError(f"Expected at least {state_dim} state dims, got {states.shape[1]} for {path}")
    return np.asarray(states[:, :state_dim], dtype=np.float32)


def build_windows(
    num_steps: int,
    future_offset: int,
    stride: int,
    max_windows: int | None,
) -> list[tuple[int, int]]:
    last_t = num_steps - future_offset
    if last_t <= 0:
        return []
    windows = [(t, t + future_offset) for t in range(0, last_t, stride)]
    if max_windows is not None:
        windows = windows[:max_windows]
    return windows


def format_duration(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{seconds:02d}s"
    if minutes:
        return f"{minutes}m{seconds:02d}s"
    return f"{seconds}s"


def episode_windows(row: dict, cfg: dict) -> list[tuple[int, int]]:
    return build_windows(
        num_steps=int(row["num_frames"]),
        future_offset=int(cfg["data"]["future_offset_frames"]),
        stride=int(cfg["data"]["stride_frames"]),
        max_windows=cfg["data"].get("max_windows_per_episode"),
    )


def count_pending_windows(row: dict, cfg: dict, output_dir: Path, resume: bool) -> int:
    windows = episode_windows(row, cfg)
    if not resume:
        return len(windows)
    episode_id = row["episode_id"]
    return sum(
        not (output_dir / f"{episode_id}_{sample_idx:04d}_t{t_index}_f{future_index}.npz").exists()
        for sample_idx, (t_index, future_index) in enumerate(windows)
    )


def pending_local_indices(
    row: dict,
    cfg: dict,
    output_dir: Path,
    resume: bool,
) -> set[int]:
    episode_id = row["episode_id"]
    indices: set[int] = set()
    for sample_idx, (t_index, future_index) in enumerate(episode_windows(row, cfg)):
        latent_path = output_dir / f"{episode_id}_{sample_idx:04d}_t{t_index}_f{future_index}.npz"
        if not (resume and latent_path.exists()):
            indices.update((t_index, future_index))
    return indices


@torch.no_grad()
def encode_episode_frames(
    extractor: DINOFeatureExtractor,
    base_frames: dict[int, np.ndarray],
    wrist_frames: dict[int, np.ndarray],
    local_indices: list[int],
    base_frame_offset: int,
    wrist_frame_offset: int,
    view_fusion: str,
    inference_batch_size: int,
) -> dict[int, np.ndarray]:
    if inference_batch_size <= 0:
        raise ValueError("inference_batch_size must be positive")

    images = []
    for index in local_indices:
        images.append(base_frames[base_frame_offset + index])
        images.append(wrist_frames[wrist_frame_offset + index])
    image_batch = np.stack(images, axis=0)

    feature_batches = []
    for start in range(0, len(image_batch), inference_batch_size):
        feature_batches.append(extractor(image_batch[start : start + inference_batch_size]).cpu())
    features = torch.cat(feature_batches, dim=0)

    fused: dict[int, np.ndarray] = {}
    for position, index in enumerate(local_indices):
        base_feature = features[2 * position]
        wrist_feature = features[2 * position + 1]
        if view_fusion == "concat":
            dim = 0 if base_feature.ndim == 2 else -1
            value = torch.cat([base_feature, wrist_feature], dim=dim)
        elif view_fusion == "mean":
            value = torch.stack([base_feature, wrist_feature], dim=0).mean(dim=0)
        else:
            raise ValueError(f"Unsupported view_fusion: {view_fusion}")
        fused[index] = value.numpy().astype(np.float32, copy=False)
    return fused


@torch.no_grad()
def extract_episode_windows(
    row: dict,
    cfg: dict,
    extractor: DINOFeatureExtractor,
    repo_root: Path,
    output_dir: Path,
    resume: bool = False,
    inference_batch_size: int = 32,
    compress: bool = True,
    preloaded_base_frames: dict[int, np.ndarray] | None = None,
    preloaded_wrist_frames: dict[int, np.ndarray] | None = None,
) -> tuple[list[dict], int]:
    episode_id = row["episode_id"]
    base_video = resolve_path(row["base_video_path"], repo_root)
    wrist_video = resolve_path(row["wrist_video_path"], repo_root)
    base_frame_offset = int(row.get("base_frame_offset", row.get("video_frame_offset", 0)))
    wrist_frame_offset = int(row.get("wrist_frame_offset", row.get("video_frame_offset", 0)))
    states = load_robot_states(row, repo_root, int(cfg["data"]["state_dim"]))

    windows = build_windows(
        num_steps=states.shape[0],
        future_offset=int(cfg["data"]["future_offset_frames"]),
        stride=int(cfg["data"]["stride_frames"]),
        max_windows=cfg["data"].get("max_windows_per_episode"),
    )
    if not windows:
        print(f"skip {episode_id}: not enough states for requested future offset")
        return [], 0

    rows = []
    output_dir.mkdir(parents=True, exist_ok=True)
    view_fusion = cfg["dino"].get("view_fusion", "concat")

    pending = []
    for sample_idx, (t_index, future_index) in enumerate(windows):
        sample_id = f"{episode_id}_{sample_idx:04d}_t{t_index}_f{future_index}"
        latent_path = output_dir / f"{sample_id}.npz"
        rows.append(
            {
                "episode_id": episode_id,
                "sample_id": sample_id,
                "t_index": t_index,
                "future_index": future_index,
                "latent_path": str(latent_path),
            }
        )
        if not (resume and latent_path.exists()):
            pending.append((t_index, future_index, latent_path))

    if pending:
        local_indices = sorted({index for t, future, _ in pending for index in (t, future)})
        base_frames = preloaded_base_frames or read_video_frames(
            base_video, {base_frame_offset + index for index in local_indices}
        )
        wrist_frames = preloaded_wrist_frames or read_video_frames(
            wrist_video, {wrist_frame_offset + index for index in local_indices}
        )
        visual_features = encode_episode_frames(
            extractor,
            base_frames,
            wrist_frames,
            local_indices,
            base_frame_offset,
            wrist_frame_offset,
            view_fusion,
            inference_batch_size,
        )

        save_npz = np.savez_compressed if compress else np.savez
        for t_index, future_index, latent_path in pending:
            save_npz(
                latent_path,
                visual_t=visual_features[t_index],
                visual_future=visual_features[future_index],
                state_t=states[t_index].astype(np.float32),
                state_future=states[future_index].astype(np.float32),
            )

    existing_count = sum(1 for sample in rows if Path(sample["latent_path"]).exists())
    print(f"{episode_id}: indexed {len(rows)} windows ({existing_count} files available)")
    return rows, len(pending)


def split_rows_by_episode(
    rows: list[dict],
    val_ratio: float,
    seed: int,
) -> tuple[list[dict], list[dict]]:
    if not 0.0 <= val_ratio < 1.0:
        raise ValueError("--val-ratio must be in [0, 1).")
    episode_ids = sorted({row["episode_id"] for row in rows})
    random.Random(seed).shuffle(episode_ids)
    val_count = int(round(len(episode_ids) * val_ratio))
    if val_count == 0:
        return rows, []
    val_episode_ids = set(episode_ids[:val_count])
    train_rows = [row for row in rows if row["episode_id"] not in val_episode_ids]
    val_rows = [row for row in rows if row["episode_id"] in val_episode_ids]
    return train_rows, val_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="latent_action_idm/configs/dino_idm.yaml")
    parser.add_argument("--episode-manifest", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--train-out", default=None)
    parser.add_argument("--val-out", default=None)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--inference-batch-size",
        type=int,
        default=None,
        help="Number of RGB frames per frozen-DINO forward pass.",
    )
    parser.add_argument(
        "--no-compress",
        action="store_true",
        help="Write uncompressed .npz files to reduce CPU time at the cost of more disk space.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip windows whose .npz files already exist and rebuild manifests from existing/new files.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed = int(cfg["project"].get("seed", 42))
    seed_everything(seed)

    repo_root = Path.cwd()
    episode_manifest = Path(args.episode_manifest or cfg["data"]["episode_manifest"])
    output_dir = Path(args.output_dir or cfg["data"]["latent_dir"])
    train_out = Path(args.train_out or cfg["data"]["latent_manifest"])
    val_out = Path(args.val_out or cfg["data"]["val_manifest"])

    if args.resume and args.overwrite:
        raise ValueError("--resume and --overwrite are mutually exclusive.")
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite and not args.resume:
        raise FileExistsError(f"{output_dir} is not empty. Pass --overwrite to append/overwrite samples, or --resume to skip existing samples.")

    device_name = args.device or cfg["dino"].get("device", "cuda")
    device = torch.device(device_name if torch.cuda.is_available() or device_name == "cpu" else "cpu")
    dino_cfg = DINOConfig(
        backend=cfg["dino"].get("backend", "transformers"),
        model_name=cfg["dino"].get("model_name", "facebook/dinov3-vitb16-pretrain-lvd1689m"),
        hub_repo=cfg["dino"].get("hub_repo", "facebookresearch/dinov3"),
        image_size=int(cfg["dino"].get("image_size", 224)),
        feature_mode=cfg["dino"].get("feature_mode", "patch_tokens"),
        num_prefix_tokens=int(cfg["dino"].get("num_prefix_tokens", 1)),
        precision=cfg["dino"].get("precision", "fp16"),
    )
    extractor = DINOFeatureExtractor(dino_cfg).to(device).eval()
    inference_batch_size = int(
        args.inference_batch_size or cfg["dino"].get("inference_batch_size", 32)
    )

    episode_rows = read_jsonl(episode_manifest)
    total_episodes = len(episode_rows)
    pending_windows = [
        count_pending_windows(row, cfg, output_dir, resume=args.resume) for row in episode_rows
    ]
    total_pending_windows = sum(pending_windows)
    print(
        f"episodes={total_episodes}, windows_to_extract={total_pending_windows}, "
        f"resume={args.resume}"
    )

    grouped_episodes: dict[tuple[Path, Path], list[dict]] = defaultdict(list)
    for row in episode_rows:
        key = (
            resolve_path(row["base_video_path"], repo_root),
            resolve_path(row["wrist_video_path"], repo_root),
        )
        grouped_episodes[key].append(row)

    print(f"video_pairs={len(grouped_episodes)}; each pair will be decoded at most once")
    all_rows = []
    extracted_windows = 0
    episode_number = 0
    start_time = time.perf_counter()
    for pair_number, ((base_video, wrist_video), group_rows) in enumerate(
        grouped_episodes.items(), start=1
    ):
        base_indices: set[int] = set()
        wrist_indices: set[int] = set()
        for row in group_rows:
            local_indices = pending_local_indices(row, cfg, output_dir, args.resume)
            base_offset = int(row.get("base_frame_offset", row.get("video_frame_offset", 0)))
            wrist_offset = int(row.get("wrist_frame_offset", row.get("video_frame_offset", 0)))
            base_indices.update(base_offset + index for index in local_indices)
            wrist_indices.update(wrist_offset + index for index in local_indices)

        print(
            f"video_pair={pair_number}/{len(grouped_episodes)} episodes={len(group_rows)} "
            f"base_frames={len(base_indices)} wrist_frames={len(wrist_indices)}"
        )
        base_frames = read_video_frames(base_video, base_indices) if base_indices else {}
        wrist_frames = read_video_frames(wrist_video, wrist_indices) if wrist_indices else {}

        for row in group_rows:
            episode_result, newly_extracted = extract_episode_windows(
                row,
                cfg,
                extractor,
                repo_root,
                output_dir,
                resume=args.resume,
                inference_batch_size=inference_batch_size,
                compress=not args.no_compress,
                preloaded_base_frames=base_frames,
                preloaded_wrist_frames=wrist_frames,
            )
            all_rows.extend(episode_result)
            extracted_windows += newly_extracted
            episode_number += 1

            elapsed = time.perf_counter() - start_time
            if total_pending_windows == 0 or extracted_windows >= total_pending_windows:
                eta_text = "0s"
            elif extracted_windows > 0:
                eta = elapsed / extracted_windows * (total_pending_windows - extracted_windows)
                eta_text = format_duration(eta)
            else:
                eta_text = "estimating"
            print(
                f"progress={episode_number}/{total_episodes} "
                f"({episode_number / max(total_episodes, 1):.1%}) "
                f"new_windows={extracted_windows}/{total_pending_windows} "
                f"elapsed={format_duration(elapsed)} eta={eta_text}"
            )

    train_rows, val_rows = split_rows_by_episode(all_rows, args.val_ratio, seed)
    write_jsonl(train_out, train_rows)
    write_jsonl(val_out, val_rows)
    print(
        f"episode split: train={len({row['episode_id'] for row in train_rows})}, "
        f"val={len({row['episode_id'] for row in val_rows})}, overlap=0"
    )
    print(f"wrote {len(train_rows)} train rows -> {train_out}")
    print(f"wrote {len(val_rows)} val rows -> {val_out}")


if __name__ == "__main__":
    main()
