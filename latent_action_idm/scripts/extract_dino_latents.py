from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from datasets.video_pair_reader import read_video_frame
from latent_action_idm.models.dino_encoder import DINOConfig, DINOFeatureExtractor, encode_two_views
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


@torch.no_grad()
def extract_episode_windows(
    row: dict,
    cfg: dict,
    extractor: DINOFeatureExtractor,
    repo_root: Path,
    output_dir: Path,
    resume: bool = False,
) -> list[dict]:
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
        return []

    rows = []
    output_dir.mkdir(parents=True, exist_ok=True)
    view_fusion = cfg["dino"].get("view_fusion", "concat")

    for sample_idx, (t_index, future_index) in enumerate(windows):
        sample_id = f"{episode_id}_{sample_idx:04d}_t{t_index}_f{future_index}"
        latent_path = output_dir / f"{sample_id}.npz"
        if resume and latent_path.exists():
            rows.append(
                {
                    "episode_id": episode_id,
                    "sample_id": sample_id,
                    "t_index": t_index,
                    "future_index": future_index,
                    "latent_path": str(latent_path),
                }
            )
            continue

        base_t = read_video_frame(base_video, frame_index=base_frame_offset + t_index)
        wrist_t = read_video_frame(wrist_video, frame_index=wrist_frame_offset + t_index)
        base_f = read_video_frame(base_video, frame_index=base_frame_offset + future_index)
        wrist_f = read_video_frame(wrist_video, frame_index=wrist_frame_offset + future_index)

        visual_t = encode_two_views(extractor, base_t, wrist_t, view_fusion=view_fusion).cpu().numpy()
        visual_future = encode_two_views(extractor, base_f, wrist_f, view_fusion=view_fusion).cpu().numpy()

        np.savez_compressed(
            latent_path,
            visual_t=visual_t.astype(np.float32),
            visual_future=visual_future.astype(np.float32),
            state_t=states[t_index].astype(np.float32),
            state_future=states[future_index].astype(np.float32),
        )
        rows.append(
            {
                "episode_id": episode_id,
                "sample_id": sample_id,
                "t_index": t_index,
                "future_index": future_index,
                "latent_path": str(latent_path),
            }
        )

    existing_count = sum(1 for sample in rows if Path(sample["latent_path"]).exists())
    print(f"{episode_id}: indexed {len(rows)} windows ({existing_count} files available)")
    return rows


def split_rows(rows: list[dict], val_ratio: float) -> tuple[list[dict], list[dict]]:
    if not 0.0 <= val_ratio < 1.0:
        raise ValueError("--val-ratio must be in [0, 1).")
    val_count = int(round(len(rows) * val_ratio))
    if val_count == 0:
        return rows, []
    return rows[:-val_count], rows[-val_count:]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="latent_action_idm/configs/dino_idm.yaml")
    parser.add_argument("--episode-manifest", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--train-out", default=None)
    parser.add_argument("--val-out", default=None)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--device", default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip windows whose .npz files already exist and rebuild manifests from existing/new files.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed_everything(int(cfg["project"].get("seed", 42)))

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

    all_rows = []
    for row in read_jsonl(episode_manifest):
        all_rows.extend(extract_episode_windows(row, cfg, extractor, repo_root, output_dir, resume=args.resume))

    train_rows, val_rows = split_rows(all_rows, args.val_ratio)
    write_jsonl(train_out, train_rows)
    write_jsonl(val_out, val_rows)
    print(f"wrote {len(train_rows)} train rows -> {train_out}")
    print(f"wrote {len(val_rows)} val rows -> {val_out}")


if __name__ == "__main__":
    main()
