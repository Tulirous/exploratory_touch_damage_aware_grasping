from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch


DEFAULT_ROOT = Path("/data/chy/hot3d_hand_lwm/data_d1_200")
DEFAULT_ORIGINAL_CHECKPOINT = (
    DEFAULT_ROOT / "frozen_v2_idm_original_decoder_checkpoints/best.pt"
)
DEFAULT_ADALN_CHECKPOINT = (
    DEFAULT_ROOT / "frozen_v2_idm_adaln_cross_checkpoints/best.pt"
)
DEFAULT_OUTPUT_DIR = DEFAULT_ROOT / "frozen_v2_idm_decoder_comparison"
LOWER_IS_BETTER = (
    "posterior/state_mae",
    "posterior/wrist_ade_mm",
    "posterior/wrist_fde_mm",
    "posterior/rotation6d_mae",
    "posterior/mano_pca_mae",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate original and AdaLN-Cross decoders trained with the same "
            "frozen v2 Hand-IDM, verify IDM identity, and compare metrics."
        )
    )
    parser.add_argument(
        "--original-checkpoint",
        default=str(DEFAULT_ORIGINAL_CHECKPOINT),
    )
    parser.add_argument(
        "--adaln-checkpoint",
        default=str(DEFAULT_ADALN_CHECKPOINT),
    )
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--ridge-alpha", type=float, default=1.0)
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def require_file(path: str | Path, label: str) -> Path:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"{label} does not exist: {resolved}")
    return resolved


def inverse_dynamics_sha256(checkpoint_path: Path) -> str:
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    digest = hashlib.sha256()
    prefix = "inverse_dynamics."
    keys = sorted(
        key for key in checkpoint["model"] if key.startswith(prefix)
    )
    if not keys:
        raise ValueError(f"no Hand-IDM parameters in {checkpoint_path}")
    for key in keys:
        tensor = checkpoint["model"][key].detach().contiguous().cpu()
        digest.update(key.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def run_evaluation(
    checkpoint: Path,
    output_dir: Path,
    suite_name: str,
    args: argparse.Namespace,
) -> Path:
    command = [
        sys.executable,
        "-m",
        "smplx_hand_lwm.scripts.run_hmwm_lawam_v0_evaluation",
        "--checkpoint",
        str(checkpoint),
        "--output-dir",
        str(output_dir),
        "--suite-name",
        suite_name,
        "--batch-size",
        str(args.batch_size),
        "--num-workers",
        str(args.num_workers),
        "--device",
        args.device,
        "--ridge-alpha",
        str(args.ridge_alpha),
    ]
    if args.skip_existing:
        command.append("--skip-existing")
    subprocess.run(command, check=True)
    return output_dir / "evaluation_bundle.json"


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def compare_split(
    original_bundle: dict[str, Any],
    adaln_bundle: dict[str, Any],
    result_key: str,
) -> dict[str, Any]:
    original = original_bundle["results"][result_key]["metrics"]
    adaln = adaln_bundle["results"][result_key]["metrics"]
    metrics: dict[str, Any] = {}
    for key in LOWER_IS_BETTER:
        original_value = float(original[key])
        adaln_value = float(adaln[key])
        delta = original_value - adaln_value
        metrics[key] = {
            "original": original_value,
            "adaln_cross": adaln_value,
            "original_minus_adaln": delta,
            "winner": (
                "original"
                if delta < 0.0
                else "adaln_cross"
                if delta > 0.0
                else "tie"
            ),
        }
    for key in LOWER_IS_BETTER:
        shuffle_key = key.replace("posterior/", "shuffle_z/")
        if shuffle_key in original and shuffle_key in adaln:
            metrics[key]["shuffle_minus_posterior"] = {
                "original": float(original[shuffle_key])
                - float(original[key]),
                "adaln_cross": float(adaln[shuffle_key])
                - float(adaln[key]),
            }
    return metrics


def main() -> None:
    args = parse_args()
    original_checkpoint = require_file(
        args.original_checkpoint,
        "original decoder checkpoint",
    )
    adaln_checkpoint = require_file(
        args.adaln_checkpoint,
        "AdaLN-Cross decoder checkpoint",
    )
    output_dir = Path(args.output_dir).expanduser().resolve()
    original_output = output_dir / "original_decoder"
    adaln_output = output_dir / "adaln_cross_decoder"
    original_bundle_path = run_evaluation(
        original_checkpoint,
        original_output,
        "Frozen-v2-IDM original decoder",
        args,
    )
    adaln_bundle_path = run_evaluation(
        adaln_checkpoint,
        adaln_output,
        "Frozen-v2-IDM AdaLN-Cross decoder",
        args,
    )

    original_idm_hash = inverse_dynamics_sha256(original_checkpoint)
    adaln_idm_hash = inverse_dynamics_sha256(adaln_checkpoint)
    if original_idm_hash != adaln_idm_hash:
        raise RuntimeError(
            "comparison is invalid: the two checkpoints do not contain "
            "identical Hand-IDM parameters"
        )
    original_bundle = load_json(original_bundle_path)
    adaln_bundle = load_json(adaln_bundle_path)
    comparison = {
        "protocol": "frozen-v2-Hand-IDM decoder isolation",
        "idm_sha256": original_idm_hash,
        "idm_parameters_identical": True,
        "checkpoints": {
            "original": str(original_checkpoint),
            "adaln_cross": str(adaln_checkpoint),
        },
        "fixed_val10": compare_split(
            original_bundle,
            adaln_bundle,
            "stage1_fixed_val10",
        ),
        "test_d1_50": compare_split(
            original_bundle,
            adaln_bundle,
            "stage1_test_d1_50",
        ),
        "interpretation": (
            "All reported errors are lower-is-better. A negative "
            "original_minus_adaln value favors the original TransformerDecoder. "
            "Test-D1 is the primary generalization result."
        ),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    comparison_path = output_dir / "decoder_comparison.json"
    comparison_path.write_text(
        json.dumps(comparison, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(comparison, indent=2, ensure_ascii=False))
    print(f"Comparison result: {comparison_path}", flush=True)


if __name__ == "__main__":
    main()
