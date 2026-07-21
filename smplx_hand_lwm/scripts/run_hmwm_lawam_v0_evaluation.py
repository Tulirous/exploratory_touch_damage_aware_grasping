from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_CHECKPOINT = (
    "/data/chy/hot3d_hand_lwm/data_d1_200/"
    "hmwm_lawam_v0_checkpoints/best.pt"
)
DEFAULT_TRAIN_MANIFEST = "/data/chy/hot3d_hand_lwm/data_d1_200/train.jsonl"
DEFAULT_VAL_MANIFEST = (
    "/data/chy/hot3d_hand_lwm/data_d1_200/fixed_val10.jsonl"
)
DEFAULT_TEST_MANIFEST = "/data/chy/hot3d_hand_lwm/test_d1_50/test.jsonl"
DEFAULT_OUTPUT_DIR = (
    "/data/chy/hot3d_hand_lwm/data_d1_200/hmwm_lawam_v0_evaluation"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the complete HMWM-LaWM-v0 train/validation/Test-D1 "
            "evaluation suite and write a combined JSON bundle."
        )
    )
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--train-manifest", default=DEFAULT_TRAIN_MANIFEST)
    parser.add_argument("--val-manifest", default=DEFAULT_VAL_MANIFEST)
    parser.add_argument("--test-manifest", default=DEFAULT_TEST_MANIFEST)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--ridge-alpha", type=float, default=1.0)
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Reuse an existing non-empty JSON result instead of rerunning it.",
    )
    return parser.parse_args()


def require_file(path: str, label: str) -> Path:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"{label} does not exist: {resolved}")
    return resolved


def command_for_module(
    module: str,
    arguments: list[str],
) -> list[str]:
    return [sys.executable, "-m", module, *arguments]


def valid_json_file(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size == 0:
        return False
    try:
        with path.open("r", encoding="utf-8") as handle:
            json.load(handle)
    except (OSError, json.JSONDecodeError):
        return False
    return True


def tail(path: Path, lines: int = 40) -> str:
    try:
        content = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return "<log unavailable>"
    return "\n".join(content[-lines:])


def run_job(
    index: int,
    total: int,
    name: str,
    command: list[str],
    result_path: Path,
    log_path: Path,
    skip_existing: bool,
) -> dict[str, Any]:
    if skip_existing and valid_json_file(result_path):
        print(f"[{index}/{total}] {name}: reuse {result_path}", flush=True)
        return {
            "status": "reused",
            "seconds": 0.0,
            "result": str(result_path),
            "log": str(log_path),
        }

    print(f"[{index}/{total}] {name}: running", flush=True)
    started = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as log_handle:
        completed = subprocess.run(
            command,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    seconds = time.perf_counter() - started
    if completed.returncode != 0:
        raise RuntimeError(
            f"{name} failed with exit code {completed.returncode}.\n"
            f"Log: {log_path}\n"
            f"Last log lines:\n{tail(log_path)}"
        )
    if not valid_json_file(result_path):
        raise RuntimeError(
            f"{name} completed but did not produce valid JSON: {result_path}.\n"
            f"Log: {log_path}\n"
            f"Last log lines:\n{tail(log_path)}"
        )
    print(
        f"[{index}/{total}] {name}: done in {seconds:.1f}s -> {result_path}",
        flush=True,
    )
    return {
        "status": "completed",
        "seconds": seconds,
        "result": str(result_path),
        "log": str(log_path),
    }


def main() -> None:
    args = parse_args()
    checkpoint = require_file(args.checkpoint, "checkpoint")
    train_manifest = require_file(args.train_manifest, "train manifest")
    val_manifest = require_file(args.val_manifest, "validation manifest")
    test_manifest = require_file(args.test_manifest, "Test-D1 manifest")
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    common = [
        "--checkpoint",
        str(checkpoint),
        "--batch-size",
        str(args.batch_size),
        "--num-workers",
        str(args.num_workers),
        "--device",
        args.device,
    ]
    ridge = ["--ridge-alpha", str(args.ridge_alpha)]
    specifications = [
        (
            "stage1_train",
            "Stage-1 train metrics",
            "smplx_hand_lwm.evaluate_stage1",
            [*common, "--manifest", str(train_manifest)],
        ),
        (
            "stage1_fixed_val10",
            "Stage-1 fixed validation metrics",
            "smplx_hand_lwm.evaluate_stage1",
            [*common, "--manifest", str(val_manifest)],
        ),
        (
            "stage1_test_d1_50",
            "Stage-1 Test-D1 metrics",
            "smplx_hand_lwm.evaluate_stage1",
            [*common, "--manifest", str(test_manifest)],
        ),
        (
            "teacher_la_fixed_val10",
            "Teacher-LA audit on fixed validation",
            "smplx_hand_lwm.audit_teacher_la",
            [
                *common,
                "--train-manifest",
                str(train_manifest),
                "--val-manifest",
                str(val_manifest),
                *ridge,
            ],
        ),
        (
            "teacher_la_test_d1_50",
            "Teacher-LA audit on Test-D1",
            "smplx_hand_lwm.audit_teacher_la",
            [
                *common,
                "--train-manifest",
                str(train_manifest),
                "--val-manifest",
                str(test_manifest),
                *ridge,
            ],
        ),
        (
            "hmwm_fixed_val10",
            "HMWM diagnostics on fixed validation",
            "smplx_hand_lwm.diagnose_hmwm",
            [
                *common,
                "--train-manifest",
                str(train_manifest),
                "--val-manifest",
                str(val_manifest),
            ],
        ),
        (
            "hmwm_test_d1_50",
            "HMWM diagnostics on Test-D1",
            "smplx_hand_lwm.diagnose_hmwm",
            [
                *common,
                "--train-manifest",
                str(train_manifest),
                "--val-manifest",
                str(test_manifest),
            ],
        ),
    ]

    run_metadata: dict[str, Any] = {}
    results: dict[str, Any] = {}
    suite_started = time.perf_counter()
    total = len(specifications)
    for index, (key, label, module, module_args) in enumerate(
        specifications, start=1
    ):
        result_path = output_dir / f"{key}.json"
        log_path = output_dir / f"{key}.log"
        command = command_for_module(
            module,
            [*module_args, "--output", str(result_path)],
        )
        run_metadata[key] = run_job(
            index=index,
            total=total,
            name=label,
            command=command,
            result_path=result_path,
            log_path=log_path,
            skip_existing=args.skip_existing,
        )
        with result_path.open("r", encoding="utf-8") as handle:
            results[key] = json.load(handle)

    bundle = {
        "suite": "HMWM-LaWM-v0 evaluation",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "total_seconds": time.perf_counter() - suite_started,
        "inputs": {
            "checkpoint": str(checkpoint),
            "train_manifest": str(train_manifest),
            "validation_manifest": str(val_manifest),
            "test_manifest": str(test_manifest),
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
            "device": args.device,
            "ridge_alpha": args.ridge_alpha,
        },
        "runs": run_metadata,
        "results": results,
        "test_protocol": (
            "Test-D1 is evaluation-only and must not be used for checkpoint "
            "selection or hyperparameter tuning."
        ),
    }
    bundle_path = output_dir / "evaluation_bundle.json"
    bundle_path.write_text(
        json.dumps(bundle, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"Complete evaluation bundle: {bundle_path}", flush=True)


if __name__ == "__main__":
    main()
