from __future__ import annotations

import argparse
from pathlib import Path
import urllib.request


HAND_LANDMARKER_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download official MediaPipe task models")
    parser.add_argument(
        "--output",
        default="human_mimic_demo/assets/hand_landmarker.task",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists() and not args.force:
        print(f"Already present: {output.resolve()}")
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".part")
    print(f"Downloading {HAND_LANDMARKER_URL}")
    try:
        urllib.request.urlretrieve(HAND_LANDMARKER_URL, temporary)
        if temporary.stat().st_size < 1_000_000:
            raise RuntimeError("Downloaded model is unexpectedly small")
        temporary.replace(output)
    finally:
        if temporary.exists():
            temporary.unlink()
    print(f"Saved: {output.resolve()} ({output.stat().st_size / 1_000_000:.1f} MB)")


if __name__ == "__main__":
    main()
