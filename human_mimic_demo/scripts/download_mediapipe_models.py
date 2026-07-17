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
    parser.add_argument(
        "--timeout",
        type=float,
        default=30.0,
        help="Network connect/read timeout in seconds",
    )
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists() and not args.force:
        print(f"Already present: {output.resolve()}")
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".part")
    print(f"Downloading {HAND_LANDMARKER_URL}")
    try:
        request = urllib.request.Request(
            HAND_LANDMARKER_URL,
            headers={"User-Agent": "human-mimic-demo/0.1"},
        )
        with urllib.request.urlopen(request, timeout=args.timeout) as response:
            total = int(response.headers.get("Content-Length", "0"))
            downloaded = 0
            with temporary.open("wb") as handle:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    handle.write(chunk)
                    downloaded += len(chunk)
                    if total > 0:
                        percent = 100.0 * downloaded / total
                        print(
                            f"\r{downloaded / 1_000_000:.1f}/{total / 1_000_000:.1f} MB "
                            f"({percent:.1f}%)",
                            end="",
                            flush=True,
                        )
                    else:
                        print(
                            f"\r{downloaded / 1_000_000:.1f} MB",
                            end="",
                            flush=True,
                        )
        print()
        if temporary.stat().st_size < 1_000_000:
            raise RuntimeError("Downloaded model is unexpectedly small")
        temporary.replace(output)
    finally:
        if temporary.exists():
            temporary.unlink()
    print(f"Saved: {output.resolve()} ({output.stat().st_size / 1_000_000:.1f} MB)")


if __name__ == "__main__":
    main()
