from __future__ import annotations

import argparse
import time

import numpy as np

from human_mimic_demo.config import load_config, nested
from human_mimic_demo.controllers.linkerhand_o6 import LinkerHandO6


def main() -> int:
    parser = argparse.ArgumentParser(description="Low-range, one-axis-at-a-time O6 motion test")
    parser.add_argument(
        "--config",
        default="human_mimic_demo/configs/demo_windows.json",
    )
    parser.add_argument("--step", type=int, default=20, help="Small closing step from 255")
    parser.add_argument("--settle", type=float, default=1.0)
    parser.add_argument("--i-understand-hardware-risk", action="store_true")
    args = parser.parse_args()
    if not args.i_understand_hardware_risk:
        print("Refusing motion: add --i-understand-hardware-risk")
        return 2
    if not 1 <= args.step <= 30:
        print("For the first diagnostic, --step must be in [1, 30]")
        return 2

    config = load_config(args.config)
    hand = LinkerHandO6(nested(config, "hand"))
    open_pose = np.asarray([255] * 6, dtype=np.int64)
    confirmation = input(
        "Secure the O6, clear the workspace, keep power disconnect reachable. "
        "Type MOVE O6 to continue: "
    )
    if confirmation != "MOVE O6":
        print("Cancelled")
        return 2

    print("Commanding safe-open pose")
    hand.command(open_pose)
    time.sleep(args.settle)
    print(f"state={hand.state()}")
    try:
        for axis in range(6):
            pose = open_pose.copy()
            pose[axis] -= args.step
            print(f"Axis {axis}: command={pose.tolist()}")
            hand.command(pose)
            time.sleep(args.settle)
            print(f"state={hand.state()}")
            hand.command(open_pose)
            time.sleep(args.settle)
    finally:
        hand.command(open_pose)
    print("PASS: completed six small axis motions and returned to open")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
