from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from human_mimic_demo.config import load_config, nested


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Read-only Linker Hand O6 SDK/CAN connection check"
    )
    parser.add_argument(
        "--config",
        default="human_mimic_demo/configs/demo_windows.json",
    )
    args = parser.parse_args()
    config = load_config(args.config)
    hand_config = nested(config, "hand")
    sdk_path = Path(str(hand_config.get("sdk_path", "")))
    if not sdk_path.is_dir():
        print(f"FAIL: Linker Hand SDK directory does not exist: {sdk_path}")
        return 1
    sys.path.insert(0, str(sdk_path))
    try:
        from LinkerHand.linker_hand_api import LinkerHandApi
    except ImportError as exc:
        print(f"FAIL: cannot import Linker Hand SDK: {exc}")
        return 1

    channel = str(hand_config.get("can_channel", "PCAN_USBBUS1"))
    modbus = str(hand_config.get("modbus", "None"))
    print(f"Opening right O6: CAN={channel!r}, MODBUS={modbus!r}")
    try:
        hand = LinkerHandApi(
            hand_type="right",
            hand_joint="O6",
            can=channel,
            modbus=modbus,
        )
        result = {
            "embedded_version": hand.get_embedded_version(),
            "serial_number": hand.get_serial_number(),
            "state": hand.get_state(),
        }
        print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
        state = result["state"]
        if state is None or len(state) != 6:
            print("FAIL: O6 state must contain six values")
            return 1
        print("PASS: O6 SDK/CAN connection and six-axis state read succeeded")
        return 0
    except Exception as exc:
        print(f"FAIL: {type(exc).__name__}: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
