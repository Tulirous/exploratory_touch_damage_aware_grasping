from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
import statistics
import sys
import time
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Safely enumerate a local D435 and verify RGB/depth streaming."
    )
    parser.add_argument("--seconds", type=float, default=5.0)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--serial", default=None, help="Optional RealSense serial number")
    parser.add_argument("--preview", action="store_true", help="Show aligned RGB/depth preview")
    parser.add_argument(
        "--output",
        default="human_mimic_demo/logs/d435_check.json",
        help="JSON report path",
    )
    return parser.parse_args()


def require_realsense():
    try:
        import pyrealsense2 as rs
    except ImportError as exc:
        raise SystemExit(
            "pyrealsense2 is not installed in this Python environment.\n"
            "Run: python -m pip install pyrealsense2"
        ) from exc
    return rs


def device_info(rs: Any, device: Any) -> dict[str, Any]:
    fields = {
        "name": rs.camera_info.name,
        "serial_number": rs.camera_info.serial_number,
        "firmware_version": rs.camera_info.firmware_version,
        "product_id": rs.camera_info.product_id,
        "usb_type_descriptor": rs.camera_info.usb_type_descriptor,
    }
    info: dict[str, Any] = {}
    for key, field in fields.items():
        try:
            info[key] = device.get_info(field) if device.supports(field) else None
        except RuntimeError:
            info[key] = None
    info["sensors"] = []
    for sensor in device.query_sensors():
        try:
            name = sensor.get_info(rs.camera_info.name)
        except RuntimeError:
            name = "unknown"
        info["sensors"].append(name)
    return info


def select_device(rs: Any, serial: str | None) -> tuple[Any, list[dict[str, Any]]]:
    devices = list(rs.context().query_devices())
    inventory = [device_info(rs, device) for device in devices]
    if not devices:
        raise RuntimeError(
            "No RealSense device was found. Check USB 3 cable/port and RealSense Viewer."
        )
    if serial is None:
        for device, info in zip(devices, inventory):
            if "435" in str(info.get("name", "")):
                return device, inventory
        return devices[0], inventory
    for device, info in zip(devices, inventory):
        if info.get("serial_number") == serial:
            return device, inventory
    raise RuntimeError(f"Requested serial {serial!r} was not found")


def preview_frames(color: Any, depth: Any, depth_scale: float) -> bool:
    try:
        import cv2
        import numpy as np
    except ImportError as exc:
        raise RuntimeError(
            "--preview requires opencv-python and numpy"
        ) from exc
    color_image = np.asanyarray(color.get_data())
    depth_image = np.asanyarray(depth.get_data())
    depth_8u = cv2.convertScaleAbs(depth_image, alpha=255.0 / max(4.0 / depth_scale, 1.0))
    depth_colormap = cv2.applyColorMap(depth_8u, cv2.COLORMAP_JET)
    combined = np.hstack((color_image, depth_colormap))
    cv2.putText(
        combined,
        "D435 RGB + aligned depth | q=quit",
        (12, 26),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        1,
        cv2.LINE_AA,
    )
    cv2.imshow("D435 connection check", combined)
    return (cv2.waitKey(1) & 0xFF) == ord("q")


def main() -> int:
    args = parse_args()
    rs = require_realsense()
    report: dict[str, Any] = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "platform": platform.platform(),
        "python": sys.version,
        "requested_stream": {
            "width": args.width,
            "height": args.height,
            "fps": args.fps,
            "seconds": args.seconds,
        },
        "success": False,
    }
    pipeline = None
    try:
        selected, inventory = select_device(rs, args.serial)
        selected_info = device_info(rs, selected)
        report["devices"] = inventory
        report["selected_device"] = selected_info
        serial = selected_info.get("serial_number")
        print(f"Found {len(inventory)} RealSense device(s)")
        print(json.dumps(selected_info, indent=2, ensure_ascii=False))

        pipeline = rs.pipeline()
        config = rs.config()
        if serial:
            config.enable_device(serial)
        config.enable_stream(rs.stream.color, args.width, args.height, rs.format.bgr8, args.fps)
        config.enable_stream(rs.stream.depth, args.width, args.height, rs.format.z16, args.fps)
        profile = pipeline.start(config)
        align = rs.align(rs.stream.color)
        depth_sensor = profile.get_device().first_depth_sensor()
        depth_scale = float(depth_sensor.get_depth_scale())
        report["depth_scale_m"] = depth_scale

        # Allow auto-exposure and stream startup to settle.
        for _ in range(15):
            pipeline.wait_for_frames(timeout_ms=2000)

        started = time.monotonic()
        frame_count = 0
        center_depths = []
        timestamps_ms = []
        while time.monotonic() - started < args.seconds:
            frames = align.process(pipeline.wait_for_frames(timeout_ms=2000))
            color = frames.get_color_frame()
            depth = frames.get_depth_frame()
            if not color or not depth:
                continue
            frame_count += 1
            timestamps_ms.append(float(frames.get_timestamp()))
            center_depth = float(depth.get_distance(args.width // 2, args.height // 2))
            if center_depth > 0.0:
                center_depths.append(center_depth)
            if args.preview and preview_frames(color, depth, depth_scale):
                break

        elapsed = max(time.monotonic() - started, 1e-6)
        measured_fps = frame_count / elapsed
        report["stream"] = {
            "frames": frame_count,
            "elapsed_s": elapsed,
            "measured_fps": measured_fps,
            "valid_center_depth_frames": len(center_depths),
            "median_center_depth_m": (
                statistics.median(center_depths) if center_depths else None
            ),
            "device_timestamp_monotonic": all(
                later > earlier for earlier, later in zip(timestamps_ms, timestamps_ms[1:])
            ),
        }
        report["success"] = (
            frame_count >= max(5, int(args.fps * min(args.seconds, 1.0) * 0.5))
            and len(center_depths) > 0
        )
        status = "PASS" if report["success"] else "FAIL"
        print(
            f"{status}: {frame_count} aligned RGB-D frames, "
            f"{measured_fps:.1f} FPS, "
            f"median center depth={report['stream']['median_center_depth_m']} m"
        )
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        print(f"FAIL: {report['error']}", file=sys.stderr)
    finally:
        if pipeline is not None:
            try:
                pipeline.stop()
            except RuntimeError:
                pass
        if args.preview:
            try:
                import cv2

                cv2.destroyAllWindows()
            except ImportError:
                pass
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Report: {output.resolve()}")
    return 0 if report["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
