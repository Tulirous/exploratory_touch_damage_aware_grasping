from __future__ import annotations

import argparse
import json
from pathlib import Path
import signal
import time
from typing import Any

import numpy as np

from human_mimic_demo.config import load_config, nested
from human_mimic_demo.controllers import DryRunArm, DryRunHand, LinkerHandO6, URRTDEArm
from human_mimic_demo.messages import TrackingSample
from human_mimic_demo.retargeting import O6Retargeter
from human_mimic_demo.safety import DemoState, ForceGuard, RelativeArmMapper
from human_mimic_demo.tracking import RealSenseMediaPipeTracker, SyntheticTracker


class JsonlLogger:
    def __init__(self, directory: str) -> None:
        output = Path(directory)
        output.mkdir(parents=True, exist_ok=True)
        name = time.strftime("session_%Y%m%d_%H%M%S.jsonl")
        self.path = output / name
        self.handle = self.path.open("w", encoding="utf-8")

    def write(self, payload: dict[str, Any]) -> None:
        self.handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
        self.handle.flush()

    def close(self) -> None:
        self.handle.close()


class MimicDemo:
    def __init__(self, config: dict, mode: str, tracker_name: str, display: bool) -> None:
        self.config = config
        self.mode = mode
        self.display = display
        self.tracker = self._make_tracker(tracker_name)
        self.arm = DryRunArm(nested(config, "arm").get("dry_run_initial_tcp"))
        self.hand = DryRunHand()
        if mode == "hardware":
            self.hand = LinkerHandO6(nested(config, "hand"))
            self.arm = URRTDEArm(nested(config, "arm"))
        self.retargeter = O6Retargeter(nested(config, "retargeting"))
        self.mapper = RelativeArmMapper.from_config(nested(config, "arm_mapping"))
        self.force_guard = ForceGuard(
            nested(config, "force_guard"), self.retargeter.open_position
        )
        self.state = DemoState.WAITING_CALIBRATION
        self.last_sample_time = 0.0
        self.tracking_timeout = float(nested(config, "safety").get("tracking_timeout_s", 0.35))
        self.confidence_threshold = float(nested(config, "camera").get("min_confidence", 0.65))
        self.logger = JsonlLogger(str(nested(config, "logging").get("directory", "human_mimic_demo/logs")))
        self.running = True
        self._last_status_print = 0.0

    def _make_tracker(self, tracker_name: str):
        if tracker_name == "synthetic":
            return SyntheticTracker()
        if tracker_name == "realsense":
            return RealSenseMediaPipeTracker(nested(self.config, "camera"))
        raise ValueError(f"Unsupported tracker: {tracker_name}")

    def calibrate(self, sample: TrackingSample) -> None:
        self.mapper.calibrate(sample.wrist_xyz_m, self.arm.current_tcp())
        self.retargeter.reset_filter()
        self.state = DemoState.DISARMED
        print("[calibrated] Human wrist and current UR5 TCP are now relative origins.")

    def toggle_arm(self) -> None:
        if self.state == DemoState.ESTOP:
            print("[blocked] Restart the process after ESTOP.")
        elif not self.mapper.calibrated:
            print("[blocked] Press 'c' to calibrate before arming.")
        elif self.state == DemoState.ARMED:
            self.state = DemoState.DISARMED
            self.arm.hold()
            print("[disarmed]")
        else:
            self.mapper.previous_target = self.arm.current_tcp()
            self.state = DemoState.ARMED
            print("[armed] Robot commands are enabled for the selected mode.")

    def estop(self) -> None:
        self.state = DemoState.ESTOP
        self.arm.hold()
        print("[ESTOP] Motion stopped. Restart is required.")

    def _key(self, sample: TrackingSample | None) -> int:
        if not self.display or sample is None or sample.frame_bgr is None:
            return -1
        import cv2

        frame = sample.frame_bgr
        cv2.putText(
            frame,
            f"state={self.state.value}  c=calibrate a=arm o=open f=fist e=estop q=quit",
            (12, 26),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )
        cv2.imshow("UR5 + O6 human mimic demo", frame)
        return cv2.waitKey(1) & 0xFF

    def _handle_key(self, key: int, sample: TrackingSample | None) -> None:
        if key == ord("q"):
            self.running = False
        elif key == ord("e"):
            self.estop()
        elif key == ord("c") and sample is not None:
            self.calibrate(sample)
        elif key == ord("a"):
            self.toggle_arm()
        elif key == ord("o") and sample is not None:
            self.retargeter.record_open(sample.hand_joints)
            print("[hand calibration] Recorded the current pose as open.")
        elif key == ord("f") and sample is not None:
            self.retargeter.record_closed(sample.hand_joints)
            print("[hand calibration] Recorded the current pose as closed/fist.")

    def step(self, sample: TrackingSample | None) -> None:
        now = time.time()
        if sample is None or not sample.valid() or sample.confidence < self.confidence_threshold:
            if self.state == DemoState.ARMED and now - self.last_sample_time > self.tracking_timeout:
                self.state = DemoState.TRACKING_LOST
                self.arm.hold()
                print("[tracking lost] Arm is holding. Press 'a' after tracking recovers.")
            return

        self.last_sample_time = now
        command, features = self.retargeter.retarget(sample.hand_joints)
        target_tcp = self.mapper.target(sample.wrist_xyz_m) if self.mapper.calibrated else None
        force = self.hand.force() if self.state == DemoState.ARMED else None
        command, force_override = self.force_guard.apply(command, force)

        if self.state == DemoState.ARMED and target_tcp is not None:
            self.arm.command_tcp(target_tcp)
            self.hand.command(command)

        self.logger.write(
            {
                "timestamp": sample.timestamp,
                "mode": self.mode,
                "state": self.state.value,
                "source": sample.source,
                "confidence": sample.confidence,
                "human_wrist_camera_m": sample.wrist_xyz_m.tolist(),
                "hand_features": features.as_array().tolist(),
                "o6_command": command.tolist(),
                "target_tcp": None if target_tcp is None else target_tcp.tolist(),
                "force_override": force_override,
            }
        )
        if now - self._last_status_print > 1.0:
            tcp_text = "uncalibrated" if target_tcp is None else np.array2string(target_tcp[:3], precision=3)
            print(f"[{self.state.value}] wrist={sample.wrist_xyz_m.round(3)} tcp={tcp_text} o6={command}")
            self._last_status_print = now

    def run(self, duration: float, auto_calibrate: bool, auto_arm: bool) -> None:
        started = time.monotonic()
        first_valid = True
        while self.running and (duration <= 0.0 or time.monotonic() - started < duration):
            sample = self.tracker.read()
            if (
                sample is not None
                and sample.valid()
                and sample.confidence >= self.confidence_threshold
                and first_valid
            ):
                if auto_calibrate:
                    self.calibrate(sample)
                if auto_arm:
                    self.toggle_arm()
                first_valid = False
            self.step(sample)
            key = self._key(sample)
            self._handle_key(key, sample)
            if sample is not None and sample.source == "synthetic":
                time.sleep(1.0 / 30.0)

    def close(self) -> None:
        try:
            self.arm.hold()
            self.arm.close()
        finally:
            self.hand.close()
            self.tracker.close()
            self.logger.close()
            if self.display:
                try:
                    import cv2

                    cv2.destroyAllWindows()
                except ImportError:
                    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="human_mimic_demo/configs/demo.json")
    parser.add_argument("--mode", choices=["dry-run", "hardware"], default="dry-run")
    parser.add_argument("--tracker", choices=["synthetic", "realsense"], default="synthetic")
    parser.add_argument("--duration", type=float, default=0.0, help="Seconds; 0 runs until quit")
    parser.add_argument("--display", action="store_true", help="Show camera view and keyboard control")
    parser.add_argument("--auto-calibrate", action="store_true")
    parser.add_argument("--auto-arm", action="store_true")
    parser.add_argument(
        "--i-understand-hardware-risk",
        action="store_true",
        help="Required acknowledgement for real UR5/O6 commands",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode == "hardware" and not args.i_understand_hardware_risk:
        raise SystemExit("Hardware mode requires --i-understand-hardware-risk")
    if args.mode == "hardware" and (args.auto_calibrate or args.auto_arm):
        raise SystemExit("Hardware mode forbids automatic calibration/arming; use the GUI keys")
    demo = MimicDemo(load_config(args.config), args.mode, args.tracker, args.display)
    signal.signal(signal.SIGINT, lambda *_: setattr(demo, "running", False))
    try:
        demo.run(args.duration, args.auto_calibrate, args.auto_arm)
    finally:
        demo.close()
        print(f"Session log: {demo.logger.path}")


if __name__ == "__main__":
    main()
