"""Save a validated physical park pose: python -m yam_common.park_pose."""

import argparse
import dataclasses
import json
import sys
from pathlib import Path

from yam_common import YAMArm, YAMArmConfig


def pose_document(config, pose):
    # Validate in exactly the coordinate system used by this arm.
    dataclasses.replace(config, rest_pose=tuple(pose), park_pose_path=None)
    return {
        "schema": "yam-park-v1",
        "rest_pose": list(pose),
        "motor_offsets": config.motor_offsets,
        "motor_directions": config.motor_directions,
        "gripper_type": config.gripper_type,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        required=True,
        type=Path,
        help="YAMArmConfig JSON (port, offsets, etc.)",
    )
    parser.add_argument("--output", required=True, type=Path)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--joints",
        type=float,
        nargs="+",
        help="Physical radians; optional gripper is 0..1",
    )
    mode.add_argument(
        "--capture",
        action="store_true",
        help="Capture current hardware pose interactively",
    )
    args = parser.parse_args()
    config = YAMArmConfig(**json.loads(args.config.read_text()))
    if args.output.exists():
        parser.error("Output already exists; choose a new filename")
    if args.capture and not sys.stdin.isatty():
        parser.error("--capture requires an interactive terminal")
    arm = None
    try:
        pose = args.joints
        if args.capture:
            arm = YAMArm(config)
            arm.connect()
            input(
                "Position the arm at a supported park pose, then press ENTER to capture and HOLD: "
            )
            if not arm.emergency_stop():
                raise RuntimeError("Stop failed; support the arm and check hardware")
            pose = arm.get_joint_pos().tolist()
        document = pose_document(config, pose)
        with args.output.open("x") as file:
            json.dump(document, file, indent=2)
            file.write("\n")
        print(
            f"Saved {args.output}. Use --robot.park_pose_path={args.output.resolve()}"
        )
        if arm is not None:
            input(
                "Arm is HOLDING. Support the arm AND payload; press ENTER to disable motors and exit: "
            )
            arm.release_after_support()
    finally:
        if arm is not None:
            arm.close()  # On interruption hold; never release automatically.


if __name__ == "__main__":
    main()
