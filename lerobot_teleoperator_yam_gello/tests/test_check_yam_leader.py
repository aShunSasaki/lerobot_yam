import math
from pathlib import Path
from types import SimpleNamespace

import draccus
import pytest
from lerobot.motors import MotorCalibration

from lerobot_teleoperator_yam_gello import check_yam_leader


ARM_JOINT_NAMES = check_yam_leader.ARM_JOINT_NAMES
YAM_JOINT_LIMITS = check_yam_leader.YAM_JOINT_LIMITS
evaluate_initial_position = check_yam_leader.evaluate_initial_position
movement_matches_direction = check_yam_leader.movement_matches_direction
normalized_action_to_radians = check_yam_leader.normalized_action_to_radians


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_gello_01_calibration_is_loadable_by_lerobot() -> None:
    calibration_path = REPO_ROOT / "calibrations" / "gello-01.json"

    with calibration_path.open() as calibration_file, draccus.config_type("json"):
        calibration = draccus.load(dict[str, MotorCalibration], calibration_file)

    assert tuple(calibration) == (*ARM_JOINT_NAMES, "gripper")
    assert tuple(item.id for item in calibration.values()) == (1, 2, 3, 4, 5, 6, 7)
    assert calibration["shoulder_lift"].drive_mode == 1
    assert calibration["gripper"].range_min == 1557
    assert calibration["gripper"].range_max == 2234


def test_main_requests_zero_pose_before_connect(monkeypatch, tmp_path: Path) -> None:
    (tmp_path / "gello-01.json").write_text("{}")
    args = SimpleNamespace(
        id="gello-01",
        calibration_dir=tmp_path,
        port="/dev/null",
        sample_count=1,
        sample_interval_s=0.0,
        zero_tolerance_rad=0.15,
        movement_threshold_rad=0.10,
        gripper_closed_max=10.0,
        gripper_open_min=90.0,
    )
    events: list[str] = []

    class FakeLeader:
        def __init__(self, config) -> None:
            self.is_connected = False

        def connect(self, calibrate: bool = True) -> None:
            events.append("connect")
            self.is_connected = True

        def disconnect(self) -> None:
            events.append("disconnect")
            self.is_connected = False

    monkeypatch.setattr(check_yam_leader, "parse_args", lambda: args)
    monkeypatch.setattr(check_yam_leader, "YAMLeaderTeleop", FakeLeader)
    monkeypatch.setattr(
        check_yam_leader,
        "run_interactive_checks",
        lambda leader, **kwargs: True,
    )
    monkeypatch.setattr(
        "builtins.input",
        lambda prompt: events.append("zero_prompt"),
    )

    assert check_yam_leader.main() == 0
    assert events[:2] == ["zero_prompt", "connect"]


def test_initial_position_is_checked_in_physical_radians() -> None:
    action = {}
    for name, (lower, upper) in YAM_JOINT_LIMITS.items():
        action[f"{name}.pos"] = 200.0 * (-lower) / (upper - lower) - 100.0
    action["gripper.pos"] = 100.0

    positions = normalized_action_to_radians(action)
    checks = evaluate_initial_position(positions, tolerance_rad=0.02)

    assert all(abs(positions[name]) < 1e-6 for name in ARM_JOINT_NAMES)
    assert all(checks.values())


def test_initial_position_rejects_nonfinite_or_nonzero_joint() -> None:
    positions = dict.fromkeys(ARM_JOINT_NAMES, 0.0)
    positions["shoulder_pan"] = 0.2
    positions["wrist_yaw"] = math.nan

    checks = evaluate_initial_position(positions, tolerance_rad=0.1)

    assert checks["shoulder_pan"] is False
    assert checks["wrist_yaw"] is False


@pytest.mark.parametrize(
    ("reference", "moved", "expected_sign", "expected"),
    [
        (0.0, 0.2, 1, True),
        (0.0, -0.2, -1, True),
        (0.0, -0.2, 1, False),
        (0.0, 0.2, -1, False),
        (0.0, 0.02, 1, False),
    ],
)
def test_direction_check_requires_the_expected_sign_and_minimum_movement(
    reference: float,
    moved: float,
    expected_sign: int,
    expected: bool,
) -> None:
    passed, delta = movement_matches_direction(
        reference,
        moved,
        expected_sign=expected_sign,
        threshold_rad=0.05,
    )

    assert passed is expected
    assert delta == pytest.approx(moved - reference)
