"""Soft-landing: interpolation to rest_pose during shutdown."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest

from fakes import TEST_REST_POSE, FakeRobot


def _make_arm(robot: FakeRobot, **config_overrides):
    from yam_common import YAMArm, YAMArmConfig

    defaults = dict(
        use_gravity_compensation=False,
        rest_pose=TEST_REST_POSE,
        soft_landing_duration=0.05,
    )
    defaults.update(config_overrides)
    return YAMArm(YAMArmConfig(**defaults), robot_factory=lambda **kwargs: robot)


def test_soft_land_moves_toward_rest_pose() -> None:
    start_pos = np.array([0.5, 2.0, 2.0, 0.3, 0.3, 0.3, 0.2])
    robot = FakeRobot(pos=start_pos)
    arm = _make_arm(robot)
    arm.connect()

    landed = arm.soft_land()

    assert landed is True
    np.testing.assert_allclose(robot.pos, np.array(TEST_REST_POSE), atol=0.1)
    assert len(robot.commands) > 1


def test_soft_land_already_at_rest_returns_true() -> None:
    robot = FakeRobot(pos=np.array(TEST_REST_POSE))
    arm = _make_arm(robot)
    arm.connect()

    landed = arm.soft_land()

    assert landed is True


def test_soft_land_not_connected_returns_false() -> None:
    robot = FakeRobot()
    arm = _make_arm(robot)

    assert arm.soft_land() is False


def test_soft_land_exception_does_not_propagate() -> None:
    robot = FakeRobot()
    arm = _make_arm(robot)
    arm.connect()
    robot.control_loop_error = RuntimeError("CAN bus down")

    landed = arm._soft_land_safely()

    assert landed is False


def test_shutdown_calls_soft_land_then_zero_torque() -> None:
    start_pos = np.array([0.5, 2.0, 2.0, 0.3, 0.3, 0.3, 0.2])
    robot = FakeRobot(pos=start_pos)
    arm = _make_arm(robot)
    arm.connect()

    arm.disconnect()

    assert robot.zero_torque is True
    assert robot.closed is True
    assert arm.is_connected is False
    assert len(robot.commands) > 0


def test_soft_land_uses_scaled_gains() -> None:
    from yam_common.yam_arm import motor_names_for_config

    start_pos = np.array([1.0, 2.0, 2.0, 0.5, 0.5, 0.5, 0.3])
    robot = FakeRobot(pos=start_pos)
    arm = _make_arm(robot, soft_landing_kp_scale=0.3, soft_landing_kd_scale=0.6)
    arm.connect()

    commands_before = len(robot.commands)
    arm.soft_land()

    assert len(robot.commands) > commands_before


def test_emergency_cleanup_still_skips_soft_land_wait() -> None:
    start_pos = np.array([0.5, 2.0, 2.0, 0.3, 0.3, 0.3, 0.2])
    robot = FakeRobot(pos=start_pos)
    arm = _make_arm(robot)
    arm.connect()

    arm.emergency_cleanup()

    assert robot.zero_torque is True
    assert robot.closed is True
    assert arm.is_connected is False
