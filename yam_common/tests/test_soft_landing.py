"""Controlled shutdown parking and emergency stop."""

from __future__ import annotations

import numpy as np
import pytest

from fakes import TEST_REST_POSE, FakeRobot


def _make_arm(robot: FakeRobot, **config_overrides):
    from yam_common import YAMArm, YAMArmConfig

    defaults = dict(
        use_gravity_compensation=False,
        rest_pose=TEST_REST_POSE,
        parking_min_duration=0.02,
        parking_max_duration=0.1,
    )
    defaults.update(config_overrides)
    return YAMArm(YAMArmConfig(**defaults), robot_factory=lambda **kwargs: robot)


# --- quintic trajectory ---

def test_quintic_boundary_conditions() -> None:
    from yam_common.yam_arm import YAMArm

    s0, ds0 = YAMArm._quintic(0.0)
    assert s0 == pytest.approx(0.0)
    assert ds0 == pytest.approx(0.0)

    s1, ds1 = YAMArm._quintic(1.0)
    assert s1 == pytest.approx(1.0)
    assert ds1 == pytest.approx(0.0)

    s_mid, ds_mid = YAMArm._quintic(0.5)
    assert s_mid == pytest.approx(0.5)
    assert ds_mid > 0


# --- park_to_rest ---

def test_park_to_rest_moves_toward_rest_pose() -> None:
    start_pos = np.array([0.5, 2.0, 2.0, 0.3, 0.3, 0.3, 0.2])
    robot = FakeRobot(pos=start_pos)
    arm = _make_arm(robot)
    arm.connect()

    parked = arm.park_to_rest()

    assert parked is True
    np.testing.assert_allclose(robot.pos, np.array(TEST_REST_POSE), atol=0.05)
    assert len(robot.commands) > 1


def test_park_to_rest_already_at_rest_returns_true() -> None:
    robot = FakeRobot(pos=np.array(TEST_REST_POSE))
    arm = _make_arm(robot)
    arm.connect()

    assert arm.park_to_rest() is True


def test_park_to_rest_not_connected_returns_false() -> None:
    robot = FakeRobot()
    arm = _make_arm(robot)

    assert arm.park_to_rest() is False


# --- controlled_shutdown ---

def test_controlled_shutdown_parks_then_releases() -> None:
    start_pos = np.array([0.5, 2.0, 2.0, 0.3, 0.3, 0.3, 0.2])
    robot = FakeRobot(pos=start_pos)
    arm = _make_arm(robot)
    arm.connect()

    arm.disconnect()

    assert robot.zero_torque is True
    assert robot.closed is True
    assert arm.is_connected is False
    assert len(robot.commands) > 0


def test_controlled_shutdown_holds_on_failure() -> None:
    robot = FakeRobot()
    arm = _make_arm(robot)
    arm.connect()
    robot.control_loop_error = RuntimeError("CAN bus down")

    arm.close()

    assert robot.closed is True
    assert arm.is_connected is False


# --- emergency_stop ---

def test_emergency_stop_holds_current_position() -> None:
    start_pos = np.array([0.5, 2.0, 2.0, 0.3, 0.3, 0.3, 0.2])
    robot = FakeRobot(pos=start_pos)
    arm = _make_arm(robot)
    arm.connect()

    arm.emergency_stop()

    assert len(robot.commands) >= 1
    np.testing.assert_allclose(robot.commands[-1], start_pos)
    assert robot.zero_torque is False


def test_emergency_stop_not_connected_is_noop() -> None:
    robot = FakeRobot()
    arm = _make_arm(robot)

    arm.emergency_stop()
    assert len(robot.commands) == 0


def test_emergency_cleanup_still_closes() -> None:
    start_pos = np.array([0.5, 2.0, 2.0, 0.3, 0.3, 0.3, 0.2])
    robot = FakeRobot(pos=start_pos)
    arm = _make_arm(robot)
    arm.connect()

    arm.emergency_cleanup()

    assert robot.closed is True
    assert arm.is_connected is False


# --- duration computation ---

def test_parking_duration_scales_with_distance() -> None:
    robot = FakeRobot()
    arm = _make_arm(robot, parking_min_duration=0.5, parking_max_duration=10.0)

    small_delta = np.array([0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01])
    large_delta = np.array([2.0, 2.0, 2.0, 1.0, 1.0, 1.0, 0.5])

    t_small = arm._compute_parking_duration(small_delta)
    t_large = arm._compute_parking_duration(large_delta)

    assert t_small < t_large
    assert t_small >= 0.5
    assert t_large <= 10.0
