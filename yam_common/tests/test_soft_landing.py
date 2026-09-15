"""Controlled shutdown parking, emergency stop, and trajectory safety."""

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
        parking_max_duration=5.0,
        parking_settle_duration=0.005,
    )
    defaults.update(config_overrides)
    return YAMArm(YAMArmConfig(**defaults), robot_factory=lambda **kwargs: robot)


# --- quintic ---

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


# --- duration computation ---

def test_parking_duration_scales_with_distance() -> None:
    robot = FakeRobot()
    arm = _make_arm(robot, parking_min_duration=0.5, parking_max_duration=10.0)

    small_delta = np.array([0.01] * 7)
    large_delta = np.array([2.0, 2.0, 2.0, 1.0, 1.0, 1.0, 0.5])

    t_small = arm._compute_parking_duration(small_delta)
    t_large = arm._compute_parking_duration(large_delta)

    assert t_small is not None
    assert t_large is not None
    assert t_small < t_large
    assert t_small >= 0.5


def test_parking_duration_returns_none_when_infeasible() -> None:
    robot = FakeRobot()
    arm = _make_arm(robot, parking_max_duration=0.5)

    huge_delta = np.array([3.0, 3.0, 3.0, 1.0, 1.0, 1.0, 0.5])
    assert arm._compute_parking_duration(huge_delta) is None


def test_parking_duration_includes_jerk_constraint() -> None:
    robot = FakeRobot()
    common = dict(parking_max_joint_velocity=10.0, parking_max_joint_acceleration=50.0, parking_max_duration=20.0)
    arm_low_jerk = _make_arm(robot, parking_max_joint_jerk=10.0, **common)
    arm_high_jerk = _make_arm(robot, parking_max_joint_jerk=1000.0, **common)

    delta = np.array([1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.3])
    t_low = arm_low_jerk._compute_parking_duration(delta)
    t_high = arm_high_jerk._compute_parking_duration(delta)

    assert t_low is not None and t_high is not None
    assert t_low > t_high


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


def test_park_to_rest_already_at_rest() -> None:
    robot = FakeRobot(pos=np.array(TEST_REST_POSE))
    arm = _make_arm(robot)
    arm.connect()

    assert arm.park_to_rest() is True


def test_park_to_rest_not_connected() -> None:
    robot = FakeRobot()
    arm = _make_arm(robot)

    assert arm.park_to_rest() is False


def test_park_to_rest_infeasible_returns_false() -> None:
    start_pos = np.array([3.0, 3.5, 3.0, 1.5, 1.5, 2.0, 0.9])
    robot = FakeRobot(pos=start_pos)
    arm = _make_arm(robot, parking_max_duration=0.1)
    arm.connect()

    assert arm.park_to_rest() is False


def test_park_to_rest_tracking_error_aborts() -> None:
    class DriftingRobot(FakeRobot):
        def get_joint_pos(self):
            pos = super().get_joint_pos()
            return pos + 0.5  # large persistent offset

    start_pos = np.array([0.5, 2.0, 2.0, 0.3, 0.3, 0.3, 0.2])
    robot = DriftingRobot(pos=start_pos)
    arm = _make_arm(robot, parking_max_tracking_error=0.1)
    arm.connect()

    assert arm.park_to_rest() is False


# --- controlled_shutdown ---

def test_controlled_shutdown_parked_result() -> None:
    from yam_common.yam_arm import ShutdownResult

    start_pos = np.array([0.5, 2.0, 2.0, 0.3, 0.3, 0.3, 0.2])
    robot = FakeRobot(pos=start_pos)
    arm = _make_arm(robot)
    arm.connect()

    result = arm.controlled_shutdown()

    assert result is ShutdownResult.PARKED
    assert robot.zero_torque is True


def test_controlled_shutdown_holding_fault_on_infeasible() -> None:
    from yam_common.yam_arm import ShutdownResult

    start_pos = np.array([3.0, 3.5, 3.0, 1.5, 1.5, 2.0, 0.9])
    robot = FakeRobot(pos=start_pos)
    arm = _make_arm(robot, parking_max_duration=0.1)
    arm.connect()

    result = arm.controlled_shutdown()

    assert result is ShutdownResult.HOLDING_FAULT
    assert robot.zero_torque is False


def test_controlled_shutdown_exception_holds() -> None:
    from yam_common.yam_arm import ShutdownResult

    robot = FakeRobot()
    arm = _make_arm(robot)
    arm.connect()
    robot.control_loop_error = RuntimeError("CAN bus down")

    result = arm.controlled_shutdown()

    assert result is ShutdownResult.HOLDING_FAULT


def test_disconnect_calls_controlled_shutdown() -> None:
    start_pos = np.array([0.5, 2.0, 2.0, 0.3, 0.3, 0.3, 0.2])
    robot = FakeRobot(pos=start_pos)
    arm = _make_arm(robot, shutdown_zero_gravity_wait_for_enter=False)
    arm.connect()

    arm.disconnect()

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


def test_emergency_cleanup_skips_parking() -> None:
    start_pos = np.array([0.5, 2.0, 2.0, 0.3, 0.3, 0.3, 0.2])
    robot = FakeRobot(pos=start_pos)
    arm = _make_arm(robot)
    arm.connect()

    arm.emergency_cleanup()

    assert robot.zero_torque is True
    assert robot.closed is True
    assert arm.is_connected is False
