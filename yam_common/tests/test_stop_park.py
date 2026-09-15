"""Regression tests for distinct stop and normal-park lifecycles."""

import json
import threading

import numpy as np
import pytest
from fakes import TEST_REST_POSE, FakeRobot
from yam_common import YAMArm, YAMArmConfig
from yam_common.safety import ShutdownResult


def make_arm(robot=None, **options):
    robot = robot or FakeRobot(pos=np.array(TEST_REST_POSE))
    config = dict(
        use_gravity_compensation=False,
        rest_pose=TEST_REST_POSE,
        parking_min_duration=0.01,
        parking_settle_duration=0.005,
    )
    config.update(options)
    arm = YAMArm(YAMArmConfig(**config), robot_factory=lambda **kw: robot)
    arm.connect()
    return arm, robot


def test_stop_has_no_deceleration_trajectory_and_latches():
    arm, robot = make_arm()
    robot.vel[:] = 2.0
    start = robot.pos.copy()
    assert arm.emergency_stop()
    assert len(robot.commands) == 1
    np.testing.assert_array_equal(robot.commands[0], start)
    assert not robot.closed and not robot.zero_torque
    assert arm.emergency_stop()
    assert len(robot.commands) == 1
    for call in (
        lambda: arm.send_action({}),
        lambda: arm.command_rest(),
        lambda: arm.zero_torque(),
        lambda: arm.update_kp_kd(np.ones(7), np.ones(7)),
    ):
        with pytest.raises(RuntimeError, match="blocked"):
            call()
    assert not arm.park_to_rest()
    arm.disconnect()
    assert not robot.closed


def test_failed_stop_is_not_reported_as_holding_or_released():
    arm, robot = make_arm()
    robot.control_loop_error = RuntimeError("CAN down")
    assert not arm.emergency_stop()
    assert arm.safety_state == "stop_failed"
    assert arm.controlled_shutdown() is ShutdownResult.STOP_FAILED
    assert not robot.closed and not robot.zero_torque


def test_normal_park_keeps_gripper_and_holds_until_explicit_release():
    robot = FakeRobot(pos=np.array(TEST_REST_POSE) + 0.01)
    arm, robot = make_arm(robot)
    grip = robot.pos[-1]
    assert arm.controlled_shutdown() is ShutdownResult.PARKED
    np.testing.assert_allclose(robot.pos[:6], TEST_REST_POSE[:6])
    assert robot.pos[-1] == grip
    assert not robot.zero_torque and not robot.closed
    arm.disconnect()
    assert not robot.closed
    arm.release_after_support()
    assert robot.closed and not arm.is_connected


def test_supported_park_may_release_only_on_success():
    arm, robot = make_arm(park_release_torque=True)
    assert arm.controlled_shutdown() is ShutdownResult.PARKED
    assert robot.closed


@pytest.mark.parametrize(
    "fault", ["missing", "timeout", "nan", "missing_velocity", "infeasible"]
)
def test_park_fault_never_releases(fault):
    arm, robot = make_arm(park_release_torque=True)
    if fault == "missing":
        arm.config.rest_pose = None
    elif fault == "timeout":
        robot.vel[:] = 1
    elif fault == "nan":
        robot.pos[0] = np.nan
    elif fault == "missing_velocity":
        original = robot.get_observations
        robot.get_observations = lambda: {
            k: v for k, v in original().items() if k != "joint_vel"
        }
    else:
        arm.config.parking_max_duration = 0.01
        robot.pos[0] = 1
    assert arm.controlled_shutdown() in (
        ShutdownResult.HOLDING_FAULT,
        ShutdownResult.STOP_FAILED,
    )
    assert not robot.closed and not robot.zero_torque


def test_stop_interrupts_park_and_no_target_follows_stop():
    arm, robot = make_arm(FakeRobot(pos=np.array(TEST_REST_POSE) + 0.05))
    sent = threading.Event()
    original = robot.command_joint_state

    def command(state):
        original(state)
        sent.set()

    robot.command_joint_state = command
    result = []
    thread = threading.Thread(target=lambda: result.append(arm.park_to_rest()))
    thread.start()
    assert sent.wait(1)
    assert arm.emergency_stop()
    count = len(robot.commands)
    thread.join(2)
    assert not thread.is_alive()
    assert result == [False]
    assert len(robot.commands) == count
    assert not robot.closed


@pytest.mark.parametrize(
    "name,value",
    [
        ("parking_min_duration", 0),
        ("parking_max_duration", float("nan")),
        ("parking_settle_vel_tolerance", 0),
        ("parking_max_tracking_error", -1),
        ("rest_pose", (99,) * 7),
    ],
)
def test_invalid_park_config(name, value):
    with pytest.raises(ValueError):
        YAMArmConfig(**{name: value})


def test_saved_pose_roundtrip_and_calibration_mismatch(tmp_path):
    from yam_common.park_pose import pose_document

    config = YAMArmConfig()
    file = tmp_path / "park.json"
    file.write_text(json.dumps(pose_document(config, TEST_REST_POSE)))
    loaded = YAMArmConfig(park_pose_path=str(file))
    assert loaded.rest_pose == TEST_REST_POSE
    offsets = dict(config.motor_offsets, shoulder_pan=0.1)
    with pytest.raises(ValueError, match="motor_offsets"):
        YAMArmConfig(park_pose_path=str(file), motor_offsets=offsets)


def test_low_level_stop_restores_gains_and_blocks_all_writers():
    from types import SimpleNamespace

    from yam_common.motor_chain_robot import MotorChainRobot
    from yam_common.utils import JointMapper

    robot = MotorChainRobot.__new__(MotorChainRobot)
    robot._command_lock = threading.RLock()
    robot._state_lock = threading.Lock()
    robot._safety_stopped = False
    robot.motor_chain = list(range(7))
    robot._joint_state = SimpleNamespace(pos=np.arange(7) / 10)
    robot.remapper = JointMapper({}, 7)
    robot._joint_limits = None
    robot._kp = robot._kd = np.zeros(7)
    robot.raise_if_unhealthy = lambda: None
    sent = []
    robot._update_locked = lambda: sent.append(robot._commands)
    robot.emergency_stop(np.ones(7) * 80, np.ones(7) * 5)
    assert len(sent) == 1
    np.testing.assert_array_equal(sent[0].pos, robot._joint_state.pos)
    np.testing.assert_array_equal(sent[0].vel, np.zeros(7))
    assert np.all(sent[0].kp == 80)
    for call in (
        lambda: robot.command_joint_pos(np.zeros(7)),
        lambda: robot.command_joint_state({"pos": np.zeros(7), "vel": np.zeros(7)}),
        robot.zero_torque_mode,
        lambda: robot.update_kp_kd(np.ones(7), np.ones(7)),
    ):
        with pytest.raises(RuntimeError, match="latched"):
            call()
