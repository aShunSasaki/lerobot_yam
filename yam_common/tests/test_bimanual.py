"""BiYAMFollower: prefix routing, coordinated stop, and shutdown result."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

from fakes import TEST_REST_POSE, FakeRobot

_REPO_ROOT = Path(__file__).resolve().parents[2]
_FOLLOWER_SRC = _REPO_ROOT / "lerobot_robot_yam"
if str(_FOLLOWER_SRC) not in sys.path:
    sys.path.insert(0, str(_FOLLOWER_SRC))


def _make_bi_follower(left_robot=None, right_robot=None):
    pytest.importorskip("lerobot")

    from lerobot_robot_yam.bi_yam_follower import BiYAMFollower
    from lerobot_robot_yam.config_bi_yam_follower import BiYAMFollowerRobotConfig
    from lerobot_robot_yam.config_yam_follower import YAMFollowerConfig
    from yam_common import YAMArm, YAMArmConfig

    left_robot = left_robot or FakeRobot(pos=np.array(TEST_REST_POSE))
    right_robot = right_robot or FakeRobot(pos=np.array(TEST_REST_POSE))

    left_cfg = YAMFollowerConfig(
        port="can0",
        use_gravity_compensation=False,
        rest_pose=TEST_REST_POSE,
        rest_min_duration=0.01,
        rest_settle_duration=0.005,
    )
    right_cfg = YAMFollowerConfig(
        port="can1",
        use_gravity_compensation=False,
        rest_pose=TEST_REST_POSE,
        rest_min_duration=0.01,
        rest_settle_duration=0.005,
    )

    config = BiYAMFollowerRobotConfig(
        left_arm_config=left_cfg,
        right_arm_config=right_cfg,
    )

    follower = BiYAMFollower(config)
    follower.left_arm._arm = YAMArm(
        YAMArmConfig(
            use_gravity_compensation=False,
            rest_pose=TEST_REST_POSE,
            rest_min_duration=0.01,
            rest_settle_duration=0.005,
        ),
        robot_factory=lambda **kw: left_robot,
    )
    follower.right_arm._arm = YAMArm(
        YAMArmConfig(
            use_gravity_compensation=False,
            rest_pose=TEST_REST_POSE,
            rest_min_duration=0.01,
            rest_settle_duration=0.005,
        ),
        robot_factory=lambda **kw: right_robot,
    )
    follower.left_arm._arm.connect()
    follower.right_arm._arm.connect()
    return follower


def test_observation_keys_have_left_right_prefix():
    follower = _make_bi_follower()
    obs = follower.get_observation()
    left_keys = [k for k in obs if k.startswith("left_")]
    right_keys = [k for k in obs if k.startswith("right_")]
    assert len(left_keys) == 7
    assert len(right_keys) == 7
    assert "left_shoulder_pan.pos" in obs
    assert "right_shoulder_pan.pos" in obs


def test_action_features_have_left_right_prefix():
    follower = _make_bi_follower()
    features = follower.action_features
    assert "left_shoulder_pan.pos" in features
    assert "right_gripper.pos" in features
    assert len(features) == 14


def test_send_action_routes_to_both_arms():
    left_robot = FakeRobot(pos=np.array(TEST_REST_POSE))
    right_robot = FakeRobot(pos=np.array(TEST_REST_POSE))
    follower = _make_bi_follower(left_robot, right_robot)

    action = {}
    for key in follower.action_features:
        action[key] = 0.0

    sent = follower.send_action(action)
    assert len(left_robot.commands) >= 1
    assert len(right_robot.commands) >= 1
    assert all(k.startswith("left_") or k.startswith("right_") for k in sent)


def test_soft_stop_stops_both_arms():
    follower = _make_bi_follower()
    result = follower.soft_stop()
    assert result is True


def test_soft_stop_one_arm_fails():
    left_robot = FakeRobot(pos=np.array(TEST_REST_POSE))
    right_robot = FakeRobot(pos=np.array(TEST_REST_POSE))
    follower = _make_bi_follower(left_robot, right_robot)
    right_robot.control_loop_error = RuntimeError("CAN down")
    result = follower.soft_stop()
    assert result is False


def test_controlled_shutdown_returns_worst_case():
    from yam_common.safety import ShutdownResult

    left_robot = FakeRobot(pos=np.array(TEST_REST_POSE) + 0.01)
    right_robot = FakeRobot(pos=np.array(TEST_REST_POSE) + 0.01)
    follower = _make_bi_follower(left_robot, right_robot)
    result = follower.controlled_shutdown()
    assert result is ShutdownResult.AT_REST
