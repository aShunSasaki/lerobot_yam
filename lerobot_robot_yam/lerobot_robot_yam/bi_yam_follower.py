"""Bimanual YAM follower: two YAMFollower arms with coordinated safety stop."""

import logging
from functools import cached_property

from lerobot.cameras.utils import make_cameras_from_configs
from lerobot.processor import RobotAction, RobotObservation
from lerobot.robots.robot import Robot
from lerobot.utils.bimanual import BimanualMixin
from lerobot.utils.decorators import check_if_not_connected

from .config_bi_yam_follower import BiYAMFollowerRobotConfig
from .config_yam_follower import YAMFollowerRobotConfig
from .yam_follower import YAMFollower

logger = logging.getLogger(__name__)


_SHUTDOWN_SEVERITY = {"not_connected": 0, "at_rest": 1, "holding_fault": 2, "stop_failed": 3}


class BiYAMFollower(BimanualMixin, Robot):
    config_class = BiYAMFollowerRobotConfig
    name = "bi_yam_follower"

    def __init__(self, config: BiYAMFollowerRobotConfig):
        super().__init__(config)
        self.config = config

        self._top_level_cam_keys = set(config.cameras)
        collisions = self._top_level_cam_keys & set(config.left_arm_config.cameras) | \
                     self._top_level_cam_keys & set(config.right_arm_config.cameras)
        if collisions:
            raise ValueError(f"Top-level camera names collide with per-arm names: {sorted(collisions)}")

        left_arm_cameras = {**config.left_arm_config.cameras, **config.cameras}
        left_cfg = config.left_arm_config
        right_cfg = config.right_arm_config

        left_arm_config = YAMFollowerRobotConfig(
            id=f"{config.id}_left" if config.id else None,
            calibration_dir=config.calibration_dir,
            port=left_cfg.port,
            bitrate=left_cfg.bitrate,
            bustype=left_cfg.bustype,
            motor_offsets=left_cfg.motor_offsets,
            motor_directions=left_cfg.motor_directions,
            kp_gains=left_cfg.kp_gains,
            kd_gains=left_cfg.kd_gains,
            joint_limits=left_cfg.joint_limits,
            gripper_limits=left_cfg.gripper_limits,
            use_gravity_compensation=left_cfg.use_gravity_compensation,
            gravity_comp_factor=left_cfg.gravity_comp_factor,
            mujoco_xml_path=left_cfg.mujoco_xml_path,
            gripper_type=left_cfg.gripper_type,
            zero_gravity_mode=left_cfg.zero_gravity_mode,
            shutdown_zero_gravity_wait_for_enter=left_cfg.shutdown_zero_gravity_wait_for_enter,
            limit_gripper_force=left_cfg.limit_gripper_force,
            lerobot_max_step=left_cfg.lerobot_max_step,
            lerobot_gripper_max_step=left_cfg.lerobot_gripper_max_step,
            rest_pose=left_cfg.rest_pose,
            rest_pose_path=getattr(left_cfg, "rest_pose_path", None),
            rest_release_torque=getattr(left_cfg, "rest_release_torque", False),
            rest_max_joint_velocity=left_cfg.rest_max_joint_velocity,
            rest_max_joint_acceleration=left_cfg.rest_max_joint_acceleration,
            rest_max_joint_jerk=left_cfg.rest_max_joint_jerk,
            rest_min_duration=left_cfg.rest_min_duration,
            rest_max_duration=left_cfg.rest_max_duration,
            rest_kp_scale=left_cfg.rest_kp_scale,
            rest_kd_scale=left_cfg.rest_kd_scale,
            rest_settle_pos_tolerance=left_cfg.rest_settle_pos_tolerance,
            rest_settle_vel_tolerance=left_cfg.rest_settle_vel_tolerance,
            rest_settle_duration=left_cfg.rest_settle_duration,
            rest_max_tracking_error=left_cfg.rest_max_tracking_error,
            rest_decel_velocity_threshold=left_cfg.rest_decel_velocity_threshold,
            cameras=left_arm_cameras,
        )

        right_arm_config = YAMFollowerRobotConfig(
            id=f"{config.id}_right" if config.id else None,
            calibration_dir=config.calibration_dir,
            port=right_cfg.port,
            bitrate=right_cfg.bitrate,
            bustype=right_cfg.bustype,
            motor_offsets=right_cfg.motor_offsets,
            motor_directions=right_cfg.motor_directions,
            kp_gains=right_cfg.kp_gains,
            kd_gains=right_cfg.kd_gains,
            joint_limits=right_cfg.joint_limits,
            gripper_limits=right_cfg.gripper_limits,
            use_gravity_compensation=right_cfg.use_gravity_compensation,
            gravity_comp_factor=right_cfg.gravity_comp_factor,
            mujoco_xml_path=right_cfg.mujoco_xml_path,
            gripper_type=right_cfg.gripper_type,
            zero_gravity_mode=right_cfg.zero_gravity_mode,
            shutdown_zero_gravity_wait_for_enter=right_cfg.shutdown_zero_gravity_wait_for_enter,
            limit_gripper_force=right_cfg.limit_gripper_force,
            lerobot_max_step=right_cfg.lerobot_max_step,
            lerobot_gripper_max_step=right_cfg.lerobot_gripper_max_step,
            rest_pose=right_cfg.rest_pose,
            rest_pose_path=getattr(right_cfg, "rest_pose_path", None),
            rest_release_torque=getattr(right_cfg, "rest_release_torque", False),
            rest_max_joint_velocity=right_cfg.rest_max_joint_velocity,
            rest_max_joint_acceleration=right_cfg.rest_max_joint_acceleration,
            rest_max_joint_jerk=right_cfg.rest_max_joint_jerk,
            rest_min_duration=right_cfg.rest_min_duration,
            rest_max_duration=right_cfg.rest_max_duration,
            rest_kp_scale=right_cfg.rest_kp_scale,
            rest_kd_scale=right_cfg.rest_kd_scale,
            rest_settle_pos_tolerance=right_cfg.rest_settle_pos_tolerance,
            rest_settle_vel_tolerance=right_cfg.rest_settle_vel_tolerance,
            rest_settle_duration=right_cfg.rest_settle_duration,
            rest_max_tracking_error=right_cfg.rest_max_tracking_error,
            rest_decel_velocity_threshold=right_cfg.rest_decel_velocity_threshold,
            cameras=right_cfg.cameras,
        )

        self.left_arm = YAMFollower(left_arm_config)
        self.right_arm = YAMFollower(right_arm_config)
        self.cameras = {**self.left_arm.cameras, **self.right_arm.cameras}

    @property
    def _motors_ft(self) -> dict[str, type]:
        return {
            **{f"left_{k}": v for k, v in self.left_arm._motors_ft.items()},
            **{f"right_{k}": v for k, v in self.right_arm._motors_ft.items()},
        }

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        out: dict[str, tuple] = {}
        for k, v in self.left_arm._cameras_ft.items():
            out[k if k in self._top_level_cam_keys else f"left_{k}"] = v
        for k, v in self.right_arm._cameras_ft.items():
            out[f"right_{k}"] = v
        return out

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        return {**self._motors_ft, **self._cameras_ft}

    @cached_property
    def action_features(self) -> dict[str, type]:
        return self._motors_ft

    @check_if_not_connected
    def get_observation(self) -> RobotObservation:
        obs: RobotObservation = {}
        for key, value in self.left_arm.get_observation().items():
            obs[key if key in self._top_level_cam_keys else f"left_{key}"] = value
        for key, value in self.right_arm.get_observation().items():
            obs[f"right_{key}"] = value
        return obs

    @check_if_not_connected
    def send_action(self, action: RobotAction) -> RobotAction:
        left_action = {k.removeprefix("left_"): v for k, v in action.items() if k.startswith("left_")}
        right_action = {k.removeprefix("right_"): v for k, v in action.items() if k.startswith("right_")}
        sent_left = self.left_arm.send_action(left_action)
        sent_right = self.right_arm.send_action(right_action)
        return {
            **{f"left_{k}": v for k, v in sent_left.items()},
            **{f"right_{k}": v for k, v in sent_right.items()},
        }

    def soft_stop(self) -> bool:
        left_ok = self.left_arm.soft_stop()
        right_ok = self.right_arm.soft_stop()
        return left_ok and right_ok

    def controlled_shutdown(self):
        from yam_common.safety import ShutdownResult
        left_result = self.left_arm.controlled_shutdown()
        right_result = self.right_arm.controlled_shutdown()
        severity_left = _SHUTDOWN_SEVERITY.get(left_result.value, 0) if left_result else 0
        severity_right = _SHUTDOWN_SEVERITY.get(right_result.value, 0) if right_result else 0
        return left_result if severity_left >= severity_right else right_result

    def wait_for_safe_release(self) -> None:
        self.left_arm.wait_for_safe_release()
        self.right_arm.wait_for_safe_release()

    def release_after_support(self) -> None:
        self.left_arm.release_after_support()
        self.right_arm.release_after_support()

    def disconnect(self) -> None:
        self.left_arm.disconnect()
        self.right_arm.disconnect()

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if exc_type is None:
            self.controlled_shutdown()
        else:
            self.disconnect()


class BiYAMFollowerRobot(BiYAMFollower):
    pass
