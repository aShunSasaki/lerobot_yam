"""Bimanual YAM GELLO leader: two YAMLeader teleoperators."""

import logging
from functools import cached_property

from lerobot.lerobot_types import RobotAction
from lerobot.teleoperators.teleoperator import Teleoperator
from lerobot.utils.bimanual import BimanualMixin
from lerobot.utils.decorators import check_if_not_connected

from .config_bi_yam_leader import BiYAMLeaderTeleopConfig
from .config_yam_leader import YAMLeaderTeleopConfig
from .yam_leader import YAMLeader

logger = logging.getLogger(__name__)


class BiYAMLeader(BimanualMixin, Teleoperator):
    config_class = BiYAMLeaderTeleopConfig
    name = "bi_yam_leader"

    def __init__(self, config: BiYAMLeaderTeleopConfig):
        super().__init__(config)
        self.config = config

        left_arm_config = YAMLeaderTeleopConfig(
            id=f"{config.id}_left" if config.id else None,
            calibration_dir=config.calibration_dir,
            port=config.left_arm_config.port,
            baudrate=config.left_arm_config.baudrate,
            read_retries=config.left_arm_config.read_retries,
            read_retry_sleep_s=config.left_arm_config.read_retry_sleep_s,
            preflight_range_check=config.left_arm_config.preflight_range_check,
            preflight_refresh_hz=config.left_arm_config.preflight_refresh_hz,
            freeze_out_of_range=config.left_arm_config.freeze_out_of_range,
            out_of_range_warn_period_s=config.left_arm_config.out_of_range_warn_period_s,
            out_of_range_tolerance=config.left_arm_config.out_of_range_tolerance,
        )

        right_arm_config = YAMLeaderTeleopConfig(
            id=f"{config.id}_right" if config.id else None,
            calibration_dir=config.calibration_dir,
            port=config.right_arm_config.port,
            baudrate=config.right_arm_config.baudrate,
            read_retries=config.right_arm_config.read_retries,
            read_retry_sleep_s=config.right_arm_config.read_retry_sleep_s,
            preflight_range_check=config.right_arm_config.preflight_range_check,
            preflight_refresh_hz=config.right_arm_config.preflight_refresh_hz,
            freeze_out_of_range=config.right_arm_config.freeze_out_of_range,
            out_of_range_warn_period_s=config.right_arm_config.out_of_range_warn_period_s,
            out_of_range_tolerance=config.right_arm_config.out_of_range_tolerance,
        )

        self.left_arm = YAMLeader(left_arm_config)
        self.right_arm = YAMLeader(right_arm_config)

    @cached_property
    def action_features(self) -> dict[str, type]:
        return {
            **{f"left_{k}": v for k, v in self.left_arm.action_features.items()},
            **{f"right_{k}": v for k, v in self.right_arm.action_features.items()},
        }

    @cached_property
    def feedback_features(self) -> dict[str, type]:
        return {}

    @check_if_not_connected
    def get_action(self) -> RobotAction:
        action: RobotAction = {}
        for key, value in self.left_arm.get_action().items():
            action[f"left_{key}"] = value
        for key, value in self.right_arm.get_action().items():
            action[f"right_{key}"] = value
        return action

    @check_if_not_connected
    def send_feedback(self, feedback: dict[str, float]) -> None:
        pass


class BiYAMLeaderTeleop(BiYAMLeader):
    pass
