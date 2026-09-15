"""Configuration for bimanual YAM GELLO Leader."""

from dataclasses import dataclass

from lerobot.teleoperators.config import TeleoperatorConfig

from .config_yam_leader import YAMLeaderConfig


@TeleoperatorConfig.register_subclass("bi_yam_leader")
@dataclass
class BiYAMLeaderTeleopConfig(TeleoperatorConfig):
    left_arm_config: YAMLeaderConfig = None
    right_arm_config: YAMLeaderConfig = None
