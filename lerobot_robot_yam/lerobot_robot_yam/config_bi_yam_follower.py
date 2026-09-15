"""Configuration for bimanual YAM Follower robot."""

from dataclasses import dataclass, field

from lerobot.cameras import CameraConfig
from lerobot.robots.config import RobotConfig

from .config_yam_follower import YAMFollowerConfig


@RobotConfig.register_subclass("bi_yam_follower")
@dataclass
class BiYAMFollowerRobotConfig(RobotConfig):
    left_arm_config: YAMFollowerConfig = field(default_factory=lambda: YAMFollowerConfig(port="can0"))
    right_arm_config: YAMFollowerConfig = field(default_factory=lambda: YAMFollowerConfig(port="can1"))
    cameras: dict[str, CameraConfig] = field(default_factory=dict)
