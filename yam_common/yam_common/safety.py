"""Latched position stop and explicit normal-completion parking for YAM.

Powered holding requires a healthy CAN/control loop. It is not a hardware E-stop.
"""

from __future__ import annotations

import enum
import functools
import json
import logging
import time
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


class ShutdownResult(enum.Enum):
    PARKED = "parked"
    HOLDING_FAULT = "holding_fault"
    STOP_FAILED = "stop_failed"
    NOT_CONNECTED = "not_connected"


def validate_park_config(cfg):
    if cfg.park_pose_path:
        if cfg.rest_pose is not None:
            raise ValueError("Specify only one of rest_pose and park_pose_path")
        data = json.loads(Path(cfg.park_pose_path).expanduser().read_text())
        if data.get("schema") != "yam-park-v1":
            raise ValueError("Unsupported park pose file")
        for key in ("motor_offsets", "motor_directions", "gripper_type"):
            if data.get(key) != getattr(cfg, key):
                raise ValueError(
                    f"Park pose {key} does not match this robot configuration"
                )
        cfg.rest_pose = tuple(data["rest_pose"])
    for name in (
        "parking_max_joint_velocity",
        "parking_max_joint_acceleration",
        "parking_max_joint_jerk",
        "parking_min_duration",
        "parking_max_duration",
        "parking_settle_pos_tolerance",
        "parking_settle_vel_tolerance",
        "parking_settle_duration",
        "parking_max_tracking_error",
    ):
        value = getattr(cfg, name)
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be positive and finite")
    if cfg.parking_min_duration > cfg.parking_max_duration:
        raise ValueError("parking_min_duration exceeds parking_max_duration")
    for name in ("parking_kp_scale", "parking_kd_scale"):
        if not 0 < getattr(cfg, name) <= 1:
            raise ValueError(f"{name} must be in (0, 1]")
    if cfg.rest_pose is not None:
        from yam_common.yam_arm import motor_names_for_config

        names = motor_names_for_config(cfg)
        pose = np.asarray(cfg.rest_pose, dtype=float)
        if pose.shape != (len(names),) or not np.all(np.isfinite(pose)):
            raise ValueError(f"rest_pose must have {len(names)} finite values")
        for name, value in zip(names, pose):
            lo, hi = (0, 1) if name == "gripper" else cfg.joint_limits[name]
            if not lo <= value <= hi:
                raise ValueError(f"rest_pose {name} is outside joint limits")


def motion_command(method):
    @functools.wraps(method)
    def guarded(self, *args, **kwargs):
        with self._motion_lock:
            self._ensure_healthy()
            if self.safety_state != "active":
                raise RuntimeError(f"YAM motion blocked: {self.safety_state}")
            try:
                return method(self, *args, **kwargs)
            except Exception:
                self.emergency_stop()
                raise

    return guarded


class SafetyLifecycle:
    def _gains(self, scale=False):
        from yam_common.yam_arm import motor_names_for_config

        names = motor_names_for_config(self.config)
        kp = np.array([self.config.kp_gains[n] for n in names])
        kd = np.array([self.config.kd_gains[n] for n in names])
        if np.any(kp <= 0) or np.any(kd <= 0):
            raise ValueError("Position holding requires positive kp and kd gains")
        if scale:
            kp *= self.config.parking_kp_scale
            kd *= self.config.parking_kd_scale
        return kp, kd

    def emergency_stop(self) -> bool:
        """Latch a zero-velocity position target; never travel to park or release torque.

        False means holding could not be confirmed; operator intervention is required.
        Further motion stays blocked even when holding fails.
        """
        with self._motion_lock:
            if self._robot is None:
                return False
            if self.safety_state in ("stopped", "stop_failed"):
                return self.safety_state == "stopped"
            self.safety_state = "stop_failed"  # fail closed before touching hardware
            try:
                kp, kd = self._gains()
                self._robot.emergency_stop(kp, kd)
                self.safety_state = "stopped"
                logger.warning(
                    "YAM stop latched. Holding current position; CAN remains connected."
                )
                return True
            except Exception:
                logger.exception(
                    "YAM STOP FAILED: powered holding is unavailable; support the arm"
                )
                return False

    def close(self):
        if self._robot is not None and self.safety_state != "parked":
            self.emergency_stop()

    def disconnect(self):
        self.close()

    def emergency_cleanup(self):
        self.emergency_stop()

    def release_after_support(self):
        """Explicit operator action: arm AND payload must already be supported.

        This disables motors and closes CAN. Never called automatically on a fault.
        """
        with self._motion_lock:
            if self._robot is not None:
                self._robot.close()
                self._robot = None
            self.safety_state = "disconnected"

    @staticmethod
    def _quintic(u):
        return 10 * u**3 - 15 * u**4 + 6 * u**5, 30 * u**2 - 60 * u**3 + 30 * u**4

    def _compute_parking_duration(self, delta):
        c = self.config
        distance = float(np.max(np.abs(delta)))
        duration = max(
            c.parking_min_duration,
            1.875 * distance / c.parking_max_joint_velocity,
            (10 / np.sqrt(3) * distance / c.parking_max_joint_acceleration) ** 0.5,
            (60 * distance / c.parking_max_joint_jerk) ** (1 / 3),
        )
        return duration if duration <= c.parking_max_duration else None

    def _feedback(self):
        self._ensure_healthy()
        pos = np.array(self._robot.get_joint_pos(), dtype=float, copy=True)
        vel = np.asarray(self._robot.get_observations()["joint_vel"], dtype=float)
        if pos.shape != (len(self.action_keys),) or vel.shape != pos.shape:
            raise RuntimeError("Incomplete joint feedback")
        if not np.all(np.isfinite(pos)) or not np.all(np.isfinite(vel)):
            raise RuntimeError("Non-finite joint feedback")
        from yam_common.yam_arm import motor_names_for_config

        for name, value in zip(motor_names_for_config(self.config), pos):
            lo, hi = (0, 1) if name == "gripper" else self.config.joint_limits[name]
            if not lo <= value <= hi:
                raise RuntimeError(f"Joint feedback outside limits: {name}")
        return pos, vel

    def _park_command(self, pos, vel, kp, kd):
        with self._motion_lock:
            if self.safety_state != "parking":
                raise RuntimeError("Parking interrupted by stop")
            self._ensure_healthy()
            self._robot.command_joint_state(
                {"pos": pos, "vel": vel, "kp": kp, "kd": kd}
            )

    def _settle(self, target, kp, kd, deadline):
        since = None
        while time.monotonic() < deadline:
            self._park_command(target, np.zeros_like(target), kp, kd)
            pos, vel = self._feedback()
            now = time.monotonic()
            if np.max(np.abs(pos - target)) > self.config.parking_max_tracking_error:
                raise RuntimeError("Parking tracking error")
            if (
                np.max(np.abs(pos - target)) <= self.config.parking_settle_pos_tolerance
                and np.max(np.abs(vel)) <= self.config.parking_settle_vel_tolerance
            ):
                since = now if since is None else since
                if now - since >= self.config.parking_settle_duration:
                    return
            else:
                since = None
            time.sleep(0.004)
        raise RuntimeError("Parking settle timeout")

    def park_to_rest(self):
        """Normal completion only. Hold first, then bounded quintic motion and settling.

        The gripper target stays at its starting position to retain a payload.
        A configured collision-free path and healthy feedback are prerequisites.
        """
        if self._robot is None:
            return False
        with self._motion_lock:
            if self.safety_state != "active":
                return False
            self.safety_state = "parking"
        try:
            if self.config.rest_pose is None:
                raise RuntimeError("No park pose configured")
            start, _ = self._feedback()
            kp, kd = self._gains()
            self._settle(
                start,
                kp,
                kd,
                time.monotonic() + 1 + self.config.parking_settle_duration,
            )
            rest = np.array(self.config.rest_pose, dtype=float)
            if len(rest) == 7:
                rest[-1] = start[-1]
            delta = rest - start
            duration = self._compute_parking_duration(delta)
            if duration is None:
                raise RuntimeError("Parking exceeds configured duration limits")
            kp, kd = self._gains(scale=True)
            t0 = time.monotonic()
            while True:
                elapsed = time.monotonic() - t0
                u = min(elapsed / duration, 1.0)
                s, ds = self._quintic(u)
                desired = start + delta * s
                pos, _ = self._feedback()
                if (
                    np.max(np.abs(pos - desired))
                    > self.config.parking_max_tracking_error
                ):
                    raise RuntimeError("Parking tracking error")
                self._park_command(desired, delta * ds / duration, kp, kd)
                if u == 1:
                    break
                time.sleep(0.004)
            self._settle(
                rest, kp, kd, time.monotonic() + 1 + self.config.parking_settle_duration
            )
            with self._motion_lock:
                if self.safety_state != "parking":
                    return False
                self.safety_state = "parked"
            return True
        except BaseException as exc:
            self.emergency_stop()
            logger.exception("Parking aborted; no automatic torque release")
            # Keep signals visible to the caller after latching the stop.
            if not isinstance(exc, Exception):
                raise
            return False

    def controlled_shutdown(self):
        if self._robot is None:
            return ShutdownResult.NOT_CONNECTED
        if self.safety_state == "parked" or self.park_to_rest():
            with self._motion_lock:
                if self.safety_state == "parked":
                    if self.config.park_release_torque:
                        self.release_after_support()
                    return ShutdownResult.PARKED
        return (
            ShutdownResult.HOLDING_FAULT
            if self.safety_state == "stopped"
            else ShutdownResult.STOP_FAILED
        )
