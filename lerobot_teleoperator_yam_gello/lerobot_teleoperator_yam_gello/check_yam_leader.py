#!/usr/bin/env python3
"""GELLO の初期位置と関節方向を単体で確認する。"""

from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean
from typing import Mapping, Sequence

from lerobot_teleoperator_yam_gello import YAMLeaderTeleop, YAMLeaderTeleopConfig


# YAM follower の既定可動範囲（rad）。
# lerobot_robot_yam/config_yam_follower.py と同じ値を使用する。
YAM_JOINT_LIMITS = {
    "shoulder_pan": (-2.767, 3.28),
    "shoulder_lift": (-0.15, 3.8),
    "elbow_flex": (-0.15, 3.28),
    "wrist_flex": (-1.72, 1.72),
    "wrist_roll": (-1.72, 1.72),
    "wrist_yaw": (-2.24, 2.24),
}

ARM_JOINT_NAMES = tuple(YAM_JOINT_LIMITS)


class JapaneseArgumentParser(argparse.ArgumentParser):
    """argparse の固定見出しを日本語で表示する。"""

    def format_help(self) -> str:
        return super().format_help().replace("usage:", "使用方法:")


@dataclass(frozen=True)
class LeaderSample:
    raw_ticks: dict[str, float]
    normalized_action: dict[str, float]
    joint_radians: dict[str, float]


def normalized_action_to_radians(action: Mapping[str, float]) -> dict[str, float]:
    """LeRobot の [-100, 100] 指令を follower の物理角度へ戻す。"""

    positions: dict[str, float] = {}
    for name, (lower, upper) in YAM_JOINT_LIMITS.items():
        value = float(action[f"{name}.pos"])
        if not math.isfinite(value):
            positions[name] = math.nan
            continue
        ratio = (value + 100.0) / 200.0
        positions[name] = lower + ratio * (upper - lower)
    return positions


def evaluate_initial_position(
    positions: Mapping[str, float],
    *,
    tolerance_rad: float,
) -> dict[str, bool]:
    """各アーム関節が 0 rad 付近にあるかを返す。"""

    return {
        name: math.isfinite(float(positions[name]))
        and abs(float(positions[name])) <= tolerance_rad
        for name in ARM_JOINT_NAMES
    }


def movement_matches_direction(
    reference: float,
    moved: float,
    *,
    expected_sign: int,
    threshold_rad: float,
) -> tuple[bool, float]:
    """基準位置から指定方向へ十分に移動したかを判定する。"""

    if expected_sign not in (-1, 1):
        raise ValueError("expected_sign は -1 または 1 である必要があります")
    delta = float(moved) - float(reference)
    passed = (
        math.isfinite(delta)
        and abs(delta) >= threshold_rad
        and delta * expected_sign > 0.0
    )
    return passed, delta


def _average_dict(samples: Sequence[Mapping[str, float]]) -> dict[str, float]:
    keys = tuple(samples[0])
    return {key: fmean(float(sample[key]) for sample in samples) for key in keys}


def collect_sample(
    leader: YAMLeaderTeleop,
    *,
    sample_count: int,
    sample_interval_s: float,
) -> LeaderSample:
    raw_samples: list[dict[str, float]] = []
    action_samples: list[dict[str, float]] = []

    for index in range(sample_count):
        action_samples.append(leader.get_action())
        raw_samples.append(
            {
                name: float(value)
                for name, value in leader.bus.sync_read(
                    "Present_Position", normalize=False
                ).items()
            }
        )
        if index + 1 < sample_count:
            time.sleep(sample_interval_s)

    normalized_action = _average_dict(action_samples)
    return LeaderSample(
        raw_ticks=_average_dict(raw_samples),
        normalized_action=normalized_action,
        joint_radians=normalized_action_to_radians(normalized_action),
    )


def print_sample(sample: LeaderSample) -> None:
    print("\n測定値:")
    for name in ARM_JOINT_NAMES:
        normalized = sample.normalized_action[f"{name}.pos"]
        radians = sample.joint_radians[name]
        print(
            f"  {name:14s} raw={sample.raw_ticks[name]:8.1f} "
            f"normalized={normalized:8.2f} physical={radians:8.4f} rad"
        )
    print(
        f"  {'gripper':14s} raw={sample.raw_ticks['gripper']:8.1f} "
        f"normalized={sample.normalized_action['gripper.pos']:8.2f}"
    )


def prompt_and_collect(
    leader: YAMLeaderTeleop,
    prompt: str,
    *,
    sample_count: int,
    sample_interval_s: float,
) -> LeaderSample:
    input(f"\n{prompt}\n準備ができたら ENTER を押してください。")
    sample = collect_sample(
        leader,
        sample_count=sample_count,
        sample_interval_s=sample_interval_s,
    )
    print_sample(sample)
    return sample


def run_interactive_checks(
    leader: YAMLeaderTeleop,
    *,
    sample_count: int,
    sample_interval_s: float,
    zero_tolerance_rad: float,
    movement_threshold_rad: float,
    gripper_closed_max: float,
    gripper_open_min: float,
) -> bool:
    results: list[tuple[str, bool]] = []

    initial = prompt_and_collect(
        leader,
        "全アーム関節を 0 rad 付近の初期姿勢に戻してください。",
        sample_count=sample_count,
        sample_interval_s=sample_interval_s,
    )
    initial_checks = evaluate_initial_position(
        initial.joint_radians,
        tolerance_rad=zero_tolerance_rad,
    )
    for name, passed in initial_checks.items():
        results.append((f"初期位置: {name}", passed))
        status = "合格" if passed else "不合格"
        print(
            f"{status}: {name} = {initial.joint_radians[name]:.4f} rad "
            f"(許容値 ±{zero_tolerance_rad:.4f} rad)"
        )

    for name in ARM_JOINT_NAMES:
        reference = prompt_and_collect(
            leader,
            f"{name} を 0 rad 付近に戻し、他の関節をできるだけ動かさないでください。",
            sample_count=sample_count,
            sample_interval_s=sample_interval_s,
        )
        reference_ok = evaluate_initial_position(
            {joint: reference.joint_radians[joint] for joint in ARM_JOINT_NAMES},
            tolerance_rad=zero_tolerance_rad,
        )[name]
        results.append((f"方向試験前の原点: {name}", reference_ok))

        positive = prompt_and_collect(
            leader,
            f"{name} だけを YAM の定義上の正方向へ動かしてください。",
            sample_count=sample_count,
            sample_interval_s=sample_interval_s,
        )
        positive_ok, positive_delta = movement_matches_direction(
            reference.joint_radians[name],
            positive.joint_radians[name],
            expected_sign=1,
            threshold_rad=movement_threshold_rad,
        )
        results.append((f"正方向: {name}", positive_ok))
        print(
            f"{'合格' if positive_ok else '不合格'}: {name} の正方向差分 "
            f"{positive_delta:+.4f} rad"
        )

        negative_reference = prompt_and_collect(
            leader,
            f"{name} を再び 0 rad 付近に戻してください。",
            sample_count=sample_count,
            sample_interval_s=sample_interval_s,
        )
        negative_reference_ok = evaluate_initial_position(
            {joint: negative_reference.joint_radians[joint] for joint in ARM_JOINT_NAMES},
            tolerance_rad=zero_tolerance_rad,
        )[name]
        results.append((f"負方向試験前の原点: {name}", negative_reference_ok))

        negative = prompt_and_collect(
            leader,
            f"{name} だけを YAM の定義上の負方向へ動かしてください。",
            sample_count=sample_count,
            sample_interval_s=sample_interval_s,
        )
        negative_ok, negative_delta = movement_matches_direction(
            negative_reference.joint_radians[name],
            negative.joint_radians[name],
            expected_sign=-1,
            threshold_rad=movement_threshold_rad,
        )
        results.append((f"負方向: {name}", negative_ok))
        print(
            f"{'合格' if negative_ok else '不合格'}: {name} の負方向差分 "
            f"{negative_delta:+.4f} rad"
        )

        prompt_and_collect(
            leader,
            f"安全のため {name} を 0 rad 付近に戻してください。",
            sample_count=sample_count,
            sample_interval_s=sample_interval_s,
        )

    closed = prompt_and_collect(
        leader,
        "グリッパーを閉じてください。",
        sample_count=sample_count,
        sample_interval_s=sample_interval_s,
    )
    closed_value = closed.normalized_action["gripper.pos"]
    closed_ok = math.isfinite(closed_value) and closed_value <= gripper_closed_max
    results.append(("グリッパー閉", closed_ok))

    opened = prompt_and_collect(
        leader,
        "グリッパーを開いてください。",
        sample_count=sample_count,
        sample_interval_s=sample_interval_s,
    )
    opened_value = opened.normalized_action["gripper.pos"]
    opened_ok = math.isfinite(opened_value) and opened_value >= gripper_open_min
    results.append(("グリッパー開", opened_ok))

    print("\n=== 試験結果 ===")
    for label, passed in results:
        print(f"{'合格' if passed else '不合格'}: {label}")

    return all(passed for _, passed in results)


def parse_args() -> argparse.Namespace:
    parser = JapaneseArgumentParser(
        description=(
            "YAM follower を接続せず、GELLO の初期位置と関節方向を確認します。"
        ),
        add_help=False,
    )
    parser._optionals.title = "オプション"
    parser.add_argument(
        "-h", "--help", action="help", help="このヘルプを表示して終了します"
    )
    parser.add_argument("--port", required=True, help="GELLO のシリアルポート")
    parser.add_argument(
        "--calibration-dir",
        type=Path,
        default=Path.cwd(),
        help="gello-01.json を含むディレクトリ",
    )
    parser.add_argument("--id", default="gello-01", help="校正ファイル名の拡張子を除いた部分")
    parser.add_argument("--sample-count", type=int, default=10, help="各姿勢で平均する測定回数")
    parser.add_argument(
        "--sample-interval-s", type=float, default=0.02, help="測定間隔（秒）"
    )
    parser.add_argument(
        "--zero-tolerance-rad", type=float, default=0.15, help="初期位置の許容誤差（rad）"
    )
    parser.add_argument(
        "--movement-threshold-rad", type=float, default=0.10, help="方向判定に必要な最小移動量（rad）"
    )
    parser.add_argument(
        "--gripper-closed-max", type=float, default=10.0, help="閉状態として許容する最大値"
    )
    parser.add_argument(
        "--gripper-open-min", type=float, default=90.0, help="開状態として許容する最小値"
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    calibration_path = args.calibration_dir / f"{args.id}.json"
    if not calibration_path.is_file():
        print(f"校正ファイルが見つかりません: {calibration_path}")
        return 2
    if args.sample_count <= 0:
        print("--sample-count は 1 以上である必要があります。")
        return 2

    config = YAMLeaderTeleopConfig(
        id=args.id,
        calibration_dir=args.calibration_dir,
        port=args.port,
        preflight_range_check=False,
        freeze_out_of_range=True,
    )
    leader = YAMLeaderTeleop(config)

    print("この試験は GELLO のみを使用し、YAM follower へ指令を送りません。")
    print("GELLO のトルクは無効のままです。各指示では対象関節だけをゆっくり動かしてください。")

    try:
        input(
            "\n接続する前に、全アーム関節を 0 rad 付近の初期姿勢に戻してください。"
            "\n準備ができたら ENTER を押してください。"
        )
        leader.connect(calibrate=False)
        passed = run_interactive_checks(
            leader,
            sample_count=args.sample_count,
            sample_interval_s=args.sample_interval_s,
            zero_tolerance_rad=args.zero_tolerance_rad,
            movement_threshold_rad=args.movement_threshold_rad,
            gripper_closed_max=args.gripper_closed_max,
            gripper_open_min=args.gripper_open_min,
        )
    except KeyboardInterrupt:
        print("\n試験を中断しました。")
        return 130
    finally:
        if leader.is_connected:
            leader.disconnect()

    if passed:
        print("\n全項目に合格しました。")
        return 0
    print("\n不合格項目があります。校正値、関節方向、raw tick の折り返しを確認してください。")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
