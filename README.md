# LeRobot YAM Plugins

YAM follower arm + GELLO leader plugins for LeRobot.

## Quickstart

Requirements: CAN interface (e.g. `can0`) and a GELLO leader (port configurable).
GELLO assembly information from [gello_mechanical](https://github.com/wuphilipp/gello_mechanical/).

```bash
uv sync
uv run lerobot-teleoperate \
  --robot.type=yam_follower \
  --robot.port=can0 \
  --robot.gripper_type=crank_4310 \
  --teleop.type=yam_leader \
  --teleop.port=/dev/ttyUSB0
```

GELLO 01 の校正ファイル（`calibrations/gello-01.json`）を使用する場合：
```bash
uv run lerobot-teleoperate \
  --robot.type=yam_follower \
  --robot.port=can0 \
  --robot.gripper_type=crank_4310 \
  --teleop.type=yam_leader \
  --teleop.port=/dev/ttyUSB0 \
  --teleop.calibration_dir=./calibrations \
  --teleop.id=gello-01
```
既存の校正ファイルを読み込む場合は `--teleop.id` が必要です。LeRobot は
`<teleop.calibration_dir>/<teleop.id>.json` の形式で校正ファイルを参照します。

## GELLO の単体動作確認

YAM follower を接続する前に、GELLO の校正値、初期位置、関節方向を
単体で確認できます。この試験は follower へ指令を送信しません。

```bash
uv run --project lerobot_teleoperator_yam_gello check-yam-leader \
  --port=/dev/ttyUSB0 \
  --calibration-dir=calibrations \
  --id=gello-01
```

画面の指示に従い、次の項目を順番に確認します。

- アーム 6 軸が初期姿勢で 0 rad から ±0.15 rad 以内であること
- 各関節を YAM の定義上の正方向へ動かしたとき、角度差分が +0.10 rad 以上になること
- 各関節を YAM の定義上の負方向へ動かしたとき、角度差分が -0.10 rad 以下になること
- グリッパーの正規化値が、閉状態で 10 以下、開状態で 90 以上になること

正負方向の基準はスクリプト側では自動判定できません。YAM の関節定義を確認しながら、
指示された物理方向へ対象関節だけをゆっくり動かしてください。各測定では raw tick も
表示されるため、`shoulder_pan` で 0–4095 tick の折り返しが発生していないかどうかも
確認してください。全項目に合格した場合は終了コード 0、不合格項目がある場合は 1、
設定エラーの場合は 2 を返します。

判定ロジックと校正 JSON のユニットテストは次のコマンドで実行できます。

```bash
uv run pytest -q lerobot_teleoperator_yam_gello/tests/test_check_yam_leader.py
```

Note: this repo expects `lerobot` 0.4.3 features (plugin discovery in the
standard CLIs). If 0.4.3 is not on PyPI, install it from
[source](https://github.com/huggingface/lerobot) instead.

## Record

```bash
uv run lerobot-record \
  --dataset.repo_id=YOUR_USERNAME/yam_teleop \
  --dataset.single_task="Teleop YAM arm (pick/place)" \
  --dataset.fps=30 \
  --dataset.num_episodes=5 \
  --dataset.episode_time_s=60 \
  --dataset.reset_time_s=15 \
  --dataset.push_to_hub=true \
  --robot.type=yam_follower \
  --robot.port=can0 \
  --robot.gripper_type=crank_4310 \
  --teleop.type=yam_leader \
  --teleop.port=/dev/ttyUSB0
```

## Policy Control

```bash
uv run lerobot-control \
  --robot.type=yam_follower \
  --robot.port=can0 \
  --policy.path=YOUR_USERNAME/yam-policy
```

## Extras

List cameras:
```bash
uv run lerobot-find-cameras opencv
```

Teleop with cameras + Rerun:
```bash
uv run lerobot-teleoperate \
  --robot.type=yam_follower \
  --robot.port=can0 \
  --robot.gripper_type=crank_4310 \
  --robot.cameras='{
    "wrist": {"type": "opencv", "index_or_path": "/dev/video6", "width": 640, "height": 480, "fps": 30},
    "front": {"type": "opencv", "index_or_path": "/dev/video4", "width": 640, "height": 480, "fps": 30}
  }' \
  --teleop.type=yam_leader \
  --teleop.port=/dev/ttyUSB0 \
  --display_data=true
```

Safety step limits (normalized per‑cycle steps: joints in [-100, 100], gripper in [0, 100]):
```bash
--robot.lerobot_max_step=1 \
--robot.lerobot_gripper_max_step=1
```

## Packages

| Package | Purpose |
| --- | --- |
| `yam-common` | Shared YAM utilities |
| `lerobot_robot_yam` | Follower robot plugin |
| `lerobot_teleoperator_yam_gello` | Leader teleoperator plugin |

Install packages separately if you only need one component (e.g., follower‑only for policy inference).
