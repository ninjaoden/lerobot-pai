#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Minimal terminal-based hardware smoke test for the NMini follower arm.

Example:
```shell
python -m lerobot.scripts.lerobot_nmini_keyboard_test \
  --robot.type=nmini_follower \
  --robot.port=/dev/ttyACM0 \
  --robot.id=nmini
```
"""

import select
import sys
import termios
import tty
from dataclasses import dataclass
from pprint import pformat

import draccus

from lerobot.robots import (  # noqa: F401
    RobotConfig,
    make_robot_from_config,
    nmini_follower,
)
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.utils import init_logging


@dataclass
class NMiniKeyboardTestConfig:
    robot: RobotConfig
    joint_step: float = 5.0
    gripper_step: float = 5.0


class RawTerminalInput:
    def __enter__(self):
        if not sys.stdin.isatty():
            raise RuntimeError("This script requires an interactive terminal.")

        self.fd = sys.stdin.fileno()
        self._attrs = termios.tcgetattr(self.fd)
        tty.setcbreak(self.fd)
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        termios.tcsetattr(self.fd, termios.TCSADRAIN, self._attrs)

    def read_key(self) -> str:
        key = sys.stdin.read(1)
        if key != "\x1b":
            return key

        # Drain any pending escape-sequence bytes so arrow keys do not leave
        # stray characters in the input stream.
        while select.select([sys.stdin], [], [], 0.0)[0]:
            sys.stdin.read(1)
        return key


def _motor_positions(robot) -> dict[str, float]:
    observation = robot.get_observation()
    return {motor: observation[f"{motor}.pos"] for motor in robot.bus.motors}


def _print_help(joint_names: list[str], selected_index: int, joint_step: float, gripper_step: float) -> None:
    print("\nNMini keyboard test")
    print("-------------------")
    print("1-7 : select joint")
    print("j   : move selected joint in the negative direction")
    print("k   : move selected joint in the positive direction")
    print("-   : decrease active step size")
    print("+/= : increase active step size")
    print("p   : print current joint positions")
    print("h/? : show this help")
    print("q   : quit")
    print(f"Selected joint : {joint_names[selected_index]}")
    print(f"Joint step     : {joint_step:.2f}")
    print(f"Gripper step   : {gripper_step:.2f}\n")


def _print_positions(positions: dict[str, float]) -> None:
    print("\nCurrent positions:")
    print(pformat({name: round(value, 2) for name, value in positions.items()}))


def _step_for_joint(joint_name: str, joint_step: float, gripper_step: float) -> float:
    if joint_name == "larm_finger_joint":
        return gripper_step
    return joint_step


@draccus.wrap()
def keyboard_test(cfg: NMiniKeyboardTestConfig) -> None:
    init_logging()
    robot = make_robot_from_config(cfg.robot)

    if cfg.robot.type != "nmini_follower":
        raise ValueError("This script only supports --robot.type=nmini_follower.")

    robot.connect()
    joint_names = list(robot.bus.motors)
    selected_index = 0
    joint_step = cfg.joint_step
    gripper_step = cfg.gripper_step

    try:
        positions = _motor_positions(robot)
        _print_help(joint_names, selected_index, joint_step, gripper_step)
        _print_positions(positions)

        with RawTerminalInput() as keyboard:
            while True:
                key = keyboard.read_key()

                if key in {"q", "\x03"}:
                    break

                if key in {"h", "?"}:
                    _print_help(joint_names, selected_index, joint_step, gripper_step)
                    continue

                if key == "p":
                    _print_positions(_motor_positions(robot))
                    continue

                if key in "1234567":
                    index = int(key) - 1
                    if index < len(joint_names):
                        selected_index = index
                        print(f"Selected joint: {joint_names[selected_index]}")
                    continue

                selected_joint = joint_names[selected_index]

                if key in {"-", "_"}:
                    if selected_joint == "larm_finger_joint":
                        gripper_step = max(gripper_step / 2.0, 0.5)
                    else:
                        joint_step = max(joint_step / 2.0, 0.5)
                    print(
                        f"Step sizes updated: joint_step={joint_step:.2f}, gripper_step={gripper_step:.2f}"
                    )
                    continue

                if key in {"=", "+"}:
                    if selected_joint == "larm_finger_joint":
                        gripper_step = min(gripper_step * 2.0, 50.0)
                    else:
                        joint_step = min(joint_step * 2.0, 45.0)
                    print(
                        f"Step sizes updated: joint_step={joint_step:.2f}, gripper_step={gripper_step:.2f}"
                    )
                    continue

                if key not in {"j", "k"}:
                    continue

                positions = _motor_positions(robot)
                delta = _step_for_joint(selected_joint, joint_step, gripper_step)
                if key == "j":
                    delta *= -1.0

                current_value = positions[selected_joint]
                target_value = current_value + delta
                action_key = f"{selected_joint}.pos"
                sent_action = robot.send_action({action_key: target_value})
                sent_value = sent_action[action_key]
                print(f"{selected_joint}: {current_value:.2f} -> {sent_value:.2f}")

    finally:
        robot.disconnect()


def main() -> None:
    register_third_party_plugins()
    keyboard_test()


if __name__ == "__main__":
    main()
