#!/usr/bin/env python

"""
Teleoperate a physical NMini arm from the keyboard.

Example:
```shell
python examples/keyboard_to_nmini/teleoperate.py \
  --robot.port=/dev/ttyACM0 \
  --robot.id=nmini \
  --urdf_path=/path/to/nmini.urdf \
  --target_frame_name=gripper_frame_link
```

This example mirrors the keyboard -> gym-hil NXMini workflow while sending the
commands to a real NMini through the existing inverse-kinematics path.
"""

import select
import sys
import termios
import tty
from dataclasses import dataclass, field
from pprint import pformat

import draccus
import numpy as np

from lerobot.model.kinematics import RobotKinematics
from lerobot.processor import RobotAction, RobotObservation, RobotProcessorPipeline
from lerobot.processor.converters import robot_action_observation_to_transition, transition_to_robot_action
from lerobot.robots.nmini_follower import NMiniFollower, NMiniFollowerConfig
from lerobot.robots.nmini_follower.robot_kinematic_processor import (
    NMiniGripperVelocityToJoint,
    NMiniInverseKinematicsEEToJoints,
)
from lerobot.robots.so_follower.robot_kinematic_processor import EEBoundsAndSafety, EEReferenceAndDelta
from lerobot.utils.robot_utils import precise_sleep


@dataclass
class _KeyboardToNMiniEEBoundsAndSafety(EEBoundsAndSafety):
    raise_on_ee_jump: bool = False

    def action(self, action: RobotAction) -> RobotAction:
        x = action["ee.x"]
        y = action["ee.y"]
        z = action["ee.z"]
        wx = action["ee.wx"]
        wy = action["ee.wy"]
        wz = action["ee.wz"]

        if None in (x, y, z, wx, wy, wz):
            raise ValueError(
                "Missing required end-effector pose components: x, y, z, wx, wy, wz must all be present in action"
            )

        pos = np.array([x, y, z], dtype=float)
        twist = np.array([wx, wy, wz], dtype=float)
        pos = np.clip(pos, self.end_effector_bounds["min"], self.end_effector_bounds["max"])

        if self._last_pos is not None:
            dpos = pos - self._last_pos
            distance = float(np.linalg.norm(dpos))
            if distance > self.max_ee_step_m and distance > 0:
                pos = self._last_pos + dpos * (self.max_ee_step_m / distance)
                if self.raise_on_ee_jump:
                    raise ValueError(f"EE jump {distance:.3f}m > {self.max_ee_step_m}m")

        self._last_pos = pos
        action["ee.x"] = float(pos[0])
        action["ee.y"] = float(pos[1])
        action["ee.z"] = float(pos[2])
        action["ee.wx"] = float(twist[0])
        action["ee.wy"] = float(twist[1])
        action["ee.wz"] = float(twist[2])
        return action


@dataclass
class KeyboardToNMiniConfig:
    robot: NMiniFollowerConfig
    urdf_path: str = ""
    target_frame_name: str = "gripper"
    position_step_m: dict[str, float] = field(
        default_factory=lambda: {"x": 0.01, "y": 0.01, "z": 0.01}
    )
    max_ee_step_m: float = 0.05
    end_effector_bounds: dict[str, list[float]] = field(
        default_factory=lambda: {"min": [-1.0, -1.0, -1.0], "max": [1.0, 1.0, 1.0]}
    )
    gripper_joint_name: str = "larm_finger_joint"
    gripper_open_velocity: float = 1.0
    gripper_close_velocity: float = -1.0
    gripper_speed_factor: float = 20.0
    startup_joint_positions: dict[str, float] = field(
        default_factory=lambda: {
            "larm_joint0": 0.0,
            "larm_joint1": 0.0,
            "larm_joint2": 0.0,
            "larm_joint3": 0.0,
            "larm_joint4": 0.0,
            "larm_joint5": 0.0,
        }
    )
    startup_gripper_position: float | None = None
    startup_move_duration_s: float = 2.0
    startup_settle_time_s: float = 0.5
    startup_move_fps: int = 30
    raise_on_ee_jump: bool = False


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

        while select.select([sys.stdin], [], [], 0.0)[0]:
            sys.stdin.read(1)
        return key


def _move_robot_to_startup_pose(robot: NMiniFollower, cfg: KeyboardToNMiniConfig) -> None:
    startup_targets = dict(cfg.startup_joint_positions)
    if cfg.startup_gripper_position is not None:
        startup_targets[cfg.gripper_joint_name] = float(cfg.startup_gripper_position)

    if not startup_targets:
        return

    unknown_joints = sorted(set(startup_targets) - set(robot.bus.motors))
    if unknown_joints:
        raise ValueError(f"Unknown startup joints for NMini: {unknown_joints}")

    observation = robot.get_observation()
    current_positions = {joint: float(observation[f"{joint}.pos"]) for joint in startup_targets}
    target_positions = {joint: float(value) for joint, value in startup_targets.items()}

    move_steps = max(int(cfg.startup_move_duration_s * cfg.startup_move_fps), 1)
    move_dt = cfg.startup_move_duration_s / move_steps if move_steps > 0 else 0.0

    print("Moving NMini to startup pose before keyboard teleoperation...")
    print({joint: round(value, 2) for joint, value in target_positions.items()})

    for step_idx in range(1, move_steps + 1):
        alpha = step_idx / move_steps
        action = {
            f"{joint}.pos": current_positions[joint] + alpha * (target_positions[joint] - current_positions[joint])
            for joint in target_positions
        }
        robot.send_action(action)
        if move_dt > 0:
            precise_sleep(move_dt)

    if cfg.startup_settle_time_s > 0:
        precise_sleep(cfg.startup_settle_time_s)


def _key_to_robot_delta_action(key: str, cfg: KeyboardToNMiniConfig) -> RobotAction | None:
    action: RobotAction = {
        "enabled": False,
        "target_x": 0.0,
        "target_y": 0.0,
        "target_z": 0.0,
        "target_wx": 0.0,
        "target_wy": 0.0,
        "target_wz": 0.0,
        "gripper_vel": 0.0,
    }

    if key == "a":
        action["enabled"] = True
        action["target_x"] = -float(cfg.position_step_m["x"])
    elif key == "d":
        action["enabled"] = True
        action["target_x"] = float(cfg.position_step_m["x"])
    elif key == "w":
        action["enabled"] = True
        action["target_y"] = -float(cfg.position_step_m["y"])
    elif key == "s":
        action["enabled"] = True
        action["target_y"] = float(cfg.position_step_m["y"])
    elif key == "r":
        action["enabled"] = True
        action["target_z"] = float(cfg.position_step_m["z"])
    elif key == "f":
        action["enabled"] = True
        action["target_z"] = -float(cfg.position_step_m["z"])
    elif key == "c":
        action["gripper_vel"] = float(cfg.gripper_close_velocity)
    elif key == "o":
        action["gripper_vel"] = float(cfg.gripper_open_velocity)
    elif key == " ":
        pass
    else:
        return None

    return action


def _joint_vector(observation: RobotObservation, joint_names: list[str]) -> np.ndarray:
    return np.array([float(observation[f"{joint}.pos"]) for joint in joint_names], dtype=float)


def _tcp_pose_from_observation(
    observation: RobotObservation, kinematics: RobotKinematics, joint_names: list[str]
) -> np.ndarray:
    q = _joint_vector(observation, joint_names)
    return kinematics.forward_kinematics(q)[:3, 3]


def _print_help(cfg: KeyboardToNMiniConfig) -> None:
    print("\nKeyboard -> NMini")
    print("-----------------")
    print("a / d : step X- / X+")
    print("w / s : step Y- / Y+")
    print("r / f : step Z+ / Z-")
    print("c / o : close / open gripper")
    print("space : hold current target")
    print("p     : print current joint positions and TCP pose")
    print("x     : move back to startup pose")
    print("h / ? : show this help")
    print("q     : quit")
    print(
        "Position step [m] : "
        f"x={cfg.position_step_m['x']:.4f}, "
        f"y={cfg.position_step_m['y']:.4f}, "
        f"z={cfg.position_step_m['z']:.4f}\n"
    )


def _print_robot_state(
    robot: NMiniFollower, observation: RobotObservation, kinematics: RobotKinematics, joint_names: list[str]
) -> None:
    tcp_pose = _tcp_pose_from_observation(observation, kinematics, joint_names)
    positions = {motor: float(observation[f"{motor}.pos"]) for motor in robot.bus.motors}

    print(f"TCP pose: [{tcp_pose[0]:+.3f}, {tcp_pose[1]:+.3f}, {tcp_pose[2]:+.3f}]")
    print("Joint positions:")
    print(pformat({name: round(value, 2) for name, value in positions.items()}))


@draccus.wrap()
def main(cfg: KeyboardToNMiniConfig) -> None:
    if not cfg.urdf_path:
        raise ValueError("Please provide --urdf_path pointing to the NMini URDF.")

    robot = NMiniFollower(cfg.robot)
    arm_joint_names = [name for name in robot.bus.motors if name != cfg.gripper_joint_name]
    kinematics_solver = RobotKinematics(
        urdf_path=cfg.urdf_path,
        target_frame_name=cfg.target_frame_name,
        joint_names=arm_joint_names,
    )

    keyboard_to_nmini = RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction](
        steps=[
            EEReferenceAndDelta(
                kinematics=kinematics_solver,
                end_effector_step_sizes={"x": 1.0, "y": 1.0, "z": 1.0},
                motor_names=arm_joint_names,
                use_latched_reference=False,
            ),
            _KeyboardToNMiniEEBoundsAndSafety(
                end_effector_bounds=cfg.end_effector_bounds,
                max_ee_step_m=cfg.max_ee_step_m,
                raise_on_ee_jump=cfg.raise_on_ee_jump,
            ),
            NMiniGripperVelocityToJoint(
                speed_factor=cfg.gripper_speed_factor,
                gripper_joint_name=cfg.gripper_joint_name,
            ),
            NMiniInverseKinematicsEEToJoints(
                kinematics=kinematics_solver,
                motor_names=arm_joint_names,
                gripper_joint_name=cfg.gripper_joint_name,
                initial_guess_current_joints=True,
            ),
        ],
        to_transition=robot_action_observation_to_transition,
        to_output=transition_to_robot_action,
    )

    robot.connect()

    try:
        _move_robot_to_startup_pose(robot, cfg)
        initial_observation = robot.get_observation()
        _print_help(cfg)
        _print_robot_state(robot, initial_observation, kinematics_solver, arm_joint_names)

        with RawTerminalInput() as keyboard:
            while True:
                key = keyboard.read_key()

                if key in {"q", "\x03"}:
                    break

                if key in {"h", "?"}:
                    _print_help(cfg)
                    continue

                if key == "p":
                    _print_robot_state(robot, robot.get_observation(), kinematics_solver, arm_joint_names)
                    continue

                if key == "x":
                    _move_robot_to_startup_pose(robot, cfg)
                    _print_robot_state(robot, robot.get_observation(), kinematics_solver, arm_joint_names)
                    continue

                keyboard_action = _key_to_robot_delta_action(key, cfg)
                if keyboard_action is None:
                    continue

                robot_observation = robot.get_observation()
                current_tcp_pose = _tcp_pose_from_observation(robot_observation, kinematics_solver, arm_joint_names)
                robot_action = keyboard_to_nmini((keyboard_action, robot_observation))
                robot.send_action(robot_action)

                print(
                    f"key={key!r} delta=[{keyboard_action['target_x']:+.3f}, "
                    f"{keyboard_action['target_y']:+.3f}, {keyboard_action['target_z']:+.3f}] "
                    f"gripper_vel={keyboard_action['gripper_vel']:+.1f} "
                    f"tcp=[{current_tcp_pose[0]:+.3f}, {current_tcp_pose[1]:+.3f}, {current_tcp_pose[2]:+.3f}]"
                )
    finally:
        robot.disconnect()


if __name__ == "__main__":
    main()
