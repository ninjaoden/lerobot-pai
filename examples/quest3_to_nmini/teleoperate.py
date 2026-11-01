#!/usr/bin/env python

"""
Teleoperate an NMini arm from a Quest 3 controller.

Example:
```shell
python examples/quest3_to_nmini/teleoperate.py \
  --robot.port=/dev/ttyACM0 \
  --robot.id=nmini \
  --urdf_path=/path/to/nmini.urdf \
  --target_frame_name=gripper_frame_link
```
"""

import time
from dataclasses import dataclass, field

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
from lerobot.robots.so_follower.robot_kinematic_processor import (
    EEBoundsAndSafety,
    EEReferenceAndDelta,
)
from lerobot.teleoperators.quest3 import MapQuest3ActionToRobotAction, Quest3, Quest3Config, Quest3Hand
from lerobot.utils.robot_utils import precise_sleep


@dataclass
class _Quest3ToNMiniEEBoundsAndSafety(EEBoundsAndSafety):
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


def _button_pressed(raw_inputs: dict, button_names: tuple[str, ...]) -> bool:
    buttons = raw_inputs.get("buttons", {})
    if not isinstance(buttons, dict):
        return False
    return any(bool(buttons.get(name, False)) for name in button_names)


def _gripper_button_labels(hand: Quest3Hand) -> tuple[str, str]:
    if hand is Quest3Hand.LEFT:
        return ("X", "Y")
    return ("A", "B")


@dataclass
class _MapQuest3ActionToNMiniAction(MapQuest3ActionToRobotAction):
    gripper_close_buttons: tuple[str, ...] = ("x", "a")
    gripper_open_buttons: tuple[str, ...] = ("y", "b")

    def action(self, action: RobotAction) -> RobotAction:
        raw_inputs = dict(action["quest3.raw_inputs"])
        action = super().action(action)

        fresh = bool(raw_inputs.get("fresh", False))
        close_pressed = _button_pressed(raw_inputs, self.gripper_close_buttons)
        open_pressed = _button_pressed(raw_inputs, self.gripper_open_buttons)

        if not fresh or close_pressed == open_pressed:
            action["gripper_vel"] = 0.0
        elif close_pressed:
            action["gripper_vel"] = float(self.gripper_close_velocity)
        else:
            action["gripper_vel"] = float(self.gripper_open_velocity)

        return action


@dataclass
class Quest3ToNMiniConfig:
    robot: NMiniFollowerConfig
    teleop: Quest3Config = field(default_factory=lambda: Quest3Config(hand=Quest3Hand.LEFT))
    urdf_path: str = ""
    target_frame_name: str = "gripper"
    fps: int = 30
    position_axis_mapping: tuple[str, str, str] = ("z", "-x", "-y")
    rotation_axis_mapping: tuple[str, str, str] = ("z", "-x", "-y")
    position_scale: float | tuple[float, float, float] = 0.2
    use_orientation: bool = True
    max_ee_step_m: float = 0.05
    end_effector_bounds: dict[str, list[float]] = field(
        default_factory=lambda: {"min": [-1.0, -1.0, -1.0], "max": [1.0, 1.0, 1.0]}
    )
    gripper_joint_name: str = "larm_finger_joint"
    gripper_open_velocity: float = 1.0
    gripper_close_velocity: float = -1.0
    gripper_open_buttons: tuple[str, ...] = ("y", "b")
    gripper_close_buttons: tuple[str, ...] = ("x", "a")
    gripper_speed_factor: float = 20.0
    status_print_interval_s: float = 1.0
    warn_if_no_fresh_data_after_s: float = 3.0
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


def _move_robot_to_startup_pose(robot: NMiniFollower, cfg: Quest3ToNMiniConfig) -> None:
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

    print("Moving NMini to startup pose before teleoperation...")
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


@draccus.wrap()
def main(cfg: Quest3ToNMiniConfig) -> None:
    if not cfg.urdf_path:
        raise ValueError("Please provide --urdf_path pointing to the NMini URDF.")

    robot = NMiniFollower(cfg.robot)
    teleop = Quest3(cfg.teleop)

    arm_joint_names = [name for name in robot.bus.motors if name != cfg.gripper_joint_name]
    kinematics_solver = RobotKinematics(
        urdf_path=cfg.urdf_path,
        target_frame_name=cfg.target_frame_name,
        joint_names=arm_joint_names,
    )

    quest3_to_nmini = RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction](
        steps=[
            _MapQuest3ActionToNMiniAction(
                position_axis_mapping=cfg.position_axis_mapping,
                rotation_axis_mapping=cfg.rotation_axis_mapping,
                position_scale=cfg.position_scale,
                use_orientation=cfg.use_orientation,
                gripper_open_velocity=cfg.gripper_open_velocity,
                gripper_close_velocity=cfg.gripper_close_velocity,
                gripper_open_buttons=cfg.gripper_open_buttons,
                gripper_close_buttons=cfg.gripper_close_buttons,
            ),
            EEReferenceAndDelta(
                kinematics=kinematics_solver,
                end_effector_step_sizes={"x": 1.0, "y": 1.0, "z": 1.0},
                motor_names=arm_joint_names,
                use_latched_reference=True,
            ),
            _Quest3ToNMiniEEBoundsAndSafety(
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
    _move_robot_to_startup_pose(robot, cfg)
    teleop.connect()

    alternate_hand = Quest3Hand.RIGHT.value if cfg.teleop.hand is Quest3Hand.LEFT else Quest3Hand.LEFT.value
    close_label, open_label = _gripper_button_labels(cfg.teleop.hand)
    print("Starting Quest3 -> NMini teleoperation loop...")
    print(
        f"Using Quest3 {cfg.teleop.hand.value} controller. "
        f"Pass --teleop.hand={alternate_hand} to switch controllers."
    )
    print(
        "Hold the Quest3 grip button to enable arm motion. "
        f"Use {close_label} to close and {open_label} to open the gripper. "
        "Release the button to hold the current gripper position."
    )

    last_status_time = 0.0
    loop_start_time = time.monotonic()

    try:
        while True:
            t0 = time.perf_counter()

            robot_obs = robot.get_observation()
            quest3_action = teleop.get_action()
            robot_action = quest3_to_nmini((quest3_action, robot_obs))
            robot.send_action(robot_action)

            now = time.monotonic()
            if cfg.status_print_interval_s > 0 and now - last_status_time >= cfg.status_print_interval_s:
                raw_inputs = dict(quest3_action["quest3.raw_inputs"])
                pos = np.asarray(quest3_action["quest3.pos"], dtype=float)
                enabled = bool(quest3_action["quest3.enabled"])
                fresh = bool(raw_inputs.get("fresh", False))
                grip_pressed = bool(raw_inputs.get("grip_pressed", False))
                close_pressed = _button_pressed(raw_inputs, cfg.gripper_close_buttons)
                open_pressed = _button_pressed(raw_inputs, cfg.gripper_open_buttons)
                source = raw_inputs.get("source") or "-"

                max_joint_delta = max(
                    abs(float(robot_action[f"{joint}.pos"]) - float(robot_obs[f"{joint}.pos"])) for joint in robot.bus.motors
                )

                print(
                    "[Quest3] "
                    f"hand={cfg.teleop.hand.value} fresh={fresh} enabled={enabled} "
                    f"grip={grip_pressed} close_btn={close_pressed} open_btn={open_pressed} source={source} "
                    f"pos=[{pos[0]:+.3f}, {pos[1]:+.3f}, {pos[2]:+.3f}] "
                    f"max_joint_delta_deg={max_joint_delta:.2f}"
                )

                if not fresh and now - loop_start_time >= cfg.warn_if_no_fresh_data_after_s:
                    connection_url = teleop.runtime.connection_url or "<unknown>"
                    print(
                        "[Quest3] No fresh controller data received yet. "
                        f"Open {connection_url} in the headset browser and move the {cfg.teleop.hand.value} controller."
                    )
                elif fresh and not enabled:
                    print(
                        "[Quest3] Controller data is arriving, but teleoperation is disabled. "
                        f"Hold the {cfg.teleop.hand.value} grip button to latch the reference pose."
                    )

                last_status_time = now

            precise_sleep(max(1.0 / cfg.fps - (time.perf_counter() - t0), 0.0))
    finally:
        teleop.disconnect()
        robot.disconnect()


if __name__ == "__main__":
    main()
