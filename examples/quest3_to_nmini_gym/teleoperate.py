#!/usr/bin/env python

"""
Teleoperate the gym-hil NXMini simulation from a Quest 3 controller.

Example:
```shell
python examples/quest3_to_nmini_gym/teleoperate.py \
  --gym_hil_repo_path=/path/to/gym-hil
```

This example mirrors the existing Quest3 -> physical NMini workflow for
translational motion and gripper buttons. The current gym-hil NXMini EE action
space does not expose wrist orientation, so controller rotation is ignored.
"""

import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import draccus
import numpy as np

from lerobot.processor import RobotAction
from lerobot.processor.converters import create_transition
from lerobot.processor.pipeline import TransitionKey
from lerobot.teleoperators.quest3 import MapQuest3ActionToRobotAction, Quest3, Quest3Config, Quest3Hand
from lerobot.utils.robot_utils import precise_sleep


def _button_pressed(raw_inputs: dict[str, Any], button_names: tuple[str, ...]) -> bool:
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
class _Quest3ToGymNMiniActionAdapter:
    position_axis_mapping: tuple[str, str, str] = ("z", "-x", "-y")
    rotation_axis_mapping: tuple[str, str, str] = ("z", "-x", "-y")
    position_scale: float | tuple[float, float, float] = 0.2
    use_orientation: bool = True
    ee_action_step_size: dict[str, float] = field(
        default_factory=lambda: {"x": 0.01, "y": 0.01, "z": 0.01}
    )
    end_effector_bounds: dict[str, list[float]] = field(
        default_factory=lambda: {"min": [0.2, -0.4, 0.0], "max": [0.3, 0.4, 0.7]}
    )
    max_ee_step_m: float = 0.05
    gripper_open_velocity: float = 1.0
    gripper_close_velocity: float = -1.0
    gripper_open_buttons: tuple[str, ...] = ("y", "b")
    gripper_close_buttons: tuple[str, ...] = ("x", "a")

    def __post_init__(self) -> None:
        self._quest3_mapper = _MapQuest3ActionToNMiniAction(
            position_axis_mapping=self.position_axis_mapping,
            rotation_axis_mapping=self.rotation_axis_mapping,
            position_scale=self.position_scale,
            use_orientation=self.use_orientation,
            gripper_open_velocity=self.gripper_open_velocity,
            gripper_close_velocity=self.gripper_close_velocity,
            gripper_open_buttons=self.gripper_open_buttons,
            gripper_close_buttons=self.gripper_close_buttons,
        )
        self._ee_step_size_xyz = np.array(
            [
                self.ee_action_step_size["x"],
                self.ee_action_step_size["y"],
                self.ee_action_step_size["z"],
            ],
            dtype=np.float32,
        )
        self._reference_tcp_pose: np.ndarray | None = None
        self._last_target_tcp_pose: np.ndarray | None = None

    @property
    def commanded_target_tcp_pose(self) -> np.ndarray | None:
        if self._last_target_tcp_pose is None:
            return None
        return self._last_target_tcp_pose.copy()

    def reset(self, current_tcp_pose: np.ndarray) -> None:
        tcp_pose = self._validate_tcp_pose(current_tcp_pose)
        self._reference_tcp_pose = None
        self._last_target_tcp_pose = tcp_pose.copy()

    def to_env_action(self, quest3_action: RobotAction, current_tcp_pose: np.ndarray) -> np.ndarray:
        tcp_pose = self._validate_tcp_pose(current_tcp_pose)
        if self._last_target_tcp_pose is None:
            self._last_target_tcp_pose = tcp_pose.copy()

        mapped_action = self._map_quest3_action(quest3_action)
        env_action = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        enabled = bool(mapped_action["enabled"])

        if enabled:
            if self._reference_tcp_pose is None:
                self._reference_tcp_pose = tcp_pose.copy()

            desired_tcp_pose = self._reference_tcp_pose + np.array(
                [
                    mapped_action["target_x"],
                    mapped_action["target_y"],
                    mapped_action["target_z"],
                ],
                dtype=np.float32,
            )
            desired_tcp_pose = np.clip(
                desired_tcp_pose,
                np.asarray(self.end_effector_bounds["min"], dtype=np.float32),
                np.asarray(self.end_effector_bounds["max"], dtype=np.float32),
            )

            delta_to_target = desired_tcp_pose - self._last_target_tcp_pose
            distance = float(np.linalg.norm(delta_to_target))
            if self.max_ee_step_m > 0 and distance > self.max_ee_step_m and distance > 0.0:
                desired_tcp_pose = self._last_target_tcp_pose + delta_to_target * (self.max_ee_step_m / distance)

            desired_delta = desired_tcp_pose - self._last_target_tcp_pose
            normalized_delta = np.divide(
                desired_delta,
                self._ee_step_size_xyz,
                out=np.zeros(3, dtype=np.float32),
                where=self._ee_step_size_xyz > 0,
            )
            normalized_delta = np.clip(normalized_delta, -1.0, 1.0).astype(np.float32)
            self._last_target_tcp_pose = self._last_target_tcp_pose + normalized_delta * self._ee_step_size_xyz
            env_action[:3] = normalized_delta
        else:
            self._reference_tcp_pose = None

        env_action[3] = self._gripper_command_from_velocity(float(mapped_action["gripper_vel"]))
        return env_action

    def _map_quest3_action(self, quest3_action: RobotAction) -> RobotAction:
        transition = create_transition(action=quest3_action)
        mapped_transition = self._quest3_mapper(transition)
        return mapped_transition[TransitionKey.ACTION]

    @staticmethod
    def _gripper_command_from_velocity(gripper_vel: float) -> float:
        if gripper_vel < 0.0:
            return 0.0
        if gripper_vel > 0.0:
            return 2.0
        return 1.0

    @staticmethod
    def _validate_tcp_pose(tcp_pose: np.ndarray) -> np.ndarray:
        pose = np.asarray(tcp_pose, dtype=np.float32)
        if pose.shape != (3,):
            raise ValueError(f"Expected tcp_pose with shape (3,), got {pose.shape}.")
        return pose


@dataclass
class Quest3ToGymNMiniConfig:
    teleop: Quest3Config = field(default_factory=lambda: Quest3Config(hand=Quest3Hand.LEFT))
    gym_hil_repo_path: str = ""
    base_env_id: str = "gym_hil/NXMiniPickCubeBase-v0"
    render_mode: str = "human"
    image_obs: bool = True
    use_viewer: bool = True
    fps: int = 30
    max_episode_steps: int = 1000
    reset_delay_seconds: float = 1.0
    position_axis_mapping: tuple[str, str, str] = ("z", "-x", "-y")
    rotation_axis_mapping: tuple[str, str, str] = ("z", "-x", "-y")
    position_scale: float | tuple[float, float, float] = 0.2
    use_orientation: bool = True
    ee_action_step_size: dict[str, float] = field(
        default_factory=lambda: {"x": 0.01, "y": 0.01, "z": 0.01}
    )
    max_ee_step_m: float = 0.05
    end_effector_bounds: dict[str, list[float]] = field(
        default_factory=lambda: {"min": [0.2, -0.4, 0.0], "max": [0.3, 0.4, 0.7]}
    )
    gripper_open_velocity: float = 1.0
    gripper_close_velocity: float = -1.0
    gripper_open_buttons: tuple[str, ...] = ("y", "b")
    gripper_close_buttons: tuple[str, ...] = ("x", "a")
    status_print_interval_s: float = 1.0
    warn_if_no_fresh_data_after_s: float = 3.0


def _ensure_gym_hil_importable(repo_path: str) -> None:
    if not repo_path:
        return

    resolved_path = Path(repo_path).expanduser().resolve()
    if not resolved_path.exists():
        raise FileNotFoundError(f"gym-hil repo path does not exist: {resolved_path}")

    resolved_str = str(resolved_path)
    if resolved_str not in sys.path:
        sys.path.insert(0, resolved_str)


def _make_env(cfg: Quest3ToGymNMiniConfig):
    _ensure_gym_hil_importable(cfg.gym_hil_repo_path)

    try:
        from gymnasium.wrappers import TimeLimit
        from gym_hil.wrappers.factory import make_env
    except ImportError as exc:
        raise ImportError(
            "gym-hil is required. Install it in the environment or pass --gym_hil_repo_path=/path/to/gym-hil."
        ) from exc

    env = make_env(
        env_id=cfg.base_env_id,
        render_mode=cfg.render_mode,
        image_obs=cfg.image_obs,
        ee_step_size=cfg.ee_action_step_size,
        use_ee_action=True,
        use_viewer=cfg.use_viewer,
        use_gripper=True,
        use_inputs_control=False,
        reset_delay_seconds=cfg.reset_delay_seconds,
    )

    if cfg.max_episode_steps > 0:
        env = TimeLimit(env, max_episode_steps=cfg.max_episode_steps)

    return env


def _extract_tcp_pose(observation: dict[str, Any]) -> np.ndarray:
    agent_pos = observation.get("agent_pos")
    if isinstance(agent_pos, dict):
        if "tcp_pose" not in agent_pos:
            raise ValueError("Expected observation['agent_pos']['tcp_pose'] in gym-hil observation.")

        tcp_pose = np.asarray(agent_pos["tcp_pose"], dtype=np.float32)
        if tcp_pose.shape == (3,):
            return tcp_pose
        if tcp_pose.shape[0] >= 3:
            return tcp_pose[:3]
        raise ValueError(f"Expected observation tcp_pose with at least 3 values, got {tcp_pose.shape}.")

    if agent_pos is None:
        raise ValueError("Expected observation['agent_pos'] in gym-hil observation.")

    flat_agent_pos = np.asarray(agent_pos, dtype=np.float32).reshape(-1)
    if flat_agent_pos.shape[0] < 3:
        raise ValueError(f"Expected flat agent_pos with at least 3 values, got {flat_agent_pos.shape}.")

    # NXMiniPickCubeGymEnv exposes a flat robot_state = [qpos(6), qvel(6), ee_pos(3)].
    return flat_agent_pos[-3:]


@draccus.wrap()
def main(cfg: Quest3ToGymNMiniConfig) -> None:
    env = _make_env(cfg)
    teleop = Quest3(cfg.teleop)

    try:
        adapter = _Quest3ToGymNMiniActionAdapter(
            position_axis_mapping=cfg.position_axis_mapping,
            rotation_axis_mapping=cfg.rotation_axis_mapping,
            position_scale=cfg.position_scale,
            use_orientation=cfg.use_orientation,
            ee_action_step_size=cfg.ee_action_step_size,
            end_effector_bounds=cfg.end_effector_bounds,
            max_ee_step_m=cfg.max_ee_step_m,
            gripper_open_velocity=cfg.gripper_open_velocity,
            gripper_close_velocity=cfg.gripper_close_velocity,
            gripper_open_buttons=cfg.gripper_open_buttons,
            gripper_close_buttons=cfg.gripper_close_buttons,
        )

        observation, _ = env.reset()
        adapter.reset(_extract_tcp_pose(observation))
        teleop.connect()

        alternate_hand = Quest3Hand.RIGHT.value if cfg.teleop.hand is Quest3Hand.LEFT else Quest3Hand.LEFT.value
        close_label, open_label = _gripper_button_labels(cfg.teleop.hand)

        print("Starting Quest3 -> gym-hil NXMini teleoperation loop...")
        print(
            f"Using Quest3 {cfg.teleop.hand.value} controller. "
            f"Pass --teleop.hand={alternate_hand} to switch controllers."
        )
        print(
            "Hold the Quest3 grip button to enable arm motion. "
            f"Use {close_label} to close and {open_label} to open the gripper. "
            "Release the button to hold the current gripper position."
        )
        print("Quest3 controller orientation is ignored because gym-hil NXMini exposes translation-only EE actions.")

        last_status_time = 0.0
        loop_start_time = time.monotonic()

        while True:
            t0 = time.perf_counter()

            quest3_action = teleop.get_action()
            env_action = adapter.to_env_action(quest3_action, _extract_tcp_pose(observation))
            observation, reward, terminated, truncated, info = env.step(env_action)
            del reward

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
                tcp_pose = _extract_tcp_pose(observation)
                target_tcp_pose = adapter.commanded_target_tcp_pose

                target_text = (
                    f"[{target_tcp_pose[0]:+.3f}, {target_tcp_pose[1]:+.3f}, {target_tcp_pose[2]:+.3f}]"
                    if target_tcp_pose is not None
                    else "[-, -, -]"
                )

                print(
                    "[Quest3Gym] "
                    f"hand={cfg.teleop.hand.value} fresh={fresh} enabled={enabled} "
                    f"grip={grip_pressed} close_btn={close_pressed} open_btn={open_pressed} source={source} "
                    f"controller_pos=[{pos[0]:+.3f}, {pos[1]:+.3f}, {pos[2]:+.3f}] "
                    f"env_action=[{env_action[0]:+.2f}, {env_action[1]:+.2f}, {env_action[2]:+.2f}, {env_action[3]:+.1f}] "
                    f"tcp=[{tcp_pose[0]:+.3f}, {tcp_pose[1]:+.3f}, {tcp_pose[2]:+.3f}] "
                    f"target_tcp={target_text}"
                )

                if not fresh and now - loop_start_time >= cfg.warn_if_no_fresh_data_after_s:
                    connection_url = teleop.runtime.connection_url or "<unknown>"
                    print(
                        "[Quest3Gym] No fresh controller data received yet. "
                        f"Open {connection_url} in the headset browser and move the {cfg.teleop.hand.value} controller."
                    )
                elif fresh and not enabled:
                    print(
                        "[Quest3Gym] Controller data is arriving, but teleoperation is disabled. "
                        f"Hold the {cfg.teleop.hand.value} grip button to latch the reference pose."
                    )

                last_status_time = now

            if info.get("succeed", False):
                print("\nSuccess! Block has been picked up.")

            if terminated or truncated:
                print("Episode ended, resetting environment.")
                observation, _ = env.reset()
                adapter.reset(_extract_tcp_pose(observation))

            precise_sleep(max(1.0 / cfg.fps - (time.perf_counter() - t0), 0.0))
    finally:
        if teleop.is_connected:
            teleop.disconnect()
        env.close()


if __name__ == "__main__":
    main()
