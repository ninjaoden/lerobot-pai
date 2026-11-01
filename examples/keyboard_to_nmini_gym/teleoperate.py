#!/usr/bin/env python

"""
Teleoperate the gym-hil NXMini simulation from the keyboard.

Example:
```shell
python examples/keyboard_to_nmini_gym/teleoperate.py \
  --gym_hil_repo_path=/path/to/gym-hil
```

This example bypasses Quest3 entirely and sends explicit Cartesian step
commands to the gym-hil NXMini environment. It is intended for isolating
direction and axis-mapping issues in the simulation path.
"""

import select
import sys
import termios
import tty
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import draccus
import numpy as np


@dataclass
class KeyboardToGymNMiniConfig:
    gym_hil_repo_path: str = ""
    base_env_id: str = "gym_hil/NXMiniPickCubeBase-v0"
    render_mode: str = "human"
    image_obs: bool = True
    use_viewer: bool = True
    max_episode_steps: int = 1000
    reset_delay_seconds: float = 1.0
    ee_action_step_size: dict[str, float] = field(
        default_factory=lambda: {"x": 0.01, "y": 0.01, "z": 0.01}
    )
    action_magnitude: float = 1.0


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


def _ensure_gym_hil_importable(repo_path: str) -> None:
    if not repo_path:
        return

    resolved_path = Path(repo_path).expanduser().resolve()
    if not resolved_path.exists():
        raise FileNotFoundError(f"gym-hil repo path does not exist: {resolved_path}")

    resolved_str = str(resolved_path)
    if resolved_str not in sys.path:
        sys.path.insert(0, resolved_str)


def _make_env(cfg: KeyboardToGymNMiniConfig):
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

    return flat_agent_pos[-3:]


def _key_to_env_action(key: str, action_magnitude: float) -> np.ndarray | None:
    action = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    magnitude = float(action_magnitude)

    if key == "a":
        action[0] = -magnitude
    elif key == "d":
        action[0] = magnitude
    elif key == "w":
        action[1] = -magnitude
    elif key == "s":
        action[1] = magnitude
    elif key == "r":
        action[2] = magnitude
    elif key == "f":
        action[2] = -magnitude
    elif key == "c":
        action[3] = 0.0
    elif key == "o":
        action[3] = 2.0
    elif key == " ":
        pass
    else:
        return None

    return action


def _print_help(cfg: KeyboardToGymNMiniConfig) -> None:
    print("\nKeyboard -> gym-hil NXMini")
    print("--------------------------")
    print("a / d : step X- / X+")
    print("w / s : step Y- / Y+")
    print("r / f : step Z+ / Z-")
    print("c / o : close / open gripper")
    print("space : hold current target")
    print("p     : print current tcp pose")
    print("x     : reset episode")
    print("h / ? : show this help")
    print("q     : quit")
    print(
        "EE step size [m] : "
        f"x={cfg.ee_action_step_size['x']:.4f}, "
        f"y={cfg.ee_action_step_size['y']:.4f}, "
        f"z={cfg.ee_action_step_size['z']:.4f}"
    )
    print(f"Action magnitude : {cfg.action_magnitude:.2f}\n")


def _format_vec(vec: np.ndarray) -> str:
    return f"[{vec[0]:+.3f}, {vec[1]:+.3f}, {vec[2]:+.3f}]"


@draccus.wrap()
def main(cfg: KeyboardToGymNMiniConfig) -> None:
    env = _make_env(cfg)

    try:
        observation, _ = env.reset()
        _print_help(cfg)
        print(f"Initial tcp pose: {_format_vec(_extract_tcp_pose(observation))}")

        with RawTerminalInput() as keyboard:
            while True:
                key = keyboard.read_key()

                if key in {"q", "\x03"}:
                    break

                if key in {"h", "?"}:
                    _print_help(cfg)
                    continue

                if key == "p":
                    print(f"Current tcp pose: {_format_vec(_extract_tcp_pose(observation))}")
                    continue

                if key == "x":
                    print("Resetting episode.")
                    observation, _ = env.reset()
                    print(f"tcp pose after reset: {_format_vec(_extract_tcp_pose(observation))}")
                    continue

                env_action = _key_to_env_action(key, cfg.action_magnitude)
                if env_action is None:
                    continue

                observation, reward, terminated, truncated, info = env.step(env_action)
                del reward

                tcp_pose = _extract_tcp_pose(observation)
                print(f"key={key!r} action={env_action.tolist()} tcp={_format_vec(tcp_pose)}")

                if info.get("succeed", False):
                    print("Success! Block has been picked up.")

                if terminated or truncated:
                    print("Episode ended, resetting environment.")
                    observation, _ = env.reset()
                    print(f"tcp pose after reset: {_format_vec(_extract_tcp_pose(observation))}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
