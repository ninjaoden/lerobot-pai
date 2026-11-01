#!/usr/bin/env python

import importlib.util
import sys
from pathlib import Path

import numpy as np


def _load_module():
    module_path = Path(__file__).resolve().parents[2] / "examples/quest3_to_nmini_gym/teleoperate.py"
    spec = importlib.util.spec_from_file_location("quest3_to_nmini_gym_teleoperate", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module from {module_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _quest3_action(*, enabled: bool, pos: list[float], buttons: dict[str, bool] | None = None, fresh: bool = True):
    from lerobot.utils.rotation import Rotation

    return {
        "quest3.enabled": enabled,
        "quest3.pos": np.asarray(pos, dtype=float),
        "quest3.rot": Rotation.from_rotvec(np.zeros(3, dtype=float)),
        "quest3.raw_inputs": {
            "fresh": fresh,
            "buttons": {} if buttons is None else dict(buttons),
        },
    }


def test_adapter_saturates_translation_until_absolute_target_is_reached():
    module = _load_module()
    adapter = module._Quest3ToGymNMiniActionAdapter(
        position_axis_mapping=("x", "y", "z"),
        rotation_axis_mapping=("x", "y", "z"),
        position_scale=0.2,
        ee_action_step_size={"x": 0.01, "y": 0.01, "z": 0.01},
        end_effector_bounds={"min": [0.0, -1.0, -1.0], "max": [1.0, 1.0, 1.0]},
        max_ee_step_m=1.0,
    )
    current_tcp_pose = np.array([0.25, 0.0, 0.2], dtype=np.float32)
    adapter.reset(current_tcp_pose)

    first = adapter.to_env_action(
        _quest3_action(enabled=True, pos=[0.0, 0.0, 0.0]),
        current_tcp_pose=current_tcp_pose,
    )
    np.testing.assert_allclose(first, np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32))

    second = adapter.to_env_action(
        _quest3_action(enabled=True, pos=[0.1, 0.0, 0.0]),
        current_tcp_pose=current_tcp_pose,
    )
    np.testing.assert_allclose(second, np.array([1.0, 0.0, 0.0, 1.0], dtype=np.float32))
    np.testing.assert_allclose(adapter.commanded_target_tcp_pose, np.array([0.26, 0.0, 0.2], dtype=np.float32))

    third = adapter.to_env_action(
        _quest3_action(enabled=True, pos=[0.1, 0.0, 0.0]),
        current_tcp_pose=current_tcp_pose,
    )
    np.testing.assert_allclose(third, np.array([1.0, 0.0, 0.0, 1.0], dtype=np.float32))
    np.testing.assert_allclose(adapter.commanded_target_tcp_pose, np.array([0.27, 0.0, 0.2], dtype=np.float32))

    fourth = adapter.to_env_action(
        _quest3_action(enabled=True, pos=[0.1, 0.0, 0.0]),
        current_tcp_pose=current_tcp_pose,
    )
    np.testing.assert_allclose(fourth, np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32))


def test_adapter_holds_position_when_disabled_and_maps_face_buttons_to_gripper():
    module = _load_module()
    adapter = module._Quest3ToGymNMiniActionAdapter(
        position_axis_mapping=("x", "y", "z"),
        rotation_axis_mapping=("x", "y", "z"),
        position_scale=1.0,
    )
    current_tcp_pose = np.array([0.25, 0.0, 0.2], dtype=np.float32)
    adapter.reset(current_tcp_pose)

    hold = adapter.to_env_action(
        _quest3_action(enabled=False, pos=[0.3, 0.2, 0.1], buttons={}),
        current_tcp_pose=current_tcp_pose,
    )
    np.testing.assert_allclose(hold, np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32))

    close = adapter.to_env_action(
        _quest3_action(enabled=False, pos=[0.0, 0.0, 0.0], buttons={"x": True}),
        current_tcp_pose=current_tcp_pose,
    )
    np.testing.assert_allclose(close, np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32))

    open_action = adapter.to_env_action(
        _quest3_action(enabled=False, pos=[0.0, 0.0, 0.0], buttons={"b": True}),
        current_tcp_pose=current_tcp_pose,
    )
    np.testing.assert_allclose(open_action, np.array([0.0, 0.0, 0.0, 2.0], dtype=np.float32))


def test_extract_tcp_pose_supports_flat_agent_pos_vector():
    module = _load_module()
    observation = {
        "agent_pos": np.array(
            [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, -0.1, -0.2, -0.3, -0.4, -0.5, -0.6, 0.25, 0.05, 0.3],
            dtype=np.float32,
        )
    }

    tcp_pose = module._extract_tcp_pose(observation)

    np.testing.assert_allclose(tcp_pose, np.array([0.25, 0.05, 0.3], dtype=np.float32))


def test_extract_tcp_pose_supports_nested_tcp_pose():
    module = _load_module()
    observation = {"agent_pos": {"tcp_pose": np.array([0.21, -0.02, 0.18], dtype=np.float32)}}

    tcp_pose = module._extract_tcp_pose(observation)

    np.testing.assert_allclose(tcp_pose, np.array([0.21, -0.02, 0.18], dtype=np.float32))
