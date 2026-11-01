#!/usr/bin/env python

import importlib.util
import sys
from pathlib import Path

import numpy as np


def _load_module():
    module_path = Path(__file__).resolve().parents[2] / "examples/keyboard_to_nmini_gym/teleoperate.py"
    spec = importlib.util.spec_from_file_location("keyboard_to_nmini_gym_teleoperate", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module from {module_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_key_to_env_action_maps_xyz_and_gripper_commands():
    module = _load_module()

    np.testing.assert_allclose(module._key_to_env_action("a", 1.0), np.array([-1.0, 0.0, 0.0, 1.0], dtype=np.float32))
    np.testing.assert_allclose(module._key_to_env_action("d", 1.0), np.array([1.0, 0.0, 0.0, 1.0], dtype=np.float32))
    np.testing.assert_allclose(module._key_to_env_action("w", 1.0), np.array([0.0, -1.0, 0.0, 1.0], dtype=np.float32))
    np.testing.assert_allclose(module._key_to_env_action("s", 1.0), np.array([0.0, 1.0, 0.0, 1.0], dtype=np.float32))
    np.testing.assert_allclose(module._key_to_env_action("r", 1.0), np.array([0.0, 0.0, 1.0, 1.0], dtype=np.float32))
    np.testing.assert_allclose(module._key_to_env_action("f", 1.0), np.array([0.0, 0.0, -1.0, 1.0], dtype=np.float32))
    np.testing.assert_allclose(module._key_to_env_action("c", 1.0), np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32))
    np.testing.assert_allclose(module._key_to_env_action("o", 1.0), np.array([0.0, 0.0, 0.0, 2.0], dtype=np.float32))
    np.testing.assert_allclose(module._key_to_env_action(" ", 1.0), np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32))
    assert module._key_to_env_action("z", 1.0) is None


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
