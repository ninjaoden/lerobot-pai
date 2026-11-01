#!/usr/bin/env python

import importlib.util
import sys
from pathlib import Path


def _load_module():
    module_path = Path(__file__).resolve().parents[2] / "examples/keyboard_to_nmini/teleoperate.py"
    spec = importlib.util.spec_from_file_location("keyboard_to_nmini_teleoperate", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module from {module_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_key_to_robot_delta_action_maps_xyz_and_gripper_commands():
    module = _load_module()
    cfg = module.KeyboardToNMiniConfig(robot=module.NMiniFollowerConfig(port="/dev/null"))

    a = module._key_to_robot_delta_action("a", cfg)
    assert a == {
        "enabled": True,
        "target_x": -0.01,
        "target_y": 0.0,
        "target_z": 0.0,
        "target_wx": 0.0,
        "target_wy": 0.0,
        "target_wz": 0.0,
        "gripper_vel": 0.0,
    }

    d = module._key_to_robot_delta_action("d", cfg)
    assert d["target_x"] == 0.01
    assert d["enabled"] is True

    w = module._key_to_robot_delta_action("w", cfg)
    assert w["target_y"] == -0.01

    s = module._key_to_robot_delta_action("s", cfg)
    assert s["target_y"] == 0.01

    r = module._key_to_robot_delta_action("r", cfg)
    assert r["target_z"] == 0.01

    f = module._key_to_robot_delta_action("f", cfg)
    assert f["target_z"] == -0.01

    c = module._key_to_robot_delta_action("c", cfg)
    assert c["enabled"] is False
    assert c["gripper_vel"] == -1.0

    o = module._key_to_robot_delta_action("o", cfg)
    assert o["enabled"] is False
    assert o["gripper_vel"] == 1.0

    hold = module._key_to_robot_delta_action(" ", cfg)
    assert hold["enabled"] is False
    assert hold["gripper_vel"] == 0.0

    assert module._key_to_robot_delta_action("z", cfg) is None
