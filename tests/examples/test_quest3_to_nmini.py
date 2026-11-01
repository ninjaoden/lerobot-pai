#!/usr/bin/env python

import importlib.util
import sys
from pathlib import Path

import numpy as np

from lerobot.processor.converters import create_transition
from lerobot.processor.pipeline import TransitionKey
from lerobot.utils.rotation import Rotation


def _load_module():
    module_path = Path(__file__).resolve().parents[2] / "examples/quest3_to_nmini/teleoperate.py"
    spec = importlib.util.spec_from_file_location("quest3_to_nmini_teleoperate", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module from {module_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_button_gripper_mapping_stops_without_face_button():
    module = _load_module()
    step = module._MapQuest3ActionToNMiniAction(
        position_axis_mapping=("x", "y", "z"),
        rotation_axis_mapping=("x", "y", "z"),
        position_scale=1.0,
    )

    transition = create_transition(
        action={
            "quest3.enabled": False,
            "quest3.pos": np.zeros(3, dtype=float),
            "quest3.rot": Rotation.from_rotvec(np.zeros(3, dtype=float)),
            "quest3.raw_inputs": {"fresh": True, "buttons": {"x": True}},
        }
    )
    result = step(transition)
    assert result[TransitionKey.ACTION]["gripper_vel"] == -1.0

    transition = create_transition(
        action={
            "quest3.enabled": False,
            "quest3.pos": np.zeros(3, dtype=float),
            "quest3.rot": Rotation.from_rotvec(np.zeros(3, dtype=float)),
            "quest3.raw_inputs": {"fresh": True, "buttons": {}},
        }
    )
    result = step(transition)
    assert result[TransitionKey.ACTION]["gripper_vel"] == 0.0

    transition = create_transition(
        action={
            "quest3.enabled": False,
            "quest3.pos": np.zeros(3, dtype=float),
            "quest3.rot": Rotation.from_rotvec(np.zeros(3, dtype=float)),
            "quest3.raw_inputs": {"fresh": True, "buttons": {"b": True}},
        }
    )
    result = step(transition)
    assert result[TransitionKey.ACTION]["gripper_vel"] == 1.0
