#!/usr/bin/env python

from types import SimpleNamespace

import numpy as np
import time

from lerobot.processor.converters import create_transition
from lerobot.processor.pipeline import TransitionKey
from lerobot.teleoperators.quest3 import MapQuest3ActionToRobotAction, Quest3, Quest3Config, Quest3Hand
from lerobot.teleoperators.utils import make_teleoperator_from_config
from lerobot.utils.rotation import Rotation


class FakeQuest3Runtime:
    def __init__(self, config: Quest3Config):
        self.config = config
        self._connected = False
        self.goal = None
        self.goal_timestamp = None
        self.connection_url = "https://127.0.0.1:8443"

    @property
    def is_connected(self) -> bool:
        return self._connected

    def connect(self) -> None:
        self._connected = True

    def disconnect(self) -> None:
        self._connected = False
        self.goal = None
        self.goal_timestamp = None

    def get_latest_goal(self, hand: str):
        assert hand == self.config.hand.value
        return self.goal, self.goal_timestamp


def _goal(position=None, rotation=None, **metadata):
    payload = {}
    if position is not None:
        payload["vr_position"] = list(position)
    if rotation is not None:
        quat = rotation.as_quat()
        payload["quaternion"] = {"x": quat[0], "y": quat[1], "z": quat[2], "w": quat[3]}
    payload.update(metadata)
    return SimpleNamespace(metadata=payload)


def test_make_teleoperator_from_config_returns_quest3():
    teleop = make_teleoperator_from_config(Quest3Config())
    assert isinstance(teleop, Quest3)


def test_quest3_get_action_tracks_relative_pose(monkeypatch):
    monkeypatch.setattr(Quest3, "runtime_cls", FakeQuest3Runtime)

    teleop = Quest3(Quest3Config())
    teleop.connect()

    runtime = teleop.runtime
    now = time.monotonic()
    runtime.goal = _goal(position=[0.2, -0.1, 0.4], rotation=Rotation.from_quat([0.0, 0.0, 0.0, 1.0]))
    runtime.goal_timestamp = now

    idle_action = teleop.get_action()
    assert idle_action["quest3.enabled"] is False
    assert np.allclose(idle_action["quest3.pos"], np.zeros(3))
    assert idle_action["quest3.raw_inputs"]["fresh"] is True

    runtime.goal = _goal(
        position=[0.2, -0.1, 0.4],
        rotation=Rotation.from_quat([0.0, 0.0, 0.0, 1.0]),
        gripActive=True,
        trigger=1.0,
        trigger_active=True,
    )
    runtime.goal_timestamp = now + 0.01

    latched_action = teleop.get_action()
    assert latched_action["quest3.enabled"] is True
    assert np.allclose(latched_action["quest3.pos"], np.zeros(3))
    assert latched_action["quest3.raw_inputs"]["trigger_pressed"] is True

    delta_rotation = Rotation.from_rotvec(np.array([0.0, 0.0, np.pi / 6.0]))
    runtime.goal = _goal(
        position=[0.35, -0.05, 0.25],
        rotation=delta_rotation,
        gripActive=True,
        trigger=1.0,
        trigger_active=True,
    )
    runtime.goal_timestamp = now + 0.02

    moved_action = teleop.get_action()
    assert moved_action["quest3.enabled"] is True
    assert np.allclose(moved_action["quest3.pos"], np.array([0.15, 0.05, -0.15]))
    assert np.allclose(moved_action["quest3.rot"].as_rotvec(), delta_rotation.as_rotvec())

    runtime.goal = _goal(gripActive=False, trigger=0.0, trigger_active=False)
    runtime.goal_timestamp = now + 0.03

    released_action = teleop.get_action()
    assert released_action["quest3.enabled"] is False
    assert np.allclose(released_action["quest3.pos"], np.zeros(3))
    assert released_action["quest3.raw_inputs"]["trigger_pressed"] is False


def test_quest3_get_action_exposes_face_buttons(monkeypatch):
    monkeypatch.setattr(Quest3, "runtime_cls", FakeQuest3Runtime)

    teleop = Quest3(Quest3Config(hand=Quest3Hand.LEFT))
    teleop.connect()

    runtime = teleop.runtime
    runtime.goal = _goal(buttons={"a": True, "b": False, "thumbstick": True})
    runtime.goal_timestamp = time.monotonic()

    action = teleop.get_action()
    buttons = action["quest3.raw_inputs"]["buttons"]

    assert buttons["a"] is True
    assert buttons["x"] is True
    assert buttons["b"] is False
    assert buttons["y"] is False
    assert buttons["thumbstick"] is True


def test_map_quest3_action_to_robot_action():
    step = MapQuest3ActionToRobotAction(
        position_axis_mapping=("-z", "-x", "y"),
        rotation_axis_mapping=("-z", "-x", "y"),
        position_scale=0.2,
    )

    transition = create_transition(
        action={
            "quest3.enabled": True,
            "quest3.pos": np.array([1.0, 2.0, 3.0]),
            "quest3.rot": Rotation.from_rotvec(np.array([0.1, 0.2, 0.3])),
            "quest3.raw_inputs": {"fresh": True, "trigger_pressed": True},
        }
    )

    result = step(transition)
    action = result[TransitionKey.ACTION]

    assert action["enabled"] is True
    assert action["target_x"] == -0.6
    assert action["target_y"] == -0.2
    assert action["target_z"] == 0.4
    assert action["target_wx"] == -0.3
    assert action["target_wy"] == -0.1
    assert action["target_wz"] == 0.2
    assert action["gripper_vel"] == -1.0
