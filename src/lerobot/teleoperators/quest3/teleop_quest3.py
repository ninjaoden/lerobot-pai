#!/usr/bin/env python

import logging
import time
from typing import Any

import numpy as np

from lerobot.processor import RobotAction
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected
from lerobot.utils.rotation import Rotation

from ..teleoperator import Teleoperator
from .config_quest3 import Quest3Config
from .quest3_runtime import Quest3Runtime

logger = logging.getLogger(__name__)


class Quest3(Teleoperator):
    """
    Quest 3 teleoperator backed by the in-tree XLeVR WebXR bridge.

    The device is intentionally robot-agnostic:
    - grip press latches a local origin for relative 6-DoF motion
    - trigger and face button state are exposed as raw input metadata
    - robot/world-frame mapping is handled by downstream processors
    """

    config_class = Quest3Config
    name = "quest3"
    runtime_cls = Quest3Runtime

    def __init__(self, config: Quest3Config):
        super().__init__(config)
        self.config = config
        self.runtime = self.runtime_cls(config)
        self._identity_rotation = Rotation.from_quat(np.array([0.0, 0.0, 0.0, 1.0], dtype=float))
        self._reset_state()

    @property
    def action_features(self) -> dict[str, type]:
        return {
            "quest3.pos": np.ndarray,
            "quest3.rot": Rotation,
            "quest3.raw_inputs": dict,
            "quest3.enabled": bool,
        }

    @property
    def feedback_features(self) -> dict[str, type]:
        return {}

    @property
    def is_connected(self) -> bool:
        return self.runtime.is_connected

    @check_if_already_connected
    def connect(self, calibrate: bool = True) -> None:
        self.runtime.connect()
        self._reset_state()
        if calibrate:
            self.calibrate()

        if self.runtime.connection_url:
            print(f"Quest3 web UI: {self.runtime.connection_url}")

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        self._anchor_position = None
        self._anchor_rotation_inv = None
        logger.info("Quest3 teleoperator uses grip-latched relative motion and does not require calibration.")

    def configure(self) -> None:
        pass

    @check_if_not_connected
    def get_action(self) -> RobotAction:
        goal, goal_timestamp = self.runtime.get_latest_goal(self.config.hand.value)
        if goal is not None:
            self._update_from_goal(goal, goal_timestamp)

        now = time.monotonic()
        if self._last_goal_timestamp is None or now - self._last_goal_timestamp > self.config.stale_timeout_s:
            self._mark_inputs_stale()

        position = np.zeros(3, dtype=float)
        rotation = self._identity_rotation

        if self._enabled and self._anchor_position is not None and self._last_raw_position is not None:
            position = self._last_raw_position - self._anchor_position

        if self._enabled and self._anchor_rotation_inv is not None and self._last_raw_rotation is not None:
            rotation = self._anchor_rotation_inv * self._last_raw_rotation

        return {
            "quest3.pos": position,
            "quest3.rot": rotation,
            "quest3.raw_inputs": dict(self._raw_inputs),
            "quest3.enabled": self._enabled,
        }

    def send_feedback(self, feedback: dict[str, Any]) -> None:
        del feedback

    @check_if_not_connected
    def disconnect(self) -> None:
        self.runtime.disconnect()
        self._reset_state()

    def _reset_state(self) -> None:
        self._enabled = False
        self._anchor_position: np.ndarray | None = None
        self._anchor_rotation_inv: Rotation | None = None
        self._last_raw_position: np.ndarray | None = None
        self._last_raw_rotation: Rotation | None = None
        self._last_goal_timestamp: float | None = None
        self._raw_inputs = self._default_raw_inputs(fresh=False)

    def _default_raw_inputs(self, *, fresh: bool) -> dict[str, Any]:
        return {
            "hand": self.config.hand.value,
            "fresh": fresh,
            "grip_pressed": False,
            "trigger_pressed": False,
            "trigger_value": 0.0,
            "thumbstick": {},
            "buttons": {},
            "source": None,
            "metadata": {},
        }

    def _update_from_goal(self, goal: Any, goal_timestamp: float | None) -> None:
        metadata = dict(getattr(goal, "metadata", None) or {})

        raw_position = self._extract_position(metadata)
        if raw_position is not None:
            self._last_raw_position = raw_position

        raw_rotation = self._extract_rotation(metadata)
        if raw_rotation is not None:
            self._last_raw_rotation = raw_rotation

        grip_pressed = bool(metadata.get("gripActive", False))
        trigger_value = float(metadata.get("trigger", 0.0) or 0.0)
        trigger_pressed = bool(metadata.get("trigger_active", trigger_value > 0.5))
        buttons = self._extract_buttons(metadata)

        self._raw_inputs = {
            "hand": self.config.hand.value,
            "fresh": True,
            "grip_pressed": grip_pressed,
            "trigger_pressed": trigger_pressed,
            "trigger_value": trigger_value,
            "thumbstick": metadata.get("thumbstick", {}),
            "buttons": buttons,
            "source": metadata.get("source"),
            "metadata": metadata,
        }

        if grip_pressed and self._anchor_position is None and self._last_raw_position is not None:
            self._anchor_position = self._last_raw_position.copy()

        if grip_pressed and self._anchor_rotation_inv is None and self._last_raw_rotation is not None:
            self._anchor_rotation_inv = self._last_raw_rotation.inv()

        if not grip_pressed:
            self._enabled = False
            self._anchor_position = None
            self._anchor_rotation_inv = None
        else:
            self._enabled = self._anchor_position is not None

        if goal_timestamp is not None:
            self._last_goal_timestamp = goal_timestamp

    def _mark_inputs_stale(self) -> None:
        self._enabled = False
        self._anchor_position = None
        self._anchor_rotation_inv = None
        self._raw_inputs = {
            **self._raw_inputs,
            "fresh": False,
            "grip_pressed": False,
            "trigger_pressed": False,
            "trigger_value": 0.0,
            "buttons": {},
        }

    def _extract_position(self, metadata: dict[str, Any]) -> np.ndarray | None:
        raw_position = metadata.get("vr_position")
        if raw_position is None:
            return None

        position = np.asarray(raw_position, dtype=float)
        if position.shape != (3,):
            raise ValueError(f"Expected a Quest3 position with shape (3,), got {position.shape}.")
        return position

    def _extract_rotation(self, metadata: dict[str, Any]) -> Rotation | None:
        quaternion = metadata.get("quaternion")
        if not isinstance(quaternion, dict):
            return None

        expected_keys = {"x", "y", "z", "w"}
        if not expected_keys.issubset(quaternion):
            return None

        quat_xyzw = np.array(
            [quaternion["x"], quaternion["y"], quaternion["z"], quaternion["w"]],
            dtype=float,
        )
        return Rotation.from_quat(quat_xyzw)

    def _extract_buttons(self, metadata: dict[str, Any]) -> dict[str, bool]:
        buttons = metadata.get("buttons", {})
        if not isinstance(buttons, dict):
            return {}

        normalized = {str(name).lower(): bool(is_pressed) for name, is_pressed in buttons.items()}

        if "x" in normalized and "a" not in normalized:
            normalized["a"] = normalized["x"]
        if "a" in normalized and "x" not in normalized:
            normalized["x"] = normalized["a"]

        if "y" in normalized and "b" not in normalized:
            normalized["b"] = normalized["y"]
        if "b" in normalized and "y" not in normalized:
            normalized["y"] = normalized["b"]

        return normalized
