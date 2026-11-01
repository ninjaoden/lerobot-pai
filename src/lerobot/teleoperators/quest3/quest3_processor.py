#!/usr/bin/env python

from dataclasses import dataclass

import numpy as np

from lerobot.configs.types import FeatureType, PipelineFeatureType, PolicyFeature
from lerobot.processor import ProcessorStepRegistry, RobotAction, RobotActionProcessorStep


def _normalize_axis(axis: str) -> tuple[int, float]:
    sign = -1.0 if axis.startswith("-") else 1.0
    base_axis = axis[1:] if axis.startswith(("-", "+")) else axis
    axis_to_index = {"x": 0, "y": 1, "z": 2}
    if base_axis not in axis_to_index:
        raise ValueError(f"Invalid axis specification '{axis}'. Expected one of +/-x, +/-y, +/-z.")
    return axis_to_index[base_axis], sign


def _expand_scale(scale: float | tuple[float, float, float] | list[float]) -> np.ndarray:
    if isinstance(scale, (tuple, list)):
        if len(scale) != 3:
            raise ValueError("position_scale must contain exactly 3 elements when passed as a tuple.")
        return np.asarray(scale, dtype=float)
    return np.asarray([scale, scale, scale], dtype=float)


def _remap_vector(
    vector: np.ndarray,
    axis_mapping: tuple[str, str, str],
    scale: float | tuple[float, float, float] | list[float] = 1.0,
) -> np.ndarray:
    remapped = np.zeros(3, dtype=float)
    scales = _expand_scale(scale)
    for output_index, axis in enumerate(axis_mapping):
        source_index, sign = _normalize_axis(axis)
        remapped[output_index] = vector[source_index] * sign * scales[output_index]
    return remapped


@ProcessorStepRegistry.register("map_quest3_action_to_robot_action")
@dataclass
class MapQuest3ActionToRobotAction(RobotActionProcessorStep):
    position_axis_mapping: tuple[str, str, str] = ("x", "y", "z")
    rotation_axis_mapping: tuple[str, str, str] = ("x", "y", "z")
    position_scale: float | tuple[float, float, float] = 1.0
    use_orientation: bool = True
    gripper_open_velocity: float = 1.0
    gripper_close_velocity: float = -1.0

    def action(self, action: RobotAction) -> RobotAction:
        enabled = bool(action.pop("quest3.enabled"))
        pos = np.asarray(action.pop("quest3.pos"), dtype=float)
        rot = action.pop("quest3.rot")
        inputs = dict(action.pop("quest3.raw_inputs"))

        if pos.shape != (3,):
            raise ValueError(f"quest3.pos must have shape (3,), got {pos.shape}.")

        fresh = bool(inputs.get("fresh", False))
        mapped_position = _remap_vector(pos, self.position_axis_mapping, self.position_scale)

        if self.use_orientation:
            mapped_rotation = _remap_vector(rot.as_rotvec(), self.rotation_axis_mapping)
        else:
            mapped_rotation = np.zeros(3, dtype=float)

        if not fresh:
            gripper_vel = 0.0
        else:
            gripper_vel = (
                self.gripper_close_velocity
                if bool(inputs.get("trigger_pressed", False))
                else self.gripper_open_velocity
            )

        action["enabled"] = enabled and fresh
        action["target_x"] = float(mapped_position[0]) if enabled and fresh else 0.0
        action["target_y"] = float(mapped_position[1]) if enabled and fresh else 0.0
        action["target_z"] = float(mapped_position[2]) if enabled and fresh else 0.0
        action["target_wx"] = float(mapped_rotation[0]) if enabled and fresh else 0.0
        action["target_wy"] = float(mapped_rotation[1]) if enabled and fresh else 0.0
        action["target_wz"] = float(mapped_rotation[2]) if enabled and fresh else 0.0
        action["gripper_vel"] = float(gripper_vel)
        return action

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        for feat in ["enabled", "pos", "rot", "raw_inputs"]:
            features[PipelineFeatureType.ACTION].pop(f"quest3.{feat}", None)

        for feat in [
            "enabled",
            "target_x",
            "target_y",
            "target_z",
            "target_wx",
            "target_wy",
            "target_wz",
            "gripper_vel",
        ]:
            features[PipelineFeatureType.ACTION][feat] = PolicyFeature(type=FeatureType.ACTION, shape=(1,))

        return features
