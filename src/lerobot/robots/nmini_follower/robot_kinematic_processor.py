#!/usr/bin/env python

from dataclasses import dataclass

import numpy as np

from lerobot.configs.types import FeatureType, PipelineFeatureType, PolicyFeature
from lerobot.processor import ProcessorStepRegistry, RobotAction
from lerobot.processor.pipeline import TransitionKey
from lerobot.robots.so_follower.robot_kinematic_processor import (
    GripperVelocityToJoint,
    InverseKinematicsEEToJoints,
)
from lerobot.utils.rotation import Rotation


@ProcessorStepRegistry.register("nmini_gripper_velocity_to_joint")
@dataclass
class NMiniGripperVelocityToJoint(GripperVelocityToJoint):
    gripper_joint_name: str = "larm_finger_joint"

    def action(self, action: RobotAction) -> RobotAction:
        observation = self.transition.get(TransitionKey.OBSERVATION).copy()
        gripper_vel = action.pop("ee.gripper_vel")

        if observation is None:
            raise ValueError("Joints observation is require for computing robot kinematics")

        gripper_key = f"{self.gripper_joint_name}.pos"
        if gripper_key not in observation:
            raise ValueError(f"Missing gripper joint observation for {gripper_key!r}.")

        if self.discrete_gripper:
            gripper_vel = (gripper_vel - 1) * self.clip_max

        delta = gripper_vel * float(self.speed_factor)
        current_gripper_pos = float(observation[gripper_key])
        action["ee.gripper_pos"] = float(np.clip(current_gripper_pos + delta, self.clip_min, self.clip_max))
        return action


@ProcessorStepRegistry.register("nmini_inverse_kinematics_ee_to_joints")
@dataclass
class NMiniInverseKinematicsEEToJoints(InverseKinematicsEEToJoints):
    gripper_joint_name: str = "larm_finger_joint"

    def action(self, action: RobotAction) -> RobotAction:
        x = action.pop("ee.x")
        y = action.pop("ee.y")
        z = action.pop("ee.z")
        wx = action.pop("ee.wx")
        wy = action.pop("ee.wy")
        wz = action.pop("ee.wz")
        gripper_pos = action.pop("ee.gripper_pos")

        if None in (x, y, z, wx, wy, wz, gripper_pos):
            raise ValueError(
                "Missing required end-effector pose components: ee.x, ee.y, ee.z, ee.wx, ee.wy, ee.wz, ee.gripper_pos must all be present in action"
            )

        observation = self.transition.get(TransitionKey.OBSERVATION).copy()
        if observation is None:
            raise ValueError("Joints observation is require for computing robot kinematics")

        try:
            q_raw = np.array([float(observation[f"{name}.pos"]) for name in self.motor_names], dtype=float)
        except KeyError as exc:
            raise ValueError(f"Missing joint observation for {exc.args[0]!r}.") from exc

        if self.initial_guess_current_joints:
            self.q_curr = q_raw
        elif self.q_curr is None:
            self.q_curr = q_raw

        t_des = np.eye(4, dtype=float)
        t_des[:3, :3] = Rotation.from_rotvec([wx, wy, wz]).as_matrix()
        t_des[:3, 3] = [x, y, z]

        q_target = self.kinematics.inverse_kinematics(self.q_curr, t_des)
        self.q_curr = q_target

        for i, name in enumerate(self.motor_names):
            action[f"{name}.pos"] = float(q_target[i])
        action[f"{self.gripper_joint_name}.pos"] = float(gripper_pos)
        return action

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        features = super().transform_features(features)
        features[PipelineFeatureType.ACTION][f"{self.gripper_joint_name}.pos"] = PolicyFeature(
            type=FeatureType.ACTION,
            shape=(1,),
        )
        return features
