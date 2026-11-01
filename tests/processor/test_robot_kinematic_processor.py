#!/usr/bin/env python

import numpy as np

from lerobot.processor.converters import create_transition
from lerobot.processor.pipeline import TransitionKey
from lerobot.robots.nmini_follower.robot_kinematic_processor import (
    NMiniGripperVelocityToJoint,
    NMiniInverseKinematicsEEToJoints,
)


class MockKinematics:
    def __init__(self):
        self.last_q_curr = None
        self.last_target = None

    def inverse_kinematics(self, q_curr: np.ndarray, target: np.ndarray) -> np.ndarray:
        self.last_q_curr = q_curr
        self.last_target = target
        return np.array([11.0, 22.0], dtype=float)


def test_gripper_velocity_to_joint_uses_named_gripper_joint():
    step = NMiniGripperVelocityToJoint(speed_factor=5.0, gripper_joint_name="larm_finger_joint")

    transition = create_transition(
        observation={
            "joint0.pos": 1.0,
            "larm_finger_joint.pos": 40.0,
            "joint1.pos": 2.0,
        },
        action={"ee.gripper_vel": 2.0},
    )

    result = step(transition)
    action = result[TransitionKey.ACTION]

    assert action["ee.gripper_pos"] == 50.0


def test_inverse_kinematics_ee_to_joints_uses_named_gripper_joint():
    kinematics = MockKinematics()
    step = NMiniInverseKinematicsEEToJoints(
        kinematics=kinematics,
        motor_names=["joint0", "joint1"],
        gripper_joint_name="larm_finger_joint",
    )

    transition = create_transition(
        observation={
            "joint0.pos": 1.0,
            "larm_finger_joint.pos": 50.0,
            "joint1.pos": 2.0,
        },
        action={
            "ee.x": 0.1,
            "ee.y": 0.2,
            "ee.z": 0.3,
            "ee.wx": 0.0,
            "ee.wy": 0.0,
            "ee.wz": 0.0,
            "ee.gripper_pos": 77.0,
        },
    )

    result = step(transition)
    action = result[TransitionKey.ACTION]

    assert np.allclose(kinematics.last_q_curr, np.array([1.0, 2.0]))
    assert action["joint0.pos"] == 11.0
    assert action["joint1.pos"] == 22.0
    assert action["larm_finger_joint.pos"] == 77.0
