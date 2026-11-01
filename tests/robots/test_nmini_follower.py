#!/usr/bin/env python

from contextlib import contextmanager
from unittest.mock import MagicMock, call, patch

import pytest

from lerobot.motors import MotorCalibration
from lerobot.robots.nmini_follower import (
    NMiniFollower,
    NMiniFollowerConfig,
)


def _make_bus_mock() -> MagicMock:
    bus = MagicMock(name="FeetechBusMock")
    bus.is_connected = False
    bus.model_resolution_table = {"sts3215": 4096}
    bus.is_calibrated = True

    def _connect():
        bus.is_connected = True

    def _disconnect(_disable=True):
        bus.is_connected = False

    bus.connect.side_effect = _connect
    bus.disconnect.side_effect = _disconnect

    @contextmanager
    def _dummy_cm():
        yield

    bus.torque_disabled.side_effect = _dummy_cm

    return bus


def _set_default_calibration(robot: NMiniFollower) -> None:
    robot.calibration = {
        motor: MotorCalibration(
            id=joint.id,
            drive_mode=0,
            homing_offset=0,
            range_min=1800 if motor != "larm_finger_joint" else 1000,
            range_max=2300 if motor != "larm_finger_joint" else 3000,
        )
        for motor, joint in robot.bus.motors.items()
    }


@pytest.fixture
def follower(tmp_path):
    bus_mock = _make_bus_mock()

    def _bus_side_effect(*_args, **kwargs):
        bus_mock.motors = kwargs["motors"]
        return bus_mock

    with (
        patch(
            "lerobot.robots.nmini_follower.nmini_follower.FeetechMotorsBus",
            side_effect=_bus_side_effect,
        ),
        patch.object(NMiniFollower, "configure", lambda self: None),
    ):
        cfg = NMiniFollowerConfig(port="/dev/null", id="test-nmini", calibration_dir=tmp_path)
        robot = NMiniFollower(cfg)
        _set_default_calibration(robot)
        yield robot
        if robot.is_connected:
            robot.disconnect()


def test_get_observation_uses_urdf_zero_reference_for_degree_motors(follower):
    raw_positions = {
        "larm_joint0": 2047,
        "larm_joint1": 2147,
        "larm_joint2": 1947,
        "larm_joint3": 2047,
        "larm_joint4": 2047,
        "larm_joint5": 2047,
    }

    def _sync_read(data_name, motors=None, normalize=True, num_retry=0):
        del data_name
        assert num_retry == follower.config.motor_read_num_retry
        selected = list(follower.bus.motors if motors is None else motors)
        if not normalize:
            return {motor: raw_positions[motor] for motor in selected}
        return {motor: 12.5 for motor in selected}

    follower.bus.sync_read.side_effect = _sync_read
    follower.connect()

    observation = follower.get_observation()

    assert observation["larm_joint0.pos"] == pytest.approx(0.0)
    assert observation["larm_joint1.pos"] == pytest.approx((100 * 360) / 4095)
    assert observation["larm_joint2.pos"] == pytest.approx((-100 * 360) / 4095)
    assert observation["larm_finger_joint.pos"] == pytest.approx(12.5)


def test_send_action_clips_degree_targets_with_recorded_ranges(follower):
    follower.connect()

    sent_action = follower.send_action(
        {
            "larm_joint0.pos": 50.0,
            "larm_finger_joint.pos": 30.0,
        }
    )

    follower.bus.sync_write.assert_has_calls(
        [
            call(
                "Goal_Position",
                {"larm_joint0": 2300},
                normalize=False,
                num_retry=follower.config.motor_write_num_retry,
            ),
            call(
                "Goal_Position",
                {"larm_finger_joint": 30.0},
                num_retry=follower.config.motor_write_num_retry,
            ),
        ]
    )
    assert sent_action["larm_joint0.pos"] == pytest.approx(((2300 - 2047) * 360) / 4095)
    assert sent_action["larm_finger_joint.pos"] == pytest.approx(30.0)


def test_calibrate_prompts_for_urdf_zero_pose(follower):
    follower.calibration = {}
    follower.bus.set_half_turn_homings.return_value = {
        motor: idx for idx, motor in enumerate(follower.bus.motors, 1)
    }
    follower.bus.record_ranges_of_motion.return_value = (
        {motor: 1000 for motor in follower.bus.motors},
        {motor: 3000 for motor in follower.bus.motors},
    )

    with patch("builtins.input", return_value="") as input_mock:
        follower.calibrate()

    prompts = [args[0] for args, _ in input_mock.call_args_list]
    assert any("URDF zero/reference pose" in prompt for prompt in prompts)
    follower.bus.set_half_turn_homings.assert_called_once()
    follower.bus.record_ranges_of_motion.assert_called_once()
