#!/usr/bin/env python

from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import pytest

from lerobot.motors import MotorCalibration
from lerobot.scripts.lerobot_setup_motors import COMPATIBLE_DEVICES
from lerobot.teleoperators.nmini_leader import NMiniLeader, NMiniLeaderConfig
from lerobot.teleoperators.utils import make_teleoperator_from_config


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


def _set_default_calibration(teleop: NMiniLeader) -> None:
    teleop.calibration = {
        motor: MotorCalibration(
            id=joint.id,
            drive_mode=0,
            homing_offset=0,
            range_min=1800 if motor != "larm_finger_joint" else 1000,
            range_max=2300 if motor != "larm_finger_joint" else 3000,
        )
        for motor, joint in teleop.bus.motors.items()
    }


@pytest.fixture
def leader(tmp_path):
    bus_mock = _make_bus_mock()

    def _bus_side_effect(*_args, **kwargs):
        bus_mock.motors = kwargs["motors"]
        return bus_mock

    with patch(
        "lerobot.teleoperators.nmini_leader.nmini_leader.FeetechMotorsBus",
        side_effect=_bus_side_effect,
    ):
        cfg = NMiniLeaderConfig(port="/dev/null", id="test-nmini-leader", calibration_dir=tmp_path)
        teleop = NMiniLeader(cfg)
        _set_default_calibration(teleop)
        yield teleop
        if teleop.is_connected:
            teleop.disconnect()


def test_make_teleoperator_from_config_returns_nmini_leader(tmp_path):
    bus_mock = _make_bus_mock()

    def _bus_side_effect(*_args, **kwargs):
        bus_mock.motors = kwargs["motors"]
        return bus_mock

    with patch(
        "lerobot.teleoperators.nmini_leader.nmini_leader.FeetechMotorsBus",
        side_effect=_bus_side_effect,
    ):
        teleop = make_teleoperator_from_config(
            NMiniLeaderConfig(port="/dev/null", id="test-nmini-leader", calibration_dir=tmp_path)
        )

    assert isinstance(teleop, NMiniLeader)


def test_get_action_uses_urdf_zero_reference_for_degree_motors(leader):
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
        assert num_retry == leader.config.motor_read_num_retry
        selected = list(leader.bus.motors if motors is None else motors)
        if not normalize:
            return {motor: raw_positions[motor] for motor in selected}
        return {motor: 12.5 for motor in selected}

    leader.bus.sync_read.side_effect = _sync_read
    leader.connect()

    action = leader.get_action()

    assert action["larm_joint0.pos"] == pytest.approx(0.0)
    assert action["larm_joint1.pos"] == pytest.approx((100 * 360) / 4095)
    assert action["larm_joint2.pos"] == pytest.approx((-100 * 360) / 4095)
    assert action["larm_finger_joint.pos"] == pytest.approx(12.5)


def test_connect_leaves_torque_disabled_for_backdrivable_leader(leader):
    leader.connect()

    leader.bus.disable_torque.assert_called_once_with()
    leader.bus.enable_torque.assert_not_called()


def test_calibrate_prompts_for_urdf_zero_pose(leader):
    leader.calibration = {}
    leader.bus.set_half_turn_homings.return_value = {motor: idx for idx, motor in enumerate(leader.bus.motors, 1)}
    leader.bus.record_ranges_of_motion.return_value = (
        {motor: 1000 for motor in leader.bus.motors},
        {motor: 3000 for motor in leader.bus.motors},
    )

    with patch("builtins.input", return_value="") as input_mock:
        leader.calibrate()

    prompts = [args[0] for args, _ in input_mock.call_args_list]
    assert any("URDF zero/reference pose" in prompt for prompt in prompts)
    leader.bus.set_half_turn_homings.assert_called_once()
    leader.bus.record_ranges_of_motion.assert_called_once()


def test_nmini_leader_is_supported_by_setup_motors_script():
    assert "nmini_leader" in COMPATIBLE_DEVICES
