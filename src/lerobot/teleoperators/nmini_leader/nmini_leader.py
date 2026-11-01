#!/usr/bin/env python

import logging
import time
from functools import cached_property
from typing import Any

from lerobot.motors import Motor, MotorCalibration, MotorNormMode
from lerobot.motors.feetech import (
    FeetechMotorsBus,
    OperatingMode,
)
from lerobot.processor import RobotAction
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected

from ..teleoperator import Teleoperator
from .config_nmini_leader import NMiniLeaderConfig

logger = logging.getLogger(__name__)

FULL_TURN_MIN_POSITION = 0


class NMiniLeader(Teleoperator):
    """Teleoperator implementation for the 6-axis NMini leader arm with gripper."""

    config_class = NMiniLeaderConfig
    name = "nmini_leader"

    def __init__(self, config: NMiniLeaderConfig):
        super().__init__(config)
        self.config = config

        norm_mode_body = MotorNormMode.DEGREES if config.use_degrees else MotorNormMode.RANGE_M100_100
        self.bus = FeetechMotorsBus(
            port=self.config.port,
            motors={
                "larm_joint0": Motor(1, "sts3215", norm_mode_body),
                "larm_joint1": Motor(2, "sts3215", norm_mode_body),
                "larm_joint2": Motor(3, "sts3215", norm_mode_body),
                "larm_joint3": Motor(4, "sts3215", norm_mode_body),
                "larm_joint4": Motor(5, "sts3215", norm_mode_body),
                "larm_joint5": Motor(6, "sts3215", norm_mode_body),
                "larm_finger_joint": Motor(7, "sts3215", MotorNormMode.RANGE_0_100),
            },
            calibration=self.calibration,
        )
        self._full_turn_motors = self._validate_full_turn_motors(config.full_turn_motors)

    def _validate_full_turn_motors(self, full_turn_motors: tuple[str, ...]) -> tuple[str, ...]:
        unknown_motors = sorted(set(full_turn_motors) - set(self.bus.motors))
        if unknown_motors:
            raise ValueError(f"Unknown full_turn_motors: {unknown_motors}")

        return full_turn_motors

    @cached_property
    def _degree_motors(self) -> tuple[str, ...]:
        return tuple(
            name for name, motor in self.bus.motors.items() if motor.norm_mode is MotorNormMode.DEGREES
        )

    def _zero_present_position(self, motor: str) -> int:
        model = self.bus.motors[motor].model
        max_res = self.bus.model_resolution_table[model] - 1
        return int(max_res / 2)

    def _raw_present_to_degrees(self, motor: str, raw_position: int) -> float:
        model = self.bus.motors[motor].model
        max_res = self.bus.model_resolution_table[model] - 1
        zero_position = self._zero_present_position(motor)
        return (raw_position - zero_position) * 360 / max_res

    def _read_motor_positions(self, motors: list[str] | None = None) -> dict[str, float]:
        motor_names = list(self.bus.motors if motors is None else motors)
        positions: dict[str, float] = {}

        degree_motors = [motor for motor in motor_names if motor in self._degree_motors]
        if degree_motors:
            raw_positions = self.bus.sync_read(
                "Present_Position",
                degree_motors,
                normalize=False,
                num_retry=self.config.motor_read_num_retry,
            )
            positions.update(
                {
                    motor: self._raw_present_to_degrees(motor, int(raw_position))
                    for motor, raw_position in raw_positions.items()
                }
            )

        other_motors = [motor for motor in motor_names if motor not in self._degree_motors]
        if other_motors:
            normalized_positions = self.bus.sync_read(
                "Present_Position", other_motors, num_retry=self.config.motor_read_num_retry
            )
            positions.update({motor: float(position) for motor, position in normalized_positions.items()})

        return positions

    @property
    def action_features(self) -> dict[str, type]:
        return {f"{motor}.pos": float for motor in self.bus.motors}

    @property
    def feedback_features(self) -> dict[str, type]:
        return {}

    @property
    def is_connected(self) -> bool:
        return self.bus.is_connected

    @check_if_already_connected
    def connect(self, calibrate: bool = True) -> None:
        self.bus.connect()
        if not self.is_calibrated and calibrate:
            logger.info(
                "Mismatch between calibration values in the motor and the calibration file or no calibration file found"
            )
            self.calibrate()

        self.configure()
        logger.info(f"{self} connected.")

    @property
    def is_calibrated(self) -> bool:
        return self.bus.is_calibrated

    def calibrate(self) -> None:
        if self.calibration:
            user_input = input(
                f"Press ENTER to use provided calibration file associated with the id {self.id}, or type 'c' and press ENTER to run calibration: "
            )
            if user_input.strip().lower() != "c":
                logger.info(f"Writing calibration file associated with the id {self.id} to the motors")
                self.bus.write_calibration(self.calibration)
                return

        logger.info(f"\nRunning calibration of {self}")
        self.bus.disable_torque()
        for motor in self.bus.motors:
            self.bus.write("Operating_Mode", motor, OperatingMode.POSITION.value)

        input(
            f"Move {self} to the URDF zero/reference pose "
            "(the pose that should correspond to joint values of 0 degrees) and press ENTER...."
        )
        homing_offsets = self.bus.set_half_turn_homings()

        range_mins: dict[str, int] = {}
        range_maxes: dict[str, int] = {}
        unknown_range_motors = [motor for motor in self.bus.motors if motor not in self._full_turn_motors]
        if unknown_range_motors:
            print(
                "Move all joints sequentially through their entire ranges "
                "of motion.\nRecording positions. Press ENTER to stop..."
            )
            range_mins, range_maxes = self.bus.record_ranges_of_motion(unknown_range_motors)

        for motor in self._full_turn_motors:
            max_position = self.bus.model_resolution_table[self.bus.motors[motor].model] - 1
            range_mins[motor] = FULL_TURN_MIN_POSITION
            range_maxes[motor] = max_position

        self.calibration = {}
        for motor, m in self.bus.motors.items():
            self.calibration[motor] = MotorCalibration(
                id=m.id,
                drive_mode=0,
                homing_offset=homing_offsets[motor],
                range_min=range_mins[motor],
                range_max=range_maxes[motor],
            )

        self.bus.write_calibration(self.calibration)
        self._save_calibration()
        print("Calibration saved to", self.calibration_fpath)

    def configure(self) -> None:
        self.bus.disable_torque()
        self.bus.configure_motors()
        for motor in self.bus.motors:
            self.bus.write("Operating_Mode", motor, OperatingMode.POSITION.value)
            self.bus.write("P_Coefficient", motor, 16)
            self.bus.write("I_Coefficient", motor, 0)
            self.bus.write("D_Coefficient", motor, 32)

            if motor == "larm_joint1":
                self.bus.write("P_Coefficient", motor, 96)

            if motor == "larm_joint2":
                self.bus.write("P_Coefficient", motor, 64)

            if motor == "larm_finger_joint":
                self.bus.write("Max_Torque_Limit", motor, 500)
                self.bus.write("Protection_Current", motor, 250)
                self.bus.write("Overload_Torque", motor, 25)

    def setup_motors(self) -> None:
        for motor in reversed(self.bus.motors):
            input(f"Connect the controller board to the '{motor}' motor only and press enter.")
            self.bus.setup_motor(motor)
            print(f"'{motor}' motor id set to {self.bus.motors[motor].id}")

    @check_if_not_connected
    def get_action(self) -> RobotAction:
        start = time.perf_counter()
        action = self._read_motor_positions()
        action = {f"{motor}.pos": val for motor, val in action.items()}
        dt_ms = (time.perf_counter() - start) * 1e3
        logger.debug(f"{self} read action: {dt_ms:.1f}ms")
        return action

    def send_feedback(self, feedback: dict[str, Any]) -> None:
        del feedback
        raise NotImplementedError

    @check_if_not_connected
    def disconnect(self) -> None:
        try:
            self.bus.disconnect(self.config.disable_torque_on_disconnect)
        except ConnectionError as exc:
            logger.warning(
                "%s lost motor communication during disconnect; forcing port close. Original error: %s",
                self,
                exc,
            )
            try:
                self.bus.port_handler.clearPort()
            except Exception:  # nosec B110
                pass
            try:
                self.bus.port_handler.is_using = False
            except Exception:  # nosec B110
                pass
            try:
                self.bus.port_handler.closePort()
            except Exception as close_exc:
                logger.warning("%s failed to force-close the motor port: %s", self, close_exc)

        logger.info(f"{self} disconnected.")
