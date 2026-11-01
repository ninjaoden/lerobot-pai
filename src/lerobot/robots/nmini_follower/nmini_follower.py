#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import time
from functools import cached_property

from lerobot.cameras.utils import make_cameras_from_configs
from lerobot.motors import Motor, MotorCalibration, MotorNormMode
from lerobot.motors.feetech import (
    FeetechMotorsBus,
    OperatingMode,
)
from lerobot.processor import RobotAction, RobotObservation
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected

from ..robot import Robot
from ..utils import ensure_safe_goal_position
from .config_nmini_follower import NMiniFollowerConfig

logger = logging.getLogger(__name__)

FULL_TURN_MIN_POSITION = 0


class NMiniFollower(Robot):
    """Follower implementation for the 6-axis NMini arm with a gripper."""

    config_class = NMiniFollowerConfig
    name = "nmini_follower"

    def __init__(self, config: NMiniFollowerConfig):
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
        self.cameras = make_cameras_from_configs(config.cameras)

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

    def _degrees_to_raw_present(self, motor: str, degrees: float) -> int:
        model = self.bus.motors[motor].model
        max_res = self.bus.model_resolution_table[model] - 1
        zero_position = self._zero_present_position(motor)
        raw_position = int((degrees * max_res / 360) + zero_position)
        calibration = self.calibration[motor]
        return min(calibration.range_max, max(calibration.range_min, raw_position))

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

    def _write_motor_positions(self, goal_positions: dict[str, float]) -> dict[str, float]:
        sent_positions: dict[str, float] = {}

        degree_goals = {
            motor: self._degrees_to_raw_present(motor, float(goal))
            for motor, goal in goal_positions.items()
            if motor in self._degree_motors
        }
        if degree_goals:
            self.bus.sync_write(
                "Goal_Position",
                degree_goals,
                normalize=False,
                num_retry=self.config.motor_write_num_retry,
            )
            sent_positions.update(
                {
                    motor: self._raw_present_to_degrees(motor, raw_goal)
                    for motor, raw_goal in degree_goals.items()
                }
            )

        other_goals = {
            motor: float(goal) for motor, goal in goal_positions.items() if motor not in self._degree_motors
        }
        if other_goals:
            self.bus.sync_write("Goal_Position", other_goals, num_retry=self.config.motor_write_num_retry)
            sent_positions.update(other_goals)

        return sent_positions

    @property
    def _motors_ft(self) -> dict[str, type]:
        return {f"{motor}.pos": float for motor in self.bus.motors}

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        return {
            cam: (self.config.cameras[cam].height, self.config.cameras[cam].width, 3) for cam in self.cameras
        }

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        return {**self._motors_ft, **self._cameras_ft}

    @cached_property
    def action_features(self) -> dict[str, type]:
        return self._motors_ft

    @property
    def is_connected(self) -> bool:
        return self.bus.is_connected and all(cam.is_connected for cam in self.cameras.values())

    @check_if_already_connected
    def connect(self, calibrate: bool = True) -> None:
        self.bus.connect()
        if not self.is_calibrated and calibrate:
            logger.info(
                "Mismatch between calibration values in the motor and the calibration file or no calibration file found"
            )
            self.calibrate()

        for cam in self.cameras.values():
            cam.connect()

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
        with self.bus.torque_disabled():
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
    def get_observation(self) -> RobotObservation:
        start = time.perf_counter()
        motor_positions = self._read_motor_positions()
        obs_dict = {f"{motor}.pos": val for motor, val in motor_positions.items()}
        dt_ms = (time.perf_counter() - start) * 1e3
        logger.debug(f"{self} read state: {dt_ms:.1f}ms")

        for cam_key, cam in self.cameras.items():
            start = time.perf_counter()
            obs_dict[cam_key] = cam.read_latest()
            dt_ms = (time.perf_counter() - start) * 1e3
            logger.debug(f"{self} read {cam_key}: {dt_ms:.1f}ms")

        return obs_dict

    @check_if_not_connected
    def send_action(self, action: RobotAction) -> RobotAction:
        goal_pos = {key.removesuffix(".pos"): val for key, val in action.items() if key.endswith(".pos")}

        if self.config.max_relative_target is not None:
            present_pos = self._read_motor_positions(list(goal_pos))
            goal_present_pos = {key: (g_pos, present_pos[key]) for key, g_pos in goal_pos.items()}
            goal_pos = ensure_safe_goal_position(goal_present_pos, self.config.max_relative_target)

        sent_goal_pos = self._write_motor_positions(goal_pos)
        return {f"{motor}.pos": val for motor, val in sent_goal_pos.items()}

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
        finally:
            for cam in self.cameras.values():
                cam.disconnect()

        logger.info(f"{self} disconnected.")
