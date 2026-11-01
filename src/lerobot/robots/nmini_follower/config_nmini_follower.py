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

from dataclasses import dataclass, field

from lerobot.cameras import CameraConfig

from ..config import RobotConfig


@RobotConfig.register_subclass("nmini_follower")
@dataclass
class NMiniFollowerConfig(RobotConfig):
    """Configuration for the NMini 6-axis follower arm."""

    port: str
    disable_torque_on_disconnect: bool = True
    motor_read_num_retry: int = 2
    motor_write_num_retry: int = 2
    max_relative_target: float | dict[str, float] | None = None
    cameras: dict[str, CameraConfig] = field(default_factory=dict)
    use_degrees: bool = True
    # Keep this empty by default because the published URDF exposes finite limits on every arm axis.
    full_turn_motors: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        super().__post_init__()

        if self.motor_read_num_retry < 0:
            raise ValueError("motor_read_num_retry must be non-negative.")

        if self.motor_write_num_retry < 0:
            raise ValueError("motor_write_num_retry must be non-negative.")

        if len(set(self.full_turn_motors)) != len(self.full_turn_motors):
            raise ValueError("full_turn_motors contains duplicates.")
