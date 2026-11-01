#!/usr/bin/env python

from dataclasses import dataclass

from ..config import TeleoperatorConfig


@TeleoperatorConfig.register_subclass("nmini_leader")
@dataclass
class NMiniLeaderConfig(TeleoperatorConfig):
    """Configuration for the NMini leader teleoperator."""

    port: str
    disable_torque_on_disconnect: bool = True
    motor_read_num_retry: int = 2
    motor_write_num_retry: int = 2
    use_degrees: bool = True
    full_turn_motors: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.motor_read_num_retry < 0:
            raise ValueError("motor_read_num_retry must be non-negative.")

        if self.motor_write_num_retry < 0:
            raise ValueError("motor_write_num_retry must be non-negative.")

        if len(set(self.full_turn_motors)) != len(self.full_turn_motors):
            raise ValueError("full_turn_motors contains duplicates.")
