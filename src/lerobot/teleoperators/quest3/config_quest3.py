#!/usr/bin/env python

from dataclasses import dataclass
from enum import Enum

from ..config import TeleoperatorConfig


class Quest3Hand(Enum):
    LEFT = "left"
    RIGHT = "right"


@TeleoperatorConfig.register_subclass("quest3")
@dataclass
class Quest3Config(TeleoperatorConfig):
    hand: Quest3Hand = Quest3Hand.RIGHT
    host_ip: str = "0.0.0.0"
    https_port: int = 8443
    websocket_port: int = 8442
    startup_timeout_s: float = 10.0
    stale_timeout_s: float = 1.0
