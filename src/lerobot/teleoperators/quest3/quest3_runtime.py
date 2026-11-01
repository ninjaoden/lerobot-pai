#!/usr/bin/env python

import asyncio
import http.server
import logging
import os
import socket
import ssl
import sys
import threading
import time
from pathlib import Path
from types import ModuleType
from typing import Any

from .config_quest3 import Quest3Config

logger = logging.getLogger(__name__)


class Quest3Runtime:
    """
    Thin adapter around XLeVR's VRMonitor.

    It keeps the XLeVR implementation as the source of truth for websocket/HTTPS setup,
    while exposing a small synchronous API that fits LeRobot's teleoperator lifecycle.
    """

    def __init__(self, config: Quest3Config):
        self.config = config
        self._xlevr_root = Path(__file__).resolve().parent.parent / "XLeVR"
        self._monitor_module: ModuleType | None = None
        self._monitor = None
        self._thread: threading.Thread | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._shutdown_event: asyncio.Event | None = None
        self._startup_done = threading.Event()
        self._startup_exception: Exception | None = None
        self._connected = False
        self._connection_url: str | None = None
        self._goal_lock = threading.Lock()
        self._goals: dict[str, tuple[Any, float]] = {}

    @property
    def is_connected(self) -> bool:
        return self._connected and self._thread is not None and self._thread.is_alive()

    @property
    def connection_url(self) -> str | None:
        return self._connection_url

    def connect(self) -> None:
        if self.is_connected:
            return

        self._startup_done.clear()
        self._startup_exception = None
        self._thread = threading.Thread(target=self._thread_main, name="quest3-vr-monitor", daemon=True)
        self._thread.start()

        if not self._startup_done.wait(timeout=self.config.startup_timeout_s):
            self.disconnect()
            raise TimeoutError("Timed out while starting XLeVR VRMonitor.")

        if self._startup_exception is not None:
            self.disconnect()
            raise RuntimeError("Failed to start XLeVR VRMonitor.") from self._startup_exception

    def disconnect(self) -> None:
        if self._loop is not None and self._shutdown_event is not None:
            self._loop.call_soon_threadsafe(self._shutdown_event.set)

        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None

        with self._goal_lock:
            self._goals.clear()

        self._loop = None
        self._shutdown_event = None
        self._monitor = None
        self._connected = False
        self._connection_url = None

    def get_latest_goal(self, hand: str) -> tuple[Any | None, float | None]:
        with self._goal_lock:
            return self._goals.get(hand, (None, None))

    def _thread_main(self) -> None:
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)

        try:
            self._loop.run_until_complete(self._run())
        except Exception as exc:  # pragma: no cover - reported through connect()
            self._startup_exception = exc
            self._startup_done.set()
        finally:
            self._connected = False
            if self._loop is not None:
                self._loop.close()
            self._loop = None

    async def _run(self) -> None:
        self._monitor_module = self._load_vr_monitor_module()
        self._patch_vr_monitor_module(self._monitor_module)

        monitor = self._monitor_module.VRMonitor()
        self._monitor = monitor
        if not monitor.initialize():
            raise RuntimeError("VRMonitor.initialize() returned False.")

        self._configure_monitor(monitor)
        self._shutdown_event = asyncio.Event()

        await monitor.https_server.start()
        if getattr(monitor.https_server, "httpd", None) is None:
            raise RuntimeError("XLeVR HTTPS server failed to start.")

        await monitor.vr_server.start()
        if getattr(monitor.vr_server, "server", None) is None or not getattr(monitor.vr_server, "is_running", False):
            raise RuntimeError(
                "XLeVR WebSocket server failed to start. Quest3 controller data will not be received."
            )

        monitor.is_running = True
        self._connected = True
        self._connection_url = self._build_connection_url()
        logger.info("Quest3 web UI: %s", self._connection_url)
        self._startup_done.set()

        try:
            while self._shutdown_event is not None and not self._shutdown_event.is_set():
                try:
                    goal = await asyncio.wait_for(monitor.command_queue.get(), timeout=0.1)
                except asyncio.TimeoutError:
                    continue

                goal_timestamp = time.monotonic()
                with self._goal_lock:
                    self._goals[goal.arm] = (goal, goal_timestamp)

                with monitor._goal_lock:
                    if goal.arm == "left":
                        monitor.left_goal = goal
                    elif goal.arm == "right":
                        monitor.right_goal = goal
                    elif goal.arm == "headset":
                        monitor.headset_goal = goal
                    monitor.latest_goal = goal
        finally:
            monitor.is_running = False
            await monitor.stop_monitoring()

    def _load_vr_monitor_module(self) -> ModuleType:
        xlevr_root = str(self._xlevr_root)
        if xlevr_root not in sys.path:
            sys.path.insert(0, xlevr_root)

        from lerobot.teleoperators.XLeVR import vr_monitor as vr_monitor_module

        return vr_monitor_module

    def _patch_vr_monitor_module(self, module: ModuleType) -> None:
        module.XLEVR_PATH = str(self._xlevr_root)

        def setup_xlevr_environment() -> None:
            xlevr_root = str(self._xlevr_root)
            if xlevr_root not in sys.path:
                sys.path.insert(0, xlevr_root)
            os.environ["PYTHONPATH"] = f"{xlevr_root}:{os.environ.get('PYTHONPATH', '')}"

        def import_xlevr_modules():
            cwd = Path.cwd()
            try:
                os.chdir(self._xlevr_root)
                from xlevr.config import XLeVRConfig
                from xlevr.inputs.base import ControlGoal, ControlMode
                from xlevr.inputs.vr_ws_server import VRWebSocketServer

                return XLeVRConfig, VRWebSocketServer, ControlGoal, ControlMode
            finally:
                os.chdir(cwd)

        async def start_https_server(https_server) -> None:
            https_server.httpd = http.server.HTTPServer(
                (https_server.config.host_ip, https_server.config.https_port),
                module.SimpleAPIHandler,
            )
            https_server.httpd.web_root_path = str(self._xlevr_root)

            certfile = Path(https_server.config.certfile)
            keyfile = Path(https_server.config.keyfile)
            if not certfile.is_absolute():
                certfile = self._xlevr_root / certfile
            if not keyfile.is_absolute():
                keyfile = self._xlevr_root / keyfile

            ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
            ssl_context.load_cert_chain(certfile=str(certfile), keyfile=str(keyfile))
            https_server.httpd.socket = ssl_context.wrap_socket(https_server.httpd.socket, server_side=True)
            https_server.server_thread = threading.Thread(target=https_server.httpd.serve_forever, daemon=True)
            https_server.server_thread.start()

        module.setup_xlevr_environment = setup_xlevr_environment
        module.import_xlevr_modules = import_xlevr_modules
        module.SimpleHTTPSServer.start = start_https_server

    def _configure_monitor(self, monitor) -> None:
        monitor.config.host_ip = self.config.host_ip
        monitor.config.https_port = self.config.https_port
        monitor.config.websocket_port = self.config.websocket_port
        monitor.config.enable_vr = True
        monitor.config.enable_keyboard = False
        monitor.config.enable_https = True

        certfile = Path(monitor.config.certfile)
        keyfile = Path(monitor.config.keyfile)
        if not certfile.is_absolute():
            monitor.config.certfile = str(self._xlevr_root / certfile)
        if not keyfile.is_absolute():
            monitor.config.keyfile = str(self._xlevr_root / keyfile)

        monitor.https_server.web_root_path = str(self._xlevr_root)

    def _build_connection_url(self) -> str:
        host_ip = self.config.host_ip
        if host_ip == "0.0.0.0":
            host_ip = self._get_local_ip()
        return f"https://{host_ip}:{self.config.https_port}"

    def _get_local_ip(self) -> str:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
                sock.connect(("8.8.8.8", 80))
                return sock.getsockname()[0]
        except OSError:
            return "localhost"
