"""Singleton source manager exposing LeFranX's VRRouterManager consumer interface, so
its teleoperators (Phase B) read webcam data unchanged via get_wrist_data /
get_landmarks_data / get_status / register_teleoperator."""

import logging
import threading
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)


class WebcamSourceManager:
    _instance: Optional["WebcamSourceManager"] = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self, source=None, camera_index: int = 0):
        if getattr(self, "_initialized", False):
            return
        self._initialized = True
        self._source = source
        self._camera_index = camera_index
        self._reference_count = 0

    def register_teleoperator(self, config: Any, teleop_name: str) -> bool:
        with self._lock:
            self._reference_count += 1
            if self._reference_count == 1:
                self._source.start(self._camera_index)
            logger.info("WebcamSourceManager: registered %s (refs=%d)", teleop_name, self._reference_count)
            return self.is_started

    def unregister_teleoperator(self, teleop_name: str) -> None:
        with self._lock:
            if self._reference_count > 0:
                self._reference_count -= 1
                if self._reference_count == 0:
                    self._source.stop()

    def _status(self) -> Dict[str, Any]:
        connected = bool(self._source is not None and self._source.has_frame)
        return {"tcp_connected": connected, "running": self.is_started,
                "reference_count": self._reference_count}

    def get_wrist_data(self) -> Tuple[Any, Dict[str, Any]]:
        wrist, _ = self._source.latest()
        return wrist, self._status()

    def get_landmarks_data(self) -> Tuple[Any, Dict[str, Any]]:
        _, landmarks = self._source.latest()
        return landmarks, self._status()

    def get_status(self) -> Dict[str, Any]:
        return self._status()

    @property
    def is_started(self) -> bool:
        return bool(self._source is not None and self._source.has_frame)
