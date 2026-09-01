"""Read the local PressureVision operator preview independently of control UDP."""

from __future__ import annotations

import mmap
from pathlib import Path
import struct
import time

import cv2
import numpy as np


DEFAULT_PV_PREVIEW_SHARE = Path("/tmp/pressurevision-preview-v1.mmap")
PREVIEW_MAGIC = b"PVPREV1\0"
PREVIEW_HEADER = struct.Struct("<8sQdIIII")
PREVIEW_HEADER_SIZE = 64


def format_gripper_positions(
    commanded: float | None,
    observed: float | None,
) -> str:
    """Format the operator's commanded and measured gripper positions."""
    command_text = "--" if commanded is None else f"{float(commanded):.1f}"
    observed_text = "--" if observed is None else f"{float(observed):.1f}"
    return f"CMD q={command_text}    OBS q={observed_text}"


def draw_gripper_position_banner(
    frame: np.ndarray,
    *,
    commanded: float | None,
    observed: float | None,
) -> np.ndarray:
    """Draw the live gripper command/read-back large enough for the operator."""
    height, width = frame.shape[:2]
    top = max(34, height - 108)
    bottom = max(top + 68, height - 34)
    right = min(width - 10, 760)
    cv2.rectangle(frame, (10, top), (right, bottom), (0, 0, 0), -1)
    cv2.putText(
        frame,
        format_gripper_positions(commanded, observed),
        (24, top + 42),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.15,
        (0, 255, 255),
        3,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        "lower q = tighter",
        (26, top + 66),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return frame


class PressureVisionPreviewSource:
    def __init__(
        self,
        path: Path | str = DEFAULT_PV_PREVIEW_SHARE,
        *,
        stale_after_s: float = 0.75,
    ):
        self.path = Path(path)
        self.stale_after_s = float(stale_after_s)
        if self.stale_after_s <= 0.0:
            raise ValueError("stale_after_s must be positive")
        self._file = None
        self._map = None
        self._identity = None

    def _refresh_mapping(self) -> bool:
        try:
            stat = self.path.stat()
        except FileNotFoundError:
            self.close()
            return False
        identity = (stat.st_dev, stat.st_ino, stat.st_size)
        if self._map is not None and identity == self._identity:
            return True
        self.close()
        if stat.st_size < PREVIEW_HEADER_SIZE:
            return False
        self._file = self.path.open("rb")
        self._map = mmap.mmap(self._file.fileno(), 0, access=mmap.ACCESS_READ)
        self._identity = identity
        return True

    def read(self, *, now_s: float | None = None) -> np.ndarray | None:
        if not self._refresh_mapping():
            return None
        now_s = time.monotonic() if now_s is None else float(now_s)
        for _ in range(2):
            first = PREVIEW_HEADER.unpack(self._map[:PREVIEW_HEADER.size])
            magic, sequence, observed_at_s, height, width, channels, payload_size = first
            if magic != PREVIEW_MAGIC or sequence % 2 or channels != 3:
                continue
            if payload_size != height * width * channels:
                return None
            end = PREVIEW_HEADER_SIZE + payload_size
            if height <= 0 or width <= 0 or end > len(self._map):
                return None
            pixels = np.frombuffer(
                self._map[PREVIEW_HEADER_SIZE:end], dtype=np.uint8
            ).copy()
            second = PREVIEW_HEADER.unpack(self._map[:PREVIEW_HEADER.size])
            if first == second and now_s - observed_at_s <= self.stale_after_s:
                return pixels.reshape((height, width, channels))
        return None

    def close(self) -> None:
        if self._map is not None:
            self._map.close()
            self._map = None
        if self._file is not None:
            self._file.close()
            self._file = None
        self._identity = None
