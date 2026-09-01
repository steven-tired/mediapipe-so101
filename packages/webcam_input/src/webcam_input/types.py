"""Payloads duck-typed to LeFranX's VR router messages (wrist_data / landmarks_data),
so the existing teleoperators read webcam data unchanged in Phase B."""

from dataclasses import dataclass
import numpy as np


@dataclass
class WristData:
    position: np.ndarray      # (3,) VR frame meters
    quaternion: np.ndarray    # (4,) [x, y, z, w] VR frame
    fist_state: str           # "open" | "closed"
    valid: bool


@dataclass
class LandmarksData:
    landmarks: np.ndarray     # (21, 3) MANO joint_pos from SingleHandDetector
    valid: bool
