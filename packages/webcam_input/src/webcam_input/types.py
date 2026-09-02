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
    image_xy: np.ndarray | None = None    # optional (21, 2), normalized OAK/RGB coordinates
    depth_m: np.ndarray | None = None     # optional (21,), aligned depth in metres
    # Host-monotonic read-completion metadata, not a sensor exposure timestamp.
    observed_at_s: float | None = None
    frame_id: int | None = None


@dataclass(frozen=True)
class WebcamSample:
    """One atomic webcam publication from a single captured frame."""

    preview_frame: np.ndarray | None
    wrist: WristData
    landmarks: LandmarksData
    observed_at_s: float | None
    frame_id: int | None
