"""Wrist depth (VR +z) estimation from RGB. Pluggable so RealSense can drop in later.

This is genuinely new — the reused SingleHandDetector gives hand-relative landmarks
but no absolute camera distance, which the arm needs for the wrist's z axis.
"""

from abc import ABC, abstractmethod
import numpy as np


class DepthStrategy(ABC):
    @abstractmethod
    def estimate_z(self, image_landmarks: np.ndarray, image_shape) -> float:
        """Return wrist +z in meters (larger = farther from camera)."""


class ScaleDepthStrategy(DepthStrategy):
    """Scale-based depth: apparent wrist→middle-MCP bone length vs a calibrated reference.

    z = ref_distance_m * (ref_bone_px / current_bone_px). Smaller hand ⇒ larger z.
    """

    def __init__(self, ref_bone_px: float = 120.0, ref_distance_m: float = 0.5):
        self.ref_bone_px = float(ref_bone_px)
        self.ref_distance_m = float(ref_distance_m)
        self._last_z = ref_distance_m

    def estimate_z(self, image_landmarks: np.ndarray, image_shape) -> float:
        h, w = image_shape[0], image_shape[1]
        wrist_px = image_landmarks[0] * np.array([w, h])
        mid_mcp_px = image_landmarks[9] * np.array([w, h])
        bone_px = float(np.linalg.norm(mid_mcp_px - wrist_px))
        if bone_px < 1e-3:
            return self._last_z
        self._last_z = self.ref_distance_m * (self.ref_bone_px / bone_px)
        return self._last_z


def sample_depth_m(depth_mm, wrist_xy_norm, radius_px: int = 6,
                   min_m: float = 0.1, max_m: float = 2.0):
    """Median valid depth (metres) in a window around the wrist pixel, or None.

    depth_mm: (H, W) uint16 aligned depth in millimetres (0 = invalid/hole). wrist_xy_norm:
    (x, y) normalized [0, 1]. Holes (0) and out-of-[min_m, max_m] readings are rejected so a
    single bad/edge pixel can't corrupt z; returns None when no valid pixel is in the window.
    """
    depth_mm = np.asarray(depth_mm)
    h, w = depth_mm.shape[:2]
    cx = int(round(float(wrist_xy_norm[0]) * w))
    cy = int(round(float(wrist_xy_norm[1]) * h))
    x0, x1 = max(0, cx - radius_px), min(w, cx + radius_px + 1)
    y0, y1 = max(0, cy - radius_px), min(h, cy + radius_px + 1)
    patch = depth_mm[y0:y1, x0:x1].astype(np.float64)
    valid = patch[patch > 0] / 1000.0
    valid = valid[(valid >= min_m) & (valid <= max_m)]
    if valid.size == 0:
        return None
    return float(np.median(valid))


class OAKDepthStrategy(DepthStrategy):
    """Real metric wrist depth from an OAK-D aligned stereo-depth frame.

    Replaces the noisy monocular ScaleDepthStrategy: the OAK capture loop calls update_depth()
    each frame with the depth image aligned to the RGB camera, and estimate_z() samples the
    wrist pixel (hole-rejected median) and temporally smooths it. Falls back to the last good
    value on a hole so a dropout never injects a spike.
    """

    def __init__(self, radius_px: int = 6, min_m: float = 0.1, max_m: float = 2.0,
                 ema_alpha: float = 0.4, default_z: float = 0.5):
        self.radius_px = int(radius_px)
        self.min_m = float(min_m)
        self.max_m = float(max_m)
        self.ema_alpha = float(ema_alpha)
        self._depth_mm = None
        self._z = float(default_z)

    def update_depth(self, depth_mm) -> None:
        """Store the latest aligned depth frame (uint16 millimetres), called by the OAK loop."""
        self._depth_mm = depth_mm

    def estimate_z(self, image_landmarks, image_shape) -> float:
        if self._depth_mm is None:
            return self._z
        wrist = np.asarray(image_landmarks, dtype=float)[0]
        z = sample_depth_m(self._depth_mm, wrist, self.radius_px, self.min_m, self.max_m)
        if z is None:
            return self._z   # hole -> hold last good
        self._z = self.ema_alpha * z + (1.0 - self.ema_alpha) * self._z
        return self._z

    def sample_landmark_depths(self, image_landmarks, image_shape) -> np.ndarray:
        image_landmarks = np.asarray(image_landmarks, dtype=float)
        depths = np.full((image_landmarks.shape[0],), np.nan, dtype=float)
        if self._depth_mm is None:
            return depths
        for index, xy in enumerate(image_landmarks):
            if np.allclose(xy, 0.0):
                continue
            z = sample_depth_m(self._depth_mm, xy, self.radius_px, self.min_m, self.max_m)
            if z is not None:
                depths[index] = z
        return depths
