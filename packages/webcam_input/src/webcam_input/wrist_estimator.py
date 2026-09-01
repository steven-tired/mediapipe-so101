"""Estimate a VR-frame wrist pose (position + quaternion) for the arm.

Reuse: the wrist *orientation* frame comes straight from SingleHandDetector.detect()'s
`mediapipe_wrist_rot` (a 3x3 hand frame in MediaPipe-world coords) — we do NOT recompute
it. New here: the metric wrist *position* (image x/y → workspace box, z via DepthStrategy)
and the MediaPipe-world → VR-frame mapping the arm IK expects.

VR frame = +x right, +y up, +z forward (away from camera). MediaPipe world ≈
+x right, +y down, +z toward camera, so the remap flips Y and Z. Exact signs are
re-validated live (off-robot) before any motion.
"""

import numpy as np
from scipy.spatial.transform import Rotation

from .depth import DepthStrategy

_CAMERA_TO_VR = np.diag([1.0, -1.0, -1.0])  # flip Y and Z


class WebcamWristEstimator:
    def __init__(self, depth: DepthStrategy, workspace_size_m: float = 0.4):
        self.depth = depth
        self.workspace_size_m = float(workspace_size_m)

    def estimate(self, wrist_rot, image_landmarks, image_shape):
        """Map a detector wrist frame + image wrist position to a VR-frame pose.

        Args:
            wrist_rot: 3x3 hand frame from SingleHandDetector (MediaPipe-world coords).
            image_landmarks: (21,2) normalized image landmarks (for wrist x/y + depth).
            image_shape: (h, w) of the source frame.
        Returns:
            (position[3] meters VR frame, quaternion[4] [x,y,z,w] VR frame)
        """
        image_landmarks = np.asarray(image_landmarks, dtype=float)

        wx, wy = image_landmarks[0]
        x = (wx - 0.5) * self.workspace_size_m          # +x right
        y = (0.5 - wy) * self.workspace_size_m          # +y up (image y is down)
        z = self.depth.estimate_z(image_landmarks, image_shape)
        position = np.array([x, y, z])

        frame_vr = _CAMERA_TO_VR @ np.asarray(wrist_rot, dtype=float)
        u, _, vt = np.linalg.svd(frame_vr)              # re-orthonormalize
        rot = u @ vt
        if np.linalg.det(rot) < 0:
            u[:, -1] *= -1
            rot = u @ vt
        quat = Rotation.from_matrix(rot).as_quat()      # [x, y, z, w]
        return position, quat
