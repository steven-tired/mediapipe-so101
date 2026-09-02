"""Clutch-hand gestures. New here — the reused detector and the
VR adapter don't expose a fist signal (in the VR path it came over the wire).

Works on either raw MediaPipe landmarks or MANO joint_pos: it compares tip-vs-PIP
distance to the wrist, which is invariant under the detector's rotation/centering.
"""

import numpy as np

_FINGERS = ((8, 6), (12, 10), (16, 14), (20, 18))  # (tip, pip)
_WRIST = 0


def count_curled_fingers(landmarks: np.ndarray) -> int:
    """Number of curled fingers (tip closer to wrist than its PIP)."""
    wrist = landmarks[_WRIST]
    curled = 0
    for tip, pip in _FINGERS:
        if np.linalg.norm(landmarks[tip] - wrist) < np.linalg.norm(landmarks[pip] - wrist):
            curled += 1
    return curled


def detect_fist(landmarks: np.ndarray, min_curled: int = 3) -> bool:
    """True if >= min_curled fingers are curled."""
    return count_curled_fingers(landmarks) >= min_curled


def detect_v_sign(landmarks: np.ndarray) -> bool:
    """True for index+middle extended with ring+pinky curled.

    The alternative to the left-fist clutch, for the right hand. It exists
    because PressureVision occupies the left hand: it is pressing the pad, so a
    left fist is not a gesture the operator can make. Unlike a fist, pinch, or
    open palm, this finger pattern is not part of an ordinary grasp cycle.

    Thumb posture is deliberately ignored: MediaPipe thumb orientation changes
    substantially as the hand rotates toward or away from the camera.
    """
    wrist = landmarks[_WRIST]

    def extended(tip: int, pip: int) -> bool:
        return np.linalg.norm(landmarks[tip] - wrist) > np.linalg.norm(landmarks[pip] - wrist)

    return bool(
        extended(8, 6)
        and extended(12, 10)
        and not extended(16, 14)
        and not extended(20, 18)
    )
