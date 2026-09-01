import numpy as np
from webcam_input.gestures import detect_fist


def _open_hand():
    kp = np.zeros((21, 3))
    for tip, pip in ((8, 6), (12, 10), (16, 14), (20, 18)):
        kp[pip] = [0, 0.05, 0]
        kp[tip] = [0, 0.10, 0]
    return kp


def _fist():
    kp = np.zeros((21, 3))
    for tip, pip in ((8, 6), (12, 10), (16, 14), (20, 18)):
        kp[pip] = [0, 0.06, 0]
        kp[tip] = [0, 0.03, 0]
    return kp


def test_open_hand_not_fist():
    assert detect_fist(_open_hand()) is False


def test_curled_hand_is_fist():
    assert detect_fist(_fist()) is True
