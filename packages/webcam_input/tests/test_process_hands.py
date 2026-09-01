import numpy as np
from webcam_input.depth import ScaleDepthStrategy
from webcam_input.wrist_estimator import WebcamWristEstimator
from webcam_input.webcam_source import WebcamSource
from webcam_input.types import WristData, LandmarksData


def _image_xy():
    lm = np.zeros((21, 2)); lm[0] = [0.5, 0.5]; lm[9] = [0.5, 0.25]
    return lm


def _joint_pos(curled=False):
    """MANO-ish joint_pos[21,3]; tips near/far from wrist for fist control."""
    kp = np.zeros((21, 3))
    for tip, pip in ((8, 6), (12, 10), (16, 14), (20, 18)):
        kp[pip] = [0, 0.06, 0]
        kp[tip] = [0, (0.03 if curled else 0.10), 0]
    return kp


def _right(curled=False):
    return (_joint_pos(curled), _image_xy(), np.eye(3))


def _left(curled=False):
    return (_joint_pos(curled), _image_xy(), np.eye(3))


def _make_source():
    return WebcamSource(WebcamWristEstimator(depth=ScaleDepthStrategy()), image_shape=(480, 640))


def test_right_with_open_left_is_valid_and_open():
    wrist, landmarks = _make_source().process_hands(right=_right(), left=_left(curled=False))
    assert isinstance(wrist, WristData) and wrist.valid
    assert isinstance(landmarks, LandmarksData) and landmarks.valid
    assert landmarks.landmarks.shape == (21, 3)
    assert wrist.fist_state == "open"


def test_left_fist_sets_closed():
    wrist, _ = _make_source().process_hands(right=_right(), left=_left(curled=True))
    assert wrist.fist_state == "closed"


def test_no_left_hand_holds_last_state_starting_paused():
    wrist, _ = _make_source().process_hands(right=_right(), left=None)
    assert wrist.fist_state == "closed"  # safety default before any left reading


def test_left_hand_alone_updates_clutch_even_without_right():
    src = _make_source()
    # left hand open, no right hand: arm pose invalid, but clutch still updates to open
    wrist, landmarks = src.process_hands(right=None, left=_left(curled=False))
    assert not wrist.valid and not landmarks.valid
    assert wrist.fist_state == "open"
    # clenching the left hand (still no right) updates the clutch to closed
    wrist2, _ = src.process_hands(right=None, left=_left(curled=True))
    assert wrist2.fist_state == "closed"


def test_no_right_hand_invalid():
    wrist, landmarks = _make_source().process_hands(right=None, left=None)
    assert not wrist.valid and not landmarks.valid
