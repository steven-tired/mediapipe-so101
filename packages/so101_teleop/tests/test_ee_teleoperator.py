import numpy as np
from webcam_input.types import LandmarksData, WristData

from lerobot_teleoperator_so101_webcam.config_so101_webcam_ee import SO101WebcamEEConfig
from lerobot_teleoperator_so101_webcam.ee_control import EE_ACTION_KEYS
from lerobot_teleoperator_so101_webcam.so101_webcam_ee import SO101WebcamEE


class FakeSource:
    """Stand-in for WebcamSource: returns a scripted (WristData, LandmarksData)."""
    def __init__(self): self._latest = None
    def start(self, idx): pass
    def stop(self): pass
    def set(self, pos, fist="open", valid=True, pinch=0.07):
        lm = np.zeros((21, 3)); lm[4] = [pinch, 0, 0]
        self._latest = (
            WristData(np.array(pos, dtype=float), np.array([0., 0., 0., 1.]), fist, valid),
            LandmarksData(lm, valid),
        )
    def latest(self): return self._latest


def _teleop():
    src = FakeSource(); src.set([0., 0., 0.])
    t = SO101WebcamEE(SO101WebcamEEConfig(), source=src)
    t.connect(calibrate=False)
    return t, src


def test_action_features_are_ee_keys():
    t, _ = _teleop()
    assert set(t.action_features) == set(EE_ACTION_KEYS)


def test_latches_reference_then_reports_displacement():
    # With the VR->robot axis transform, VR y (hand height) maps to robot z (EE height),
    # so moving the hand up should latch ~0 then report a positive target_z displacement.
    t, src = _teleop()
    src.set([0.0, 0.1, 0.0]); first = t.get_action()      # rising edge latches ref here
    assert abs(first["target_z"]) < 1e-9                   # displacement ~0 at latch
    src.set([0.0, 0.15, 0.0]); second = t.get_action()
    assert second["target_z"] > 0.04                       # raised hand +0.05 -> gripper up


def test_clutch_closed_disables():
    t, src = _teleop()
    src.set([0.1, 0.0, 0.0], fist="closed")
    a = t.get_action()
    assert a["enabled"] is False
    assert a["gripper_vel"] == 0.0


def test_invalid_hand_disables():
    t, src = _teleop()
    src.set([0.1, 0.0, 0.0], valid=False)
    a = t.get_action()
    assert a["enabled"] is False
    assert a["gripper_vel"] == 0.0
