import numpy as np

from webcam_input.types import LandmarksData, WristData

from lerobot_teleoperator_so101_webcam.config_so101_webcam import SO101WebcamConfig
from lerobot_teleoperator_so101_webcam.control import MOTORS
from lerobot_teleoperator_so101_webcam.so101_webcam import SO101Webcam


class FakeManager:
    def __init__(self, wrist, landmarks):
        self._wrist, self._landmarks = wrist, landmarks
    def register_teleoperator(self, config, name): return True
    def unregister_teleoperator(self, name): pass
    def get_wrist_data(self): return self._wrist, {}
    def get_landmarks_data(self): return self._landmarks, {}


def _open_hand(pos):
    lm = np.zeros((21, 3)); lm[4] = [0.12, 0, 0]  # wide pinch
    return (WristData(np.array(pos), np.array([0.0, 0.0, 0.0, 1.0]), "open", True),
            LandmarksData(lm, True))


def test_action_features_lists_six_motors():
    teleop = SO101Webcam(SO101WebcamConfig())
    assert set(teleop.action_features) == {f"{m}.pos" for m in MOTORS}


def test_get_action_maps_when_tracked_and_open():
    wrist, lm = _open_hand([0.1, 0.0, 0.5])
    teleop = SO101Webcam(SO101WebcamConfig(), source_manager=FakeManager(wrist, lm))
    teleop.connect(calibrate=False)
    action = teleop.get_action()
    assert set(action) == {f"{m}.pos" for m in MOTORS}
    assert action["shoulder_pan.pos"] > 0.0           # hand to the right
    assert -100.0 <= action["shoulder_pan.pos"] <= 100.0


def test_clutch_closed_holds_rest_pose():
    wrist, lm = _open_hand([0.1, 0.0, 0.5])
    wrist.fist_state = "closed"                        # clutch engaged
    teleop = SO101Webcam(SO101WebcamConfig(), source_manager=FakeManager(wrist, lm))
    teleop.connect(calibrate=False)
    action = teleop.get_action()
    assert action == {f"{m}.pos": 0.0 for m in MOTORS}


def test_invalid_hand_holds_last_action():
    wrist, lm = _open_hand([0.1, 0.0, 0.5])
    wrist.valid = False
    teleop = SO101Webcam(SO101WebcamConfig(), source_manager=FakeManager(wrist, lm))
    teleop.connect(calibrate=False)
    action = teleop.get_action()
    assert action == {f"{m}.pos": 0.0 for m in MOTORS}
