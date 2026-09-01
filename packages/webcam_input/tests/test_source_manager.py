import numpy as np
from webcam_input.webcam_source_manager import WebcamSourceManager
from webcam_input.types import WristData, LandmarksData


class _FakeSource:
    def __init__(self):
        self.started = False
        self._latest = (
            WristData(np.array([0.1, 0.2, 0.5]), np.array([0.0, 0.0, 0.0, 1.0]), "open", True),
            LandmarksData(np.zeros((21, 3)), True),
        )

    def start(self, camera_index=0):
        self.started = True

    def stop(self):
        self.started = False

    def latest(self):
        return self._latest

    @property
    def has_frame(self):
        return self.started


def test_interface_parity_and_refcount():
    WebcamSourceManager._instance = None
    mgr = WebcamSourceManager(source=_FakeSource())

    assert mgr.register_teleoperator(config=None, teleop_name="arm") is True
    assert mgr.is_started

    wrist, status = mgr.get_wrist_data()
    assert status["tcp_connected"] is True
    assert isinstance(wrist, WristData)

    landmarks, _ = mgr.get_landmarks_data()
    assert isinstance(landmarks, LandmarksData)

    assert mgr.register_teleoperator(config=None, teleop_name="hand") is True
    mgr.unregister_teleoperator("hand")
    assert mgr.is_started
    mgr.unregister_teleoperator("arm")
    assert not mgr.is_started
