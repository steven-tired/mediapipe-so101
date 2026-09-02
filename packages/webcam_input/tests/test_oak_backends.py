"""OAKCamera must speak whichever depthai the venv actually has.

The pipeline API changed incompatibly at depthai v3. The code targeted v3 only,
the LeRobot runtime venv had 2.32, and the mismatch surfaced as
`'depthai.node.Camera' object has no attribute 'build'` -- thrown from inside a
recording run, after the workspace cameras were already open. No test caught it
because none of them built a pipeline.

These do, against a fake depthai for each API. No OAK, no USB.
"""

import sys
import types

import numpy as np
import pytest

from webcam_input.oak_camera import V2, V3, OAKCamera, detect_backend


class FakeQueue:
    def __init__(self, payload):
        self._payload = payload

    def get(self):
        return self._payload


class FakeFrame:
    def __init__(self, array):
        self._array = array

    def getCvFrame(self):
        return self._array

    def getFrame(self):
        return self._array


class FakeNode:
    """Records every call, so a test can assert which API path was walked."""

    def __init__(self, kind, log):
        self.kind = kind
        self.log = log
        self.out = self
        self.left = self
        self.right = self
        self.depth = self
        self.preview = self
        self.input = self

    def __getattr__(self, name):
        def call(*args, **kwargs):
            self.log.append(f"{self.kind}.{name}")
            if name == "build":
                return self
            if name == "requestOutput":
                return self
            if name == "createOutputQueue":
                return FakeQueue(FakeFrame(np.zeros((4, 4), dtype=np.uint16)))
            return None
        return call


class FakePipeline:
    def __init__(self, log):
        self.log = log

    def create(self, node_kind):
        self.log.append(f"create:{node_kind}")
        return FakeNode(node_kind, self.log)

    def start(self):
        self.log.append("pipeline.start")

    def stop(self):
        self.log.append("pipeline.stop")


class FakeDevice:
    def __init__(self, pipeline):
        self.log = pipeline.log
        self.log.append("Device(pipeline)")

    def getOutputQueue(self, name, maxSize, blocking):
        self.log.append(f"getOutputQueue:{name}")
        return FakeQueue(FakeFrame(np.zeros((4, 4), dtype=np.uint16)))

    def close(self):
        self.log.append("device.close")


def fake_depthai(version, log):
    """A depthai stand-in exposing only what OAKCamera.start touches."""
    dai = types.ModuleType("depthai")
    dai.__version__ = version

    class Enum:
        THE_480_P = "480p"
        BGR = "bgr"

    node = types.SimpleNamespace(
        MonoCamera="MonoCamera",
        StereoDepth=types.SimpleNamespace(
            PresetMode=types.SimpleNamespace(ROBOTICS="robotics", HIGH_DENSITY="high_density")
        ),
        Camera="Camera",
        ColorCamera="ColorCamera",
        XLinkOut="XLinkOut",
    )
    # `pipeline.create(dai.node.StereoDepth)` needs a hashable marker, and the class
    # above is only the enum holder; give the node table a plain name for creation.
    node.StereoDepth.__str__ = lambda self: "StereoDepth"
    dai.node = node
    dai.MonoCameraProperties = types.SimpleNamespace(SensorResolution=Enum)
    dai.ColorCameraProperties = types.SimpleNamespace(ColorOrder=Enum)
    dai.CameraBoardSocket = types.SimpleNamespace(CAM_A="CAM_A")
    dai.ImgFrame = types.SimpleNamespace(Type=types.SimpleNamespace(BGR888i="BGR888i"))
    dai.Pipeline = lambda: FakePipeline(log)
    dai.Device = FakeDevice
    return dai


@pytest.fixture
def installed(monkeypatch):
    def install(version):
        log = []
        monkeypatch.setitem(sys.modules, "depthai", fake_depthai(version, log))
        return log
    return install


def test_detect_backend_reads_the_major_version():
    assert detect_backend("2.32.0.0") == V2
    assert detect_backend("3.7.1") == V3
    with pytest.raises(RuntimeError):
        detect_backend("not-a-version")


def test_v2_install_takes_the_device_path_not_camera_build(installed):
    log = installed("2.32.0.0")
    cam = OAKCamera()
    cam.start()
    assert cam.backend == V2
    assert "create:ColorCamera" in log and "create:XLinkOut" in log
    assert "Device(pipeline)" in log
    assert not any(entry.endswith(".build") for entry in log), (
        "the v2 path must not call Camera.build -- that is the v3-only call that "
        "killed the recorder"
    )


def test_v3_install_takes_the_camera_build_path(installed):
    log = installed("3.7.1")
    cam = OAKCamera()
    cam.start()
    assert cam.backend == V3
    assert "Camera.build" in log and "Camera.requestOutput" in log
    assert "pipeline.start" in log
    assert "Device(pipeline)" not in log


def test_backend_can_be_pinned_against_the_installed_version(installed):
    log = installed("3.7.1")
    cam = OAKCamera(backend=V2)
    cam.start()
    assert cam.backend == V2
    assert "Device(pipeline)" in log


def test_an_unknown_backend_is_refused_before_any_hardware():
    with pytest.raises(ValueError):
        OAKCamera(backend="v4")


@pytest.mark.parametrize("version,expected", [("2.32.0.0", V2), ("3.7.1", V3)])
def test_read_returns_rgb_and_depth_on_both_backends(installed, version, expected):
    installed(version)
    cam = OAKCamera()
    cam.start()
    rgb, depth = cam.read()
    assert rgb.shape == (4, 4)
    assert depth.dtype == np.uint16
    assert cam.backend == expected


def test_stop_closes_the_device_on_v2_and_the_pipeline_on_v3(installed):
    log = installed("2.32.0.0")
    cam = OAKCamera()
    cam.start()
    cam.stop()
    assert "device.close" in log and "pipeline.stop" not in log

    log = installed("3.7.1")
    cam = OAKCamera()
    cam.start()
    cam.stop()
    assert "pipeline.stop" in log and "device.close" not in log


def test_stop_is_idempotent(installed):
    installed("2.32.0.0")
    cam = OAKCamera()
    cam.start()
    cam.stop()
    cam.stop()
