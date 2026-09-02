"""OAKCamera: stream aligned (RGB, depth) frames from an OAK-D for metric wrist depth.

The OAK supplies BOTH the RGB frame MediaPipe runs on AND a stereo-depth frame aligned to
that RGB camera (so a wrist pixel in RGB indexes the same pixel in depth). This replaces the
laptop webcam + monocular ScaleDepthStrategy with one camera giving clean metric depth.

**Two depthai backends, because the pipeline API changed incompatibly at v3 and v3 is not
always the one you want.** v3 builds a camera node and pulls output queues off it
(`Camera.build` / `requestOutput` / `createOutputQueue`, with `pipeline.start()`); v2 wires
a `ColorCamera` to `XLinkOut` nodes and opens a `dai.Device`. Only one depthai can be
installed in a venv at a time, so `backend="auto"` reads the installed version and picks the
matching path. Pin it with `backend="v2"` / `"v3"` when you want the failure to be loud
instead of adaptive.

This exists because the code targeted v3 exclusively while the LeRobot runtime venv had
depthai 2.32, so every OAK path died at `'depthai.node.Camera' object has no attribute
'build'` -- after the cameras were open. depthai stays an optional dependency, imported
lazily, so the rest of webcam_input works without it.
"""

import numpy as np

V2, V3 = "v2", "v3"
_BACKENDS = (V2, V3)


def detect_backend(version: str) -> str:
    """Map a depthai version string onto the pipeline API it speaks."""
    major = version.split(".", 1)[0]
    if not major.isdigit():
        raise RuntimeError(f"cannot read a depthai major version from {version!r}")
    return V3 if int(major) >= 3 else V2


class OAKCamera:
    def __init__(self, rgb_size=(640, 480), fps: int = 30, backend: str = "auto"):
        if backend != "auto" and backend not in _BACKENDS:
            raise ValueError(f"backend must be 'auto', 'v2' or 'v3', not {backend!r}")
        self.rgb_size = rgb_size
        self.fps = int(fps)
        self.backend = backend
        self._pipeline = None
        self._device = None
        self._rgb_q = None
        self._depth_q = None

    def _resolve_backend(self, dai) -> str:
        if self.backend != "auto":
            return self.backend
        return detect_backend(dai.__version__)

    def start(self) -> None:
        import depthai as dai

        backend = self._resolve_backend(dai)
        pipeline = dai.Pipeline()
        stereo = self._build_stereo(dai, pipeline)
        if backend == V3:
            self._start_v3(dai, pipeline, stereo)
        else:
            self._start_v2(dai, pipeline, stereo)
        self.backend = backend
        self._pipeline = pipeline

    def _build_stereo(self, dai, pipeline):
        """Mono pair into StereoDepth, aligned to the colour camera. Same on both APIs."""
        mono_left = pipeline.create(dai.node.MonoCamera)
        mono_left.setCamera("left")
        mono_left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_480_P)
        mono_right = pipeline.create(dai.node.MonoCamera)
        mono_right.setCamera("right")
        mono_right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_480_P)

        stereo = pipeline.create(dai.node.StereoDepth)
        # ROBOTICS is v3's name for what v2 calls HIGH_DENSITY. Ask for whichever the
        # installed enum actually has rather than guessing from the version.
        presets = dai.node.StereoDepth.PresetMode
        preset = getattr(presets, "ROBOTICS", None) or getattr(presets, "HIGH_DENSITY")
        stereo.setDefaultProfilePreset(preset)
        stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)   # align depth to the colour camera
        stereo.setLeftRightCheck(True)
        # When aligned to the colour cam, depth is warped to its FOV; the output width must be a
        # multiple of 16, so pin it to the RGB size (which is) instead of the raw ISP width.
        stereo.setOutputSize(self.rgb_size[0], self.rgb_size[1])
        mono_left.out.link(stereo.left)
        mono_right.out.link(stereo.right)
        return stereo

    def _start_v3(self, dai, pipeline, stereo) -> None:
        cam_rgb = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)
        rgb_out = cam_rgb.requestOutput(self.rgb_size, dai.ImgFrame.Type.BGR888i, fps=self.fps)
        self._rgb_q = rgb_out.createOutputQueue()
        self._depth_q = stereo.depth.createOutputQueue()
        pipeline.start()

    def _start_v2(self, dai, pipeline, stereo) -> None:
        cam_rgb = pipeline.create(dai.node.ColorCamera)
        cam_rgb.setBoardSocket(dai.CameraBoardSocket.CAM_A)
        cam_rgb.setPreviewSize(*self.rgb_size)
        cam_rgb.setInterleaved(False)      # HxWx3, the layout getCvFrame returns anyway
        cam_rgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
        cam_rgb.setFps(self.fps)

        rgb_out = pipeline.create(dai.node.XLinkOut)
        rgb_out.setStreamName("rgb")
        cam_rgb.preview.link(rgb_out.input)
        depth_out = pipeline.create(dai.node.XLinkOut)
        depth_out.setStreamName("depth")
        stereo.depth.link(depth_out.input)

        # v2 starts the pipeline by opening the device on it. blocking=False with a short
        # queue keeps read() on the newest frame instead of draining a backlog, which is
        # what the v3 output queues do by default.
        self._device = dai.Device(pipeline)
        self._rgb_q = self._device.getOutputQueue("rgb", maxSize=4, blocking=False)
        self._depth_q = self._device.getOutputQueue("depth", maxSize=4, blocking=False)

    def read(self):
        """Return (rgb_bgr HxWx3 uint8, depth_mm HxW uint16 aligned to the RGB camera)."""
        rgb = self._rgb_q.get().getCvFrame()
        depth = np.asarray(self._depth_q.get().getFrame())
        return rgb, depth

    def stop(self) -> None:
        if self._pipeline is not None:
            if self._device is not None:
                self._device.close()      # v2: the device owns the running pipeline
            else:
                self._pipeline.stop()
            self._pipeline = None
            self._device = None
            self._rgb_q = None
            self._depth_q = None


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", default="auto", choices=("auto", V2, V3))
    args = parser.parse_args()

    cam = OAKCamera(backend=args.backend)
    cam.start()
    print(f"depthai backend: {cam.backend}")
    try:
        for i in range(10):
            rgb, depth = cam.read()
            h, w = depth.shape[:2]
            centre = depth[h // 2, w // 2]
            valid = int((depth > 0).sum())
            print(f"frame {i}: rgb={rgb.shape} depth={depth.shape} "
                  f"centre={centre}mm valid_px={valid}/{depth.size} "
                  f"range=[{depth[depth>0].min() if valid else 0},{depth.max()}]mm")
    finally:
        cam.stop()
