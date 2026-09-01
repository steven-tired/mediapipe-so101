"""OAKCamera: stream aligned (RGB, depth) frames from an OAK-D for metric wrist depth.

The OAK supplies BOTH the RGB frame MediaPipe runs on AND a stereo-depth frame aligned to
that RGB camera (so a wrist pixel in RGB indexes the same pixel in depth). This replaces the
laptop webcam + monocular ScaleDepthStrategy with one camera giving clean metric depth.

Requires depthai >= 3 and a stereo OAK-D (verified via oak_probe.py). depthai is an optional
dependency -- imported lazily so the rest of webcam_input works without it.
"""

import numpy as np


class OAKCamera:
    def __init__(self, rgb_size=(640, 480), fps: int = 30):
        self.rgb_size = rgb_size
        self.fps = int(fps)
        self._pipeline = None
        self._rgb_q = None
        self._depth_q = None

    def start(self) -> None:
        import depthai as dai

        pipeline = dai.Pipeline()

        mono_left = pipeline.create(dai.node.MonoCamera)
        mono_left.setCamera("left")
        mono_left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_480_P)
        mono_right = pipeline.create(dai.node.MonoCamera)
        mono_right.setCamera("right")
        mono_right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_480_P)

        stereo = pipeline.create(dai.node.StereoDepth)
        stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.ROBOTICS)
        stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)   # align depth to the colour camera
        stereo.setLeftRightCheck(True)
        # When aligned to the colour cam, depth is warped to its FOV; the output width must be a
        # multiple of 16, so pin it to the RGB size (which is) instead of the raw ISP width.
        stereo.setOutputSize(self.rgb_size[0], self.rgb_size[1])
        mono_left.out.link(stereo.left)
        mono_right.out.link(stereo.right)

        cam_rgb = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)
        rgb_out = cam_rgb.requestOutput(self.rgb_size, dai.ImgFrame.Type.BGR888i, fps=self.fps)

        self._rgb_q = rgb_out.createOutputQueue()
        self._depth_q = stereo.depth.createOutputQueue()
        pipeline.start()
        self._pipeline = pipeline

    def read(self):
        """Return (rgb_bgr HxWx3 uint8, depth_mm HxW uint16 aligned to the RGB camera)."""
        rgb = self._rgb_q.get().getCvFrame()
        depth = np.asarray(self._depth_q.get().getFrame())
        return rgb, depth

    def stop(self) -> None:
        if self._pipeline is not None:
            self._pipeline.stop()
            self._pipeline = None


if __name__ == "__main__":
    cam = OAKCamera()
    cam.start()
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
