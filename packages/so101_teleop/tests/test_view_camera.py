import unittest

import cv2
import numpy as np

from lerobot_teleoperator_so101_webcam.programs import view_camera


class ViewCameraOverlayTest(unittest.TestCase):
    def test_detects_red_cube_and_black_tag_centers(self):
        frame = np.full((480, 640, 3), 210, dtype=np.uint8)
        cv2.rectangle(frame, (300, 200), (340, 240), (0, 0, 220), -1)
        cv2.rectangle(frame, (370, 260), (470, 360), (0, 0, 0), -1)
        cv2.rectangle(frame, (390, 280), (420, 310), (255, 255, 255), -1)

        cube = view_camera.detect_red_cube(frame)
        tag = view_camera.detect_apriltag_square(frame)

        self.assertIsNotNone(cube)
        self.assertIsNotNone(tag)
        self.assertAlmostEqual(cube.center[0], 320, delta=2)
        self.assertAlmostEqual(cube.center[1], 220, delta=2)
        self.assertAlmostEqual(tag.center[0], 420, delta=5)
        self.assertAlmostEqual(tag.center[1], 310, delta=5)

    def test_alignment_status_uses_reference_targets(self):
        status = view_camera.alignment_status(
            cube_center=(315, 225),
            tag_center=(420, 292),
            cube_target=(320, 220),
            tag_target=(416, 297),
            tolerance_px=30,
        )

        self.assertTrue(status.ok)
        self.assertEqual(status.cube_delta, (-5, 5))
        self.assertEqual(status.tag_delta, (4, -5))

    def test_loads_dp50_and_dp100_alignment_profiles(self):
        dp50 = view_camera.load_alignment_profile("dp50")
        dp100 = view_camera.load_alignment_profile("dp100")

        self.assertEqual(dp50.name, "dp50")
        self.assertEqual(dp50.cube_target, (286, 118))
        self.assertEqual(dp50.tag_target, (421, 143))
        self.assertEqual(dp100.name, "dp100")
        self.assertEqual(dp100.cube_target, (320, 220))
        self.assertEqual(dp100.tag_target, (416, 297))

    def test_unknown_alignment_profile_errors_clearly(self):
        with self.assertRaisesRegex(ValueError, "Unknown camera alignment profile"):
            view_camera.load_alignment_profile("missing")

    def test_reference_overlay_blends_same_size_frame(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        reference = np.full_like(frame, 200)

        overlay = view_camera.draw_reference_overlay(frame, reference, alpha=0.25, label="test")

        self.assertEqual(overlay.shape, frame.shape)
        self.assertEqual(int(overlay[100, 100, 0]), 50)


if __name__ == "__main__":
    unittest.main()
