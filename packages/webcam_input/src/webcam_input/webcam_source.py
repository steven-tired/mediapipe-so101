"""WebcamSource: turn reused-detector outputs into VR-router-compatible payloads.

`process_hands` is pure (no camera) so it is unit-testable. Capture runs ONE MediaPipe
`Hands(max_num_hands=2)` (so both hands are tracked on the same frame) and reuses
SingleHandDetector's static transforms to produce, per hand, the tuple
(joint_pos[21,3] MANO, image_xy[21,2] normalized, wrist_rot[3,3]).

Right hand drives the arm + fingers; left hand is the clutch.
"""

import threading

import numpy as np

from .gestures import detect_fist
from .types import WristData, LandmarksData
from .wrist_estimator import WebcamWristEstimator


class WebcamSource:
    def __init__(self, wrist_estimator: WebcamWristEstimator, image_shape=(480, 640)):
        self.wrist_estimator = wrist_estimator
        self.image_shape = image_shape
        self._last_fist = "closed"  # safety: start paused
        # capture state (populated by start()/start_oak())
        self._cap = None
        self._oak = None
        self._hands = None
        self._thread = None
        self._running = False
        # Annotated BGR frame for an external preview. Set here, not only in
        # start_oak(): latest_frame() is part of the public API and must not
        # depend on which start path was taken.
        self._latest_frame = None
        self._mp_draw = None
        self._mp_conns = None
        self._lock = threading.Lock()
        self._latest = (WristData(np.zeros(3), np.array([0.0, 0.0, 0.0, 1.0]), self._last_fist, False),
                        LandmarksData(np.zeros((21, 3)), False))
        self._has_frame = False

    def process_hands(self, right, left):
        """Return (WristData, LandmarksData) from per-hand detector outputs.

        Each hand arg is (joint_pos[21,3] MANO, image_xy[21,2], wrist_rot[3,3]) or None.
        The clutch is updated from the LEFT hand whether or not the RIGHT hand is present.
        """
        if left is not None:
            self._last_fist = "closed" if detect_fist(np.asarray(left[0], dtype=float)) else "open"
        fist_state = self._last_fist

        if right is None:
            return (WristData(np.zeros(3), np.array([0.0, 0.0, 0.0, 1.0]), fist_state, False),
                    LandmarksData(np.zeros((21, 3)), False))

        joint_pos, image_xy, wrist_rot = right
        joint_pos = np.asarray(joint_pos, dtype=float)
        position, quat = self.wrist_estimator.estimate(wrist_rot, image_xy, self.image_shape)
        return (WristData(position, quat, fist_state, True),
                LandmarksData(joint_pos, True))

    @staticmethod
    def split_results(results):
        """MediaPipe Hands(max_num_hands=2) result -> (right_tuple, left_tuple).

        Reuses SingleHandDetector's static transforms (parse_keypoint_3d,
        estimate_frame_from_hand_points, OPERATOR2MANO_*). On a non-selfie camera a
        physical RIGHT hand is labeled 'Left' (control) and a physical LEFT hand 'Right'
        (clutch).
        """
        from .detector import SingleHandDetector, OPERATOR2MANO_RIGHT, OPERATOR2MANO_LEFT
        if not results.multi_hand_world_landmarks or not results.multi_handedness:
            return None, None
        right = left = None
        for i, handed in enumerate(results.multi_handedness):
            label = handed.classification[0].label
            world = SingleHandDetector.parse_keypoint_3d(results.multi_hand_world_landmarks[i])
            image_xy = np.array([[p.x, p.y] for p in results.multi_hand_landmarks[i].landmark])
            centered = world - world[0:1, :]
            wrist_rot = SingleHandDetector.estimate_frame_from_hand_points(centered)
            if label == "Left":   # physical right hand -> control
                joint_pos = centered @ wrist_rot @ OPERATOR2MANO_RIGHT
                right = (joint_pos, image_xy, wrist_rot)
            else:                  # physical left hand -> clutch
                joint_pos = centered @ wrist_rot @ OPERATOR2MANO_LEFT
                left = (joint_pos, image_xy, wrist_rot)
        return right, left

    # --- camera capture (one MediaPipe Hands(2); reuses SingleHandDetector math) ---

    def start(self, camera_index: int = 0):
        if self._running:
            return
        import cv2
        import mediapipe as mp
        self._cap = cv2.VideoCapture(camera_index)
        if not self._cap.isOpened():
            raise RuntimeError(f"Could not open camera {camera_index}")
        self._hands = mp.solutions.hands.Hands(
            static_image_mode=False, max_num_hands=2,
            min_detection_confidence=0.8, min_tracking_confidence=0.8,
        )
        self._mp_draw = mp.solutions.drawing_utils
        self._mp_conns = mp.solutions.hands.HAND_CONNECTIONS
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        if self._cap is not None:
            self._cap.release()
        if self._oak is not None:
            self._oak.stop()
        if self._hands is not None:
            self._hands.close()

    def _loop(self):
        import cv2
        while self._running:
            ok, frame_bgr = self._cap.read()
            if not ok:
                continue
            self.image_shape = frame_bgr.shape[:2]
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            results = self._hands.process(rgb)
            right, left = self.split_results(results)
            wrist, landmarks = self.process_hands(right=right, left=left)
            annotated = frame_bgr.copy()   # draw landmarks for the preview window
            if results.multi_hand_landmarks:
                for hand_lms in results.multi_hand_landmarks:
                    self._mp_draw.draw_landmarks(annotated, hand_lms, self._mp_conns)
            with self._lock:
                self._latest = (wrist, landmarks)
                self._has_frame = wrist.valid
                self._latest_frame = annotated

    def start_oak(self):
        """Capture from an OAK-D (clean stereo depth) instead of a cv2 webcam.

        Feeds metric depth to the wrist estimator and runs the SAME process_hands path, so the
        recorder's hand tracking matches teleop_viz_ee.py --oak. depthai is imported lazily.
        """
        if self._running:
            return
        import mediapipe as mp

        from .depth import OAKDepthStrategy
        from .oak_camera import OAKCamera

        self._oak_depth = OAKDepthStrategy(radius_px=6, ema_alpha=0.4)
        self.wrist_estimator.depth = self._oak_depth   # wrist z now comes from OAK metric depth
        self._oak = OAKCamera(rgb_size=(640, 480), fps=30)
        self._oak.start()
        self._hands = mp.solutions.hands.Hands(
            static_image_mode=False, max_num_hands=2,
            min_detection_confidence=0.8, min_tracking_confidence=0.8,
        )
        self._mp_draw = mp.solutions.drawing_utils
        self._mp_conns = mp.solutions.hands.HAND_CONNECTIONS
        self.oak_failed = False     # set True if the OAK device crashes (X_LINK) mid-stream
        self._running = True
        self._thread = threading.Thread(target=self._loop_oak, daemon=True)
        self._thread.start()

    def _loop_oak(self):
        import cv2
        while self._running:
            try:
                rgb_bgr, depth = self._oak.read()
            except Exception as e:
                # OAK-D-Lite occasionally crashes the USB link (X_LINK_ERROR). Fail GRACEFULLY:
                # flag it, stop reading, and let the caller end the session cleanly (the recorder's
                # discard/restore protects the dataset). Reopen the script to re-init the device.
                self.oak_failed = True
                self._has_frame = False
                print(f"\n[OAK] camera crashed ({type(e).__name__}). Press ESC to end -- the session "
                      "will be discarded safely; restart the script to reconnect the OAK.\n")
                return
            self._oak_depth.update_depth(depth)
            self.image_shape = rgb_bgr.shape[:2]
            rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)
            results = self._hands.process(rgb)
            right, left = self.split_results(results)
            wrist, landmarks = self.process_hands(right=right, left=left)
            annotated = rgb_bgr.copy()   # draw landmarks for the preview window
            if results.multi_hand_landmarks:
                for hand_lms in results.multi_hand_landmarks:
                    self._mp_draw.draw_landmarks(annotated, hand_lms, self._mp_conns)
            with self._lock:
                self._latest = (wrist, landmarks)
                self._has_frame = wrist.valid
                self._latest_frame = annotated

    def latest_frame(self):
        """Latest annotated BGR hand-cam frame for a preview window (None until first frame)."""
        with self._lock:
            return self._latest_frame

    def latest(self):
        with self._lock:
            return self._latest

    @property
    def has_frame(self) -> bool:
        with self._lock:
            return self._has_frame
