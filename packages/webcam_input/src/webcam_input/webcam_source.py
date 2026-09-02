"""WebcamSource: turn reused-detector outputs into VR-router-compatible payloads.

`process_hands` is pure (no camera) so it is unit-testable. Capture runs ONE MediaPipe
`Hands(max_num_hands=2)` (so both hands are tracked on the same frame) and reuses
SingleHandDetector's static transforms to produce, per hand, the tuple
(joint_pos[21,3] MANO, image_xy[21,2] normalized, wrist_rot[3,3]).

Right hand drives the arm + fingers; left hand is the clutch.
"""

import threading
import time

import numpy as np

from .gestures import detect_fist
from .types import WristData, LandmarksData, WebcamSample
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
        self._mp_draw = None
        self._mp_conns = None
        self._lock = threading.Lock()
        # ONE locked publication, not three fields written in sequence, and created
        # here rather than in start_oak(): latest_frame()/latest()/latest_sample() are
        # public API and must not depend on which start path was taken. The recorder's
        # 3 s hand-startup gate advances on `frame_id` changing and reads validity from
        # the same object, so a torn read -- this iteration's id beside the previous
        # iteration's wrist -- would miscount the dwell.
        self._latest_sample = WebcamSample(
            preview_frame=None,
            wrist=WristData(np.zeros(3), np.array([0.0, 0.0, 0.0, 1.0]), self._last_fist, False),
            landmarks=LandmarksData(np.zeros((21, 3)), False),
            observed_at_s=None,
            frame_id=None,
        )
        self._next_frame_id = 0
        self._has_frame = False

    def process_hands(self, right, left, observed_at_s=None, frame_id=None):
        """Return (WristData, LandmarksData) from per-hand detector outputs.

        Each hand arg is (joint_pos[21,3] MANO, image_xy[21,2], wrist_rot[3,3]) or None.
        The clutch is updated from the LEFT hand whether or not the RIGHT hand is present.
        Timing metadata is the host-monotonic frame read-completion observation, not exposure time.
        """
        if left is not None:
            self._last_fist = "closed" if detect_fist(np.asarray(left[0], dtype=float)) else "open"
        fist_state = self._last_fist

        if right is None:
            return (WristData(np.zeros(3), np.array([0.0, 0.0, 0.0, 1.0]), fist_state, False),
                    LandmarksData(
                        np.zeros((21, 3)),
                        False,
                        observed_at_s=observed_at_s,
                        frame_id=frame_id,
                    ))

        joint_pos, image_xy, wrist_rot = right
        joint_pos = np.asarray(joint_pos, dtype=float)
        image_xy = np.asarray(image_xy, dtype=float)
        position, quat = self.wrist_estimator.estimate(wrist_rot, image_xy, self.image_shape)

        # Per-landmark depth only exists on a depth-bearing strategy (OAK); the monocular
        # one has no such method and the field stays None.
        depth_m = None
        depth_strategy = getattr(self.wrist_estimator, "depth", None)
        sample_landmark_depths = getattr(depth_strategy, "sample_landmark_depths", None)
        if callable(sample_landmark_depths):
            depth_m = sample_landmark_depths(image_xy, self.image_shape)

        return (WristData(position, quat, fist_state, True),
                LandmarksData(
                    joint_pos,
                    True,
                    image_xy=image_xy,
                    depth_m=depth_m,
                    observed_at_s=observed_at_s,
                    frame_id=frame_id,
                ))

    def _publish_sample(self, preview_frame, wrist, landmarks) -> None:
        sample = WebcamSample(
            preview_frame=preview_frame,
            wrist=wrist,
            landmarks=landmarks,
            observed_at_s=landmarks.observed_at_s,
            frame_id=landmarks.frame_id,
        )
        with self._lock:
            self._latest_sample = sample
            self._has_frame = wrist.valid

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
            self._thread = None
        # Each handle is dropped as it is released, so a second stop() is a no-op.
        # The recorder registers this on an ExitStack and also calls it from its own
        # teardown, and MediaPipe's Hands.close() raises "Closing SolutionBase._graph
        # which is already None" the second time -- which turned an ordinary ESC into
        # a traceback after the episode was already on disk.
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        if self._oak is not None:
            self._oak.stop()
            self._oak = None
        if self._hands is not None:
            self._hands.close()
            self._hands = None

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
            observed_at_s = time.perf_counter()
            frame_id = self._next_frame_id
            self._next_frame_id += 1
            wrist, landmarks = self.process_hands(
                right=right, left=left, observed_at_s=observed_at_s, frame_id=frame_id,
            )
            annotated = frame_bgr.copy()   # draw landmarks for the preview window
            if results.multi_hand_landmarks:
                for hand_lms in results.multi_hand_landmarks:
                    self._mp_draw.draw_landmarks(annotated, hand_lms, self._mp_conns)
            self._publish_sample(annotated, wrist, landmarks)

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
            observed_at_s = time.perf_counter()
            frame_id = self._next_frame_id
            self._next_frame_id += 1
            wrist, landmarks = self.process_hands(
                right=right, left=left, observed_at_s=observed_at_s, frame_id=frame_id,
            )
            annotated = rgb_bgr.copy()   # draw landmarks for the preview window
            if results.multi_hand_landmarks:
                for hand_lms in results.multi_hand_landmarks:
                    self._mp_draw.draw_landmarks(annotated, hand_lms, self._mp_conns)
            self._publish_sample(annotated, wrist, landmarks)

    def latest_frame(self):
        """Latest annotated BGR hand-cam frame for a preview window (None until first frame)."""
        with self._lock:
            return self._latest_sample.preview_frame

    def latest_sample(self) -> WebcamSample:
        """Return one locked frame/wrist/landmark publication from a single iteration."""
        with self._lock:
            return self._latest_sample

    def latest(self):
        with self._lock:
            return self._latest_sample.wrist, self._latest_sample.landmarks

    @property
    def has_frame(self) -> bool:
        with self._lock:
            return self._has_frame
