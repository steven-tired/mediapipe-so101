import logging

import numpy as np
from lerobot.teleoperators.teleoperator import Teleoperator

from .config_so101_webcam_ee import SO101WebcamEEConfig
from .ee_control import EE_ACTION_KEYS, ee_action_from_hand

logger = logging.getLogger(__name__)

_THUMB_TIP = 4
_INDEX_TIP = 8


class SO101WebcamEE(Teleoperator):
    """Webcam end-effector teleoperator for the SO-101.

    Emits EE-delta actions for LeRobot's IK pipeline. Right hand drives the gripper
    pose differentially (reference latched when the clutch engages); closed LEFT fist
    or lost tracking disables motion. Pinch controls the gripper.

    Opens its own `WebcamSource` directly (see `connect()`) rather than sharing a
    `WebcamSourceManager`, so it cannot be connected at the same time as the
    `so101_webcam` joint teleoperator on the same camera index.
    """

    config_class = SO101WebcamEEConfig
    name = "so101_webcam_ee"

    def __init__(self, config: SO101WebcamEEConfig, source=None):
        super().__init__(config)
        self.config = config
        self._source = source
        self._connected = False
        self._ref = None
        self._prev_enabled = False

    @property
    def action_features(self) -> dict:
        return {k: float for k in EE_ACTION_KEYS}

    @property
    def feedback_features(self) -> dict:
        return {}

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def is_calibrated(self) -> bool:
        return True

    def connect(self, calibrate: bool = True) -> None:
        if self._source is None:
            from webcam_input.depth import ScaleDepthStrategy
            from webcam_input.webcam_source import WebcamSource
            from webcam_input.wrist_estimator import WebcamWristEstimator

            est = WebcamWristEstimator(ScaleDepthStrategy(), workspace_size_m=self.config.workspace_size_m)
            self._source = WebcamSource(est)
        self._source.start(self.config.camera_index)
        self._ref = None
        self._prev_enabled = False
        self._connected = True
        logger.info("%s connected.", self)

    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        pass

    def get_action(self) -> dict:
        # NOTE: this teleoperator latches its own wrist reference on the clutch rising
        # edge (see `connect()`/below) and emits target_x/y/z as displacement-in-meters
        # since that latch. The downstream `EEReferenceAndDelta` MUST therefore be wired
        # with end_effector_step_sizes={"x": 1.0, "y": 1.0, "z": 1.0} so its own latch
        # point stays in lockstep with ours (no additional per-step scaling).
        if not self._connected:
            raise RuntimeError(f"{self} is not connected.")
        wrist, landmarks = self._source.latest()

        enabled = bool(wrist.valid and wrist.fist_state != "closed")
        if enabled and (not self._prev_enabled or self._ref is None):
            self._ref = np.asarray(wrist.position, dtype=float).copy()   # latch reference
        if not enabled:
            self._ref = None
            displacement = np.zeros(3)
        else:
            displacement = np.asarray(wrist.position, dtype=float) - self._ref

        lm = np.asarray(landmarks.landmarks, dtype=float)
        pinch = float(np.linalg.norm(lm[_THUMB_TIP] - lm[_INDEX_TIP]))

        self._prev_enabled = enabled
        return ee_action_from_hand(displacement, pinch, enabled, self.config)

    def send_feedback(self, feedback: dict) -> None:
        pass

    def disconnect(self) -> None:
        if self._source is not None:
            self._source.stop()
        self._connected = False
        logger.info("%s disconnected.", self)
