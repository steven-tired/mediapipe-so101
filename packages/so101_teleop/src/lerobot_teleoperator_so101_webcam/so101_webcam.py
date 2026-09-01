import logging

from lerobot.teleoperators.teleoperator import Teleoperator

from .config_so101_webcam import SO101WebcamConfig
from .control import (
    MOTORS,
    REST_ACTION,
    clamp_joints,
    ema,
    rate_limit,
    retarget,
)

logger = logging.getLogger(__name__)


class SO101Webcam(Teleoperator):
    """Webcam hand-tracking teleoperator for the SO-101.

    Right hand drives the arm + gripper; a closed LEFT fist is the clutch (pause).
    """

    config_class = SO101WebcamConfig
    name = "so101_webcam"

    def __init__(self, config: SO101WebcamConfig, source_manager=None):
        super().__init__(config)
        self.config = config
        self._manager = source_manager
        self._connected = False
        self._last_action = dict(REST_ACTION)

    @property
    def action_features(self) -> dict:
        return {f"{m}.pos": float for m in MOTORS}

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
        if self._manager is None:
            from webcam_input.depth import ScaleDepthStrategy
            from webcam_input.webcam_source import WebcamSource
            from webcam_input.webcam_source_manager import WebcamSourceManager
            from webcam_input.wrist_estimator import WebcamWristEstimator

            estimator = WebcamWristEstimator(
                ScaleDepthStrategy(), workspace_size_m=self.config.workspace_size_m
            )
            source = WebcamSource(estimator)
            self._manager = WebcamSourceManager(source=source, camera_index=self.config.camera_index)

        self._manager.register_teleoperator(self.config, self.name)
        self._last_action = dict(REST_ACTION)
        self._connected = True
        logger.info("%s connected.", self)

    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        pass

    def get_action(self) -> dict:
        if not self._connected:
            raise RuntimeError(f"{self} is not connected.")

        wrist, _ = self._manager.get_wrist_data()
        landmarks, _ = self._manager.get_landmarks_data()

        # Clutch (closed left fist) or lost tracking -> hold last safe action.
        if (not wrist.valid) or wrist.fist_state == "closed":
            return dict(self._last_action)

        target = retarget(wrist.position, wrist.quaternion, landmarks.landmarks, self.config)
        target = ema(target, self._last_action, self.config.smoothing)
        target = rate_limit(target, self._last_action, self.config.max_delta)
        target = clamp_joints(target)
        self._last_action = target
        return dict(target)

    def send_feedback(self, feedback: dict) -> None:
        pass

    def disconnect(self) -> None:
        if self._manager is not None:
            self._manager.unregister_teleoperator(self.name)
        self._connected = False
        logger.info("%s disconnected.", self)
