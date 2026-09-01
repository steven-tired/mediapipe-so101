"""PressureVision as a grip-strength modulator.

MediaPipe owns arm motion and grasp/release. This adapter only decides how far
to close inside a configured span while a MediaPipe grasp is already active, so
a PressureVision fault can make the grip wrong but can never drop the object or
open the claw.
"""

from __future__ import annotations

from lerobot_teleoperator_so101_webcam.grip.contract import GripInput
from lerobot_teleoperator_so101_webcam.grip.mediapipe import RELEASE_POS

__all__ = ["PressureVisionGripperController"]


class PressureVisionGripperController:
    """Maps PressureVision severity onto a gripper-position span.

    `zero_pos` is the loosest PV-commanded grip and `one_pos` the firmest; on an
    SO-101 gripper a lower position value is more closed, so `one_pos < zero_pos`.
    The span comes from the object profile, not from this class.
    """

    def __init__(
        self,
        *,
        zero_pos: float,
        one_pos: float,
        release_pos: float = RELEASE_POS,
    ) -> None:
        self.zero_pos = float(zero_pos)
        self.one_pos = float(one_pos)
        self.release_pos = float(release_pos)
        self.current_command: float | None = None

    def reset(self) -> None:
        self.current_command = None

    def step(self, grip: GripInput, actual_pos: float) -> float:
        # Explicit MediaPipe release is the ONLY transition that opens the claw.
        # It is checked before validity: a release must work even with PV dead.
        if grip.explicit_release:
            self.reset()
            return self.release_pos

        # Missing, out-of-range, stale, or no grasp: hold. A silent sender keeps
        # the current clamp indefinitely rather than slackening.
        if not grip.valid or not grip.grasp_active or grip.severity is None:
            return self.current_command if self.current_command is not None else actual_pos

        severity = min(1.0, max(0.0, float(grip.severity)))
        span = self.zero_pos - self.one_pos
        self.current_command = self.zero_pos - severity * span
        return self.current_command
