"""The default gripper controller: MediaPipe pinch drives the claw.

This is the validated path lifted out of `WebcamEEController.step` unchanged.
The constants, the overdrive, and the asymmetric smoothing are the tuned values
that path has always used; do not retune them here without hardware evidence.
"""

from __future__ import annotations

from .contract import GripInput

__all__ = [
    "GRIP_CLOSE_ALPHA",
    "GRIP_OPEN_ALPHA",
    "GRIP_OVERDRIVE",
    "RELEASE_POS",
    "MediaPipeGripperController",
]

# 0 = clamped shut, 100 = fully open.
#
# ASYMMETRIC EMA: close fast and firm, open slow. The slow open is what stops the
# claw loosening when pinch tracking jitters mid-lift — a brief spurious "open"
# reading barely moves the command. GRIP_OVERDRIVE shifts the whole command
# toward closed so a normal (not fully touching) pinch still reaches a firm grip.
GRIP_CLOSE_ALPHA = 0.7
GRIP_OPEN_ALPHA = 0.15
GRIP_OVERDRIVE = 18.0

# Release goes to the calibrated centre, not to 100. `joint_center()` returns
# 50.0 for a RANGE_0_100 gripper and the clutch/ready pose has always used it.
# Full open would be a behaviour change, not a clarification.
RELEASE_POS = 50.0


class MediaPipeGripperController:
    """Maps MediaPipe pinch intensity to a gripper position command."""

    def __init__(
        self,
        *,
        close_alpha: float = GRIP_CLOSE_ALPHA,
        open_alpha: float = GRIP_OPEN_ALPHA,
        overdrive: float = GRIP_OVERDRIVE,
        release_pos: float = RELEASE_POS,
    ) -> None:
        self.close_alpha = close_alpha
        self.open_alpha = open_alpha
        self.overdrive = overdrive
        self.release_pos = release_pos
        self.current_command: float | None = None

    def reset(self) -> None:
        self.current_command = None

    def step(self, grip: GripInput, actual_pos: float) -> float:
        # Explicit release is the ONLY transition that opens the claw.
        if grip.explicit_release:
            self.reset()
            return self.release_pos

        # Missing, out-of-range, stale, or no grasp: hold. Never open on absence.
        if not grip.valid or not grip.grasp_active or grip.severity is None:
            return self.current_command if self.current_command is not None else actual_pos

        raw = max(0.0, (1.0 - grip.severity) * 100.0 - self.overdrive)
        if self.current_command is None:
            self.current_command = raw
        else:
            alpha = self.close_alpha if raw < self.current_command else self.open_alpha
            self.current_command = alpha * raw + (1.0 - alpha) * self.current_command
        return self.current_command
