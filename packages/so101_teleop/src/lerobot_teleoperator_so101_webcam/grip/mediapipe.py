"""The default gripper controller: MediaPipe pinch drives the claw.

This is the validated path lifted out of `WebcamEEController.step` unchanged.
The constants, the overdrive, and the asymmetric smoothing are the tuned values
that path has always used; do not retune them here without hardware evidence.
"""

from __future__ import annotations

from ..ee_control import GRIP_RELEASE_ALPHA, grip_ratchet
from .contract import GripInput

__all__ = [
    "GRIP_CLOSE_ALPHA",
    "GRIP_OPEN_ALPHA",
    "GRIP_OVERDRIVE",
    "GRIP_MODES",
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

# "tracked" is the validated arm and must stay the default: the recorded
# datasets and the trained policies assume it, so switching it silently would
# contaminate its own baseline. "latched" is arm B of the grip comparison — the
# same EMA with the opening weight taken to zero once the operator has
# committed to a grasp (see `ee_control.grip_ratchet`).
GRIP_MODES = ("tracked", "latched")


class MediaPipeGripperController:
    """Maps MediaPipe pinch intensity to a gripper position command."""

    def __init__(
        self,
        *,
        close_alpha: float = GRIP_CLOSE_ALPHA,
        open_alpha: float = GRIP_OPEN_ALPHA,
        overdrive: float = GRIP_OVERDRIVE,
        release_pos: float = RELEASE_POS,
        grip_mode: str = "tracked",
    ) -> None:
        if grip_mode not in GRIP_MODES:
            raise ValueError(f"unknown grip_mode {grip_mode!r}; expected one of {GRIP_MODES}")
        self.close_alpha = close_alpha
        self.open_alpha = open_alpha
        self.overdrive = overdrive
        self.release_pos = release_pos
        self.grip_mode = grip_mode
        self.current_command: float | None = None
        self.latched = False
        self.open_frames = 0

    def reset(self) -> None:
        self.current_command = None
        # Without this the ratchet survives a clutch, so releasing the fist
        # would resume a grasp the operator has already abandoned.
        self.latched = False
        self.open_frames = 0

    def step(self, grip: GripInput, actual_pos: float) -> float:
        # Explicit release is the ONLY transition that opens the claw.
        if grip.explicit_release:
            self.reset()
            return self.release_pos

        # Missing, out-of-range, stale, or no grasp: hold. Never open on absence.
        if not grip.valid or not grip.grasp_active or grip.severity is None:
            return self.current_command if self.current_command is not None else actual_pos

        raw = max(0.0, (1.0 - grip.severity) * 100.0 - self.overdrive)
        if self.grip_mode == "latched":
            self.current_command, self.latched, self.open_frames = grip_ratchet(
                raw,
                self.current_command,
                self.latched,
                self.open_frames,
                close_alpha=self.close_alpha,
                # Not `open_alpha`: the tracked arm's slow open exists to resist
                # loosening, a job the ratchet already does outright. Keeping
                # both only adds lag — measured on foam as a badly lagging
                # release. See GRIP_RELEASE_ALPHA.
                open_alpha=GRIP_RELEASE_ALPHA,
            )
            return self.current_command

        if self.current_command is None:
            self.current_command = raw
        else:
            alpha = self.close_alpha if raw < self.current_command else self.open_alpha
            self.current_command = alpha * raw + (1.0 - alpha) * self.current_command
        return self.current_command
