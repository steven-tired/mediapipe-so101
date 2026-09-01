"""Hold the arm until the operator's hand has been continuously visible.

A half-tracked hand should not be able to jog the arm while a camera is still
settling, so the gate measures how long the hand has been *continuously* seen
and the caller keeps the arm locked until that reaches the dwell period. Any
dropout resets the clock — this is a continuity requirement, not a total.

Pure timing logic: it takes a validity flag and a timestamp, and knows nothing
about which camera produced them. It was written for the IR branch, which is
why it lived there, but the PressureVision recorder needs the same gate.
"""

from __future__ import annotations

#: Wrist-roll span, in degrees, that still counts as the same continuous hold.
MAX_WRIST_ROLL_RANGE_DEG = 45.0

#: Seconds the hand must stay visible before the arm unlocks.
HAND_STARTUP_DWELL_S = 3.0

__all__ = ["MAX_WRIST_ROLL_RANGE_DEG", "HAND_STARTUP_DWELL_S", "ContinuousHandStartupGate"]


class ContinuousHandStartupGate:
    required_s: float = HAND_STARTUP_DWELL_S
    detected_since_s: float | None = None

    def update(self, *, hand_valid: bool, observed_at_s: float) -> float:
        if not hand_valid:
            self.detected_since_s = None
            return 0.0
        if self.detected_since_s is None:
            self.detected_since_s = float(observed_at_s)
        return max(0.0, float(observed_at_s) - self.detected_since_s)
