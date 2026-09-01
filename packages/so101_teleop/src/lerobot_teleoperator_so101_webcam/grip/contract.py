"""The seam between hand tracking and gripper command.

MediaPipe always owns arm motion and grasp/release authority. A gripper
controller only decides *how far* to close while a grasp is active. This keeps
the optional PressureVision path from ever being able to open the gripper.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

__all__ = ["GripInput", "GripperController", "STALE_AFTER_S", "make_grip_input"]

#: A grip signal older than this is treated as invalid. Staleness is decided
#: here, at the composition layer, so controllers stay clock-free.
STALE_AFTER_S = 0.5


@dataclass(frozen=True, kw_only=True)
class GripInput:
    """One frame of grip intent.

    Keyword-only on purpose: four of the five fields are bool/float and a
    positional call site would silently reorder them.
    """

    #: MediaPipe says a grasp should be held this frame.
    grasp_active: bool
    #: MediaPipe explicitly released (clutch). The only thing that opens the gripper.
    explicit_release: bool
    #: Normalised grip intensity in [0, 1]; 1 = grip as hard as possible.
    #: None when this frame carries no usable measurement.
    severity: float | None
    #: False when the measurement is missing, out of range, or stale.
    valid: bool
    #: When the measurement was taken, seconds on the caller's clock.
    observed_at_s: float


@runtime_checkable
class GripperController(Protocol):
    def reset(self) -> None: ...

    def step(self, grip: GripInput, actual_pos: float) -> float: ...


def make_grip_input(
    *,
    grasp_active: bool,
    explicit_release: bool,
    severity: float | None,
    observed_at_s: float,
    now_s: float,
    stale_after_s: float = STALE_AFTER_S,
) -> GripInput:
    """Build a GripInput, stamping `valid` from measurement age and range.

    This is the only place `observed_at_s` is read. A sender that goes silent
    therefore produces `valid=False` frames, which every controller holds on —
    a dead sender keeps the gripper where it is rather than dropping the object.
    """
    fresh = (now_s - observed_at_s) <= stale_after_s
    in_range = severity is not None and 0.0 <= severity <= 1.0
    return GripInput(
        grasp_active=grasp_active,
        explicit_release=explicit_release,
        severity=severity,
        valid=bool(fresh and in_range),
        observed_at_s=observed_at_s,
    )
