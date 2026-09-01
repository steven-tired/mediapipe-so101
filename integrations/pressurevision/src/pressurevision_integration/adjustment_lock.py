"""Hold the grip where the operator left it when PressureVision loses contact.

PressureVision infers contact from an image, and that inference drops out — a
finger rolls, something occludes the view, and PV reports no contact for a few
frames. A gripper that followed PV directly would slacken every time that
happened and drop the object.

So the operator squeezes to set the grip force, and when contact is lost for
`PV_ADJUSTMENT_CONFIRM_RELEASE_S` the command **latches** at wherever it is: the
arm keeps holding at that force indefinitely. Re-contact has to persist for
`PV_ADJUSTMENT_RESUME_CONTACT_S` before adjustment resumes, so one flickering
frame cannot unlatch a held object. While adjusting, the command may only go
*tighter* than the anchor, never looser.

The latched anchor is also the training signal: normalised into the gripper's
span it becomes the `teacher` label that `record_so101_pv_ee` writes into the
dataset, which is what `research/train_grip_residual_head.py` learns from.

This is a pure state machine over timestamps and target positions — no robot, no
kinematics, no camera. It was extracted from `WebcamEEController`, where it was
inlined and could only be tested through a fake robot and a full IK pipeline.
"""

from __future__ import annotations

__all__ = [
    "PVAdjustmentLock",
    "PV_ADJUSTMENT_CONFIRM_RELEASE_S",
    "PV_ADJUSTMENT_RESUME_CONTACT_S",
]

#: Contact must stay lost this long before the grip latches. Long enough that a
#: brief dropout mid-squeeze does not freeze the grip the operator is still
#: setting; short enough that letting go reads as "done" rather than a pause.
PV_ADJUSTMENT_CONFIRM_RELEASE_S = 1.0

#: Re-contact must persist this long before a latched grip resumes adjusting.
#: One frame of flicker on a held object must not unlatch it.
PV_ADJUSTMENT_RESUME_CONTACT_S = 0.15


class PVAdjustmentLock:
    """Decide the grip target from live PV plus the contact history."""

    def __init__(
        self,
        *,
        confirm_release_s: float = PV_ADJUSTMENT_CONFIRM_RELEASE_S,
        resume_contact_s: float = PV_ADJUSTMENT_RESUME_CONTACT_S,
    ) -> None:
        self.confirm_release_s = float(confirm_release_s)
        self.resume_contact_s = float(resume_contact_s)
        self.clear()

    # -- state ---------------------------------------------------------------

    def clear(self, *, event: str | None = None) -> None:
        """Forget the grasp. `event` is only recorded if there was one to forget."""
        had_adjustment = bool(
            getattr(self, "contact_seen", False)
            or getattr(self, "anchor_target", None) is not None
            or getattr(self, "release_since_s", None) is not None
        )
        self.locked = False
        self.anchor_target: float | None = None
        self.contact_seen = False
        self.release_since_s: float | None = None
        self.release_elapsed_s: float | None = None
        self.last_contact_at_s: float | None = None
        self.recontact_since_s: float | None = None
        self.event: str | None = event if (event is not None and had_adjustment) else None

    def lock(self, target: float) -> None:
        """Latch at `target`. The caller must also seed its proposal machine."""
        self.anchor_target = float(target)
        self.locked = True
        self.event = "lock"

    def state(self, *, grip_active: bool) -> str:
        """`inactive` | `locked` | `temporary_hold` | `adjusting`, for telemetry."""
        if not grip_active:
            return "inactive"
        if self.locked:
            return "locked"
        if self.release_since_s is not None:
            return "temporary_hold"
        return "adjusting"

    # -- per frame -----------------------------------------------------------

    def update(
        self,
        *,
        live_target: float | None,
        pressure_active: bool,
        observed_at_s: float,
        current_command: float,
        previous_target: float | None,
    ) -> float | None:
        """The grip target for this frame.

        `live_target` is what PV wants right now (None when it has no opinion),
        `current_command` is where the gripper is being commanded — the position
        a latch would anchor at — and `previous_target` is last frame's result.
        """
        if pressure_active:
            return self._with_contact(live_target, observed_at_s)
        return self._without_contact(live_target, observed_at_s, current_command, previous_target)

    def _with_contact(self, live_target, observed_at_s):
        self.contact_seen = True
        resumed_temporary_hold = self.release_since_s is not None
        self.release_since_s = None
        self.release_elapsed_s = None
        self.last_contact_at_s = observed_at_s

        if self.locked:
            if self.recontact_since_s is None:
                self.recontact_since_s = observed_at_s
                self.event = "recontact_started"
            if observed_at_s - self.recontact_since_s < self.resume_contact_s:
                # Not yet convinced this is a real regrasp. Keep holding.
                return self.anchor_target
            self.locked = False
            self.event = "recontact_resume"
        elif resumed_temporary_hold:
            self.event = "contact_resumed"

        self.recontact_since_s = None
        if self.anchor_target is None or live_target is None:
            return live_target
        # Tighter than the anchor is allowed; looser is not.
        return min(self.anchor_target, live_target)

    def _without_contact(self, live_target, observed_at_s, current_command, previous_target):
        self.recontact_since_s = None
        if self.locked:
            return self.anchor_target
        if not self.contact_seen:
            # Nothing has been grasped yet, so there is nothing to hold on to.
            return live_target
        if self.release_since_s is None:
            self.release_since_s = observed_at_s
            self.event = "contact_lost"
        self.release_elapsed_s = observed_at_s - self.release_since_s
        if self.release_elapsed_s >= self.confirm_release_s:
            self.lock(current_command)
            return self.anchor_target
        return previous_target if previous_target is not None else live_target
