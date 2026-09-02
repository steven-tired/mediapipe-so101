"""Drive the PV grip runtime through the core's gripper contract.

`WebcamEEController` knows two things about a gripper: the `GripperController`
protocol (`reset`, `step`) and the optional `observe_frame` hook. The runtime
speaks neither -- it consumes raw landmarks and returns a decision. This adapter
is the translation, and it is the only reason the core never has to import this
package: the controller satisfies both by duck typing.

The severity the contract carries is the pinch map already normalised, so
`(1 - severity) * 100` is exactly the base command the runtime expects -- the
overdrive lives in the gripper, downstream of this, which is why the two agree.
"""

from __future__ import annotations

from lerobot_teleoperator_so101_webcam.grip.contract import GripInput
from lerobot_teleoperator_so101_webcam.grip.mediapipe import RELEASE_POS

from .pv_grip_controller import PressureVisionGripRuntime

__all__ = ["PVGripAdapter"]


class PVGripAdapter:
    """A `GripperController` backed by `PressureVisionGripRuntime`."""

    def __init__(self, runtime: PressureVisionGripRuntime, *, release_pos: float = RELEASE_POS):
        self.runtime = runtime
        self.release_pos = float(release_pos)
        self._frame = None
        self.current_command: float | None = None

    # -- the controller's optional sensing hook -------------------------------

    def observe_frame(self, landmarks, *, pinch: float, enabled: bool, observed_at_s: float) -> None:
        """Stash the frame for the `step` that follows it in the same tick."""
        self._frame = (landmarks, float(pinch), bool(enabled), float(observed_at_s))

    # -- GripperController ----------------------------------------------------

    def reset(self) -> None:
        self.runtime.reset_mappers(event="middle_reset")
        self._frame = None
        self.current_command = None

    # -- the controller's optional disarm hook --------------------------------

    def disarm(self, grip: GripInput, actual_pos: float, *, transition: str) -> None:
        """The hand is gone: expire the baseline and make PV re-earn control.

        A grasp the operator has walked away from must not resume from the zero
        PV calibrated before they left -- the object may have been put down, the
        hand may have moved on the pad, and the sensor would not know. Rearming
        costs one baseline frame; not rearming costs force applied against a
        scene the sensor can no longer see. This is the behaviour the episodes
        under `local/evidence/` were recorded with.
        """
        self._frame = None
        self.current_command = None
        self.runtime.reset_mappers(event=f"{transition}_reset")
        reason = self.runtime.reset_pressure_source()
        severity = 0.0 if grip.severity is None else float(grip.severity)
        self.runtime.reset_control(
            (1.0 - severity) * 100.0,
            float(actual_pos),
            transition,
            reason=reason,
        )

    def step(self, grip: GripInput, actual_pos: float) -> float:
        # Explicit release is MediaPipe's alone. It must work with the PV sender
        # dead, so it never consults the runtime.
        if grip.explicit_release:
            self.reset()
            return self.release_pos

        if self._frame is None:
            raise RuntimeError("PVGripAdapter.step needs observe_frame first")
        landmarks, pinch, enabled, observed_at_s = self._frame
        self._frame = None

        # Missing, out-of-range, stale, or no grasp: hold. Never open on absence
        # -- the same rule the MediaPipe controller follows.
        if not grip.valid or not grip.grasp_active or grip.severity is None:
            return self.current_command if self.current_command is not None else actual_pos

        decision = self.runtime.update(
            base_gripper=(1.0 - float(grip.severity)) * 100.0,
            landmarks=landmarks,
            pinch=pinch,
            enabled=enabled,
            current_command=float(actual_pos),
            observed_at_s=observed_at_s,
        )
        self.current_command = float(decision.actual_gripper)
        return self.current_command
