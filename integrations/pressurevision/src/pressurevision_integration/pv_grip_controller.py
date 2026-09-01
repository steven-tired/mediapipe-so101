"""The whole PressureVision grip layer, outside the EE controller.

`WebcamEEController` used to inline all of this: the range mapper, the
adjustment lock, the proposal state machine, the closure limiter and the
shadow/apply split. That made every PV behaviour reachable only through a fake
robot and a full IK pipeline, and it put PressureVision inside the core package
that is supposed to run without it.

Here it is one object that owns the PV pipeline and nothing else:

    reading -> range/relative mapper -> PVAdjustmentLock -> proposal machine
            -> shadow-or-apply -> closure limiter -> commanded position

The controller keeps arm motion, the pinch->grip base command, and grasp/release
authority. It hands this runtime a base command and gets a gripper position
back, so PressureVision can still make the grip wrong but can never open the
claw or move the arm.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from lerobot_teleoperator_so101_webcam.grip.proposal import (
    GRIP_CLOSE_ALPHA,
    GRIP_OPEN_ALPHA,
    GRIP_OVERDRIVE,
    MAX_PRESSURE_GRIP_STEP,
    PRESSURE_MIN_QUALITY,
    PressureControlDecision,
    PressureProposalStateMachine,
)
from lerobot_teleoperator_so101_webcam.gripper_hardware import (
    GripperClosureLimiter,
    GripperClosureLimits,
)

from .adjustment_lock import PVAdjustmentLock
from .protocol import inactive_pressure
from .pv_object_profile import PressureVisionObjectProfile
from .pv_relative_mapping import (
    RELATIVE_PRESSURE_LOW_PASS_HZ,
    SOFT_DIRECT_PRESSURE_LOW_PASS_HZ,
    PressureRangeMapper,
    RelativePressureMapper,
)

__all__ = [
    "PV_MAPPINGS",
    "SOFT_PRECISE_PRESSURE_ZERO_POS",
    "SOFT_PRECISE_PRESSURE_ONE_POS",
    "CARTON_SPAN_PRESSURE_ZERO_POS",
    "CARTON_SPAN_PRESSURE_ONE_POS",
    "PVGripDecision",
    "PressureVisionGripRuntime",
    "pressure_range_mapping_contract",
]

#: Calibrated position spans, measured on the objects they are named for.
SOFT_PRECISE_PRESSURE_ZERO_POS = 28.0
SOFT_PRECISE_PRESSURE_ONE_POS = 22.0
CARTON_SPAN_PRESSURE_ZERO_POS = 32.0
CARTON_SPAN_PRESSURE_ONE_POS = 20.0

#: `absolute` and `relative` predate the range mapper and are kept because
#: recorded episodes reference them by name.
PV_MAPPINGS = (
    "absolute",
    "relative",
    "soft_direct",
    "soft_precise",
    "carton_span",
    "hard_profile",
)

#: Mappings whose span is fixed by the mapping itself, not by an object profile.
_PROFILE_FREE_MAPPINGS = ("soft_direct", "soft_precise", "carton_span")


def pressure_range_mapping_contract(
    pv_mapping: str,
    *,
    object_profile: PressureVisionObjectProfile | None = None,
    max_grip_step: float = MAX_PRESSURE_GRIP_STEP,
) -> dict[str, float | bool | str] | None:
    """Serializable range-mapping parameters shared by control and evidence.

    Control builds its mapper from this dict and the recorder writes the same
    dict into the episode, so a recorded grip can be reproduced from the episode
    alone. Returns None for the mappings that predate the range mapper.
    """
    if pv_mapping == "soft_direct":
        positions = (100.0, 100.0, 0.0)
        cutoff_hz, stabilize = SOFT_DIRECT_PRESSURE_LOW_PASS_HZ, False
    elif pv_mapping == "soft_precise":
        positions = (100.0, SOFT_PRECISE_PRESSURE_ZERO_POS, SOFT_PRECISE_PRESSURE_ONE_POS)
        cutoff_hz, stabilize = SOFT_DIRECT_PRESSURE_LOW_PASS_HZ, False
    elif pv_mapping == "carton_span":
        positions = (100.0, CARTON_SPAN_PRESSURE_ZERO_POS, CARTON_SPAN_PRESSURE_ONE_POS)
        cutoff_hz, stabilize = SOFT_DIRECT_PRESSURE_LOW_PASS_HZ, False
    elif pv_mapping == "hard_profile":
        if object_profile is None:
            raise ValueError("hard_profile mapping contract requires an object profile")
        positions = (
            float(object_profile.open_pos),
            float(object_profile.light_pos),
            float(object_profile.hard_pos),
        )
        cutoff_hz, stabilize = RELATIVE_PRESSURE_LOW_PASS_HZ, True
    else:
        return None
    release_pos, pressure_zero_pos, pressure_one_pos = positions
    return {
        "mapping": pv_mapping,
        "release_pos": release_pos,
        "pressure_zero_pos": pressure_zero_pos,
        "pressure_one_pos": pressure_one_pos,
        "cutoff_hz": float(cutoff_hz),
        "stabilize": stabilize,
        "max_grip_step_per_control_frame": float(max_grip_step),
    }


@dataclass(frozen=True)
class PVGripDecision:
    """One frame of PV grip output."""

    #: The position to command this frame.
    actual_gripper: float
    #: The full record for telemetry and the episode.
    control: PressureControlDecision
    #: True when `actual_gripper` came from the legacy pinch path because PV is
    #: running in shadow: the caller must keep smoothing that path itself.
    shadow: bool


class PressureVisionGripRuntime:
    """Own the PV grip pipeline for one teleop or deploy session."""

    def __init__(
        self,
        pressure_source,
        *,
        initial_gripper: float,
        middle_gripper: float,
        pressure_shadow: bool = False,
        pressure_apply: bool = False,
        object_profile: PressureVisionObjectProfile | None = None,
        object_profile_sha256: str | None = None,
        trial_protocol=None,
        pv_mapping: str = "absolute",
        pressure_max_grip_step: float | None = None,
        gripper_closure_limits: GripperClosureLimits | None = None,
        grip_overdrive: float = GRIP_OVERDRIVE,
    ) -> None:
        if pressure_source is None:
            raise ValueError("PressureVisionGripRuntime requires a pressure_source")
        # Applying PV to a real gripper stays opt-in: the default of an
        # unspecified flag must be "observe", never "squeeze".
        if not pressure_shadow and not pressure_apply:
            raise ValueError(
                "pressure apply is disabled until Stage 3 physical authorization; "
                "pressure_apply must be explicit"
            )
        if pv_mapping not in PV_MAPPINGS:
            raise ValueError(f"unknown pv_mapping {pv_mapping!r}")
        if pressure_apply and object_profile is None and pv_mapping not in _PROFILE_FREE_MAPPINGS:
            raise ValueError(
                "pressure_apply requires a hard object profile unless pv_mapping is "
                "soft_direct, soft_precise, or carton_span"
            )
        if pv_mapping in ("relative", "hard_profile") and object_profile is None:
            raise ValueError(f"{pv_mapping} PV mapping requires an object profile")
        if pv_mapping in _PROFILE_FREE_MAPPINGS and object_profile is not None:
            raise ValueError(f"{pv_mapping} PV mapping does not use an object profile")

        self.pressure_source = pressure_source
        self.pressure_shadow = bool(pressure_shadow)
        self.pressure_apply = bool(pressure_apply)
        self.object_profile = object_profile
        self.object_profile_sha256 = object_profile_sha256
        self.trial_protocol = trial_protocol
        self.pv_mapping = pv_mapping
        self.grip_overdrive = float(grip_overdrive)
        self.middle_gripper = float(middle_gripper)
        self.pressure_max_grip_step = (
            MAX_PRESSURE_GRIP_STEP
            if pressure_max_grip_step is None
            else float(pressure_max_grip_step)
        )
        if not math.isfinite(self.pressure_max_grip_step) or self.pressure_max_grip_step <= 0.0:
            raise ValueError("pressure_max_grip_step must be positive")

        self._closure_limiter = (
            None
            if gripper_closure_limits is None
            else GripperClosureLimiter(gripper_closure_limits)
        )
        if self._closure_limiter is not None and not self.pressure_apply:
            raise ValueError("gripper_closure_limits require pressure_apply")

        self.last_pressure = None
        self.last_pressure_control: PressureControlDecision | None = None
        self.last_relative_grip = None
        self._latest_gripper_telemetry = None

        self._relative_mapper = (
            RelativePressureMapper(
                max_closure=float(object_profile.light_pos - object_profile.hard_pos),
                cutoff_hz=RELATIVE_PRESSURE_LOW_PASS_HZ,
                stabilize=True,
            )
            if object_profile is not None and pv_mapping == "relative"
            else None
        )
        self._relative_target: float | None = None
        self._relative_fallback_target = float(initial_gripper)

        self.mapping_contract = pressure_range_mapping_contract(
            pv_mapping,
            object_profile=object_profile,
            max_grip_step=self.pressure_max_grip_step,
        )
        self._range_mapper = None
        if self.mapping_contract is not None:
            contract = self.mapping_contract
            self._range_mapper = PressureRangeMapper(
                release_pos=float(contract["release_pos"]),
                pressure_zero_pos=float(contract["pressure_zero_pos"]),
                pressure_one_pos=float(contract["pressure_one_pos"]),
                cutoff_hz=float(contract["cutoff_hz"]),
                stabilize=bool(contract["stabilize"]),
            )
        self._range_target: float | None = None
        self._range_fallback_target = (
            None if self._range_mapper is None else self._range_mapper.release_pos
        )

        self._lock = PVAdjustmentLock()
        self._proposal = PressureProposalStateMachine(
            initial_gripper=float(initial_gripper),
            fallback_overdrive=self.grip_overdrive,
            # A PV object profile consumes the calibrated continuous value.
            # Area/offset quality remains telemetry, while only unavailable or
            # stale packets block the proposal. IR keeps its existing quality
            # gate because it has no equivalent contact-gated scalar contract.
            min_quality=(
                0.0
                if object_profile is not None or self._range_mapper is not None
                else PRESSURE_MIN_QUALITY
            ),
            max_grip_step=self.pressure_max_grip_step,
            close_alpha=1.0 if self._mapped else GRIP_CLOSE_ALPHA,
            open_alpha=1.0 if self._mapped else GRIP_OPEN_ALPHA,
            target_resolver=self._target_resolver(),
            resolve_baseline_target=self._mapped,
        )

    # -- construction helpers -------------------------------------------------

    @property
    def _mapped(self) -> bool:
        """True when a mapper already owns smoothing, so the proposal must not."""
        return self._relative_mapper is not None or self._range_mapper is not None

    def _target_resolver(self):
        object_profile = self.object_profile
        if object_profile is None and self._range_mapper is None:
            return None

        def resolve(reading):
            if self._range_mapper is not None:
                if self._range_target is not None:
                    return self._range_target
                return self._range_fallback_target
            if self._relative_mapper is not None:
                if self._relative_target is not None:
                    return self._relative_target
                return self._relative_fallback_target
            if bool(getattr(reading, "active", False)):
                return object_profile.target_for_pressure(reading.pressure_0_1)
            level = getattr(reading, "level", None)
            n_levels = getattr(reading, "n_levels", None)
            if level is None or n_levels is None:
                raise ValueError("pressure reading has no wire level")
            return object_profile.target_for_level(int(level), int(n_levels))

        return resolve

    # -- telemetry ------------------------------------------------------------

    @property
    def pressure_state(self) -> str:
        return "legacy" if self._proposal is None else self._proposal.state

    @property
    def pressure_raw_gripper(self) -> float | None:
        return None if self._proposal is None else self._proposal.raw_gripper

    @property
    def pressure_grip_smoothed(self) -> float | None:
        return None if self._proposal is None else self._proposal.smoothed_gripper

    @property
    def adjustment_locked(self) -> bool:
        return self._lock.locked

    @property
    def adjustment_anchor_target(self) -> float | None:
        return self._lock.anchor_target

    @property
    def adjustment_event(self) -> str | None:
        return self._lock.event

    @property
    def adjustment_state(self) -> str | None:
        if self._range_mapper is None:
            return None
        grip_active = not (
            self.last_relative_grip is None
            or self.last_relative_grip.status == "right_grasp_inactive"
        )
        return self._lock.state(grip_active=grip_active)

    @property
    def adjustment_teacher(self) -> float | None:
        """The latched grip as a 0..1 fraction of the span — the training label."""
        anchor = self._lock.anchor_target
        if anchor is None or self._range_mapper is None:
            return None
        target = (
            anchor
            if self._lock.locked or self._range_target is None
            else self._range_target
        )
        span = self._range_mapper.pressure_zero_pos - self._range_mapper.pressure_one_pos
        return min(1.0, max(0.0, (self._range_mapper.pressure_zero_pos - target) / span))

    # -- lifecycle ------------------------------------------------------------

    def seed(self, gripper: float, *, reset_smoothed: bool = True) -> None:
        self._proposal.seed(float(gripper), reset_smoothed=reset_smoothed)

    def set_telemetry(self, telemetry) -> None:
        """Supply the latest rate-limited motor readback to relative PV mapping."""
        self._latest_gripper_telemetry = telemetry

    def reset_mappers(self, *, event: str | None = None) -> None:
        """Forget the grasp: mappers, adjustment lock, closure limiter."""
        self.last_pressure = None
        if self._relative_mapper is not None:
            self._relative_mapper.reset()
            self._relative_target = None
        if self._range_mapper is not None:
            self._range_mapper.reset()
            self._range_target = None
            self._range_fallback_target = self._range_mapper.release_pos
        self._lock.clear(event=event)
        self.last_relative_grip = None
        if self._closure_limiter is not None:
            self._closure_limiter.reset()

    def reset_pressure_source(self) -> str | None:
        """Reset the sender, returning a reason string if it raised."""
        reset = getattr(self.pressure_source, "reset", None)
        if not callable(reset):
            return None
        try:
            reset()
        except Exception as exc:
            return f"pressure_reset_error:{type(exc).__name__}:{exc}"
        return None

    def reset_control(
        self,
        base_gripper: float,
        actual_gripper: float,
        transition: str,
        *,
        reason: str | None = None,
    ) -> PressureControlDecision:
        proposal = self._proposal.reset(
            base_gripper,
            transition=transition,
            middle_gripper=self.middle_gripper,
            reason=reason,
        )
        self.last_pressure_control = proposal.with_actual(actual_gripper)
        return self.last_pressure_control

    def close(self) -> None:
        self.reset_mappers()
        close = getattr(self.pressure_source, "close", None)
        self.pressure_source = None
        if callable(close):
            close()

    # -- per frame ------------------------------------------------------------

    def observe(self, landmarks, *, pinch: float, enabled: bool):
        """Read the sender. A raising or silent sender becomes an inactive reading."""
        try:
            reading = self.pressure_source.update(landmarks, pinch=pinch, enabled=enabled)
        except Exception:
            return inactive_pressure("pressure_error", available=False)
        if reading is None:
            return inactive_pressure("pressure_unavailable", available=False)
        return reading

    def update(
        self,
        *,
        base_gripper: float,
        landmarks,
        pinch: float,
        enabled: bool,
        current_command: float,
        observed_at_s: float,
        smooth_legacy=None,
    ) -> PVGripDecision:
        """Run one control frame and return the position to command.

        `smooth_legacy(raw)` is the caller's own pinch-path smoother; it is used
        only in shadow mode, where PV computes a proposal but the legacy command
        is what actually reaches the gripper.
        """
        self._lock.event = None
        pressure = self.observe(landmarks, pinch=pinch, enabled=enabled)
        self.last_pressure = pressure

        if self._relative_mapper is not None:
            self._update_relative(base_gripper, pressure, observed_at_s)

        control_pressure = pressure
        if self._range_mapper is not None:
            control_pressure = self._update_range(
                base_gripper, pressure, current_command, observed_at_s
            )

        proposal = self._proposal.update(base_gripper, control_pressure)
        if self.pressure_shadow:
            if smooth_legacy is None:
                raise ValueError("shadow mode needs the caller's legacy smoother")
            actual_grip = float(smooth_legacy())
        else:
            actual_grip = float(proposal.proposed_gripper)

        limited = None
        if self._closure_limiter is not None:
            release_requested = bool(
                self.last_relative_grip is not None
                and self.last_relative_grip.status == "right_grasp_inactive"
            )
            limited = self._closure_limiter.update(
                requested_pos=actual_grip,
                last_commanded_pos=current_command,
                telemetry=self._latest_gripper_telemetry,
                observed_at_s=observed_at_s,
                release_requested=release_requested,
            )
            actual_grip = float(limited.actual_pos)

        if limited is not None and limited.fault_latched:
            control = PressureControlDecision(
                base_gripper=float(proposal.base_gripper),
                proposed_gripper=float(proposal.proposed_gripper),
                actual_gripper=actual_grip,
                state="closure_limited",
                fault_latched=True,
                reason=limited.reason,
            )
        else:
            control = proposal.with_actual(actual_grip)
        self.last_pressure_control = control
        return PVGripDecision(
            actual_gripper=actual_grip,
            control=control,
            shadow=self.pressure_shadow,
        )

    def _update_relative(self, base_gripper, pressure, observed_at_s) -> None:
        latest = self._latest_gripper_telemetry
        previous_reference = (
            None if self.last_relative_grip is None else self.last_relative_grip.reference_pos
        )
        self.last_relative_grip = self._relative_mapper.update(
            base_gripper_pos=base_gripper,
            pressure=pressure,
            observed_gripper_pos=(None if latest is None else latest.observed_gripper_pos),
            control_observed_at_s=observed_at_s,
            motor_observed_at_s=(None if latest is None else getattr(latest, "observed_at_s", None)),
        )
        if previous_reference is None and self.last_relative_grip.reference_pos is not None:
            # The first frame that establishes a contact reference: without this
            # the proposal would slew toward it from wherever it started.
            self._proposal.seed(self.last_relative_grip.reference_pos, reset_smoothed=True)
        self._relative_target = self.last_relative_grip.target_pos
        previous_target = self.pressure_raw_gripper
        self._relative_fallback_target = float(
            base_gripper if previous_target is None else previous_target
        )

    def _update_range(self, base_gripper, pressure, current_command, observed_at_s):
        self.last_relative_grip = self._range_mapper.update(
            base_gripper_pos=base_gripper,
            pressure=pressure,
            control_observed_at_s=observed_at_s,
        )
        live_target = self.last_relative_grip.target_pos
        released = self.last_relative_grip.status == "right_grasp_inactive"
        previous_target = self._range_target
        if released:
            self._lock.clear(event="explicit_release_clear")
            self._range_target = live_target
        else:
            self._range_target = self._lock.update(
                live_target=live_target,
                pressure_active=bool(getattr(pressure, "active", False)),
                observed_at_s=observed_at_s,
                current_command=current_command,
                previous_target=previous_target,
            )
            if self._lock.event == "lock":
                # A fresh latch is where the proposal must restart from, or it
                # would slew back out of the position it just latched at.
                self._range_fallback_target = self._lock.anchor_target
                self._proposal.seed(self._lock.anchor_target, reset_smoothed=True)
        raw_gripper = self.pressure_raw_gripper
        if self._range_target is not None:
            self._range_fallback_target = self._range_target
        elif raw_gripper is not None:
            self._range_fallback_target = float(raw_gripper)
        if released:
            # Right-hand release must remain available even when the PV sender
            # is stale or unavailable. It is an explicit binary command, not a
            # pressure inference.
            return inactive_pressure(
                "baseline",
                available=True,
                quality=1.0,
                roi_mode=getattr(pressure, "roi_mode", None),
            )
        return pressure
