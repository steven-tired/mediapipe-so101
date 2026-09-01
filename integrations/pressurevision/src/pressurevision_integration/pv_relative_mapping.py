"""Pure per-grasp relative PressureVision mapping for shadow and bounded apply."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math

import numpy as np

from lerobot_teleoperator_so101_webcam.ee_control import GRIP_LATCH_ENTER, GRIP_LATCH_EXIT


RELATIVE_PRESSURE_LOW_PASS_HZ = 3.0
SOFT_DIRECT_PRESSURE_LOW_PASS_HZ = 1.0
TRACK_HOLD_BASELINE_HZ = 0.3
TRACK_HOLD_WINDOW_S = 0.2
TRACK_HOLD_ENTER_DELTA = 0.2
TRACK_HOLD_EXIT_DELTA = 0.08
TRACK_HOLD_SETTLE_S = 0.4
TRACK_HOLD_MAX_OUTPUT_RATE_PER_S = 1.0


@dataclass(frozen=True)
class RelativeGripDecision:
    grasp_active: bool
    reference_pos: float | None
    relative_closure: float | None
    target_pos: float | None
    status: str
    track_hold: TrackHoldDecision | None = None


@dataclass(frozen=True)
class TrackHoldDecision:
    state: str
    input_value: float
    baseline: float
    robust_residual: float
    output_value: float
    transition: str | None


class PressureTrackHoldStabilizer:
    """Freeze steady closure while retaining deliberate relative adjustment.

    Input is the existing 3 Hz filtered continuous PressureVision value.  A
    slower baseline absorbs drift while HOLD freezes the output exactly.  A
    sustained residual enters TRACK; tracking is relative to that local
    baseline so drift accumulated during HOLD is not applied to the gripper.
    """

    def __init__(
        self,
        *,
        baseline_hz: float = TRACK_HOLD_BASELINE_HZ,
        detection_window_s: float = TRACK_HOLD_WINDOW_S,
        enter_delta: float = TRACK_HOLD_ENTER_DELTA,
        exit_delta: float = TRACK_HOLD_EXIT_DELTA,
        settle_s: float = TRACK_HOLD_SETTLE_S,
        max_output_rate_per_s: float = TRACK_HOLD_MAX_OUTPUT_RATE_PER_S,
    ):
        values = {
            "baseline_hz": baseline_hz,
            "detection_window_s": detection_window_s,
            "enter_delta": enter_delta,
            "exit_delta": exit_delta,
            "settle_s": settle_s,
            "max_output_rate_per_s": max_output_rate_per_s,
        }
        for name, value in values.items():
            value = float(value)
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be positive and finite")
            setattr(self, name, value)
        if self.exit_delta >= self.enter_delta:
            raise ValueError("exit_delta must be smaller than enter_delta")
        self.reset()

    def reset(self) -> None:
        self.state = "HOLD"
        self._baseline: float | None = None
        self._output: float | None = None
        self._previous_t: float | None = None
        self._residuals: deque[tuple[float, float]] = deque()
        self._settle_since: float | None = None
        self._track_input_origin: float | None = None
        self._track_output_origin: float | None = None

    def update(self, value: float, observed_at_s: float) -> TrackHoldDecision:
        value = float(np.clip(value, 0.0, 1.0))
        observed_at_s = float(observed_at_s)
        if not math.isfinite(observed_at_s):
            raise ValueError("observed_at_s must be finite")
        if self._previous_t is not None and observed_at_s < self._previous_t:
            raise ValueError("observed_at_s must be monotonic")

        if self._baseline is None:
            self._baseline = value
            self._output = value
            self._previous_t = observed_at_s
            self._residuals.append((observed_at_s, 0.0))
            return TrackHoldDecision("HOLD", value, value, 0.0, value, None)

        dt_s = observed_at_s - self._previous_t
        alpha = 1.0 - math.exp(-2.0 * math.pi * self.baseline_hz * dt_s)
        self._baseline += alpha * (value - self._baseline)
        residual = value - self._baseline
        self._residuals.append((observed_at_s, residual))
        window_start = observed_at_s - self.detection_window_s
        while self._residuals and self._residuals[0][0] < window_start:
            self._residuals.popleft()
        robust_residual = float(np.median([sample for _, sample in self._residuals]))

        transition = None
        if self.state == "HOLD" and abs(robust_residual) >= self.enter_delta:
            self.state = "TRACK"
            self._track_input_origin = self._baseline
            self._track_output_origin = self._output
            self._settle_since = None
            transition = "HOLD_TO_TRACK"

        if self.state == "TRACK":
            desired = float(
                np.clip(
                    self._track_output_origin + value - self._track_input_origin,
                    0.0,
                    1.0,
                )
            )
            max_step = self.max_output_rate_per_s * dt_s
            self._output += float(np.clip(desired - self._output, -max_step, max_step))
            if abs(robust_residual) <= self.exit_delta:
                if self._settle_since is None:
                    self._settle_since = observed_at_s
                elif observed_at_s - self._settle_since >= self.settle_s:
                    self.state = "HOLD"
                    self._settle_since = None
                    transition = "TRACK_TO_HOLD"
            else:
                self._settle_since = None

        self._previous_t = observed_at_s
        return TrackHoldDecision(
            self.state,
            value,
            self._baseline,
            robust_residual,
            self._output,
            transition,
        )


class RelativePressureMapper:
    """Latch one observed aperture per grasp; pressure only adds closure.

    The pinch-derived command is used only for the established binary grasp
    hysteresis. It never scales the relative closure.
    """

    def __init__(
        self,
        *,
        max_closure: float,
        cutoff_hz: float | None = None,
        stabilize: bool = False,
    ):
        max_closure = float(max_closure)
        if not math.isfinite(max_closure) or max_closure <= 0.0:
            raise ValueError("max_closure must be positive and finite")
        if cutoff_hz is not None:
            cutoff_hz = float(cutoff_hz)
            if not math.isfinite(cutoff_hz) or cutoff_hz <= 0.0:
                raise ValueError("cutoff_hz must be positive and finite")
        self.max_closure = max_closure
        self.cutoff_hz = cutoff_hz
        self.grasp_active = False
        self.reference_pos: float | None = None
        self._reference_requested_at_s: float | None = None
        self._filtered_pressure = 0.0
        self._filter_observed_at_s: float | None = None
        self._track_hold = PressureTrackHoldStabilizer() if stabilize else None

    def reset(self) -> None:
        self.grasp_active = False
        self.reference_pos = None
        self._reference_requested_at_s = None
        self._filtered_pressure = 0.0
        self._filter_observed_at_s = None
        if self._track_hold is not None:
            self._track_hold.reset()

    def _low_pass(self, raw_pressure: float, observed_at_s: float | None) -> float:
        if self.cutoff_hz is None:
            return raw_pressure
        if observed_at_s is None or not math.isfinite(float(observed_at_s)):
            dt_s = 1.0 / 30.0
        elif self._filter_observed_at_s is None:
            self._filter_observed_at_s = float(observed_at_s)
            return self._filtered_pressure
        else:
            observed_at_s = float(observed_at_s)
            dt_s = max(0.0, observed_at_s - self._filter_observed_at_s)
            self._filter_observed_at_s = observed_at_s
        alpha = 1.0 - math.exp(-2.0 * math.pi * self.cutoff_hz * dt_s)
        self._filtered_pressure += alpha * (raw_pressure - self._filtered_pressure)
        return self._filtered_pressure

    def update(
        self,
        *,
        base_gripper_pos: float,
        pressure,
        observed_gripper_pos: float | None,
        control_observed_at_s: float | None = None,
        motor_observed_at_s: float | None = None,
    ) -> RelativeGripDecision:
        base_gripper_pos = float(base_gripper_pos)
        if self.grasp_active:
            if base_gripper_pos >= GRIP_LATCH_EXIT:
                self.reset()
        elif base_gripper_pos <= GRIP_LATCH_ENTER:
            self.grasp_active = True

        if not self.grasp_active:
            return RelativeGripDecision(False, None, None, None, "right_grasp_inactive")

        pressure_active = bool(
            pressure is not None
            and getattr(pressure, "available", False)
            and getattr(pressure, "active", False)
        )
        if not pressure_active:
            if self.reference_pos is None:
                self._reference_requested_at_s = None
                return RelativeGripDecision(True, None, None, None, "waiting_pressure")
            pressure_0_1 = 0.0
            status = "holding_reference"

        elif self.reference_pos is None:
            if self._reference_requested_at_s is None:
                self._reference_requested_at_s = control_observed_at_s
            readback_is_new = (
                self._reference_requested_at_s is None
                or motor_observed_at_s is None
                or float(motor_observed_at_s) >= self._reference_requested_at_s
            )
            if (
                observed_gripper_pos is None
                or not math.isfinite(float(observed_gripper_pos))
                or not readback_is_new
            ):
                return RelativeGripDecision(True, None, None, None, "waiting_position_readback")
            self.reference_pos = float(observed_gripper_pos)
            pressure_0_1 = float(
                np.clip(getattr(pressure, "pressure_0_1", 0.0), 0.0, 1.0)
            )
            status = "active"
        else:
            pressure_0_1 = float(
                np.clip(getattr(pressure, "pressure_0_1", 0.0), 0.0, 1.0)
            )
            status = "active"

        filtered_pressure = self._low_pass(pressure_0_1, control_observed_at_s)
        track_hold = None
        stabilized_pressure = filtered_pressure
        if self._track_hold is not None:
            if control_observed_at_s is None:
                raise ValueError("stabilized relative mapping requires an observation time")
            track_hold = self._track_hold.update(filtered_pressure, control_observed_at_s)
            stabilized_pressure = track_hold.output_value
        relative_closure = stabilized_pressure * self.max_closure
        target = float(np.clip(self.reference_pos - relative_closure, 0.0, 100.0))
        return RelativeGripDecision(
            True,
            self.reference_pos,
            relative_closure,
            target,
            status,
            track_hold,
        )


class PressureRangeMapper:
    """Map the stabilized PV scalar onto one explicit gripper position range.

    The right-hand pinch is only the grab/release gate.  While released the
    target is ``release_pos``.  During a grasp, PV zero and one map to
    ``pressure_zero_pos`` and ``pressure_one_pos`` respectively.  This covers
    the supported policies without tying any one to a contact reference:

    * soft_direct: release=100, zero=100, one=0
    * soft_precise: release=100, zero=28, one=22
    * carton_span: release=100, zero=32, one=20
    * hard_profile: release=open, zero=light, one=hard
    """

    def __init__(
        self,
        *,
        release_pos: float,
        pressure_zero_pos: float,
        pressure_one_pos: float,
        cutoff_hz: float | None = None,
        stabilize: bool = False,
    ):
        positions = {
            "release_pos": release_pos,
            "pressure_zero_pos": pressure_zero_pos,
            "pressure_one_pos": pressure_one_pos,
        }
        for name, value in positions.items():
            value = float(value)
            if not math.isfinite(value) or not 0.0 <= value <= 100.0:
                raise ValueError(f"{name} must be finite and in [0, 100]")
            setattr(self, name, value)
        if not self.pressure_one_pos < self.pressure_zero_pos <= self.release_pos:
            raise ValueError(
                "pressure_one_pos < pressure_zero_pos <= release_pos is required"
            )
        if cutoff_hz is not None:
            cutoff_hz = float(cutoff_hz)
            if not math.isfinite(cutoff_hz) or cutoff_hz <= 0.0:
                raise ValueError("cutoff_hz must be positive and finite")
        self.cutoff_hz = cutoff_hz
        self._track_hold = PressureTrackHoldStabilizer() if stabilize else None
        self.reset()

    def reset(self) -> None:
        self.grasp_active = False
        self._filtered_pressure = 0.0
        self._filter_observed_at_s: float | None = None
        if self._track_hold is not None:
            self._track_hold.reset()

    def _low_pass(self, raw_pressure: float, observed_at_s: float | None) -> float:
        if self.cutoff_hz is None:
            return raw_pressure
        if observed_at_s is None or not math.isfinite(float(observed_at_s)):
            dt_s = 1.0 / 30.0
        elif self._filter_observed_at_s is None:
            self._filter_observed_at_s = float(observed_at_s)
            return self._filtered_pressure
        else:
            observed_at_s = float(observed_at_s)
            dt_s = max(0.0, observed_at_s - self._filter_observed_at_s)
            self._filter_observed_at_s = observed_at_s
        alpha = 1.0 - math.exp(-2.0 * math.pi * self.cutoff_hz * dt_s)
        self._filtered_pressure += alpha * (raw_pressure - self._filtered_pressure)
        return self._filtered_pressure

    def update(
        self,
        *,
        base_gripper_pos: float,
        pressure,
        control_observed_at_s: float | None = None,
    ) -> RelativeGripDecision:
        base_gripper_pos = float(base_gripper_pos)
        if self.grasp_active:
            if base_gripper_pos >= GRIP_LATCH_EXIT:
                self.reset()
        elif base_gripper_pos <= GRIP_LATCH_ENTER:
            self.grasp_active = True

        if not self.grasp_active:
            return RelativeGripDecision(
                False,
                None,
                0.0,
                self.release_pos,
                "right_grasp_inactive",
            )

        pressure_available = bool(
            pressure is not None and getattr(pressure, "available", False)
        )
        if not pressure_available:
            return RelativeGripDecision(
                True,
                self.pressure_zero_pos,
                None,
                None,
                "waiting_pressure",
            )

        pressure_active = bool(getattr(pressure, "active", False))
        pressure_0_1 = (
            float(np.clip(getattr(pressure, "pressure_0_1", 0.0), 0.0, 1.0))
            if pressure_active
            else 0.0
        )
        filtered_pressure = self._low_pass(pressure_0_1, control_observed_at_s)
        track_hold = None
        stabilized_pressure = filtered_pressure
        if self._track_hold is not None:
            if control_observed_at_s is None:
                raise ValueError("stabilized range mapping requires an observation time")
            track_hold = self._track_hold.update(filtered_pressure, control_observed_at_s)
            stabilized_pressure = track_hold.output_value

        span = self.pressure_zero_pos - self.pressure_one_pos
        relative_closure = stabilized_pressure * span
        target = float(
            np.clip(self.pressure_zero_pos - relative_closure, 0.0, 100.0)
        )
        return RelativeGripDecision(
            True,
            self.pressure_zero_pos,
            relative_closure,
            target,
            "active" if pressure_active else "pressure_zero",
            track_hold,
        )
