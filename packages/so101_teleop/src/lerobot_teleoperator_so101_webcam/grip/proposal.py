"""Pressure-driven gripper proposal policy: how far to close, and when to stop.

A hardware-independent state machine over `disarmed -> armed -> fault_latched`.
It takes a base gripper command plus a pressure reading and proposes a new
command, rate-limited to `MAX_PRESSURE_GRIP_STEP` per step and smoothed with an
asymmetric EMA (closing is followed quickly, relaxing slowly). An unavailable,
low-quality, or stale reading latches a fault and holds the last safe proposal
rather than guessing a position.

It imports nothing sensor-specific — no PressureVision, no thermal. Both the PV
integration and the private IR line drive it with their own readings, which is
why it lives in the core `grip/` package next to the contract rather than
inside either consumer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


GRIP_CLOSE_ALPHA = 0.7
GRIP_OPEN_ALPHA = 0.15
GRIP_OVERDRIVE = 18.0
PRESSURE_MIN_QUALITY = 0.5
MAX_PRESSURE_GRIP_STEP = 2.0


def apply_pressure_overdrive(
    base_gripper: float,
    fallback_overdrive: float,
    pressure,
    min_quality: float = PRESSURE_MIN_QUALITY,
) -> float:
    if pressure is None:
        return max(0.0, float(base_gripper) - float(fallback_overdrive))
    if not pressure.available or pressure.quality < min_quality:
        return max(0.0, float(base_gripper) - float(fallback_overdrive))
    if not pressure.active:
        return float(np.clip(base_gripper, 0.0, 100.0))
    overdrive = float(np.clip(pressure.pressure_0_1, 0.0, 1.0)) * float(
        fallback_overdrive
    )
    return max(0.0, float(base_gripper) - overdrive)


@dataclass(frozen=True)
class PressureControlDecision:
    base_gripper: float
    proposed_gripper: float
    actual_gripper: float
    state: str
    fault_latched: bool
    reason: str


@dataclass(frozen=True)
class PressureProposalDecision:
    base_gripper: float
    raw_gripper: float
    proposed_gripper: float
    state: str
    fault_latched: bool
    reason: str

    def with_actual(self, actual_gripper: float) -> PressureControlDecision:
        return PressureControlDecision(
            base_gripper=self.base_gripper,
            proposed_gripper=self.proposed_gripper,
            actual_gripper=float(actual_gripper),
            state=self.state,
            fault_latched=self.fault_latched,
            reason=self.reason,
        )


class PressureProposalStateMachine:
    """Hardware-independent configured-source proposal transition state."""

    def __init__(
        self,
        *,
        initial_gripper: float,
        fallback_overdrive: float = GRIP_OVERDRIVE,
        min_quality: float = PRESSURE_MIN_QUALITY,
        max_grip_step: float = MAX_PRESSURE_GRIP_STEP,
        close_alpha: float = GRIP_CLOSE_ALPHA,
        open_alpha: float = GRIP_OPEN_ALPHA,
        target_resolver: Callable[[object], float | None] | None = None,
        resolve_baseline_target: bool = False,
    ):
        self.fallback_overdrive = float(fallback_overdrive)
        self.min_quality = float(min_quality)
        self.max_grip_step = float(max_grip_step)
        self.close_alpha = float(close_alpha)
        self.open_alpha = float(open_alpha)
        self.target_resolver = target_resolver
        self.resolve_baseline_target = bool(resolve_baseline_target)
        self.state = "disarmed"
        self.raw_gripper: float | None = float(initial_gripper)
        self.smoothed_gripper: float | None = None

    def seed(self, gripper: float, *, reset_smoothed: bool = False) -> None:
        self.raw_gripper = float(gripper)
        if reset_smoothed:
            self.smoothed_gripper = float(gripper)

    def _bounded_target(self, target: float) -> float:
        previous = self.raw_gripper
        if previous is None:
            previous = float(target)
        bounded = float(
            np.clip(
                target,
                previous - self.max_grip_step,
                previous + self.max_grip_step,
            )
        )
        self.raw_gripper = bounded
        return bounded

    def _resolved_target(self, pressure) -> float:
        if self.target_resolver is None:
            raise ValueError("no target resolver configured")
        target = self.target_resolver(pressure)
        if target is None:
            raise ValueError("target resolver returned no target")
        return float(target)

    def _smooth(self, raw_gripper: float, *, hold: bool) -> float:
        if self.smoothed_gripper is None:
            self.smoothed_gripper = float(raw_gripper)
        elif not hold:
            alpha = (
                self.close_alpha
                if raw_gripper < self.smoothed_gripper
                else self.open_alpha
            )
            self.smoothed_gripper = (
                alpha * raw_gripper + (1.0 - alpha) * self.smoothed_gripper
            )
        return float(self.smoothed_gripper)

    def _decision(
        self,
        base_gripper: float,
        raw_gripper: float,
        proposed_gripper: float,
        reason: str,
    ) -> PressureProposalDecision:
        return PressureProposalDecision(
            base_gripper=float(base_gripper),
            raw_gripper=float(raw_gripper),
            proposed_gripper=float(proposed_gripper),
            state=self.state,
            fault_latched=self.state == "fault_latched",
            reason=str(reason),
        )

    def update(self, base_gripper: float, pressure) -> PressureProposalDecision:
        if pressure is not None and getattr(pressure, "fresh", True) is False:
            raw_gripper = self.raw_gripper
            if raw_gripper is None:
                raw_gripper = float(base_gripper)
            proposed_gripper = self.smoothed_gripper
            if proposed_gripper is None:
                proposed_gripper = float(raw_gripper)
            return self._decision(
                base_gripper,
                raw_gripper,
                proposed_gripper,
                getattr(pressure, "status", "thermal_pending"),
            )

        valid = (
            pressure is not None
            and pressure.available
            and pressure.quality >= self.min_quality
        )
        hold = False
        if not valid:
            reason = getattr(pressure, "status", "pressure_unavailable")
            self.state = "fault_latched"
            if self.smoothed_gripper is not None:
                raw_gripper = float(self.raw_gripper)
                hold = True
            else:
                previous = self.raw_gripper
                if previous is None:
                    previous = float(base_gripper)
                safe_target = max(float(base_gripper), float(previous))
                raw_gripper = self._bounded_target(safe_target)
        else:
            is_baseline = not pressure.active and pressure.status == "baseline"
            if is_baseline:
                self.state = "armed"
                baseline_target = float(base_gripper)
                if self.target_resolver is not None and (
                    self.resolve_baseline_target
                    or getattr(pressure, "level", None) == 0
                ):
                    try:
                        baseline_target = self._resolved_target(pressure)
                    except Exception as exc:
                        self.state = "fault_latched"
                        raw_gripper = self.raw_gripper
                        if raw_gripper is None:
                            raw_gripper = float(base_gripper)
                        hold = self.smoothed_gripper is not None
                        reason = f"target_resolver_error:{type(exc).__name__}:{exc}"
                        if not hold:
                            raw_gripper = self._bounded_target(max(float(base_gripper), raw_gripper))
                        proposed_gripper = self._smooth(raw_gripper, hold=hold)
                        return self._decision(
                            base_gripper,
                            raw_gripper,
                            proposed_gripper,
                            reason,
                        )
                raw_gripper = self._bounded_target(baseline_target)
                reason = pressure.status
            elif self.state == "fault_latched":
                raw_gripper = float(self.raw_gripper)
                hold = True
                reason = "fault_latched"
            elif self.state == "armed" and pressure.active:
                try:
                    if self.target_resolver is None:
                        target = apply_pressure_overdrive(
                            base_gripper,
                            self.fallback_overdrive,
                            pressure,
                            self.min_quality,
                        )
                    else:
                        target = self._resolved_target(pressure)
                except Exception as exc:
                    # A profile/wire mismatch is a control fault, not a reason to
                    # guess a gripper position. Hold the last safe proposal and
                    # require a fresh baseline before any later closure.
                    self.state = "fault_latched"
                    raw_gripper = self.raw_gripper
                    if raw_gripper is None:
                        raw_gripper = float(base_gripper)
                    hold = self.smoothed_gripper is not None
                    reason = f"target_resolver_error:{type(exc).__name__}:{exc}"
                    if not hold:
                        raw_gripper = self._bounded_target(max(float(base_gripper), raw_gripper))
                    proposed_gripper = self._smooth(raw_gripper, hold=hold)
                    return self._decision(
                        base_gripper,
                        raw_gripper,
                        proposed_gripper,
                        reason,
                    )
                raw_gripper = self._bounded_target(target)
                reason = pressure.status
            else:
                self.state = "disarmed"
                raw_gripper = self._bounded_target(base_gripper)
                reason = "pressure_disarmed"

        proposed_gripper = self._smooth(raw_gripper, hold=hold)
        return self._decision(
            base_gripper,
            raw_gripper,
            proposed_gripper,
            reason,
        )

    def reset(
        self,
        base_gripper: float,
        *,
        transition: str,
        middle_gripper: float,
        reason: str | None = None,
    ) -> PressureProposalDecision:
        self.state = "disarmed"
        if transition == "middle":
            proposed_gripper = float(middle_gripper)
            self.raw_gripper = proposed_gripper
            self.smoothed_gripper = None
        else:
            proposed_gripper = self.smoothed_gripper
            if proposed_gripper is None:
                proposed_gripper = self.raw_gripper
            if proposed_gripper is None:
                proposed_gripper = float(middle_gripper)
            proposed_gripper = float(proposed_gripper)
            self.raw_gripper = proposed_gripper
            self.smoothed_gripper = proposed_gripper
        return self._decision(
            base_gripper,
            proposed_gripper,
            proposed_gripper,
            transition if reason is None else reason,
        )
