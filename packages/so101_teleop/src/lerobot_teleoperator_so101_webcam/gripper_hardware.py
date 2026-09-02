"""SO-101 gripper hardware: closure limits, telemetry readback, grip targets.

This is servo-level support for the gripper — reading `Present_Current` /
`Present_Load` / `Present_Temperature` off the Feetech bus, rate-limiting that
readback, capping closure, and picking grip targets from recorded current.
None of it is specific to any sensing modality.

It lived in the private sensing line purely by accident of history: the grip
experiments there were the first thing that needed servo telemetry, so the
module was written where they were. Not one line of it is sensor-specific. It
belongs here, where the arm is.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from statistics import mean
import time
from typing import Callable


GRIPPER = "gripper"


@dataclass(frozen=True)
class TelemetrySnapshot:
    t: float
    gripper_pos: float
    goal_gripper_pos: float
    present_current: int | None
    present_load: int | None
    present_temperature: int | None


@dataclass(frozen=True)
class GripperRuntimeTelemetry:
    observed_at_s: float
    observed_gripper_pos: float
    present_current: int | None
    present_load: int | None
    present_temperature: int | None


@dataclass(frozen=True)
class GripperClosureLimits:
    max_load: float
    max_current: float
    max_position_lag: float

    def __post_init__(self) -> None:
        for name, value in (
            ("max_load", self.max_load),
            ("max_current", self.max_current),
            ("max_position_lag", self.max_position_lag),
        ):
            if not math.isfinite(float(value)) or float(value) <= 0.0:
                raise ValueError(f"{name} must be positive and finite")


@dataclass(frozen=True)
class GripperClosureLimitDecision:
    requested_pos: float
    actual_pos: float
    fault_latched: bool
    reason: str
    position_lag: float | None


class GripperClosureLimiter:
    """Latch the last sent aperture when gripper feedback crosses a limit.

    Lower gripper positions mean more closure.  Once latched, a request may
    still open the gripper but may not close beyond the last command.  Only an
    explicit operator release clears the latch.
    """

    def __init__(
        self,
        limits: GripperClosureLimits,
        *,
        max_telemetry_age_s: float = 0.5,
        feedback_settle_s: float = 0.2,
    ):
        if not math.isfinite(float(max_telemetry_age_s)) or max_telemetry_age_s <= 0.0:
            raise ValueError("max_telemetry_age_s must be positive and finite")
        if not math.isfinite(float(feedback_settle_s)) or feedback_settle_s <= 0.0:
            raise ValueError("feedback_settle_s must be positive and finite")
        self.limits = limits
        self.max_telemetry_age_s = float(max_telemetry_age_s)
        self.feedback_settle_s = float(feedback_settle_s)
        self.latched_pos: float | None = None
        self.reason: str | None = None
        self._tracked_commanded_pos: float | None = None
        self._command_stable_since_s: float | None = None

    def reset(self) -> None:
        self.latched_pos = None
        self.reason = None
        self._tracked_commanded_pos = None
        self._command_stable_since_s = None

    def update(
        self,
        *,
        requested_pos: float,
        last_commanded_pos: float,
        telemetry: GripperRuntimeTelemetry | None,
        observed_at_s: float,
        release_requested: bool = False,
    ) -> GripperClosureLimitDecision:
        requested_pos = float(requested_pos)
        last_commanded_pos = float(last_commanded_pos)
        if release_requested:
            self.reset()
            return GripperClosureLimitDecision(
                requested_pos,
                max(requested_pos, last_commanded_pos),
                False,
                "explicit_release",
                None,
            )

        if self._tracked_commanded_pos != last_commanded_pos:
            self._tracked_commanded_pos = last_commanded_pos
            self._command_stable_since_s = float(observed_at_s)

        position_lag = None
        trip_reason = None
        if telemetry is None:
            trip_reason = "closure_limit_telemetry_unavailable"
        elif observed_at_s - float(telemetry.observed_at_s) > self.max_telemetry_age_s:
            trip_reason = "closure_limit_telemetry_stale"
        elif telemetry.present_load is None or telemetry.present_current is None:
            trip_reason = "closure_limit_telemetry_incomplete"
        else:
            position_lag = abs(
                float(telemetry.observed_gripper_pos) - last_commanded_pos
            )
            feedback_settled = (
                self._command_stable_since_s is not None
                and float(telemetry.observed_at_s) - self._command_stable_since_s
                >= self.feedback_settle_s
            )
            if abs(float(telemetry.present_current)) >= self.limits.max_current:
                trip_reason = "closure_limit_current"
            elif (
                feedback_settled
                and abs(float(telemetry.present_load)) >= self.limits.max_load
            ):
                trip_reason = "closure_limit_load"
            elif (
                feedback_settled
                and position_lag >= self.limits.max_position_lag
            ):
                trip_reason = "closure_limit_position_lag"

        telemetry_blocked = trip_reason in {
            "closure_limit_telemetry_unavailable",
            "closure_limit_telemetry_stale",
            "closure_limit_telemetry_incomplete",
        }
        if self.latched_pos is None and telemetry_blocked:
            return GripperClosureLimitDecision(
                requested_pos,
                max(requested_pos, last_commanded_pos),
                True,
                str(trip_reason),
                position_lag,
            )
        if self.latched_pos is None and trip_reason is not None:
            self.latched_pos = last_commanded_pos
            self.reason = trip_reason
        if self.latched_pos is None:
            return GripperClosureLimitDecision(
                requested_pos, requested_pos, False, "within_closure_limits", position_lag
            )
        return GripperClosureLimitDecision(
            requested_pos,
            max(requested_pos, self.latched_pos),
            True,
            str(self.reason),
            position_lag,
        )


def read_gripper_runtime_telemetry(
    robot,
    *,
    clock: Callable[[], float] = time.perf_counter,
) -> GripperRuntimeTelemetry:
    """Read position, effort proxies, and temperature for a live PV trial."""
    observation = robot.get_observation()
    present_current = _read_reg(robot, "Present_Current", GRIPPER)
    present_temperature = _read_reg(robot, "Present_Temperature", GRIPPER)
    return GripperRuntimeTelemetry(
        observed_at_s=float(clock()),
        observed_gripper_pos=float(observation["gripper.pos"]),
        present_current=present_current,
        present_load=_read_reg(robot, "Present_Load", GRIPPER),
        present_temperature=present_temperature,
    )


class GripperTelemetrySampler:
    """Rate-limit serial readback and retain the last timestamped sample."""

    def __init__(
        self,
        *,
        interval_s: float = 0.2,
        clock: Callable[[], float] = time.perf_counter,
    ):
        if interval_s <= 0.0:
            raise ValueError("interval_s must be positive")
        self.interval_s = float(interval_s)
        self.clock = clock
        self.latest: GripperRuntimeTelemetry | None = None
        self._last_attempt_at_s: float | None = None

    def poll(self, robot, *, force: bool = False) -> GripperRuntimeTelemetry | None:
        now = float(self.clock())
        if (
            not force
            and self._last_attempt_at_s is not None
            and now - self._last_attempt_at_s < self.interval_s
        ):
            return self.latest
        self._last_attempt_at_s = now
        try:
            self.latest = read_gripper_runtime_telemetry(robot, clock=self.clock)
        except BUS_FAULTS:
            # Ride out a bus fault on the last good sample. A `KeyError` or
            # `TypeError` from here is a wrong register or a wrong robot, not a
            # hiccup, and keeping a stale sample would hide it for the whole run.
            pass
        return self.latest


def slow_close_waypoints(start: float, target: float, steps: int) -> list[float]:
    if steps <= 0:
        raise ValueError("steps must be positive")
    delta = (target - start) / steps
    return [round(start + delta * (i + 1), 6) for i in range(steps)]


def _numeric(values: list[int | float | None]) -> list[float]:
    return [float(value) for value in values if value is not None]


def summarize_target_current(samples: list[TelemetrySnapshot]) -> dict[str, float]:
    currents = _numeric([sample.present_current for sample in samples])
    loads = _numeric([sample.present_load for sample in samples])
    temperatures = _numeric([sample.present_temperature for sample in samples])
    return {
        "mean_current": mean(currents) if currents else 0.0,
        "max_current": max(currents) if currents else 0.0,
        "mean_load": mean(loads) if loads else 0.0,
        "max_temperature": max(temperatures) if temperatures else 0.0,
    }


def serialize_telemetry_snapshot(
    sample: TelemetrySnapshot,
    *,
    target: float,
    sample_index: int,
) -> dict[str, float | int | None]:
    return {
        "target": float(target),
        "sample_index": int(sample_index),
        "t": float(sample.t),
        "gripper_pos": float(sample.gripper_pos),
        "goal_gripper_pos": float(sample.goal_gripper_pos),
        "present_current": sample.present_current,
        "present_load": sample.present_load,
        "present_temperature": sample.present_temperature,
    }


def choose_three_grip_targets(records: list[dict[str, float]], min_current_gap: float) -> dict[str, float]:
    ordered = sorted(records, key=lambda record: record["mean_current"])
    selected: tuple[dict[str, float], dict[str, float], dict[str, float]] | None = None
    selected_key: tuple[float, float, float] | None = None
    for low_index, low in enumerate(ordered):
        for med_index, med in enumerate(ordered[low_index + 1 :], start=low_index + 1):
            if med["mean_current"] - low["mean_current"] < min_current_gap:
                continue
            for high in ordered[med_index + 1 :]:
                if high["mean_current"] - med["mean_current"] >= min_current_gap:
                    candidate = (low, med, high)
                    candidate_key = (
                        candidate[0]["mean_current"],
                        candidate[1]["mean_current"],
                        candidate[2]["mean_current"],
                    )
                    if selected_key is None or candidate_key > selected_key:
                        selected = candidate
                        selected_key = candidate_key

    if selected is None:
        raise ValueError("could not find three separated grip targets")

    return {
        "low": selected[0]["target"],
        "med": selected[1]["target"],
        "high": selected[2]["target"],
    }


#: Faults the bus itself can raise on a good call: the port dropped, or the servo
#: answered with an error packet. `ConnectionError` -- and LeRobot's
#: `DeviceNotConnectedError`, and pyserial's `SerialException` -- are all `OSError`.
#:
#: Deliberately not `Exception`: a register name that is not in the control table
#: raises `KeyError`, and swallowing that turned a typo into three permanently
#: blank telemetry columns with nothing logged anywhere. A fault we cannot read
#: through is a missing value; a fault in this file is a bug and must be loud.
BUS_FAULTS = (OSError, RuntimeError)


def _read_reg(robot, reg: str, motor: str) -> int | None:
    try:
        return int(robot.bus.read(reg, motor, normalize=False, num_retry=5))
    except BUS_FAULTS:
        return None


def read_gripper_telemetry(robot, goal_gripper_pos: float, t: float) -> TelemetrySnapshot:
    observation = robot.get_observation()
    return TelemetrySnapshot(
        t=t,
        gripper_pos=float(observation["gripper.pos"]),
        goal_gripper_pos=float(goal_gripper_pos),
        present_current=_read_reg(robot, "Present_Current", GRIPPER),
        present_load=_read_reg(robot, "Present_Load", GRIPPER),
        present_temperature=_read_reg(robot, "Present_Temperature", GRIPPER),
    )
