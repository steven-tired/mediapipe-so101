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


#: Effort registers read across every joint, not just the gripper.
JOINT_EFFORT_REGISTERS = ("Present_Load", "Present_Current", "Present_Temperature")


def read_joint_effort(robot) -> dict[str, dict[str, int | None]]:
    """Load, current and temperature for every joint, by register sync_read.

    The gripper servo measures its own closing torque -- the normal force it
    applies -- and not the tangential load a held object's weight produces. So
    a carton twice as heavy looks the same to it. The arm joints do carry that
    weight: `shoulder_lift` holding 250 g and 750 g at one pose differ in
    current to first order, and that is the only channel on this robot where
    payload appears at all.

    One sync_read per register rather than six single reads per joint: this bus
    already drops status packets under load, and eighteen transactions per
    sample would make that worse for no benefit.

    A register that cannot be read comes back None for every joint rather than
    raising. A wrong register name still raises -- swallowing that is what left
    five dataset columns constant zero on 2026-09-02.
    """
    readings: dict[str, dict[str, int | None]] = {
        motor: {} for motor in robot.bus.motors
    }
    for register in JOINT_EFFORT_REGISTERS:
        try:
            values = robot.bus.sync_read(register, normalize=False, num_retry=3)
        except BUS_FAULTS:
            values = {}
        for motor in readings:
            raw = values.get(motor)
            readings[motor][register] = None if raw is None else int(raw)
    return readings


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


@dataclass(frozen=True)
class DeadbandStep:
    """One commanded step of a deadband staircase and what the jaw actually did."""

    step_size: float
    commanded_delta: float
    readback_delta: float
    readback_spread: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.step_size) or self.step_size <= 0.0:
            raise ValueError("step_size must be positive and finite")


def readback_spread(samples: list[TelemetrySnapshot]) -> float:
    """Peak-to-peak gripper readback across a dwell.

    Peak-to-peak rather than a standard deviation because this encoder is
    quantized and its hold noise is frequently exactly zero: what "resolvable"
    means here is "larger than anything the jaw does while the command is
    held", and that is a range, not a spread around a mean.
    """
    if not samples:
        raise ValueError("readback_spread needs at least one sample")
    positions = [float(sample.gripper_pos) for sample in samples]
    return max(positions) - min(positions)


def _tread_moved(step: DeadbandStep, noise_floor: float) -> bool:
    """Did this one tread move the jaw, the way it was told to, past the floor?

    A response of the right size in the wrong direction is the carton settling
    or the jaw unloading, not the step being resolved.
    """
    return (
        abs(step.readback_delta) > noise_floor
        and step.readback_delta * step.commanded_delta > 0.0
    )


def smallest_resolvable_step(
    steps: list[DeadbandStep],
    *,
    noise_floor: float,
) -> float | None:
    """Smallest swept size that moved the jaw on *every* tread of its ramp.

    Every tread, not any tread, and this is the whole point of the gate. A ramp
    stepped below the breakout deadband still advances -- it just advances in
    stick-slip bursts, several dead treads and then a jump, because it is the
    *accumulated* command error that eventually breaks static friction, not the
    step. Scoring "any tread moved" would therefore certify the very `0.2` step
    that moved nothing in the 2026-08-31 trials, given enough treads.

    A ramp controller needs the other property: that each commanded step buys a
    proportional amount of jaw. Returns `None` when no swept size had it, which
    is a result and not an error.
    """
    if not math.isfinite(noise_floor) or noise_floor < 0.0:
        raise ValueError("noise_floor must be non-negative and finite")
    by_size: dict[float, list[DeadbandStep]] = {}
    for step in steps:
        by_size.setdefault(step.step_size, []).append(step)
    resolved = [
        size
        for size, group in by_size.items()
        if all(_tread_moved(step, noise_floor) for step in group)
    ]
    return min(resolved) if resolved else None


def breakout_offset(steps: list[DeadbandStep], *, noise_floor: float) -> float | None:
    """Commanded travel accumulated before the jaw first moved, along one ramp.

    This is the deadband itself, in the units the servo is commanded in, and
    unlike a step size it should not depend much on how the ramp was walked.
    `None` means the ramp never broke out within the treads it was given.
    """
    accumulated = 0.0
    for step in steps:
        accumulated += step.commanded_delta
        if _tread_moved(step, noise_floor):
            return abs(accumulated)
    return None


def tracking_ratio(steps: list[DeadbandStep]) -> float:
    """Jaw travel actually delivered per unit of commanded travel along a ramp.

    One is a ramp that goes where it is told. Near zero is a ramp being
    swallowed by the deadband. Above one is a ramp that stuck and then released
    what it had stored.
    """
    if not steps:
        raise ValueError("tracking_ratio needs at least one tread")
    commanded = sum(abs(step.commanded_delta) for step in steps)
    if commanded == 0.0:
        raise ValueError("tracking_ratio needs a ramp that commanded some travel")
    return sum(abs(step.readback_delta) for step in steps) / commanded


def rank_correlation(xs: list[float], ys: list[float]) -> float:
    """Spearman correlation with tie-averaged ranks.

    Ties are averaged rather than broken because `Present_Load` is quantized to
    multiples of four and `Present_Current` spans about sixteen counts, so a
    tie-breaking rank would read structure into what is really one value.
    Returns 0.0 when either channel is constant, which is the honest answer for
    "does this channel respond to grip depth": it does not.
    """
    if len(xs) != len(ys):
        raise ValueError("rank_correlation needs equal-length inputs")
    if len(xs) < 2:
        raise ValueError("rank_correlation needs at least two points")
    rx, ry = _average_ranks(xs), _average_ranks(ys)
    mx, my = mean(rx), mean(ry)
    dx = [r - mx for r in rx]
    dy = [r - my for r in ry]
    denominator = math.sqrt(sum(d * d for d in dx) * sum(d * d for d in dy))
    if denominator == 0.0:
        return 0.0
    return sum(a * b for a, b in zip(dx, dy)) / denominator


def _average_ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda i: float(values[i]))
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        stop = start
        while stop + 1 < len(order) and float(values[order[stop + 1]]) == float(values[order[start]]):
            stop += 1
        shared = (start + stop) / 2.0
        for index in order[start : stop + 1]:
            ranks[index] = shared
        start = stop + 1
    return ranks


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
