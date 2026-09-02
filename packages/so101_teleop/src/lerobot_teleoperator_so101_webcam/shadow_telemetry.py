"""Fail-soft CSV telemetry for grip shadow and apply runs.

One finalized row per control action, written without ever affecting control:
any logging error disables the logger instead of propagating. Losing telemetry
is acceptable; interrupting a run that is holding an object is not.

The column set here is the part every grip sensor shares -- control timing, the
pressure reading, and the gripper proposal. A sensor with its own columns
supplies them through `extra_fields` + `extra_row`, so this module never has to
know that PressureVision or a thermal camera exists.

Timestamps carried by a reading are host read-completion observations. They do
not represent camera exposure synchronization.
"""

from __future__ import annotations

import csv
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

__all__ = [
    "CONTROL_SHADOW_FIELDS",
    "ShadowTelemetrySample",
    "ShadowTelemetryLogger",
    "csv_bool",
    "first_attr",
    "milliseconds",
]

CONTROL_SHADOW_FIELDS = (
    "schema_version",
    "tick",
    "control_observed_at_s",
    "oak_observed_at_s",
    "thermal_observed_at_s",
    "sensor_skew_ms",
    "oak_age_ms",
    "thermal_age_ms",
    "loop_period_ms",
    "control_latency_ms",
    "state",
    "pinch",
    "roi_mode",
    "roi_x",
    "roi_y",
    "roi_width",
    "roi_height",
    "baseline_ready",
    "pressure",
    "quality",
    "pressure_available",
    "pressure_status",
    "base_gripper_pos",
    "proposed_gripper_pos",
    "actual_gripper_pos",
    "command_sent",
    "fault_latched",
    "fallback_used",
    "fallback_reason",
)


@dataclass(frozen=True)
class ShadowTelemetrySample:
    """One control frame, as the caller saw it.

    A sensor extension subclasses this to add its own columns; the base row is
    built from these fields plus whatever the reading itself carries.
    """

    control_observed_at_s: float
    state: str
    pinch: float
    roi_mode: str | None
    pressure: object | None
    baseline_ready: bool
    base_gripper_pos: float
    proposed_gripper_pos: float
    actual_gripper_pos: float
    fault_latched: bool
    fallback_used: bool
    fallback_reason: str | None
    pressure_status: str | None = None


def milliseconds(value: float | None) -> float | None:
    return None if value is None else float(value) * 1000.0


def csv_bool(value: bool) -> str:
    return "true" if value else "false"


def first_attr(obj, *names):
    """The first of `names` the reading actually carries.

    The column names are the schema's and cannot change without invalidating
    recorded evidence, but the readings behind them do not agree on attribute
    names: the thermal path calls its timestamp `thermal_observed_at_s`, the PV
    protocol calls the same thing `observed_at_s`. Resolving here is what keeps
    a column from silently coming out empty for one of the two senders.
    """
    for name in names:
        value = getattr(obj, name, None)
        if value is not None:
            return value
    return None


class ShadowTelemetryLogger:
    """Write one finalized row per caller action without affecting control."""

    def __init__(
        self,
        path: str | Path,
        *,
        schema_version: str,
        clock: Callable[[], float] = time.perf_counter,
        extra_fields: tuple[str, ...] = (),
        extra_row: Callable[..., dict] | None = None,
        log_prefix: str = "[shadow]",
    ):
        if bool(extra_fields) != bool(extra_row is not None):
            raise ValueError("extra_fields and extra_row must be given together")
        self.path = Path(path)
        self.schema_version = str(schema_version)
        self._clock = clock
        self._extra_row = extra_row
        self._log_prefix = log_prefix
        self._file = None
        self._writer = None
        self._tick = 0
        self._previous_control_observed_at_s: float | None = None
        self._warned = False
        self.enabled = False
        self.extra_fields = tuple(extra_fields)
        self.fieldnames = CONTROL_SHADOW_FIELDS + self.extra_fields
        try:
            self._file = self.path.open("w", newline="", encoding="utf-8")
            self._writer = csv.DictWriter(
                self._file,
                fieldnames=self.fieldnames,
                lineterminator="\n",
            )
            self._writer.writeheader()
            self._file.flush()
            self.enabled = True
        except Exception as exc:
            self._disable(exc)

    def _disable(self, exc: Exception) -> None:
        if not self._warned:
            print(f"{self._log_prefix} disabled after logging error: {exc}")
            self._warned = True
        self.enabled = False
        file_obj = self._file
        self._file = None
        self._writer = None
        if file_obj is not None:
            try:
                file_obj.close()
            except Exception:
                pass

    def _row(
        self,
        sample: ShadowTelemetrySample,
        *,
        command_sent: bool,
        finalized_at_s: float,
        motor_telemetry=None,
    ) -> dict:
        pressure = sample.pressure
        roi = getattr(pressure, "roi", None)
        control_t = float(sample.control_observed_at_s)
        loop_period_s = (
            None
            if self._previous_control_observed_at_s is None
            else control_t - self._previous_control_observed_at_s
        )
        pressure_status = (
            getattr(pressure, "status", None)
            if pressure is not None
            else sample.pressure_status
        )
        row = {
            "schema_version": self.schema_version,
            "tick": self._tick,
            "control_observed_at_s": control_t,
            "oak_observed_at_s": getattr(pressure, "oak_observed_at_s", None),
            "thermal_observed_at_s": first_attr(pressure, "thermal_observed_at_s", "observed_at_s"),
            "sensor_skew_ms": milliseconds(getattr(pressure, "sensor_skew_s", None)),
            "oak_age_ms": milliseconds(getattr(pressure, "oak_age_s", None)),
            "thermal_age_ms": milliseconds(first_attr(pressure, "thermal_age_s", "age_s")),
            "loop_period_ms": milliseconds(loop_period_s),
            "control_latency_ms": milliseconds(finalized_at_s - control_t),
            "state": sample.state,
            "pinch": sample.pinch,
            "roi_mode": sample.roi_mode,
            "roi_x": getattr(roi, "x", None),
            "roi_y": getattr(roi, "y", None),
            "roi_width": getattr(roi, "width", None),
            "roi_height": getattr(roi, "height", None),
            "baseline_ready": csv_bool(sample.baseline_ready),
            "pressure": getattr(pressure, "pressure_0_1", None),
            "quality": getattr(pressure, "quality", None),
            "pressure_available": csv_bool(bool(getattr(pressure, "available", False))),
            "pressure_status": pressure_status,
            "base_gripper_pos": sample.base_gripper_pos,
            "proposed_gripper_pos": sample.proposed_gripper_pos,
            "actual_gripper_pos": sample.actual_gripper_pos,
            "command_sent": csv_bool(command_sent),
            "fault_latched": csv_bool(sample.fault_latched),
            "fallback_used": csv_bool(sample.fallback_used),
            "fallback_reason": sample.fallback_reason,
        }
        if self._extra_row is not None:
            extra = self._extra_row(
                sample,
                pressure=pressure,
                motor_telemetry=motor_telemetry,
                finalized_at_s=finalized_at_s,
            )
            # Checked here rather than left to DictWriter: a mismatch is a
            # programming error, and finding it in the row keeps the failure
            # inside the fail-soft path instead of raising into the control loop.
            if set(extra) != set(self.extra_fields):
                raise ValueError(
                    "extra_row returned "
                    f"{sorted(set(extra) ^ set(self.extra_fields))} outside extra_fields"
                )
            row.update(extra)
        return row

    def finalize(
        self,
        sample: ShadowTelemetrySample | None,
        *,
        command_sent: bool,
        motor_telemetry=None,
    ) -> None:
        if not self.enabled or sample is None:
            return
        try:
            finalized_at_s = self._clock()
            self._writer.writerow(
                self._row(
                    sample,
                    command_sent=command_sent,
                    finalized_at_s=finalized_at_s,
                    motor_telemetry=motor_telemetry,
                )
            )
            self._file.flush()
            self._previous_control_observed_at_s = float(sample.control_observed_at_s)
            self._tick += 1
        except Exception as exc:
            self._disable(exc)

    def close(self) -> None:
        file_obj = self._file
        self._file = None
        self._writer = None
        self.enabled = False
        if file_obj is None:
            return
        try:
            file_obj.close()
        except Exception as exc:
            self._disable(exc)
