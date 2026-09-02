"""The PressureVision columns of the shadow-telemetry CSV — dataset schema v7.

The generic logger lives in the core package; this module supplies only what PV
adds: the wire/staleness columns, the trial-protocol identity, the servo
telemetry, and the adjustment-lock state that produces the teacher label
`research/train_grip_residual_head.py` trains on.

Bumping `PV_SHADOW_SCHEMA_VERSION` invalidates existing evidence: the recorded
episodes under `local/evidence/` were written at "7", and a reader keys off it.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from lerobot_teleoperator_so101_webcam.shadow_telemetry import (
    ShadowTelemetryLogger,
    ShadowTelemetrySample,
    csv_bool,
    first_attr,
    milliseconds,
)

__all__ = [
    "PV_SHADOW_FIELDS",
    "PV_SHADOW_SCHEMA_VERSION",
    "PVShadowTelemetryLogger",
    "PVShadowTelemetrySample",
    "pv_shadow_row",
]

PV_SHADOW_SCHEMA_VERSION = "7"
PV_SHADOW_FIELDS = (
    "pv_sequence",
    "pv_source_observed_at_s",
    "pv_sent_at_s",
    "pv_received_at_s",
    "pv_frame_age_ms",
    "pressure_level",
    "pressure_n_levels",
    "pressure_mode",
    "object_id",
    "object_profile_sha256",
    "trial_index",
    "phase_index",
    "expected_level",
    "trial_phase",
    "commanded_gripper_pos",
    "motor_observed_at_s",
    "motor_sample_age_ms",
    "motor_sample_valid",
    "observed_gripper_pos",
    "observed_gripper_pos_valid",
    "present_current",
    "present_current_valid",
    "present_load",
    "present_load_valid",
    "present_temperature",
    "present_temperature_valid",
    "pv_adjustment_state",
    "pv_adjustment_event",
    "pv_adjustment_anchor_target",
    "pv_adjustment_release_since_s",
    "pv_adjustment_release_elapsed_s",
    "pv_adjustment_last_contact_at_s",
    "pv_adjustment_recontact_since_s",
    "relative_reference_pos",
    "relative_closure",
    "relative_mapping_status",
    "relative_track_hold_state",
    "relative_track_hold_residual",
    "relative_track_hold_output",
)


@dataclass(frozen=True)
class PVShadowTelemetrySample(ShadowTelemetrySample):
    """A control frame plus the PV columns. Every addition is optional, so a
    frame that had no PV reading still writes a complete row."""

    pressure_level: int | None = None
    pressure_n_levels: int | None = None
    pressure_mode: str | None = None
    object_id: str | None = None
    object_profile_sha256: str | None = None
    trial_index: int | None = None
    phase_index: int | None = None
    expected_level: int | None = None
    trial_phase: str | None = None
    relative_reference_pos: float | None = None
    relative_closure: float | None = None
    relative_mapping_status: str | None = None
    relative_track_hold_state: str | None = None
    relative_track_hold_residual: float | None = None
    relative_track_hold_output: float | None = None
    pv_adjustment_state: str | None = None
    pv_adjustment_event: str | None = None
    pv_adjustment_anchor_target: float | None = None
    pv_adjustment_release_since_s: float | None = None
    pv_adjustment_release_elapsed_s: float | None = None
    pv_adjustment_last_contact_at_s: float | None = None
    pv_adjustment_recontact_since_s: float | None = None


def pv_shadow_row(sample, *, pressure, motor_telemetry, finalized_at_s) -> dict:
    """The PV half of one row. Read with getattr so a base sample still works."""
    motor_observed_at_s = getattr(motor_telemetry, "observed_at_s", None)
    observed_gripper_pos = getattr(motor_telemetry, "observed_gripper_pos", None)
    present_current = getattr(motor_telemetry, "present_current", None)
    present_load = getattr(motor_telemetry, "present_load", None)
    present_temperature = getattr(motor_telemetry, "present_temperature", None)
    return {
        "pv_sequence": first_attr(pressure, "pv_sequence", "sequence"),
        "pv_source_observed_at_s": first_attr(
            pressure, "thermal_observed_at_s", "observed_at_s"
        ),
        "pv_sent_at_s": first_attr(pressure, "pv_sent_at_s", "sent_at_s"),
        "pv_received_at_s": first_attr(pressure, "pv_received_at_s", "received_at_s"),
        "pv_frame_age_ms": milliseconds(first_attr(pressure, "thermal_age_s", "age_s")),
        "pressure_level": getattr(sample, "pressure_level", None),
        "pressure_n_levels": getattr(sample, "pressure_n_levels", None),
        "pressure_mode": getattr(sample, "pressure_mode", None),
        "object_id": getattr(sample, "object_id", None),
        "object_profile_sha256": getattr(sample, "object_profile_sha256", None),
        "trial_index": getattr(sample, "trial_index", None),
        "phase_index": getattr(sample, "phase_index", None),
        "expected_level": getattr(sample, "expected_level", None),
        "trial_phase": getattr(sample, "trial_phase", None),
        # Keep actual_gripper_pos for backward compatibility. It is the command
        # selected by the controller, not a bus read.
        "commanded_gripper_pos": sample.actual_gripper_pos,
        "motor_observed_at_s": motor_observed_at_s,
        "motor_sample_age_ms": milliseconds(
            None
            if motor_observed_at_s is None
            else finalized_at_s - float(motor_observed_at_s)
        ),
        "motor_sample_valid": csv_bool(motor_telemetry is not None),
        "observed_gripper_pos": observed_gripper_pos,
        "observed_gripper_pos_valid": csv_bool(observed_gripper_pos is not None),
        "present_current": present_current,
        "present_current_valid": csv_bool(present_current is not None),
        "present_load": present_load,
        "present_load_valid": csv_bool(present_load is not None),
        "present_temperature": present_temperature,
        "present_temperature_valid": csv_bool(present_temperature is not None),
        "pv_adjustment_state": getattr(sample, "pv_adjustment_state", None),
        "pv_adjustment_event": getattr(sample, "pv_adjustment_event", None),
        "pv_adjustment_anchor_target": getattr(sample, "pv_adjustment_anchor_target", None),
        "pv_adjustment_release_since_s": getattr(sample, "pv_adjustment_release_since_s", None),
        "pv_adjustment_release_elapsed_s": getattr(sample, "pv_adjustment_release_elapsed_s", None),
        "pv_adjustment_last_contact_at_s": getattr(sample, "pv_adjustment_last_contact_at_s", None),
        "pv_adjustment_recontact_since_s": getattr(sample, "pv_adjustment_recontact_since_s", None),
        "relative_reference_pos": getattr(sample, "relative_reference_pos", None),
        "relative_closure": getattr(sample, "relative_closure", None),
        "relative_mapping_status": getattr(sample, "relative_mapping_status", None),
        "relative_track_hold_state": getattr(sample, "relative_track_hold_state", None),
        "relative_track_hold_residual": getattr(sample, "relative_track_hold_residual", None),
        "relative_track_hold_output": getattr(sample, "relative_track_hold_output", None),
    }


class PVShadowTelemetryLogger(ShadowTelemetryLogger):
    """The core logger with the PV columns and schema version already wired."""

    def __init__(
        self,
        path: str | Path,
        *,
        clock: Callable[[], float] = time.perf_counter,
        log_prefix: str = "[pv-sidecar]",
    ):
        super().__init__(
            path,
            schema_version=PV_SHADOW_SCHEMA_VERSION,
            clock=clock,
            extra_fields=PV_SHADOW_FIELDS,
            extra_row=pv_shadow_row,
            log_prefix=log_prefix,
        )
