"""The PressureVision columns — dataset schema v7.

These columns are what `research/train_grip_residual_head.py` trains on, so a
column that silently comes out empty is a training bug, not a logging one. The
tests therefore pin the wire columns against the real `PressureReading` rather
than a duck-typed stand-in.
"""

import csv
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from lerobot_teleoperator_so101_webcam.shadow_telemetry import CONTROL_SHADOW_FIELDS
from pressurevision_integration.protocol import PressureReading, PressureROI
from pressurevision_integration.pv_shadow_telemetry import (
    PV_SHADOW_FIELDS,
    PV_SHADOW_SCHEMA_VERSION,
    PVShadowTelemetryLogger,
    PVShadowTelemetrySample,
)

READING = PressureReading(
    pressure_0_1=0.5,
    active=True,
    quality=0.75,
    available=True,
    status="active",
    roi=PressureROI(x=2, y=3, width=4, height=5),
    roi_mode="tips",
    observed_at_s=9.96,
    age_s=0.04,
    sequence=12,
    sent_at_s=9.97,
    received_at_s=9.98,
)

MOTOR = SimpleNamespace(
    observed_at_s=9.99,
    observed_gripper_pos=27.4,
    present_current=18,
    present_load=31,
    present_temperature=34,
)


def _sample(**overrides):
    fields = dict(
        control_observed_at_s=10.0,
        state="MOVING",
        pinch=0.04,
        roi_mode="tips",
        pressure=READING,
        baseline_ready=True,
        base_gripper_pos=60.0,
        proposed_gripper_pos=51.0,
        actual_gripper_pos=42.0,
        fault_latched=False,
        fallback_used=False,
        fallback_reason=None,
    )
    fields.update(overrides)
    return PVShadowTelemetrySample(**fields)


def _row(tmp_path: Path, sample, *, motor_telemetry=MOTOR, name="pv.csv"):
    path = tmp_path / name
    logger = PVShadowTelemetryLogger(path, clock=lambda: 10.01)
    logger.finalize(sample, command_sent=True, motor_telemetry=motor_telemetry)
    logger.close()
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return next(reader), tuple(reader.fieldnames)


def test_the_schema_version_is_frozen_at_the_recorded_evidence():
    """`local/evidence/` was recorded at "7" and a reader keys off it; bumping
    this invalidates that evidence rather than extending it."""
    assert PV_SHADOW_SCHEMA_VERSION == "7"


def test_pv_columns_follow_the_shared_ones(tmp_path: Path):
    _, fieldnames = _row(tmp_path, _sample())

    assert fieldnames == CONTROL_SHADOW_FIELDS + PV_SHADOW_FIELDS


def test_the_wire_columns_are_populated_by_a_real_pv_reading(tmp_path: Path):
    """The regression this exists for: PressureReading names these `sequence`,
    `sent_at_s`, `observed_at_s`, while the columns keep their v7 names. Reading
    only the v7 names would leave every wire column blank without an error."""
    row, _ = _row(tmp_path, _sample())

    assert row["schema_version"] == "7"
    assert row["pv_sequence"] == "12"
    assert row["pv_source_observed_at_s"] == "9.96"
    assert row["pv_sent_at_s"] == "9.97"
    assert row["pv_received_at_s"] == "9.98"
    assert float(row["pv_frame_age_ms"]) == pytest.approx(40.0)
    # The same timestamp also fills the shared column it has always shared.
    assert row["thermal_observed_at_s"] == "9.96"


def test_the_command_is_distinguished_from_the_rate_limited_motor_readback(tmp_path: Path):
    row, _ = _row(tmp_path, _sample())

    assert row["actual_gripper_pos"] == "42.0"
    assert row["commanded_gripper_pos"] == "42.0"
    assert float(row["motor_sample_age_ms"]) == pytest.approx(20.0)
    assert row["motor_sample_valid"] == "true"
    assert row["observed_gripper_pos"] == "27.4"
    assert row["observed_gripper_pos_valid"] == "true"
    assert row["present_current"] == "18"
    assert row["present_current_valid"] == "true"
    assert row["present_load"] == "31"
    assert row["present_load_valid"] == "true"
    assert row["present_temperature"] == "34"
    assert row["present_temperature_valid"] == "true"


def test_a_missing_motor_sample_is_marked_not_invented(tmp_path: Path):
    row, _ = _row(tmp_path, _sample(), motor_telemetry=None)

    assert row["motor_observed_at_s"] == ""
    assert row["motor_sample_age_ms"] == ""
    assert row["motor_sample_valid"] == "false"
    assert row["observed_gripper_pos_valid"] == "false"
    assert row["present_current_valid"] == "false"
    assert row["present_load_valid"] == "false"
    assert row["present_temperature_valid"] == "false"


def test_the_adjustment_lock_state_is_written_including_its_absences(tmp_path: Path):
    """These are the teacher-label columns. A `None` must reach the CSV as a
    blank: a recontact clock that has not started is not a recontact at t=0."""
    row, _ = _row(
        tmp_path,
        _sample(
            pv_adjustment_state="temporary_hold",
            pv_adjustment_event="contact_lost",
            pv_adjustment_anchor_target=27.0,
            pv_adjustment_release_since_s=9.95,
            pv_adjustment_release_elapsed_s=0.05,
            pv_adjustment_last_contact_at_s=9.94,
            pv_adjustment_recontact_since_s=None,
        ),
    )

    assert row["pv_adjustment_state"] == "temporary_hold"
    assert row["pv_adjustment_event"] == "contact_lost"
    assert row["pv_adjustment_anchor_target"] == "27.0"
    assert row["pv_adjustment_release_since_s"] == "9.95"
    assert row["pv_adjustment_release_elapsed_s"] == "0.05"
    assert row["pv_adjustment_last_contact_at_s"] == "9.94"
    assert row["pv_adjustment_recontact_since_s"] == ""


def test_the_relative_mapping_columns_round_trip(tmp_path: Path):
    row, _ = _row(
        tmp_path,
        _sample(
            relative_reference_pos=27.0,
            relative_closure=1.5,
            relative_mapping_status="active",
            relative_track_hold_state="HOLD",
            relative_track_hold_residual=0.04,
            relative_track_hold_output=0.75,
        ),
    )

    assert row["relative_reference_pos"] == "27.0"
    assert row["relative_closure"] == "1.5"
    assert row["relative_mapping_status"] == "active"
    assert row["relative_track_hold_state"] == "HOLD"
    assert row["relative_track_hold_residual"] == "0.04"
    assert row["relative_track_hold_output"] == "0.75"


def test_a_frame_with_no_pv_reading_still_writes_a_complete_row(tmp_path: Path):
    """A dead sender must not cost the row. The pressure columns go blank; the
    control columns the recorder needs stay."""
    row, fieldnames = _row(tmp_path, _sample(pressure=None, roi_mode=None))

    assert set(row) == set(fieldnames)
    assert row["pv_sequence"] == ""
    assert row["pressure"] == ""
    assert row["base_gripper_pos"] == "60.0"
    assert row["commanded_gripper_pos"] == "42.0"


def test_an_inactive_reading_keeps_its_provenance(tmp_path: Path):
    """`active=False` means "no usable pressure", not "no packet" — the wire
    columns still identify which packet said so."""
    row, _ = _row(tmp_path, _sample(pressure=replace(READING, active=False, status="no_contact")))

    assert row["pressure_status"] == "no_contact"
    assert row["pv_sequence"] == "12"


# --- assembling a row from the runtime ---

class _BaselineThenPress:
    def __init__(self):
        self.calls = 0

    def update(self, landmarks, *, pinch, enabled):
        self.calls += 1
        if self.calls == 1:
            return PressureReading(
                pressure_0_1=0.0, active=False, quality=1.0,
                available=True, status="baseline",
            )
        return PressureReading(
            pressure_0_1=0.5, active=True, quality=1.0,
            available=True, status="active", sequence=self.calls,
        )


def _runtime():
    from pressurevision_integration.pv_grip_controller import PressureVisionGripRuntime

    return PressureVisionGripRuntime(
        _BaselineThenPress(),
        initial_gripper=50.0,
        middle_gripper=50.0,
        pv_mapping="carton_span",
        pressure_apply=True,
    )


def _drive(runtime, frames=12):
    import numpy as np

    command = 50.0
    for tick in range(frames):
        command = runtime.update(
            base_gripper=20.0,
            landmarks=np.zeros((21, 3)),
            pinch=0.03,
            enabled=True,
            current_command=command,
            observed_at_s=tick * 0.1,
        ).actual_gripper
    return command


def test_a_frame_before_the_first_decision_has_nothing_to_report():
    """Inventing a row would put a fabricated baseline into the training data."""
    from pressurevision_integration.pv_shadow_telemetry import pv_shadow_sample

    assert pv_shadow_sample(
        _runtime(), control_observed_at_s=1.0, state="MOVING", pinch=0.03
    ) is None


def test_the_sample_carries_the_runtime_s_decision_and_lock_state():
    from pressurevision_integration.pv_shadow_telemetry import pv_shadow_sample

    runtime = _runtime()
    command = _drive(runtime)
    sample = pv_shadow_sample(
        runtime, control_observed_at_s=1.2, state="MOVING", pinch=0.03
    )

    assert sample.actual_gripper_pos == pytest.approx(command)
    assert sample.base_gripper_pos == pytest.approx(20.0)
    assert sample.pressure_mode == "pv_carton_span_apply"
    assert sample.baseline_ready is True
    assert sample.fallback_used is False
    assert sample.pv_adjustment_state is not None


def test_the_assembled_sample_writes_a_complete_v7_row(tmp_path: Path):
    """The end the evidence actually depends on: runtime state in, v7 row out."""
    from pressurevision_integration.pv_shadow_telemetry import pv_shadow_sample

    runtime = _runtime()
    _drive(runtime)
    sample = pv_shadow_sample(
        runtime, control_observed_at_s=1.2, state="MOVING", pinch=0.03
    )
    row, fieldnames = _row(tmp_path, sample, motor_telemetry=None, name="assembled.csv")

    assert fieldnames == CONTROL_SHADOW_FIELDS + PV_SHADOW_FIELDS
    assert row["schema_version"] == "7"
    assert row["pressure_mode"] == "pv_carton_span_apply"
    assert row["pv_sequence"] != ""
