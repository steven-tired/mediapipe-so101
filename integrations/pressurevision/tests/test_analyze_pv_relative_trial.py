import csv

import pytest

from analyze_pv_relative_trial import (
    analyze_relative_trial,
    analyze_track_hold,
    write_gripper_position_artifacts,
    write_track_hold_artifacts,
)


def test_relative_trial_reports_only_signal_tracking_and_motor_metrics():
    rows = [
        {
            "state": "MOVING",
            "pressure_status": "active",
            "pressure": "0",
            "commanded_gripper_pos": "26",
            "motor_observed_at_s": "1.0",
            "observed_gripper_pos": "27",
            "present_current": "12",
            "present_temperature": "33",
        },
        {
            "state": "MOVING",
            "pressure_status": "pv_abstain_continuous",
            "pressure": "0.5",
            "commanded_gripper_pos": "25",
            "motor_observed_at_s": "1.0",
            "observed_gripper_pos": "27",
            "present_current": "12",
            "present_temperature": "33",
        },
        {
            "state": "MOVING",
            "pressure_status": "active",
            "pressure": "1",
            "commanded_gripper_pos": "24",
            "motor_observed_at_s": "1.2",
            "observed_gripper_pos": "25",
            "present_current": "-30",
            "present_temperature": "35",
        },
    ]

    report = analyze_relative_trial(rows)

    assert report["data_complete"] is True
    assert set(report) == {"data_complete", "signal", "tracking", "motor", "track_hold"}
    assert report["signal"]["zero_fraction"] == pytest.approx(1 / 3)
    assert report["tracking"]["motor_samples"] == 2
    assert report["tracking"]["median_absolute_error"] == 1.0
    assert report["motor"]["peak_absolute_current"] == 30.0
    assert report["motor"]["max_temperature_c"] == 35.0


def test_relative_trial_marks_old_sidecar_without_motor_readback_incomplete():
    report = analyze_relative_trial([
        {"state": "MOVING", "pressure_status": "active", "pressure": "0.4"}
    ])

    assert report["data_complete"] is False
    assert report["signal"]["zero_fraction"] == 0.0
    assert report["tracking"]["motor_samples"] == 0
    assert report["motor"]["peak_absolute_current"] is None


def test_gripper_position_artifacts_use_unique_motor_samples(tmp_path):
    rows = [
        {
            "motor_observed_at_s": "10.0",
            "commanded_gripper_pos": "30",
            "observed_gripper_pos": "31",
        },
        {
            "motor_observed_at_s": "10.0",
            "commanded_gripper_pos": "29",
            "observed_gripper_pos": "31",
        },
        {
            "motor_observed_at_s": "10.2",
            "commanded_gripper_pos": "28",
            "observed_gripper_pos": "29",
        },
    ]

    artifacts = write_gripper_position_artifacts(rows, tmp_path)

    with (tmp_path / "gripper_position.csv").open(newline="", encoding="utf-8") as handle:
        exported = list(csv.DictReader(handle))
    assert artifacts["rows"] == 2
    assert [float(row["time_s"]) for row in exported] == pytest.approx([0.0, 0.2])
    assert [float(row["absolute_error"]) for row in exported] == [1.0, 1.0]
    assert (tmp_path / "gripper_position.png").stat().st_size > 0


def test_track_hold_summary_and_artifacts(tmp_path):
    rows = [
        {
            "control_observed_at_s": str(10.0 + index * 0.1),
            "pressure": str(pressure),
            "relative_track_hold_residual": str(residual),
            "relative_track_hold_output": str(output),
            "relative_track_hold_state": state,
            "proposed_gripper_pos": str(30 - output * 2),
            "actual_gripper_pos": "31",
        }
        for index, (pressure, residual, output, state) in enumerate(
            [
                (0.2, 0.0, 0.2, "HOLD"),
                (0.25, 0.05, 0.2, "HOLD"),
                (0.5, 0.3, 0.3, "TRACK"),
                (0.6, 0.4, 0.4, "TRACK"),
                (0.45, 0.02, 0.4, "HOLD"),
            ]
        )
    ]

    summary = analyze_track_hold(rows)
    artifacts = write_track_hold_artifacts(rows, tmp_path)

    assert summary == {
        "valid_rows": 5,
        "state_rows": {"HOLD": 3, "TRACK": 2},
        "transitions": {"hold_to_track": 1, "track_to_hold": 1},
        "hold_episodes": 2,
        "max_within_hold_output_range": 0.0,
        "output_range": [0.2, 0.4],
    }
    with (tmp_path / "track_hold_timeseries.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        exported = list(csv.DictReader(handle))
    assert artifacts["rows"] == 5
    assert [float(row["time_s"]) for row in exported] == pytest.approx(
        [0.0, 0.1, 0.2, 0.3, 0.4]
    )
    assert (tmp_path / "track_hold_timeseries.png").stat().st_size > 0
