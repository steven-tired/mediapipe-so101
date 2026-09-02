"""The generic half of the shadow-telemetry CSV.

The columns here are what every grip sensor shares. A sensor's own columns are
covered by that sensor's tests; what matters here is the base row, the tick and
loop-period bookkeeping, and above all that a logging failure never reaches the
control loop.
"""

import csv
from pathlib import Path
from types import SimpleNamespace

import pytest

from lerobot_teleoperator_so101_webcam.shadow_telemetry import (
    CONTROL_SHADOW_FIELDS,
    ShadowTelemetryLogger,
    ShadowTelemetrySample,
    first_attr,
)


def _reading(**overrides):
    fields = dict(
        pressure_0_1=0.5,
        active=True,
        quality=0.75,
        available=True,
        status="active",
        roi=SimpleNamespace(x=2, y=3, width=4, height=5),
        oak_observed_at_s=9.94,
        thermal_observed_at_s=9.96,
        sensor_skew_s=0.02,
        oak_age_s=0.06,
        thermal_age_s=0.04,
    )
    fields.update(overrides)
    return SimpleNamespace(**fields)


def _sample(*, control_observed_at_s=10.0, pressure=True):
    return ShadowTelemetrySample(
        control_observed_at_s=control_observed_at_s,
        state="MOVING",
        pinch=0.04,
        roi_mode="tips" if pressure else None,
        pressure=_reading() if pressure else None,
        baseline_ready=True,
        base_gripper_pos=60.0,
        proposed_gripper_pos=51.0,
        actual_gripper_pos=42.0,
        fault_latched=False,
        fallback_used=False,
        fallback_reason=None,
    )


def _logger(path, **kwargs):
    kwargs.setdefault("schema_version", "1")
    return ShadowTelemetryLogger(path, **kwargs)


def test_header_is_exact_and_diagnostics_stay_nullable(tmp_path: Path):
    path = tmp_path / "shadow.csv"
    clock = iter((10.005, 10.025)).__next__
    logger = _logger(path, clock=clock)

    logger.finalize(_sample(), command_sent=False)
    logger.finalize(_sample(control_observed_at_s=10.02, pressure=False), command_sent=True)
    logger.close()

    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)

    assert tuple(reader.fieldnames) == CONTROL_SHADOW_FIELDS
    assert rows[0]["schema_version"] == "1"
    assert [row["tick"] for row in rows] == ["0", "1"]
    assert float(rows[0]["sensor_skew_ms"]) == pytest.approx(20.0)
    assert float(rows[0]["oak_age_ms"]) == pytest.approx(60.0)
    assert float(rows[0]["thermal_age_ms"]) == pytest.approx(40.0)
    # The first row has no predecessor, so there is no loop period to report.
    assert rows[0]["loop_period_ms"] == ""
    assert float(rows[1]["loop_period_ms"]) == pytest.approx(20.0)
    assert float(rows[0]["control_latency_ms"]) == pytest.approx(5.0)
    assert rows[0]["roi_mode"] == "tips"
    assert [rows[0][n] for n in ("roi_x", "roi_y", "roi_width", "roi_height")] == ["2", "3", "4", "5"]
    assert rows[0]["command_sent"] == "false"
    assert rows[1]["command_sent"] == "true"
    # A frame with no reading writes blanks, never zeros: 0.0 pressure and
    # "no measurement" must stay distinguishable to a reader.
    assert rows[1]["oak_observed_at_s"] == ""
    assert rows[1]["sensor_skew_ms"] == ""
    assert rows[1]["roi_x"] == ""
    assert rows[1]["pressure"] == ""


def test_the_schema_version_is_the_caller_s(tmp_path: Path):
    path = tmp_path / "v9.csv"
    logger = _logger(path, schema_version="9", clock=lambda: 10.01)
    logger.finalize(_sample(), command_sent=True)
    logger.close()

    with path.open(newline="", encoding="utf-8") as handle:
        assert next(csv.DictReader(handle))["schema_version"] == "9"


def test_write_failure_prints_once_disables_and_never_raises(tmp_path: Path, capsys):
    """Losing telemetry is acceptable; interrupting a run holding an object is not."""
    logger = _logger(tmp_path / "shadow.csv", clock=lambda: 10.01, log_prefix="[test]")

    class BrokenWriter:
        def writerow(self, _row):
            raise OSError("disk unavailable")

    logger._writer = BrokenWriter()
    logger.finalize(_sample(), command_sent=True)
    logger.finalize(_sample(), command_sent=True)
    logger.close()

    assert not logger.enabled
    output = capsys.readouterr().out
    assert output.count("[test] disabled") == 1
    assert "disk unavailable" in output


def test_a_missing_sample_is_not_a_row(tmp_path: Path):
    path = tmp_path / "none.csv"
    logger = _logger(path, clock=lambda: 10.01)
    logger.finalize(None, command_sent=True)
    logger.close()

    with path.open(newline="", encoding="utf-8") as handle:
        assert list(csv.DictReader(handle)) == []


# --- the extension seam ---

def test_extra_fields_and_extra_row_must_arrive_together(tmp_path: Path):
    with pytest.raises(ValueError, match="together"):
        _logger(tmp_path / "a.csv", extra_fields=("x",))
    with pytest.raises(ValueError, match="together"):
        _logger(tmp_path / "b.csv", extra_row=lambda *a, **k: {})


def test_an_extension_appends_its_columns_after_the_shared_ones(tmp_path: Path):
    path = tmp_path / "ext.csv"
    logger = _logger(
        path,
        clock=lambda: 10.01,
        extra_fields=("sensor_x", "sensor_y"),
        extra_row=lambda sample, *, pressure, motor_telemetry, finalized_at_s: {
            "sensor_x": sample.pinch,
            "sensor_y": getattr(motor_telemetry, "present_load", None),
        },
    )
    logger.finalize(_sample(), command_sent=True, motor_telemetry=SimpleNamespace(present_load=31))
    logger.close()

    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        row = next(reader)

    assert tuple(reader.fieldnames) == CONTROL_SHADOW_FIELDS + ("sensor_x", "sensor_y")
    assert row["sensor_x"] == "0.04"
    assert row["sensor_y"] == "31"


def test_an_extension_that_returns_the_wrong_columns_disables_rather_than_raises(
    tmp_path: Path, capsys
):
    """A field-set mismatch is a programming error, but it surfaces through the
    fail-soft path so it cannot take the control loop down with it."""
    logger = _logger(
        tmp_path / "bad.csv",
        clock=lambda: 10.01,
        extra_fields=("sensor_x",),
        extra_row=lambda *a, **k: {"sensor_typo": 1.0},
        log_prefix="[test]",
    )

    logger.finalize(_sample(), command_sent=True)

    assert not logger.enabled
    assert "sensor_typo" in capsys.readouterr().out


# --- the two reading dialects ---

def test_a_column_resolves_whichever_name_the_reading_uses():
    """The thermal path and the PV protocol name the same timestamp
    differently; a column that read only one would come out empty for the other.
    """
    assert first_attr(SimpleNamespace(thermal_age_s=0.04), "thermal_age_s", "age_s") == 0.04
    assert first_attr(SimpleNamespace(age_s=0.04), "thermal_age_s", "age_s") == 0.04
    assert first_attr(SimpleNamespace(), "thermal_age_s", "age_s") is None
