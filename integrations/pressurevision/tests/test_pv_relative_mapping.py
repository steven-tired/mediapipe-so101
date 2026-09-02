import math

import pytest

from pressurevision_integration.protocol import PressureReading
from pressurevision_integration.pv_relative_mapping import (
    RELATIVE_PRESSURE_LOW_PASS_HZ,
    PressureRangeMapper,
    PressureTrackHoldStabilizer,
    RelativePressureMapper,
)


def _pressure(value=0.0, *, active=False, available=True):
    return PressureReading(
        pressure_0_1=value,
        active=active,
        quality=1.0,
        available=available,
        status="active" if active else "baseline",
    )


def test_relative_mapping_uses_pinch_only_as_binary_gate_and_latches_readback():
    mapper = RelativePressureMapper(max_closure=2.0)

    inactive = mapper.update(
        base_gripper_pos=70.0,
        pressure=_pressure(1.0, active=True),
        observed_gripper_pos=28.0,
    )
    waiting = mapper.update(
        base_gripper_pos=20.0,
        pressure=_pressure(active=False),
        observed_gripper_pos=27.0,
    )
    active = mapper.update(
        base_gripper_pos=0.0,
        pressure=_pressure(0.75, active=True),
        observed_gripper_pos=27.0,
    )
    changed_pinch = mapper.update(
        base_gripper_pos=25.0,
        pressure=_pressure(0.75, active=True),
        observed_gripper_pos=24.0,
    )

    assert inactive.status == "right_grasp_inactive"
    assert waiting.status == "waiting_pressure"
    assert active.reference_pos == 27.0
    assert active.relative_closure == pytest.approx(1.5)
    assert active.target_pos == pytest.approx(25.5)
    assert changed_pinch.target_pos == pytest.approx(25.5)


def test_soft_direct_maps_full_pressure_range_and_right_release_opens():
    mapper = PressureRangeMapper(
        release_pos=100.0,
        pressure_zero_pos=100.0,
        pressure_one_pos=0.0,
    )

    released = mapper.update(
        base_gripper_pos=80.0,
        pressure=_pressure(1.0, active=True),
    )
    zero = mapper.update(
        base_gripper_pos=20.0,
        pressure=_pressure(0.0, active=True),
    )
    full = mapper.update(
        base_gripper_pos=20.0,
        pressure=_pressure(1.0, active=True),
    )
    opened = mapper.update(
        base_gripper_pos=80.0,
        pressure=_pressure(1.0, active=True),
    )

    assert released.target_pos == 100.0
    assert zero.target_pos == 100.0
    assert full.target_pos == 0.0
    assert full.relative_closure == 100.0
    assert opened.status == "right_grasp_inactive"
    assert opened.target_pos == 100.0


def test_hard_profile_maps_only_the_labeled_light_to_hard_range():
    mapper = PressureRangeMapper(
        release_pos=95.0,
        pressure_zero_pos=25.0,
        pressure_one_pos=23.5,
    )

    half = mapper.update(
        base_gripper_pos=20.0,
        pressure=_pressure(0.5, active=True),
    )

    assert half.reference_pos == 25.0
    assert half.relative_closure == pytest.approx(0.75)
    assert half.target_pos == pytest.approx(24.25)


def test_range_mapping_holds_when_pv_is_unavailable_during_grasp():
    mapper = PressureRangeMapper(
        release_pos=100.0,
        pressure_zero_pos=100.0,
        pressure_one_pos=0.0,
    )

    decision = mapper.update(
        base_gripper_pos=20.0,
        pressure=_pressure(0.5, active=True, available=False),
    )

    assert decision.status == "waiting_pressure"
    assert decision.target_pos is None


def test_relative_mapping_waits_for_readback_and_resets_on_release():
    mapper = RelativePressureMapper(max_closure=2.0)

    waiting = mapper.update(
        base_gripper_pos=20.0,
        pressure=_pressure(1.0, active=True),
        observed_gripper_pos=None,
    )
    latched = mapper.update(
        base_gripper_pos=20.0,
        pressure=_pressure(1.0, active=True),
        observed_gripper_pos=26.0,
    )
    released = mapper.update(
        base_gripper_pos=65.0,
        pressure=_pressure(1.0, active=True),
        observed_gripper_pos=26.0,
    )

    assert waiting.status == "waiting_position_readback"
    assert latched.target_pos == 24.0
    assert released.status == "right_grasp_inactive"
    assert released.reference_pos is None


def test_relative_mapping_does_not_latch_a_position_sample_from_before_pressure():
    mapper = RelativePressureMapper(max_closure=2.0)

    stale = mapper.update(
        base_gripper_pos=20.0,
        pressure=_pressure(0.5, active=True),
        observed_gripper_pos=40.0,
        control_observed_at_s=10.0,
        motor_observed_at_s=9.9,
    )
    fresh = mapper.update(
        base_gripper_pos=20.0,
        pressure=_pressure(0.5, active=True),
        observed_gripper_pos=27.0,
        control_observed_at_s=10.1,
        motor_observed_at_s=10.05,
    )

    assert stale.status == "waiting_position_readback"
    assert fresh.reference_pos == 27.0
    assert fresh.target_pos == 26.0


def test_relative_mapping_time_aware_low_pass_starts_at_reference_and_has_no_overshoot():
    mapper = RelativePressureMapper(
        max_closure=2.0,
        cutoff_hz=RELATIVE_PRESSURE_LOW_PASS_HZ,
    )

    first = mapper.update(
        base_gripper_pos=20.0,
        pressure=_pressure(1.0, active=True),
        observed_gripper_pos=27.0,
        control_observed_at_s=10.0,
    )
    second = mapper.update(
        base_gripper_pos=20.0,
        pressure=_pressure(1.0, active=True),
        observed_gripper_pos=27.0,
        control_observed_at_s=10.0 + 1.0 / 30.0,
    )
    baseline = mapper.update(
        base_gripper_pos=20.0,
        pressure=_pressure(0.0, active=False),
        observed_gripper_pos=27.0,
        control_observed_at_s=10.0 + 2.0 / 30.0,
    )

    alpha = 1.0 - math.exp(-2.0 * math.pi * 3.0 / 30.0)
    assert first.target_pos == 27.0
    assert second.relative_closure == pytest.approx(2.0 * alpha)
    assert second.target_pos == pytest.approx(27.0 - 2.0 * alpha)
    assert second.target_pos < baseline.target_pos < 27.0
    assert baseline.status == "holding_reference"


def test_relative_mapping_optional_track_hold_stabilizes_the_shadow_target():
    mapper = RelativePressureMapper(max_closure=2.0, cutoff_hz=3.0, stabilize=True)

    decisions = [
        mapper.update(
            base_gripper_pos=20.0,
            pressure=_pressure(index / 30.0, active=True),
            observed_gripper_pos=27.0,
            control_observed_at_s=10.0 + index / 30.0,
        )
        for index in range(31)
    ]

    assert decisions[0].track_hold.state == "HOLD"
    assert decisions[0].target_pos == 27.0
    assert any(decision.track_hold.state == "TRACK" for decision in decisions)
    assert decisions[-1].target_pos < 27.0
    assert all(25.0 <= decision.target_pos <= 27.0 for decision in decisions)


def test_track_hold_freezes_slow_drift_exactly():
    stabilizer = PressureTrackHoldStabilizer()

    decisions = [
        stabilizer.update(0.5 + 0.08 * index / 90.0, index / 30.0)
        for index in range(91)
    ]

    assert {decision.state for decision in decisions} == {"HOLD"}
    assert {decision.output_value for decision in decisions} == {0.5}


def test_track_hold_tracks_an_obvious_change_then_freezes_the_new_value():
    stabilizer = PressureTrackHoldStabilizer()
    decisions = []
    for index in range(121):
        t = index / 30.0
        if t < 1.0:
            value = 0.2
        elif t < 1.5:
            value = 0.2 + 1.2 * (t - 1.0)
        else:
            value = 0.8
        decisions.append(stabilizer.update(value, t))

    transitions = [decision.transition for decision in decisions if decision.transition]
    outputs = [decision.output_value for decision in decisions]
    final_hold = next(
        index
        for index, decision in enumerate(decisions)
        if decision.transition == "TRACK_TO_HOLD"
    )

    assert "HOLD_TO_TRACK" in transitions
    assert "TRACK_TO_HOLD" in transitions
    assert max(outputs) > 0.5
    assert len(set(outputs[final_hold:])) == 1


def test_track_hold_rejects_non_monotonic_time_and_invalid_hysteresis():
    with pytest.raises(ValueError, match="exit_delta"):
        PressureTrackHoldStabilizer(enter_delta=0.1, exit_delta=0.1)

    stabilizer = PressureTrackHoldStabilizer()
    stabilizer.update(0.5, 2.0)
    with pytest.raises(ValueError, match="monotonic"):
        stabilizer.update(0.6, 1.0)
