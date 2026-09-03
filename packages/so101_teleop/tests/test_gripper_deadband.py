"""The deadband calibration's decision rules, exercised on recorded shapes.

Every case here is a shape the 2026-08-31 A/B collection actually produced: a
commanded step smaller than the encoder can resolve, a `Present_Load` channel
quantized to multiples of four and constant while holding, and a readback that
moves the wrong way while the carton settles. The point of the gate is to
report those honestly rather than to read a response into them.
"""

import pytest

from lerobot_teleoperator_so101_webcam.gripper_hardware import (
    DeadbandStep,
    breakout_offset,
    TelemetrySnapshot,
    rank_correlation,
    readback_spread,
    smallest_resolvable_step,
    tracking_ratio,
)


def _snapshot(pos, *, t=0.0, current=12, load=64):
    return TelemetrySnapshot(
        t=t,
        gripper_pos=pos,
        goal_gripper_pos=26.0,
        present_current=current,
        present_load=load,
        present_temperature=38,
    )


def _step(size, *, sign=-1.0, readback_delta=0.0):
    return DeadbandStep(
        step_size=size,
        commanded_delta=sign * size,
        readback_delta=readback_delta,
        readback_spread=0.0,
    )


def test_a_held_command_that_never_moved_has_a_zero_noise_floor():
    # trial03: readback held at exactly 26.426 for all 162 remaining steps.
    held = [_snapshot(26.426, t=index * 0.1) for index in range(162)]
    assert readback_spread(held) == 0.0


def test_readback_spread_needs_a_sample():
    with pytest.raises(ValueError):
        readback_spread([])


def test_the_02_step_resolves_nothing():
    # The four A/B slots: a 0.2 command and a readback that did not move.
    steps = [_step(0.2, readback_delta=delta) for delta in (0.0, 0.0, -0.047)]
    assert smallest_resolvable_step(steps, noise_floor=0.05) is None


def test_a_ramp_that_sticks_and_slips_is_not_resolved_by_its_step():
    """The trap this metric exists for.

    A 0.2 ramp does eventually advance -- static friction breaks once enough
    command error has piled up -- so scoring "any tread moved" would certify
    the exact step that moved nothing in the trials. Every tread must move.
    """
    stick_slip = [_step(0.2, readback_delta=d) for d in (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.7)]
    clean = [_step(3.0, readback_delta=d) for d in (-2.7, -2.8, -2.7)]
    assert smallest_resolvable_step(stick_slip + clean, noise_floor=0.05) == 3.0


def test_the_smallest_step_that_moves_every_tread_wins():
    steps = [
        _step(0.5, readback_delta=0.0),
        _step(1.0, readback_delta=-0.02),
        _step(2.0, readback_delta=-0.31),
        _step(2.0, readback_delta=-0.29),
        _step(5.0, readback_delta=-4.175),
    ]
    assert smallest_resolvable_step(steps, noise_floor=0.05) == 2.0


def test_the_breakout_offset_is_the_command_piled_up_before_the_jaw_moved():
    ramp = [_step(0.5, readback_delta=d) for d in (0.0, 0.0, 0.0, -1.6, -0.4)]
    assert breakout_offset(ramp, noise_floor=0.05) == pytest.approx(2.0)


def test_a_ramp_that_never_broke_out_reports_no_offset():
    ramp = [_step(0.2, readback_delta=0.0) for _ in range(5)]
    assert breakout_offset(ramp, noise_floor=0.05) is None


def test_a_deadband_swallowed_ramp_tracks_near_zero_and_a_free_one_near_one():
    swallowed = [_step(0.5, readback_delta=0.0) for _ in range(4)]
    free = [_step(3.0, readback_delta=-2.9) for _ in range(4)]
    assert tracking_ratio(swallowed) == 0.0
    assert tracking_ratio(free) == pytest.approx(2.9 / 3.0)


def test_tracking_ratio_needs_a_ramp():
    with pytest.raises(ValueError):
        tracking_ratio([])


def test_a_response_in_the_wrong_direction_does_not_count():
    # trial01: commanded +0.297, readback went -0.389. That is the carton
    # settling, not the step being resolved.
    steps = [_step(0.3, sign=+1.0, readback_delta=-0.389)]
    assert smallest_resolvable_step(steps, noise_floor=0.05) is None


def test_noise_floor_must_be_non_negative():
    with pytest.raises(ValueError):
        smallest_resolvable_step([_step(1.0, readback_delta=-1.0)], noise_floor=-1.0)


def test_a_step_size_must_be_positive():
    with pytest.raises(ValueError):
        DeadbandStep(step_size=0.0, commanded_delta=0.0, readback_delta=0.0, readback_spread=0.0)


def test_a_constant_load_channel_correlates_with_nothing():
    # Present_Load equalled 64.0 at every sample in trial03's hold.
    depth = [-26.4, -25.4, -24.4, -23.4]
    assert rank_correlation(depth, [64.0] * 4) == 0.0


def test_a_quantized_channel_that_tracks_depth_still_correlates():
    depth = [-26.4, -25.4, -24.4, -23.4]
    quantized = [64.0, 64.0, 68.0, 72.0]
    assert rank_correlation(depth, quantized) == pytest.approx(0.9486832, abs=1e-6)


def test_rank_correlation_rejects_mismatched_and_degenerate_input():
    with pytest.raises(ValueError):
        rank_correlation([1.0, 2.0], [1.0])
    with pytest.raises(ValueError):
        rank_correlation([1.0], [1.0])


def _sweep(contact_at, *, n=30, free_effort=(2, 3), rise=6.0):
    """A closing sweep: free space, then effort rising after contact."""
    positions, efforts = [], []
    for index in range(n):
        pos = 60.0 - index * 1.0
        positions.append(pos)
        if pos > contact_at:
            efforts.append(free_effort[index % len(free_effort)])
        else:
            efforts.append(free_effort[0] + rise * (contact_at - pos + 1))
    return positions, efforts


def test_contact_onset_is_the_position_where_effort_leaves_free_space():
    from lerobot_teleoperator_so101_webcam.gripper_hardware import find_contact_onset

    positions, efforts = _sweep(35.0)
    onset = find_contact_onset(positions, efforts, baseline_samples=12)
    assert onset.detected
    assert onset.position == pytest.approx(35.0)


def test_a_single_spike_is_not_contact():
    """One sample over threshold is what this bus does on a dropped packet."""
    from lerobot_teleoperator_so101_webcam.gripper_hardware import find_contact_onset

    positions, efforts = _sweep(40.0)          # contact at index 20, inside the sweep
    efforts[15] = 400                          # a spike while still in free space
    onset = find_contact_onset(positions, efforts, baseline_samples=12, consecutive=3)
    assert onset.position == pytest.approx(40.0), "the spike must not become the contact point"


def test_an_effort_channel_too_flat_to_see_contact_reports_nothing():
    """The honest answer for Present_Load quantized to multiples of four."""
    from lerobot_teleoperator_so101_webcam.gripper_hardware import find_contact_onset

    positions = [60.0 - i for i in range(30)]
    flat = [64] * 30
    onset = find_contact_onset(positions, flat, baseline_samples=12)
    assert onset.detected is False
    assert onset.position is None


def test_a_baseline_taken_while_already_touching_is_refused_by_length():
    from lerobot_teleoperator_so101_webcam.gripper_hardware import find_contact_onset

    with pytest.raises(ValueError):
        find_contact_onset([1.0, 2.0], [1, 2], baseline_samples=12)


def test_strain_is_the_travel_past_contact_over_the_object_width():
    from lerobot_teleoperator_so101_webcam.gripper_hardware import compression_strain

    assert compression_strain(30.0, 27.0, 60.0) == pytest.approx(0.05)
    assert compression_strain(30.0, 31.0, 60.0) < 0, "not yet in contact"
    with pytest.raises(ValueError):
        compression_strain(30.0, 27.0, 0.0)
