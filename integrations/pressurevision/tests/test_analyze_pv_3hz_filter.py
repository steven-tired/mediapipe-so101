import math

import pytest

from analyze_pv_3hz_filter import replay


def test_replay_initializes_at_reference_and_filters_without_overshoot():
    rows = [
        {
            "control_observed_at_s": "10.0",
            "relative_reference_pos": "20.0",
            "relative_closure": "2.0",
        },
        {
            "control_observed_at_s": str(10.0 + 1.0 / 30.0),
            "relative_reference_pos": "20.0",
            "relative_closure": "2.0",
        },
        {
            "control_observed_at_s": str(10.0 + 2.0 / 30.0),
            "relative_reference_pos": "20.0",
            "relative_closure": "0.0",
        },
    ]

    samples, summary = replay(rows, cutoff_hz=3.0)
    alpha = 1.0 - math.exp(-2.0 * math.pi * 3.0 / 30.0)

    assert samples[0]["raw_target_pos"] == 18.0
    assert samples[0]["filtered_target_pos"] == 20.0
    assert samples[1]["filtered_target_pos"] == pytest.approx(20.0 - 2.0 * alpha)
    assert all(18.0 <= row["filtered_target_pos"] <= 20.0 for row in samples)
    assert summary["cutoff_hz"] == 3.0
    assert summary["samples"] == 3
    assert summary["nominal_30hz"]["alpha"] == pytest.approx(alpha)


def test_replay_resets_filter_when_reference_changes():
    rows = [
        {
            "control_observed_at_s": "1.0",
            "relative_reference_pos": "20.0",
            "relative_closure": "2.0",
        },
        {
            "control_observed_at_s": "1.1",
            "relative_reference_pos": "30.0",
            "relative_closure": "2.0",
        },
    ]

    samples, _ = replay(rows, cutoff_hz=3.0)

    assert samples[1]["filtered_target_pos"] == 30.0
