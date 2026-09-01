import pytest

from analyze_pv_track_hold import (
    fit_three_point_anchors,
    load_intent_trace,
    replay_intent_events,
    replay_steady_holds,
)


def _steady_trial(trial_index, label, values):
    return [
        {
            "row_type": "sample",
            "phase": "steady",
            "trial_status": "complete",
            "trial_index": trial_index,
            "target_label": label,
            "t": index / 30.0,
            "sum_kpa": value,
        }
        for index, value in enumerate(values)
    ]


def test_three_point_anchors_use_median_of_trial_medians():
    trials = {
        0: _steady_trial(0, "light", [10.0, 12.0]),
        1: _steady_trial(1, "light", [14.0, 16.0]),
        2: _steady_trial(2, "medium", [50.0, 54.0]),
        3: _steady_trial(3, "hard", [90.0, 94.0]),
    }

    anchors = fit_three_point_anchors(trials, discard_s=0.0)

    assert anchors == {"light": 13.0, "medium": 52.0, "hard": 92.0}


def test_steady_replay_freezes_output_despite_mapped_drift():
    trials = {
        0: _steady_trial(0, "light", [10.0 + index * 0.1 for index in range(120)]),
        1: _steady_trial(1, "medium", [50.0] * 120),
        2: _steady_trial(2, "hard", [90.0] * 120),
    }
    anchors = {"light": 10.0, "medium": 50.0, "hard": 90.0}

    _records, summaries = replay_steady_holds(
        trials,
        anchors=anchors,
        discard_s=0.0,
        max_closure=2.0,
    )

    assert max(summary["gripper_movement"] for summary in summaries) == 0.0
    assert sum(summary["track_transitions"] for summary in summaries) == 0


def test_intent_replay_reports_obvious_rise_response():
    times = [index / 30.0 for index in range(91)]
    values = [0.1 if t < 0.5 else min(0.9, 0.1 + 1.2 * (t - 0.5)) for t in times]

    _records, events = replay_intent_events(
        times,
        values,
        events=(("rise", 0.0),),
    )

    assert events[0]["response_s"] is not None
    assert events[0]["response_s"] < 1.0


def test_intent_trace_starts_at_right_grasp_latch():
    rows = [
        {
            "state": "MOVING",
            "base_gripper_pos": str(base),
            "control_observed_at_s": str(index / 30.0),
            "pressure": str(index / 10.0),
        }
        for index, base in enumerate((70.0, 40.0, 30.0, 20.0))
    ]

    times, values = load_intent_trace(rows)

    assert times[0] == 0.0
    assert values[0] == pytest.approx(0.2)
