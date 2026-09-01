import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import read_pad_snapshots as rp


def _rows(light, hard, none=(), blown=0.0):
    out = []
    for label, values in (("none", none), ("light", light), ("hard", hard)):
        for i, v in enumerate(values):
            out.append({
                "label": label, "index": i, "blown_percent": blown,
                "mean_kpa_in_contact": v, "sum_kpa": v * 100,
                "contact_px": int(v * 50), "max_kpa": v * 2,
            })
    return out


def test_separated_levels_report_a_large_dprime():
    report = rp.summarise(_rows(light=[4.0, 4.2, 3.8], hard=[12.0, 12.4, 11.6]))
    assert report["metrics"]["mean_kpa_in_contact"]["dprime_light_hard"] > 2.5


def test_overlapping_levels_report_a_small_dprime():
    """The case the operator is asking about: a gap that looks real until it is
    measured against how much each level wanders."""
    report = rp.summarise(_rows(light=[4.0, 8.0, 6.0], hard=[7.0, 5.0, 9.0]))
    assert report["metrics"]["mean_kpa_in_contact"]["dprime_light_hard"] < 2.5


def test_clipped_snapshots_are_counted_so_they_are_not_read_into():
    report = rp.summarise(_rows(light=[4.0, 4.2], hard=[12.0, 12.4], blown=5.0))
    assert report["blown_frames"] == 4


def test_a_single_press_per_level_cannot_be_judged():
    report = rp.summarise(_rows(light=[4.0], hard=[12.0]))
    d = report["metrics"]["mean_kpa_in_contact"]["dprime_light_hard"]
    assert d != d      # nan


def test_the_no_contact_level_is_compared_too():
    report = rp.summarise(
        _rows(light=[4.0, 4.2, 3.8], hard=[12.0, 12.4, 11.6], none=[0.0, 0.1, 0.0])
    )
    assert report["metrics"]["mean_kpa_in_contact"]["dprime_none_light"] > 2.5
    assert report["counts"]["none"] == 3


def test_every_metric_is_reported_not_just_the_chosen_one():
    report = rp.summarise(_rows(light=[4.0, 4.2], hard=[12.0, 12.4]))
    assert set(report["metrics"]) == set(rp.METRICS)


def _positioned(spec):
    """spec maps position -> {label: [values]}."""
    rows = []
    for position, labels in spec.items():
        for label, values in labels.items():
            for i, v in enumerate(values):
                rows.append({
                    "label": label, "index": i, "position": position,
                    "blown_percent": 0.0, "mean_kpa_in_contact": v,
                    "sum_kpa": v * 100, "contact_px": int(v * 50), "max_kpa": v * 2,
                })
    return rows


def test_positions_are_reported_before_pooling():
    """The complaint is that only some spots respond, and pooling a spot that
    works with one that does not gives a middling d' describing neither."""
    report = rp.summarise(_positioned({
        1: {"light": [3.0, 3.2, 3.1], "hard": [10.0, 10.4, 9.6]},   # works
        2: {"light": [0.0, 0.0, 0.1], "hard": [0.0, 0.2, 0.0]},     # dead
    }))

    assert report["by_position"][1]["dprime_light_hard"] > 2.5
    assert report["by_position"][2]["dprime_light_hard"] < 2.5


def test_pooling_hides_a_dead_position():
    """Why the split has to be reported per position: pooled, the two spots
    above give a number that looks workable while half the pad is dead."""
    spec = {
        1: {"light": [3.0, 3.2, 3.1], "hard": [10.0, 10.4, 9.6]},
        2: {"light": [0.0, 0.0, 0.1], "hard": [0.0, 0.2, 0.0]},
    }
    report = rp.summarise(_positioned(spec))
    pooled = report["metrics"]["mean_kpa_in_contact"]["dprime_light_hard"]

    assert pooled < report["by_position"][1]["dprime_light_hard"]


def test_an_unlabelled_capture_still_reports_one_position():
    rows = _rows(light=[4.0, 4.2, 3.8], hard=[12.0, 12.4, 11.6])
    report = rp.summarise(rows)

    assert list(report["by_position"]) == [1]
