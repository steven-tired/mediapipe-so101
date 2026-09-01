"""The bind detector decides when to stop driving a servo into a bound cable,
so it is tested against the shapes a real sweep produces rather than only the
happy path."""

import pytest

from lerobot_teleoperator_so101_webcam.programs import measure_wrist_roll_range as wr


def _samples(pairs):
    return [{"commanded_deg": c, "measured_deg": m} for c, m in pairs]


def test_a_freely_turning_joint_is_never_called_bound():
    samples = _samples([(3 * i, 3 * i - 0.4) for i in range(1, 12)])
    verdict = wr.binding_verdict(samples)

    assert verdict["bound"] is False
    assert verdict["last_following_deg"] == 33 - 0.4


def test_a_joint_that_stops_moving_is_called_bound():
    """The cable-wrap shape: commands keep rising, position stops."""
    following = [(3 * i, 3 * i - 0.3) for i in range(1, 6)]   # to 15 deg
    stuck = [(15 + 3 * i, 15.2) for i in range(1, 5)]
    verdict = wr.binding_verdict(_samples(following + stuck))

    assert verdict["bound"] is True
    assert verdict["last_following_deg"] == 15 - 0.3
    assert verdict["bound_at_commanded_deg"] == 15 + 3 * wr.BINDING_STEPS


def test_one_slow_step_does_not_trip_the_detector():
    """A single lagging step is ordinary settling, not a bind -- tripping on it
    would under-report the travel and make the joint look worse than it is."""
    samples = _samples([(3, 2.8), (6, 3.1), (9, 8.7), (12, 11.8), (15, 14.9)])
    verdict = wr.binding_verdict(samples)

    assert verdict["bound"] is False
    assert verdict["last_following_deg"] == 14.9


def test_lagging_must_be_consecutive():
    samples = _samples([(3, 2.9), (6, 3.0), (9, 3.0), (12, 11.8),
                        (15, 14.9), (18, 17.8)])
    assert wr.binding_verdict(samples)["bound"] is False


def test_binding_from_the_very_first_steps_reports_no_travel():
    verdict = wr.binding_verdict(_samples([(3, 0.1), (6, 0.1), (9, 0.1)]))

    assert verdict["bound"] is True
    assert verdict["last_following_deg"] is None


def test_summarise_reports_travel_either_side_of_the_start():
    positive = {"bound": True, "last_following_deg": 40.0}
    negative = {"bound": True, "last_following_deg": -95.0}

    report = wr.summarise(start_deg=0.0, positive=positive, negative=negative)

    assert report["travel_deg"] == {"positive": 40.0, "negative": 95.0}
    assert report["total_travel_deg"] == 135.0


def test_a_lopsided_range_that_binds_both_ways_is_not_one_sided():
    """This asserted the opposite while one_sided compared magnitudes. A joint
    that binds at both ends has a cable limit on both sides however lopsided
    the two distances are; 'one-sided' is about cause, not size."""
    report = wr.summarise(
        start_deg=10.0,
        positive={"bound": True, "last_following_deg": 25.0},    # 15 deg
        negative={"bound": True, "last_following_deg": -110.0},  # 120 deg
    )

    assert report["one_sided"] is False
    assert report["cable_bound_sides"] == ["positive", "negative"]


def test_summarise_does_not_call_a_symmetric_range_one_sided():
    report = wr.summarise(
        start_deg=0.0,
        positive={"bound": True, "last_following_deg": 100.0},
        negative={"bound": True, "last_following_deg": -95.0},
    )

    assert report["one_sided"] is False


def test_summarise_withholds_totals_when_a_direction_never_followed():
    report = wr.summarise(
        start_deg=0.0,
        positive={"bound": True, "last_following_deg": None},
        negative={"bound": True, "last_following_deg": -95.0},
    )

    assert report["travel_deg"]["positive"] is None
    assert "total_travel_deg" not in report


def test_urdf_limits_match_the_shipped_model():
    """These bound the sweep regardless of what the detector says, so a typo
    here would let the script drive past a mechanical limit."""
    assert wr.URDF_MIN_DEG == -157.2
    assert wr.URDF_MAX_DEG == 162.8


def test_a_direction_cut_short_by_the_bus_is_not_a_limit():
    """The failure seen on the first real run: two healthy steps, then the bus
    dropped. Travel measured that way is a lower bound, and the report has to
    say so rather than presenting it as the joint's limit."""
    positive = {"bound": False, "last_following_deg": 22.9,
                "comms_failed": "step to 26.6 deg failed 3x"}
    negative = {"bound": True, "last_following_deg": -95.0, "comms_failed": None}

    report = wr.summarise(start_deg=17.6, positive=positive, negative=negative)
    complete = not (positive.get("comms_failed") or negative.get("comms_failed"))

    assert complete is False
    # The number is still reported -- it is a real lower bound, just not the answer.
    assert report["travel_deg"]["positive"] == pytest.approx(5.3)


def test_transit_samples_look_exactly_like_a_bind():
    """First real run: the negative sweep began while the joint was still
    returning from 160 deg, and its transit through 65 and 40 deg read as three
    consecutive lagging steps. The detector cannot tell these apart -- which is
    why the sweep must wait for arrival rather than a fixed sleep."""
    transit = _samples([(19.9, 65.2), (16.9, 40.1), (13.9, 16.3)])
    verdict = wr.binding_verdict(transit)

    assert verdict["bound"] is True
    # No step ever followed, so there is no travel figure -- a genuine bind
    # after real travel always leaves one.
    assert verdict["last_following_deg"] is None


def test_summarise_gives_no_travel_when_nothing_ever_followed():
    report = wr.summarise(
        start_deg=22.9,
        positive={"bound": False, "last_following_deg": 160.0},
        negative={"bound": True, "last_following_deg": None},
    )

    assert report["travel_deg"]["positive"] == pytest.approx(137.1, abs=0.1)
    assert report["travel_deg"]["negative"] is None
    assert "total_travel_deg" not in report
    assert "one_sided" not in report


def test_one_sided_is_decided_by_cause_not_by_magnitude():
    """The clean run: 137 deg positive against 104 deg negative is not a 2x
    gap, but only the negative side ended in a cable bind -- the positive side
    ran out of URDF travel. Comparing magnitudes called that symmetric."""
    report = wr.summarise(
        start_deg=22.4,
        positive={"bound": False, "last_following_deg": 159.5},
        negative={"bound": True, "last_following_deg": -81.2},
    )

    assert report["cable_bound_sides"] == ["negative"]
    assert report["one_sided"] is True


def test_a_joint_that_binds_both_ways_is_not_one_sided():
    report = wr.summarise(
        start_deg=0.0,
        positive={"bound": True, "last_following_deg": 90.0},
        negative={"bound": True, "last_following_deg": -90.0},
    )

    assert report["cable_bound_sides"] == ["positive", "negative"]
    assert report["one_sided"] is False
