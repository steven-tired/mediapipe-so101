"""Paired lift and slip boundaries collected within one grasp.

The steps are the 2026-09-02 calibration's: tighten 2.0, loosen 0.5. The point
of pairing them inside one grasp is that their order is not fixed — trial01
lifted and then slipped, trial05 suggests the reverse — and only a within-grasp
comparison can say which way ACT's own threshold errs.
"""

import pytest

from lerobot_teleoperator_so101_webcam.grip.runtime import (
    LoosenRampConfig,
    PairedBoundaryProtocol,
    StallConfig,
    TightenRampConfig,
)

STALLED = [-19.32, 35.82]


def lifting(t):
    return [-19.32 - 1.2 * t, 35.82 - 2.8 * t]


def _protocol(**loosen_overrides):
    loosen = dict(step=0.5, interval_s=0.5, ceiling_pos=60.0)
    loosen.update(loosen_overrides)
    return PairedBoundaryProtocol(
        tighten=TightenRampConfig(step=2.0, interval_s=1.0, floor_pos=20.0),
        loosen=LoosenRampConfig(**loosen),
        stall=StallConfig(window_s=2.0),
    )


def _run(protocol, *, seconds, body_at, policy_target=28.0, hz=10.0, events=()):
    """Drive the protocol, firing operator events at their scheduled times."""
    pending = sorted(events)
    actual = policy_target
    trace = []
    for index in range(int(seconds * hz)):
        t = index / hz
        while pending and pending[0][0] <= t:
            _, event = pending.pop(0)
            getattr(protocol, event)()
        target, label = protocol.update(
            t=t, policy_target=policy_target, actual_pos=actual, body_command=body_at(t)
        )
        actual = target
        trace.append((t, target, label))
    return trace


def test_a_stall_is_tightened_and_the_depth_that_broke_it_is_the_lift_boundary():
    protocol = _protocol()
    _run(protocol, seconds=10.0, body_at=lambda t: STALLED if t < 6.0 else lifting(t - 6.0))
    assert protocol.lift_boundary is not None
    assert protocol.lift_boundary < 28.0


def test_the_loosen_ramp_runs_after_a_lift_and_stops_on_the_drop():
    protocol = _protocol()
    trace = _run(
        protocol,
        seconds=8.0,
        body_at=lambda t: lifting(t),
        events=[(1.0, "confirm_lift"), (5.0, "mark_drop")],
    )
    loosens = [label for _, _, label in trace if label["action"] == "loosen"]
    assert loosens, "the ramp never stepped"
    assert all(label["delta_q"] == pytest.approx(0.5) for label in loosens)
    assert protocol.slip_boundary is not None
    assert protocol.slip_boundary > 28.0, "loosening opens the jaw"
    assert protocol.phase == "done"


def test_the_slip_boundary_is_readback_and_not_the_command():
    """Only about 90% of a commanded loosen becomes travel on this gripper."""
    protocol = _protocol()
    protocol.confirm_lift()
    actual = 28.0
    last_ramp_target = jaw_at_drop = None
    for index in range(40):
        if index == 30:
            protocol.mark_drop()
            jaw_at_drop = actual
        target, label = protocol.update(
            t=index / 10.0, policy_target=28.0, actual_pos=actual, body_command=lifting(index / 10.0)
        )
        if label["phase"] == "loosening":
            last_ramp_target = target
        # The jaw lags its command, as it does on the bench.
        actual += 0.9 * (target - actual)

    assert protocol.slip_boundary == pytest.approx(jaw_at_drop)
    # Loosening opens the jaw and the jaw trails the command, so the readback
    # is the tighter of the two. Recording the command would overstate how far
    # open the grasp actually was when it let go.
    assert protocol.slip_boundary < last_ramp_target


def test_the_body_is_frozen_only_while_loosening():
    """A carton ACT has already set down cannot be dropped."""
    protocol = _protocol()
    trace = _run(
        protocol,
        seconds=6.0,
        body_at=lambda t: lifting(t),
        events=[(1.0, "confirm_lift"), (4.0, "mark_drop")],
    )
    frozen = [t for t, _, label in trace if label["freeze_body"]]
    assert min(frozen) >= 1.0
    assert max(frozen) < 4.1
    assert protocol.freeze_body is False


def test_the_loosen_ramp_stops_at_the_ceiling_without_a_boundary():
    protocol = _protocol(ceiling_pos=30.0)
    trace = _run(
        protocol, seconds=8.0, body_at=lambda t: lifting(t), events=[(0.5, "confirm_lift")]
    )
    assert trace[-1][2]["at_ceiling"] is True
    assert protocol.slip_boundary is None, "nothing dropped, so nothing is labelled"
    assert trace[-1][1] == pytest.approx(30.0)


def test_the_whole_ramp_is_kept_because_the_keypress_lags_the_drop():
    protocol = _protocol()
    _run(
        protocol,
        seconds=6.0,
        body_at=lambda t: lifting(t),
        events=[(1.0, "confirm_lift"), (4.0, "mark_drop")],
    )
    loosening = [row for row in protocol.trace if row["phase"] == "loosening"]
    assert len(loosening) > 10
    assert all("t" in row and "actual_pos" in row for row in loosening)


def test_a_lift_confirmed_without_any_stall_still_collects_a_slip_boundary():
    """Both branches continue into the loosen ramp, or the two boundaries would
    be measured on disjoint populations and could never be compared."""
    protocol = _protocol()
    _run(
        protocol,
        seconds=6.0,
        body_at=lambda t: lifting(t),
        events=[(0.5, "confirm_lift"), (4.0, "mark_drop")],
    )
    assert protocol.lift_boundary is None, "ACT lifted on its own; no tightening bought it"
    assert protocol.slip_boundary is not None


def test_the_steps_must_be_sized_from_a_calibration():
    with pytest.raises(ValueError):
        LoosenRampConfig(step=0.0, interval_s=0.5)
    with pytest.raises(ValueError):
        LoosenRampConfig(step=0.5, interval_s=0.0)
    with pytest.raises(ValueError):
        LoosenRampConfig(step=0.5, interval_s=0.5, ceiling_pos=101.0)
