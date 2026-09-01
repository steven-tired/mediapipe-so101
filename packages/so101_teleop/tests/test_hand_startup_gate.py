"""The startup gate measures a *continuous* hold, not a total.

It arrived from the IR branch with no tests. The behaviour worth pinning is the
reset: a gate that accumulated across dropouts would unlock the arm for a hand
that flickered in and out, which is exactly the case it exists to catch.
"""

import pytest

from lerobot_teleoperator_so101_webcam.hand_startup_gate import (
    HAND_STARTUP_DWELL_S,
    MAX_WRIST_ROLL_RANGE_DEG,
    ContinuousHandStartupGate,
)


def test_an_unseen_hand_holds_at_zero():
    gate = ContinuousHandStartupGate()

    assert gate.update(hand_valid=False, observed_at_s=10.0) == 0.0


def test_held_time_accumulates_from_the_first_sighting():
    gate = ContinuousHandStartupGate()
    gate.update(hand_valid=True, observed_at_s=100.0)

    assert gate.update(hand_valid=True, observed_at_s=102.5) == pytest.approx(2.5)


def test_a_dropout_resets_the_clock():
    gate = ContinuousHandStartupGate()
    gate.update(hand_valid=True, observed_at_s=100.0)
    assert gate.update(hand_valid=True, observed_at_s=102.9) == pytest.approx(2.9)

    assert gate.update(hand_valid=False, observed_at_s=103.0) == 0.0
    assert gate.update(hand_valid=True, observed_at_s=103.1) == 0.0


def test_a_flickering_hand_never_reaches_the_dwell():
    """The case the gate exists for: seen/lost/seen must not sum to a pass."""
    gate = ContinuousHandStartupGate()
    t = 0.0
    for _ in range(20):
        assert gate.update(hand_valid=True, observed_at_s=t) < HAND_STARTUP_DWELL_S
        t += HAND_STARTUP_DWELL_S - 0.1
        gate.update(hand_valid=False, observed_at_s=t)
        t += 0.1


def test_a_steady_hand_reaches_the_dwell():
    gate = ContinuousHandStartupGate()
    gate.update(hand_valid=True, observed_at_s=0.0)

    assert gate.update(hand_valid=True, observed_at_s=HAND_STARTUP_DWELL_S) >= gate.required_s


def test_time_never_runs_backwards():
    """An out-of-order timestamp clamps to zero rather than reporting negative held time."""
    gate = ContinuousHandStartupGate()
    gate.update(hand_valid=True, observed_at_s=50.0)

    assert gate.update(hand_valid=True, observed_at_s=49.0) == 0.0


def test_constants_are_the_documented_values():
    assert MAX_WRIST_ROLL_RANGE_DEG == 45.0
    assert HAND_STARTUP_DWELL_S == 3.0
