"""The latch that keeps a held object from being dropped when PV loses contact.

Extracted from WebcamEEController, where it was inlined and could only be
reached through a fake robot and a full IK pipeline. These tests drive it
directly, so each one names a single contact history and the target it produces.

The behaviour is not new and must not drift: `local/evidence/` was recorded with
it, and the latched anchor is the teacher label those datasets carry.
"""

import pytest

from pressurevision_integration.adjustment_lock import (
    PV_ADJUSTMENT_CONFIRM_RELEASE_S,
    PV_ADJUSTMENT_RESUME_CONTACT_S,
    PVAdjustmentLock,
)


def drive(lock, frames, *, current_command=30.0):
    """Run (live_target, pressure_active, t) frames, threading previous_target."""
    previous = None
    out = []
    for live_target, active, t in frames:
        previous = lock.update(live_target=live_target, pressure_active=active,
                               observed_at_s=t, current_command=current_command,
                               previous_target=previous)
        out.append(previous)
    return out


def test_a_fresh_lock_is_not_latched():
    lock = PVAdjustmentLock()

    assert lock.locked is False
    assert lock.anchor_target is None
    assert lock.state(grip_active=True) == "adjusting"


def test_before_any_contact_the_live_target_passes_straight_through():
    """Nothing has been grasped, so there is nothing to hold on to."""
    lock = PVAdjustmentLock()

    assert drive(lock, [(40.0, False, 0.0), (35.0, False, 5.0)]) == [40.0, 35.0]
    assert lock.locked is False


def test_contact_follows_the_live_target():
    lock = PVAdjustmentLock()

    assert drive(lock, [(40.0, True, 0.0), (32.0, True, 0.1)]) == [40.0, 32.0]
    assert lock.contact_seen is True


def test_losing_contact_briefly_holds_without_latching():
    lock = PVAdjustmentLock()
    out = drive(lock, [(40.0, True, 0.0), (25.0, False, 0.3), (25.0, False, 0.9)])

    assert out[1] == out[2] == 40.0, "holds last target, does not follow live"
    assert lock.locked is False
    assert lock.state(grip_active=True) == "temporary_hold"


def test_contact_lost_past_the_confirm_window_latches_at_the_current_command():
    lock = PVAdjustmentLock()
    out = drive(lock, [(40.0, True, 0.0),
                       (25.0, False, 0.5),
                       (25.0, False, PV_ADJUSTMENT_CONFIRM_RELEASE_S + 0.5)],
                current_command=28.0)

    assert lock.locked is True
    assert lock.anchor_target == 28.0
    assert out[-1] == 28.0
    assert lock.event == "lock"
    assert lock.state(grip_active=True) == "locked"


def test_the_latch_anchors_where_the_gripper_is_not_where_pv_wanted():
    """The anchor is the *commanded* position -- the force actually being applied."""
    lock = PVAdjustmentLock()
    drive(lock, [(40.0, True, 0.0), (10.0, False, 0.0), (10.0, False, 2.0)],
          current_command=33.0)

    assert lock.anchor_target == 33.0


def test_a_latched_grip_holds_indefinitely_with_no_contact():
    lock = PVAdjustmentLock()
    drive(lock, [(40.0, True, 0.0), (25.0, False, 0.0), (25.0, False, 2.0)],
          current_command=28.0)

    held = lock.update(live_target=5.0, pressure_active=False, observed_at_s=60.0,
                       current_command=28.0, previous_target=28.0)

    assert held == 28.0, "a dead PV sender must not slacken a held grip"


def test_a_single_flickering_contact_frame_does_not_unlatch():
    """The case the resume window exists for: one frame of noise on a held object."""
    lock = PVAdjustmentLock()
    drive(lock, [(40.0, True, 0.0), (25.0, False, 0.0), (25.0, False, 2.0)],
          current_command=28.0)

    blip = lock.update(live_target=90.0, pressure_active=True, observed_at_s=2.05,
                       current_command=28.0, previous_target=28.0)

    assert blip == 28.0
    assert lock.locked is True
    assert lock.event == "recontact_started"


def test_sustained_recontact_resumes_adjusting():
    lock = PVAdjustmentLock()
    drive(lock, [(40.0, True, 0.0), (25.0, False, 0.0), (25.0, False, 2.0)],
          current_command=28.0)
    lock.update(live_target=26.0, pressure_active=True, observed_at_s=3.0,
                current_command=28.0, previous_target=28.0)

    # Comfortably past the window rather than exactly on it: `3.0 + 0.15 - 3.0`
    # lands a hair *under* 0.15 in binary floating point, and pinning the exact
    # boundary would test the arithmetic rather than the behaviour.
    resumed = lock.update(live_target=26.0, pressure_active=True,
                          observed_at_s=3.0 + 2 * PV_ADJUSTMENT_RESUME_CONTACT_S,
                          current_command=28.0, previous_target=28.0)

    assert lock.locked is False
    assert lock.event == "recontact_resume"
    assert resumed == 26.0


def test_after_latching_the_grip_may_tighten_but_never_loosen():
    """While an anchor stands, a looser live target is clamped to the anchor."""
    lock = PVAdjustmentLock()
    drive(lock, [(40.0, True, 0.0), (25.0, False, 0.0), (25.0, False, 2.0)],
          current_command=28.0)
    for t in (3.0, 3.0 + 2 * PV_ADJUSTMENT_RESUME_CONTACT_S):
        lock.update(live_target=28.0, pressure_active=True, observed_at_s=t,
                    current_command=28.0, previous_target=28.0)

    tighter = lock.update(live_target=20.0, pressure_active=True, observed_at_s=4.0,
                          current_command=28.0, previous_target=28.0)
    looser = lock.update(live_target=45.0, pressure_active=True, observed_at_s=4.1,
                         current_command=28.0, previous_target=20.0)

    assert tighter == 20.0, "squeezing harder is allowed"
    assert looser == 28.0, "relaxing is clamped to the anchor"


def test_the_release_timer_restarts_after_a_recovered_dropout():
    """Two sub-threshold dropouts must not add up to a latch."""
    lock = PVAdjustmentLock()
    frames = [(40.0, True, 0.0)]
    t = 0.0
    for _ in range(6):
        frames.append((25.0, False, t + PV_ADJUSTMENT_CONFIRM_RELEASE_S - 0.1))
        t += PV_ADJUSTMENT_CONFIRM_RELEASE_S - 0.1
        frames.append((25.0, True, t + 0.01))
        t += 0.01
    drive(lock, frames)

    assert lock.locked is False


def test_contact_resumed_is_reported_after_a_temporary_hold():
    lock = PVAdjustmentLock()
    drive(lock, [(40.0, True, 0.0), (25.0, False, 0.3)])

    lock.update(live_target=30.0, pressure_active=True, observed_at_s=0.5,
                current_command=28.0, previous_target=40.0)

    assert lock.event == "contact_resumed"
    assert lock.release_since_s is None


def test_clear_forgets_the_grasp_and_reports_the_event():
    lock = PVAdjustmentLock()
    drive(lock, [(40.0, True, 0.0), (25.0, False, 0.0), (25.0, False, 2.0)],
          current_command=28.0)

    lock.clear(event="middle_reset")

    assert lock.locked is False
    assert lock.anchor_target is None
    assert lock.contact_seen is False
    assert lock.event == "middle_reset"


def test_clearing_a_lock_that_never_grasped_reports_nothing():
    """The event marks a real transition, so a no-op reset must not emit one."""
    lock = PVAdjustmentLock()

    lock.clear(event="middle_reset")

    assert lock.event is None


def test_state_is_inactive_whenever_the_grasp_is_not_active():
    lock = PVAdjustmentLock()
    drive(lock, [(40.0, True, 0.0), (25.0, False, 0.0), (25.0, False, 2.0)])

    assert lock.locked is True
    assert lock.state(grip_active=False) == "inactive"


def test_the_windows_are_the_documented_values():
    assert PV_ADJUSTMENT_CONFIRM_RELEASE_S == 1.0
    assert PV_ADJUSTMENT_RESUME_CONTACT_S == 0.15
