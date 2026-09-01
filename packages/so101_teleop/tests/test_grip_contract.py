"""Characterisation tests for the MediaPipe gripper path.

The numbers here are not chosen; they are what `WebcamEEController.step` already
computed before the contract existed:

    raw          = max(0, gripper_pos_from_pinch(pinch) - GRIP_OVERDRIVE)
    alpha        = GRIP_CLOSE_ALPHA if raw < current else GRIP_OPEN_ALPHA
    current      = alpha * raw + (1 - alpha) * current

with GRIP_OVERDRIVE = 18.0, GRIP_CLOSE_ALPHA = 0.7, GRIP_OPEN_ALPHA = 0.15, and
0 = clamped shut, 100 = fully open. `severity` is the same signal normalised:
severity = 1 - gripper_pos_from_pinch / 100.
"""

import pytest

from lerobot_teleoperator_so101_webcam.grip.contract import GripInput, make_grip_input
from lerobot_teleoperator_so101_webcam.grip.mediapipe import (
    GRIP_CLOSE_ALPHA,
    GRIP_OPEN_ALPHA,
    GRIP_OVERDRIVE,
    RELEASE_POS,
    MediaPipeGripperController,
)


def held(**kw):
    base = dict(grasp_active=True, explicit_release=False, severity=0.5,
                valid=True, observed_at_s=1.0)
    base.update(kw)
    return GripInput(**base)


def test_release_position_is_the_calibrated_centre_not_full_open():
    # joint_center() returns 50.0 for a RANGE_0_100 gripper, and the clutch pose
    # has always used that. Full open (100) would be a behaviour change.
    assert RELEASE_POS == 50.0


def test_explicit_release_returns_the_release_position_and_clears_state():
    c = MediaPipeGripperController()
    c.step(held(severity=0.9), actual_pos=30.0)
    assert c.current_command is not None
    assert c.step(held(explicit_release=True), actual_pos=30.0) == RELEASE_POS
    assert c.current_command is None


def test_first_valid_sample_is_taken_raw_without_smoothing():
    c = MediaPipeGripperController()
    # severity 0.0 = hand fully open -> position 100, minus overdrive
    assert c.step(held(severity=0.0), actual_pos=50.0) == 100.0 - GRIP_OVERDRIVE


def test_overdrive_clamps_at_fully_closed():
    c = MediaPipeGripperController()
    # severity 1.0 = tightest pinch -> position 0, overdrive cannot go below 0
    assert c.step(held(severity=1.0), actual_pos=50.0) == 0.0


def test_closing_is_fast_and_opening_is_slow():
    c = MediaPipeGripperController()
    c.current_command = 60.0
    closing = c.step(held(severity=0.8), actual_pos=60.0)      # raw = 20 - 18 = 2
    assert closing == pytest.approx(GRIP_CLOSE_ALPHA * 2.0 + (1 - GRIP_CLOSE_ALPHA) * 60.0)

    c.current_command = 10.0
    opening = c.step(held(severity=0.1), actual_pos=10.0)      # raw = 90 - 18 = 72
    assert opening == pytest.approx(GRIP_OPEN_ALPHA * 72.0 + (1 - GRIP_OPEN_ALPHA) * 10.0)
    # the asymmetry is the anti-loosening behaviour, not an accident
    assert GRIP_OPEN_ALPHA < GRIP_CLOSE_ALPHA


def test_invalid_input_holds_the_current_command():
    c = MediaPipeGripperController()
    c.current_command = 31.0
    assert c.step(held(severity=None, valid=False), actual_pos=30.0) == 31.0
    assert c.current_command == 31.0


def test_invalid_input_before_any_command_falls_back_to_the_measured_position():
    c = MediaPipeGripperController()
    assert c.step(held(severity=None, valid=False), actual_pos=27.5) == 27.5


def test_no_grasp_holds_rather_than_opening():
    c = MediaPipeGripperController()
    c.current_command = 22.0
    assert c.step(held(grasp_active=False), actual_pos=22.0) == 22.0


def test_stale_measurement_is_marked_invalid_at_the_composition_layer():
    fresh = make_grip_input(grasp_active=True, explicit_release=False, severity=0.9,
                            observed_at_s=10.0, now_s=10.2)
    stale = make_grip_input(grasp_active=True, explicit_release=False, severity=0.9,
                            observed_at_s=10.0, now_s=10.8)
    assert fresh.valid is True
    assert stale.valid is False
    # and a stale frame therefore holds
    c = MediaPipeGripperController()
    c.current_command = 28.0
    assert c.step(stale, actual_pos=28.4) == 28.0


def test_out_of_range_severity_is_invalid():
    bad = make_grip_input(grasp_active=True, explicit_release=False, severity=1.4,
                          observed_at_s=1.0, now_s=1.0)
    assert bad.valid is False


def test_grip_input_is_keyword_only():
    with pytest.raises(TypeError):
        GripInput(True, False, 0.5, True, 1.0)
