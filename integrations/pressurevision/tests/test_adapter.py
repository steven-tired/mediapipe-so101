"""PressureVision may modulate grip STRENGTH. It may never open the gripper.

The position span is inherited from the live controller's range mapper:

    severity = clip((zero_pos - target) / (zero_pos - one_pos), 0, 1)

so severity 0 maps to `zero_pos` (the loosest PV-commanded grip) and severity 1
to `one_pos` (the firmest). Lower position value = more closed.
"""

import pytest

from lerobot_teleoperator_so101_webcam.grip.contract import GripInput, make_grip_input
from lerobot_teleoperator_so101_webcam.grip.mediapipe import RELEASE_POS
from pressurevision_integration.adapter import PressureVisionGripperController


def held(**kw):
    base = dict(grasp_active=True, explicit_release=False, severity=0.5,
                valid=True, observed_at_s=1.0)
    base.update(kw)
    return GripInput(**base)


def test_severity_maps_across_the_configured_position_span():
    c = PressureVisionGripperController(zero_pos=32.0, one_pos=20.0)
    assert c.step(held(severity=0.0), actual_pos=32.0) == pytest.approx(32.0)
    c.reset()
    assert c.step(held(severity=1.0), actual_pos=32.0) == pytest.approx(20.0)
    c.reset()
    assert c.step(held(severity=0.5), actual_pos=32.0) == pytest.approx(26.0)


def test_severity_is_clamped_to_the_span():
    c = PressureVisionGripperController(zero_pos=32.0, one_pos=20.0)
    assert c.step(held(severity=5.0, valid=True), actual_pos=30.0) == pytest.approx(20.0)


def test_invalid_pressure_holds_the_command():
    c = PressureVisionGripperController(zero_pos=32.0, one_pos=20.0)
    c.current_command = 27.0
    assert c.step(held(severity=None, valid=False), actual_pos=27.5) == 27.0


def test_a_silent_sender_holds_indefinitely_rather_than_dropping_the_object():
    c = PressureVisionGripperController(zero_pos=32.0, one_pos=20.0)
    c.step(held(severity=0.9), actual_pos=32.0)
    clamped = c.current_command
    for tick in range(1, 200):                       # sender has gone quiet
        stale = make_grip_input(grasp_active=True, explicit_release=False,
                                severity=0.9, observed_at_s=1.0, now_s=1.0 + tick)
        assert stale.valid is False
        assert c.step(stale, actual_pos=clamped) == clamped


def test_explicit_release_is_the_only_open_transition():
    c = PressureVisionGripperController(zero_pos=32.0, one_pos=20.0)
    c.current_command = 27.0
    assert c.step(held(explicit_release=True, severity=None, valid=False),
                  actual_pos=27.5) == RELEASE_POS
    assert c.current_command is None


def test_no_grasp_holds_rather_than_opening():
    c = PressureVisionGripperController(zero_pos=32.0, one_pos=20.0)
    c.current_command = 24.0
    assert c.step(held(grasp_active=False, severity=1.0), actual_pos=24.0) == 24.0


def test_pv_cannot_command_looser_than_the_configured_zero_position():
    """PV modulates strength inside the span; it cannot slacken past zero_pos."""
    c = PressureVisionGripperController(zero_pos=32.0, one_pos=20.0)
    commands = [c.step(held(severity=s), actual_pos=32.0) for s in (0.0, 0.2, 0.9, 0.0)]
    assert max(commands) <= 32.0
    assert min(commands) >= 20.0


def test_adapter_satisfies_the_public_protocol():
    from lerobot_teleoperator_so101_webcam.grip.contract import GripperController
    assert isinstance(PressureVisionGripperController(zero_pos=32.0, one_pos=20.0),
                      GripperController)
