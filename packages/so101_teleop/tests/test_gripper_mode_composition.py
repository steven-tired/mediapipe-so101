"""--gripper-mode selects the controller; PV stays optional and grip-only."""

import pytest

from lerobot_teleoperator_so101_webcam.grip.compose import GRIPPER_MODES, build_gripper
from lerobot_teleoperator_so101_webcam.grip.contract import GripInput, GripperController
from lerobot_teleoperator_so101_webcam.grip.mediapipe import (
    RELEASE_POS,
    MediaPipeGripperController,
)


def test_default_mode_is_mediapipe():
    assert GRIPPER_MODES[0] == "mediapipe"
    assert isinstance(build_gripper(), MediaPipeGripperController)


def test_every_mode_satisfies_the_protocol():
    assert isinstance(build_gripper("mediapipe"), GripperController)
    assert isinstance(
        build_gripper("pressurevision", zero_pos=32.0, one_pos=20.0), GripperController
    )


def test_pressurevision_mode_requires_an_explicit_span():
    with pytest.raises(ValueError, match="grip-zero-pos"):
        build_gripper("pressurevision")


def test_unknown_mode_is_rejected():
    with pytest.raises(ValueError, match="unknown gripper mode"):
        build_gripper("thermal")


@pytest.mark.parametrize("mode,kw", [
    ("mediapipe", {}),
    ("pressurevision", {"zero_pos": 32.0, "one_pos": 20.0}),
])
def test_no_mode_can_open_the_gripper_without_an_explicit_release(mode, kw):
    c = build_gripper(mode, **kw)
    c.current_command = 24.0
    # every kind of bad input, in both modes, holds
    for grip in (
        GripInput(grasp_active=True, explicit_release=False, severity=None,
                  valid=False, observed_at_s=0.0),
        GripInput(grasp_active=False, explicit_release=False, severity=1.0,
                  valid=True, observed_at_s=0.0),
    ):
        assert c.step(grip, actual_pos=24.0) == 24.0
    # only the explicit release opens it
    released = c.step(
        GripInput(grasp_active=True, explicit_release=True, severity=None,
                  valid=False, observed_at_s=0.0),
        actual_pos=24.0,
    )
    assert released == RELEASE_POS
