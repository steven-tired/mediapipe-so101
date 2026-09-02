"""Wiring for the grip-mode/grip-map arms and the opt-in wrist roll.

`ee_control.grip_ratchet` and the pinch maps are unit-tested on their own; these
cover that arm B is actually reachable through the controller, that arm A is
untouched, and that a clutch clears the ratchet -- without which releasing the
fist would resume a grasp the operator had already abandoned. They also cover
the `observe_frame` hook, which is how a sensing gripper gets the frame without
the core ever importing an integration package.
"""

from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from lerobot_teleoperator_so101_webcam.ee_control import (
    GRIP_LATCH_EXIT_FRAMES,
    gripper_pos_from_pinch,
    raw_grip_from_pinch,
)
from lerobot_teleoperator_so101_webcam.ee_controller import (
    MAX_WRIST_ROLL_RANGE_DEG,
    WebcamEEController,
    bounded_wrist_roll_delta,
)
from lerobot_teleoperator_so101_webcam.grip.mediapipe import (
    GRIP_OVERDRIVE,
    MediaPipeGripperController,
)
from webcam_input.types import LandmarksData, WristData
from lerobot_teleoperator_so101_webcam.config_so101_webcam_ee import SO101WebcamEEConfig


class _Pipeline:
    def __call__(self, transition):
        _, joints = transition
        return dict(joints)

    def reset(self):
        pass


# The real config, not a copy of its defaults: axis indices and signs are the
# one thing that must not drift silently between here and the arm.
CFG = SO101WebcamEEConfig()


def _controller(grip_mode="tracked", **kwargs):
    motors = {
        "shoulder_pan": SimpleNamespace(norm_mode=SimpleNamespace(value="degrees")),
        "wrist_flex": SimpleNamespace(norm_mode=SimpleNamespace(value="degrees")),
        "gripper": SimpleNamespace(norm_mode=SimpleNamespace(value="range_0_100")),
    }
    robot = SimpleNamespace(bus=SimpleNamespace(motors=motors))
    kin = SimpleNamespace(
        forward_kinematics=lambda q: np.eye(4),
        inverse_kinematics=lambda q, pose, *, position_weight=1.0, orientation_weight=0.01: q,
    )
    controller = WebcamEEController(
        robot, kin, CFG, use_oak=True, grip_mode=grip_mode, **kwargs
    )
    controller.pipeline = _Pipeline()
    controller.seed({"shoulder_pan.pos": 0.0, "wrist_flex.pos": 90.0, "gripper.pos": 50.0})
    return controller


def _step(controller, *, pinch=0.08, clutch=False, valid=True, roll_deg=0.0):
    points = [[0.0, 0.0, 0.0] for _ in range(21)]
    points[8] = [pinch, 0.0, 0.0]
    wrist = WristData(
        position=np.zeros(3),
        quaternion=Rotation.from_euler("x", roll_deg, degrees=True).as_quat(),
        fist_state="closed" if clutch else "open",
        valid=valid,
    )
    return controller.step(wrist, LandmarksData(landmarks=points, valid=valid))


def _grasp(controller, frames=14):
    """Start from an open hand, close hard enough to commit, return the command."""
    _step(controller, pinch=0.12)
    for _ in range(frames):
        _step(controller, pinch=0.02)
    return controller.grip_smoothed


# --- construction ---

def test_default_is_the_validated_tracked_arm():
    controller = _controller()
    assert controller.grip_mode == "tracked"
    assert controller.grip_map == "overdrive"
    assert controller.grip_overdrive == pytest.approx(GRIP_OVERDRIVE)


def test_unknown_grip_mode_is_rejected_at_construction():
    with pytest.raises(ValueError, match="grip_mode"):
        _controller("ratchet")


def test_unknown_grip_map_is_rejected_at_construction():
    with pytest.raises(ValueError, match="grip_map"):
        _controller(grip_map="linear")


def test_span_map_drops_the_overdrive():
    """Narrowing the pinch range is what replaces the overdrive; applying both
    would reintroduce the clipped dead zone span mode exists to remove."""
    controller = _controller(grip_map="span")

    assert controller.grip_overdrive == pytest.approx(0.0)
    assert controller.gripper.overdrive == pytest.approx(0.0)


def test_grip_mode_is_refused_when_the_gripper_is_injected():
    """It configures the default gripper. Accepting it silently next to an
    injected controller would leave the caller believing arm B is running."""
    with pytest.raises(ValueError, match="injected gripper"):
        _controller("latched", gripper=MediaPipeGripperController())


def test_wrist_roll_range_is_bounded():
    with pytest.raises(ValueError, match="wrist_roll_range_deg"):
        _controller(wrist_roll_range_deg=MAX_WRIST_ROLL_RANGE_DEG + 1.0)
    with pytest.raises(ValueError, match="wrist_roll_range_deg"):
        _controller(wrist_roll_range_deg=-1.0)


def test_wrist_roll_gain_is_bounded():
    with pytest.raises(ValueError, match="wrist_roll_gain"):
        _controller(wrist_roll_gain=0.0)
    with pytest.raises(ValueError, match="wrist_roll_gain"):
        _controller(wrist_roll_gain=4.5)


# --- wrist roll ---

def test_roll_opt_in_replaces_fixed_orientation_with_relative_tool_roll():
    fixed = _controller()
    fixed.build(np.zeros(3))
    enabled = _controller(wrist_roll_range_deg=20.0)
    enabled.build(np.zeros(3))

    # By name, not isinstance: test_core_runs_without_pressurevision reloads
    # ee_controller, which rebinds the class, and an identity check would then
    # depend on test order.
    def has_fixed_orientation(controller):
        return any(type(s).__name__ == "FixedEEOrientation" for s in controller.pipeline.steps)

    assert has_fixed_orientation(fixed)
    assert not has_fixed_orientation(enabled)


def test_roll_delta_wraps_and_clamps_about_the_latched_hand_pose():
    reference = np.deg2rad(179.0)
    across_wrap = Rotation.from_euler("x", -179.0, degrees=True).as_quat()
    far = Rotation.from_euler("x", -120.0, degrees=True).as_quat()

    assert np.rad2deg(
        bounded_wrist_roll_delta(across_wrap, reference, 20.0)
    ) == pytest.approx(2.0)
    assert np.rad2deg(
        bounded_wrist_roll_delta(far, reference, 20.0)
    ) == pytest.approx(20.0)


def test_roll_gain_amplifies_before_the_output_range_clamp():
    reference = 0.0
    ten_deg = Rotation.from_euler("x", 10.0, degrees=True).as_quat()
    twenty_deg = Rotation.from_euler("x", 20.0, degrees=True).as_quat()

    assert np.rad2deg(
        bounded_wrist_roll_delta(ten_deg, reference, 30.0, gain=2.0)
    ) == pytest.approx(20.0)
    assert np.rad2deg(
        bounded_wrist_roll_delta(twenty_deg, reference, 30.0, gain=2.0)
    ) == pytest.approx(30.0)


def test_roll_stays_zero_until_it_is_opted_into():
    """The default arm must keep the validated fixed-down orientation even when
    the operator's hand happens to be rolled."""
    controller = _controller()
    controller.step(
        WristData(
            position=np.zeros(3),
            quaternion=Rotation.from_euler("x", 30.0, degrees=True).as_quat(),
            fist_state="open",
            valid=True,
        ),
        LandmarksData(landmarks=[[0.0, 0.0, 0.0] for _ in range(21)], valid=True),
    )

    assert controller.roll_ref is None


def test_the_roll_reference_latches_on_re_engage():
    controller = _controller(wrist_roll_range_deg=20.0)
    _step(controller, roll_deg=30.0)
    first = controller.roll_ref

    _step(controller, clutch=True)
    _step(controller, roll_deg=-10.0)

    assert first == pytest.approx(np.deg2rad(30.0))
    assert controller.roll_ref == pytest.approx(np.deg2rad(-10.0))


# --- the ratchet, through the controller ---

def test_latched_arm_holds_the_grip_through_an_open_jitter_burst():
    controller = _controller("latched")
    firm = _grasp(controller)
    assert controller.grip_latched is True

    for _ in range(GRIP_LATCH_EXIT_FRAMES - 1):
        _step(controller, pinch=0.12)

    assert controller.grip_latched is True
    assert controller.grip_smoothed == pytest.approx(firm)


def test_tracked_arm_loosens_under_the_same_burst():
    """Arm A for contrast: the same jitter does move the command, which is the
    behaviour the slow-open EMA damps rather than prevents."""
    controller = _controller("tracked")
    firm = _grasp(controller)

    for _ in range(GRIP_LATCH_EXIT_FRAMES - 1):
        _step(controller, pinch=0.12)

    assert controller.grip_smoothed > firm


def test_a_sustained_open_releases_the_latched_arm():
    controller = _controller("latched")
    firm = _grasp(controller)

    for _ in range(GRIP_LATCH_EXIT_FRAMES + 8):
        _step(controller, pinch=0.12)

    assert controller.grip_latched is False
    assert controller.grip_smoothed > firm


def test_clutch_clears_the_ratchet():
    controller = _controller("latched")
    _grasp(controller)
    assert controller.grip_latched is True

    _step(controller, clutch=True)

    assert controller.grip_latched is False
    assert controller.grip_open_frames == 0
    assert controller.grip_smoothed is None


def test_lost_tracking_holds_without_disturbing_the_ratchet():
    """HOLD returns before the grip block, so a dropped hand must neither
    release the grasp nor advance the release debounce."""
    controller = _controller("latched")
    firm = _grasp(controller)

    for _ in range(GRIP_LATCH_EXIT_FRAMES + 8):
        joints, state = _step(controller, pinch=0.12, valid=False)
        assert joints is None and state == "HOLD"

    assert controller.grip_latched is True
    assert controller.grip_smoothed == pytest.approx(firm)


def test_release_is_prompt_once_the_debounce_clears():
    """Measured on foam: release badly lagged the hand. The cause was the
    ratchet inheriting the tracked arm's slow-open EMA, which exists to resist
    loosening -- a job the ratchet already does, so keeping both only adds lag.
    """
    controller = _controller("latched")
    _grasp(controller)

    for _ in range(GRIP_LATCH_EXIT_FRAMES):
        _step(controller, pinch=0.12)
    assert controller.grip_latched is False

    # Three frames past the debounce is ~100 ms at 30 fps.
    for _ in range(3):
        _step(controller, pinch=0.12)
    open_command = gripper_pos_from_pinch(0.12, CFG) - GRIP_OVERDRIVE

    assert controller.grip_smoothed > 0.9 * open_command


# --- the grip map reaches the command ---

def test_span_map_keeps_a_mid_pinch_off_the_closed_rail():
    """The overdrive map clips a 0.038 pinch onto 0; the span map is what buys
    that resolution back, so the two must not agree there."""
    overdrive = _controller()
    span = _controller(grip_map="span")
    for _ in range(20):
        _step(overdrive, pinch=0.038)
        _step(span, pinch=0.038)

    assert overdrive.grip_smoothed == pytest.approx(0.0)
    assert span.grip_smoothed > 1.0
    assert span.grip_smoothed == pytest.approx(
        raw_grip_from_pinch(0.038, CFG, grip_map="span"), abs=1.0
    )


# --- the duck-typed sensing hook ---

class _SensingGripper(MediaPipeGripperController):
    def __init__(self):
        super().__init__()
        self.frames = []

    def observe_frame(self, landmarks, *, pinch, enabled, observed_at_s):
        self.frames.append((landmarks, pinch, enabled, observed_at_s))


def test_a_sensing_gripper_is_handed_the_frame():
    gripper = _SensingGripper()
    controller = _controller(gripper=gripper)

    _step(controller, pinch=0.05)

    assert len(gripper.frames) == 1
    _, pinch, enabled, _ = gripper.frames[0]
    assert pinch == pytest.approx(0.05)
    assert enabled is True


def test_the_hook_is_optional():
    """The plain MediaPipe gripper has no observe_frame; stepping must not care."""
    controller = _controller()

    joints, state = _step(controller, pinch=0.05)

    assert state == "MOVING" and joints is not None


def test_a_clutched_frame_is_not_observed():
    """The clutch returns to the middle pose before any grip work, so a sensing
    gripper must not be told a grasp frame happened."""
    gripper = _SensingGripper()
    controller = _controller(gripper=gripper)

    _step(controller, pinch=0.05, clutch=True)

    assert gripper.frames == []


# --- which gesture parks the arm ---

def _v_sign_points(pinch=0.08):
    """index+middle extended, ring+pinky curled."""
    points = np.zeros((21, 3))
    for (tip, pip), extended in zip(((8, 6), (12, 10), (16, 14), (20, 18)),
                                    (True, True, False, False)):
        points[pip] = [0.0, 0.06, 0.0]
        points[tip] = [0.0, 0.10 if extended else 0.03, 0.0]
    points[4] = [pinch, 0.10, 0.0]
    return points


def _step_points(controller, points, *, valid=True):
    wrist = WristData(
        position=np.zeros(3),
        quaternion=np.array([0.0, 0.0, 0.0, 1.0]),
        fist_state="open",
        valid=valid,
    )
    return controller.step(wrist, LandmarksData(landmarks=points, valid=valid))


def test_the_validated_fist_clutch_is_the_default():
    """The 100-episode pick-place dataset was recorded through it."""
    assert _controller().middle_gesture == "fist"


def test_an_unknown_middle_gesture_is_rejected():
    with pytest.raises(ValueError, match="middle_gesture"):
        _controller(middle_gesture="wave")


def test_the_fist_clutch_ignores_a_v_sign():
    controller = _controller()

    _, state = _step_points(controller, _v_sign_points())

    assert state == "MOVING"


def test_the_v_sign_mode_ignores_a_fist():
    """PressureVision occupies the left hand, so a left fist is not a gesture
    the operator can make in that mode -- it must not park the arm."""
    controller = _controller(middle_gesture="right_v")

    _, state = _step(controller, clutch=True)

    assert state == "MOVING"


def test_a_raw_v_sign_freezes_before_the_dwell_parks_the_arm(monkeypatch):
    """Moving during the dwell would make every attempt to park the arm start
    with a lurch."""
    import lerobot_teleoperator_so101_webcam.ee_controller as module

    now = [10.0]
    monkeypatch.setattr(module.time, "monotonic", lambda: now[0])
    controller = _controller(middle_gesture="right_v")

    joints, state = _step_points(controller, _v_sign_points())
    assert (joints, state) == (None, "HOLD")
    assert controller.middle_gesture_seen and not controller.middle_gesture_active

    now[0] += module.MIDDLE_GESTURE_HOLD_S
    joints, state = _step_points(controller, _v_sign_points())

    assert state == "MIDDLE"
    assert controller.middle_gesture_active


def test_a_single_stray_v_frame_does_not_restart_the_dwell_clock(monkeypatch):
    import lerobot_teleoperator_so101_webcam.ee_controller as module

    now = [10.0]
    monkeypatch.setattr(module.time, "monotonic", lambda: now[0])
    controller = _controller(middle_gesture="right_v")
    _step_points(controller, _v_sign_points())

    # A frame without the gesture clears the clock, so the dwell starts over.
    now[0] += 0.3
    _step(controller)
    now[0] += 0.3
    _, state = _step_points(controller, _v_sign_points())

    assert state == "HOLD"
