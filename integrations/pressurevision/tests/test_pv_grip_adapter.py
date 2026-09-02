"""The PV runtime reached through the core's gripper contract.

The runtime itself is covered by test_pv_grip_controller. What matters here is
the seam: that `WebcamEEController` can drive PressureVision without importing
this package, and that MediaPipe keeps grasp/release authority through it.
"""

from types import SimpleNamespace

import numpy as np
import pytest

from lerobot_teleoperator_so101_webcam.ee_controller import WebcamEEController
from lerobot_teleoperator_so101_webcam.grip.contract import GripInput
from lerobot_teleoperator_so101_webcam.grip.mediapipe import RELEASE_POS
from pressurevision_integration.protocol import PressureReading
from pressurevision_integration.pv_grip_adapter import PVGripAdapter
from pressurevision_integration.pv_grip_controller import PressureVisionGripRuntime
from webcam_input.types import LandmarksData, WristData

GRASPING = 20.0
CFG = SimpleNamespace(
    ee_x_idx=2, ee_x_sign=1.0,
    ee_y_idx=0, ee_y_sign=-1.0,
    ee_z_idx=1, ee_z_sign=1.0,
    grip_pinch_min=0.02, grip_pinch_max=0.12, grip_sign=1.0,
)


class FakeSource:
    """A sender that reports no contact first, then a steady press.

    The baseline frames are not padding: the proposal machine stays disarmed
    until it has seen one, so a sender that opens with contact never gets to
    drive the command at all.
    """

    def __init__(self, pressure=0.5, *, baseline_frames=1):
        self.pressure = pressure
        self.baseline_frames = baseline_frames
        self.calls = 0
        self.closes = 0

    def update(self, landmarks, *, pinch, enabled):
        self.calls += 1
        if self.calls <= self.baseline_frames:
            return PressureReading(
                pressure_0_1=0.0,
                active=False,
                quality=1.0,
                available=True,
                status="baseline",
            )
        return PressureReading(
            pressure_0_1=self.pressure,
            active=True,
            quality=1.0,
            available=True,
            status="active",
        )

    def close(self):
        self.closes += 1


def _adapter(source=None, **kwargs):
    kwargs.setdefault("pv_mapping", "carton_span")
    kwargs.setdefault("pressure_apply", True)
    runtime = PressureVisionGripRuntime(
        source or FakeSource(),
        initial_gripper=50.0,
        middle_gripper=50.0,
        **kwargs,
    )
    return PVGripAdapter(runtime)


# severity 0.8 -> a base command of 20, below GRIP_LATCH_ENTER, so the range
# mapper reads it as a live grasp rather than an open hand.
def _grip(*, severity=0.8, valid=True, grasp_active=True, explicit_release=False, t=0.0):
    return GripInput(
        grasp_active=grasp_active,
        explicit_release=explicit_release,
        severity=severity,
        valid=valid,
        observed_at_s=t,
    )


def _observe(adapter, *, pinch=0.03, enabled=True, t=0.0):
    adapter.observe_frame(np.zeros((21, 3)), pinch=pinch, enabled=enabled, observed_at_s=t)


# --- the contract ---

def _drive(adapter, frames=20, *, start=50.0):
    command = start
    for tick in range(frames):
        _observe(adapter, t=tick * 0.1)
        command = adapter.step(_grip(t=tick * 0.1), actual_pos=command)
    return command


def test_pv_drives_the_command_into_the_mapped_span():
    """One frame cannot get there: the proposal is rate-limited, so reaching
    the span is a slew, not a jump. carton_span maps 0..1 onto 32..20, so a
    steady half-press settles near the middle of that span."""
    command = _drive(_adapter())

    assert 20.0 <= command <= 32.0
    assert command == pytest.approx(26.0, abs=0.5)


def test_pv_cannot_take_over_before_it_has_a_baseline():
    """A sender that opens with contact has no zero to measure against. The
    command must stay on the pinch path rather than trust it."""
    adapter = _adapter(FakeSource(baseline_frames=0))

    command = _drive(adapter)

    assert adapter.runtime.pressure_state == "disarmed"
    # The pinch base is 20 (severity 0.8); the command tracks it, not the span.
    assert command == pytest.approx(20.0)


def test_explicit_release_never_consults_the_runtime():
    """The safety contract: PV can weaken a grasp but must never be able to
    keep the claw shut. Release has to work with the sender dead."""
    source = FakeSource()
    adapter = _adapter(source)
    _observe(adapter)
    adapter.step(_grip(), actual_pos=50.0)
    calls_before = source.calls

    command = adapter.step(_grip(explicit_release=True), actual_pos=25.0)

    assert command == pytest.approx(RELEASE_POS)
    assert source.calls == calls_before


def test_an_invalid_frame_holds_rather_than_opening():
    adapter = _adapter()
    _observe(adapter)
    held = adapter.step(_grip(), actual_pos=50.0)

    _observe(adapter)
    command = adapter.step(_grip(valid=False), actual_pos=50.0)

    assert command == pytest.approx(held)


def test_the_first_invalid_frame_holds_the_arm_s_own_position():
    """With no command of its own yet, holding means the position the arm is
    actually at -- not a guess, and not open."""
    adapter = _adapter()
    _observe(adapter)

    assert adapter.step(_grip(valid=False), actual_pos=41.0) == pytest.approx(41.0)


def test_stepping_without_a_frame_is_a_programming_error():
    """Silently reusing the last frame would feed the runtime a stale reading
    and call it fresh."""
    adapter = _adapter()
    with pytest.raises(RuntimeError, match="observe_frame"):
        adapter.step(_grip(), actual_pos=50.0)


def test_reset_forgets_the_grasp():
    adapter = _adapter()
    _observe(adapter)
    adapter.step(_grip(), actual_pos=50.0)

    adapter.reset()

    assert adapter.current_command is None
    assert adapter.runtime.adjustment_anchor_target is None


# --- through the real controller ---

class _Pipeline:
    def __call__(self, transition):
        _, joints = transition
        return dict(joints)

    def reset(self):
        pass


def _controller(gripper):
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
    controller = WebcamEEController(robot, kin, CFG, use_oak=True, gripper=gripper)
    controller.pipeline = _Pipeline()
    controller.seed({"shoulder_pan.pos": 0.0, "wrist_flex.pos": 90.0, "gripper.pos": 50.0})
    return controller


def _step(controller, *, pinch=0.03, clutch=False):
    points = [[0.0, 0.0, 0.0] for _ in range(21)]
    points[8] = [pinch, 0.0, 0.0]
    wrist = WristData(
        position=np.zeros(3),
        quaternion=np.array([0.0, 0.0, 0.0, 1.0]),
        fist_state="closed" if clutch else "open",
        valid=True,
    )
    return controller.step(wrist, LandmarksData(landmarks=points, valid=True))


def test_the_controller_drives_pv_without_importing_it():
    source = FakeSource()
    adapter = _adapter(source)
    controller = _controller(adapter)

    for _ in range(20):
        joints, state = _step(controller)

    assert state == "MOVING"
    assert source.calls == 20
    assert 20.0 <= joints["gripper.pos"] <= 32.0
    assert adapter.runtime.pressure_state == "armed"


def test_the_clutch_clears_the_pv_grasp():
    """A clutch abandons the grasp, so the adjustment lock must not survive it
    and resume a hold the operator has already let go of."""
    source = FakeSource()
    adapter = _adapter(source)
    controller = _controller(adapter)
    _step(controller)

    joints, state = _step(controller, clutch=True)

    assert state == "MIDDLE"
    assert joints["gripper.pos"] == pytest.approx(50.0)
    assert adapter.current_command is None
    assert adapter.runtime.adjustment_anchor_target is None


def test_a_dead_sender_does_not_stall_the_arm():
    """A raising sender becomes an inactive reading, not an exception that
    takes the control loop down mid-grasp."""

    class Broken(FakeSource):
        def update(self, landmarks, *, pinch, enabled):
            raise OSError("sender gone")

    controller = _controller(_adapter(Broken()))

    joints, state = _step(controller)

    assert state == "MOVING" and joints is not None
