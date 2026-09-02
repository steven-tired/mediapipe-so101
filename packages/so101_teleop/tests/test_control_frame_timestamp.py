"""The control frame's timestamp must be a clock read, and it must advance.

It was migrated as `wrist.observed_at_s if hasattr(wrist, "observed_at_s") else
0.0`. WristData has no such field, so every frame reported time 0.0. Nothing
raised: the `hasattr` guard turned a missing attribute into a constant.

What that constant destroyed, on a connected robot:

  * the PV low-pass computed `dt = 0` every frame, so `alpha = 0`, so the
    filtered pressure never left its initial 0.0 -- the operator pressed the pad
    as hard as they could and the commanded gripper position stayed pinned at
    the mapping's zero end, while telemetry still logged the reading as active;
  * every duration the adjustment lock measures (the 1.0 s release latch, the
    0.15 s re-contact) was `t - t = 0`, so the lock could never latch.

One dead value, every PV symptom. These tests assert the property that was lost:
the timestamp advances between frames and is not sourced from the payload.
"""

import ast
import inspect

import numpy as np
import pytest

from lerobot_teleoperator_so101_webcam.ee_controller import WebcamEEController
from webcam_input.types import LandmarksData, WristData


class RecordingGripper:
    """A sensing gripper that only remembers the times it was handed."""

    def __init__(self):
        self.times = []
        self.current_command = 50.0

    def observe_frame(self, landmarks, *, pinch, enabled, observed_at_s):
        self.times.append(observed_at_s)

    def step(self, grip, actual_pos):
        self.times.append(grip.observed_at_s)
        return 50.0

    def reset(self):
        pass


def _step_source():
    return inspect.getsource(WebcamEEController.step)


def test_the_timestamp_is_read_from_a_clock_not_from_the_payload():
    """A payload field can go missing; `time.perf_counter()` cannot."""
    tree = ast.parse(_step_source().replace("    def step", "def step", 1))
    clock_reads = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        and node.func.attr == "perf_counter"
    ]
    assert clock_reads, "step() must read the clock once for the control frame"


def test_the_timestamp_is_never_defaulted_when_an_attribute_is_absent():
    """The exact shape of the defect: hasattr(...) hiding a missing field."""
    source = _step_source()
    assert 'hasattr(wrist, "observed_at_s")' not in source
    assert "observed_at_s\") else 0.0" not in source


@pytest.mark.parametrize("attribute", ["observed_at_s"])
def test_wrist_data_still_does_not_carry_the_field_that_was_assumed(attribute):
    """Guard the guard: if WristData ever gains this, the defect stops being
    reachable by that route and this file's premise needs revisiting."""
    wrist = WristData(np.zeros(3), np.array([0.0, 0.0, 0.0, 1.0]), "open", True)
    assert not hasattr(wrist, attribute)


def test_a_frozen_time_base_pins_the_pv_mapping_at_its_zero_end():
    """The consequence, end to end, on the real runtime -- no robot, no sender.

    This is the failure the bench found, reduced: the same maximum press, the
    same mapper, differing only in whether the control frame's clock advances.
    """
    from pressurevision_integration.pv_grip_controller import PressureVisionGripRuntime
    from pressurevision_integration.pv_pressure import PressureReading

    class MaximumPress:
        def update(self, landmarks, *, pinch, enabled):
            return PressureReading(
                pressure_0_1=1.0, active=True, quality=1.0,
                available=True, status="active", sequence=1,
            )

    def closures(times):
        runtime = PressureVisionGripRuntime(
            MaximumPress(), initial_gripper=50.0, middle_gripper=50.0,
            pv_mapping="carton_span", pressure_apply=True,
        )
        command = 50.0
        out = []
        for observed_at_s in times:
            command = runtime.update(
                base_gripper=20.0, landmarks=np.zeros((21, 3)), pinch=0.03,
                enabled=True, current_command=command, observed_at_s=observed_at_s,
            ).actual_gripper
            out.append(runtime.last_relative_grip.relative_closure)
        return out

    frozen = closures([0.0] * 12)
    advancing = closures([tick * 0.1 for tick in range(12)])

    assert frozen == [0.0] * 12, "a frozen clock must be what pins the mapping, not something else"
    assert advancing[-1] > 11.0, "with time advancing the mapping must reach the span"
    # And the mapping still reports itself active in BOTH cases -- which is why
    # the telemetry looked healthy while the gripper never moved.
    assert frozen != advancing
