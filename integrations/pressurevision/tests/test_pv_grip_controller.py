"""The PV grip runtime: mapper -> adjustment lock -> proposal -> closure limiter.

These test the wiring. The lock's own semantics are pinned in
`test_adjustment_lock.py` against the original inline implementation.
"""

import pytest

from lerobot_teleoperator_so101_webcam.gripper_hardware import GripperClosureLimits
from pressurevision_integration.protocol import PressureReading
from pressurevision_integration.pv_grip_controller import (
    PV_MAPPINGS,
    PressureVisionGripRuntime,
    pressure_range_mapping_contract,
)
from pressurevision_integration.pv_object_profile import PressureVisionObjectProfile

#: Below GRIP_LATCH_ENTER, so the range mapper reports a live grasp.
GRASPING = 20.0
#: Above GRIP_LATCH_EXIT: the right hand has let go.
RELEASED = 80.0

LIMITS = GripperClosureLimits(max_load=500.0, max_current=400.0, max_position_lag=8.0)


class FakeSource:
    """Returns whatever reading it is told to, and counts resets and closes."""

    def __init__(self, reading=None):
        self.reading = reading
        self.resets = 0
        self.closes = 0
        self.raises = False
        self.calls = []

    def update(self, landmarks, *, pinch, enabled):
        self.calls.append((landmarks, pinch, enabled))
        if self.raises:
            raise RuntimeError("sender died")
        return self.reading

    def reset(self):
        self.resets += 1

    def close(self):
        self.closes += 1


def reading(pressure, *, active=True, available=True, quality=1.0):
    return PressureReading(
        pressure_0_1=pressure,
        active=active,
        quality=quality,
        available=available,
        status="active" if active else "baseline",
    )


def runtime(source=None, **kwargs):
    kwargs.setdefault("pv_mapping", "carton_span")
    kwargs.setdefault("pressure_apply", True)
    return PressureVisionGripRuntime(
        source if source is not None else FakeSource(reading(0.0, active=False)),
        initial_gripper=kwargs.pop("initial_gripper", 50.0),
        middle_gripper=50.0,
        **kwargs,
    )


def drive(rt, *, pressure, t, base=GRASPING, command=None):
    """One frame. `command` defaults to whatever the last frame commanded."""
    if command is None:
        command = getattr(rt, "_test_last_command", 50.0)
    rt.pressure_source.reading = pressure
    decision = rt.update(
        base_gripper=base,
        landmarks=None,
        pinch=0.01,
        enabled=True,
        current_command=command,
        observed_at_s=t,
    )
    rt._test_last_command = decision.actual_gripper
    return decision


# -- construction ------------------------------------------------------------


def test_a_pressure_source_is_required():
    with pytest.raises(ValueError, match="requires a pressure_source"):
        PressureVisionGripRuntime(None, initial_gripper=50.0, middle_gripper=50.0)


def test_applying_pressure_to_the_gripper_must_be_asked_for_explicitly():
    with pytest.raises(ValueError, match="must be explicit"):
        PressureVisionGripRuntime(
            FakeSource(), initial_gripper=50.0, middle_gripper=50.0
        )


def test_shadow_alone_is_enough_to_construct():
    assert runtime(pressure_apply=False, pressure_shadow=True).pressure_shadow


def test_unknown_mapping_is_refused():
    with pytest.raises(ValueError, match="unknown pv_mapping"):
        runtime(pv_mapping="nonesuch")


@pytest.mark.parametrize("mapping", ["relative", "hard_profile"])
def test_profile_mappings_require_a_profile(mapping):
    # In shadow, so the check under test is the mapping's, not apply's.
    with pytest.raises(ValueError, match="requires an object profile"):
        runtime(pv_mapping=mapping, pressure_apply=False, pressure_shadow=True)


@pytest.mark.parametrize("mapping", ["soft_direct", "soft_precise", "carton_span"])
def test_span_mappings_refuse_a_profile(mapping):
    profile = PressureVisionObjectProfile(
        object_id="box", arm_id="so101", open_pos=100.0, light_pos=32.0, hard_pos=20.0
    )
    with pytest.raises(ValueError, match="does not use an object profile"):
        runtime(pv_mapping=mapping, object_profile=profile)


def test_applying_an_unmapped_policy_requires_a_profile():
    with pytest.raises(ValueError, match="requires a hard object profile"):
        runtime(pv_mapping="absolute")


def test_closure_limits_are_meaningless_without_apply():
    with pytest.raises(ValueError, match="require pressure_apply"):
        runtime(
            pressure_apply=False,
            pressure_shadow=True,
            gripper_closure_limits=LIMITS,
        )


def test_max_grip_step_must_be_positive():
    with pytest.raises(ValueError, match="must be positive"):
        runtime(pressure_max_grip_step=0.0)


# -- the mapping contract ----------------------------------------------------


def test_the_contract_carries_the_calibrated_carton_span():
    contract = pressure_range_mapping_contract("carton_span")
    assert contract["pressure_zero_pos"] == 32.0
    assert contract["pressure_one_pos"] == 20.0
    assert contract["release_pos"] == 100.0
    assert contract["stabilize"] is False


def test_the_hard_profile_contract_comes_from_the_profile_and_stabilizes():
    profile = PressureVisionObjectProfile(
        object_id="mug", arm_id="so101", open_pos=90.0, light_pos=40.0, hard_pos=25.0
    )
    contract = pressure_range_mapping_contract("hard_profile", object_profile=profile)
    assert (contract["release_pos"], contract["pressure_zero_pos"], contract["pressure_one_pos"]) == (
        90.0,
        40.0,
        25.0,
    )
    # A calibrated object is the one case worth the track-hold filter.
    assert contract["stabilize"] is True


@pytest.mark.parametrize("mapping", ["absolute", "relative"])
def test_the_mappings_that_predate_the_range_mapper_have_no_contract(mapping):
    assert pressure_range_mapping_contract(mapping) is None


def test_every_named_mapping_either_has_a_contract_or_is_one_of_the_two_legacy_ones():
    """Guards the guard: a typo in PV_MAPPINGS would silently drop a mapping."""
    contracted = {
        m for m in PV_MAPPINGS
        if m not in ("absolute", "relative", "hard_profile")
        and pressure_range_mapping_contract(m) is not None
    }
    assert contracted == {"soft_direct", "soft_precise", "carton_span"}


def test_the_runtime_builds_its_mapper_from_its_own_contract():
    rt = runtime()
    assert rt._range_mapper.pressure_zero_pos == rt.mapping_contract["pressure_zero_pos"]
    assert rt._range_mapper.pressure_one_pos == rt.mapping_contract["pressure_one_pos"]


# -- per-frame control -------------------------------------------------------


def test_pressure_closes_the_gripper_within_the_span():
    rt = runtime()
    first = drive(rt, pressure=reading(1.0), t=0.0, command=50.0)
    later = [drive(rt, pressure=reading(1.0), t=0.1 * n) for n in range(1, 20)]

    assert first.actual_gripper < 50.0
    assert later[-1].actual_gripper == pytest.approx(20.0, abs=0.5)
    # Never outside the configured span, however hard PV pushes.
    assert all(d.actual_gripper >= 20.0 for d in later)


def test_the_command_is_rate_limited_to_the_contracted_step():
    rt = runtime()
    step = rt.mapping_contract["max_grip_step_per_control_frame"]
    previous = 50.0
    for n in range(10):
        current = drive(rt, pressure=reading(1.0), t=0.1 * n, command=previous).actual_gripper
        assert previous - current <= step + 1e-9
        previous = current


def test_losing_contact_latches_the_grip_and_publishes_a_teacher_label():
    rt = runtime()
    for n in range(20):
        drive(rt, pressure=reading(0.5), t=0.1 * n)
    assert rt.adjustment_state == "adjusting"
    assert rt.adjustment_teacher is None  # nothing anchored yet

    # Contact drops out. Below the confirmation window this is only a hold.
    drive(rt, pressure=reading(0.0, active=False), t=2.5)
    assert rt.adjustment_state == "temporary_hold"
    assert not rt.adjustment_locked

    drive(rt, pressure=reading(0.0, active=False), t=4.0)
    assert rt.adjustment_locked
    assert rt.adjustment_state == "locked"
    anchor = rt.adjustment_anchor_target
    assert anchor is not None
    # The teacher is the anchor as a fraction of the 32..20 span.
    assert rt.adjustment_teacher == pytest.approx((32.0 - anchor) / 12.0)
    assert 0.0 <= rt.adjustment_teacher <= 1.0


def test_a_latched_grip_holds_while_pv_reports_nothing():
    rt = runtime()
    for n in range(20):
        drive(rt, pressure=reading(0.5), t=0.1 * n)
    drive(rt, pressure=reading(0.0, active=False), t=2.5)
    drive(rt, pressure=reading(0.0, active=False), t=4.0)
    anchor = rt.adjustment_anchor_target

    held = [drive(rt, pressure=reading(0.0, active=False), t=4.0 + n) for n in range(1, 6)]

    assert all(d.actual_gripper == pytest.approx(anchor) for d in held)
    assert rt.adjustment_locked


def test_one_flickering_contact_frame_does_not_unlatch_a_held_object():
    rt = runtime()
    for n in range(20):
        drive(rt, pressure=reading(0.5), t=0.1 * n)
    drive(rt, pressure=reading(0.0, active=False), t=2.5)
    drive(rt, pressure=reading(0.0, active=False), t=4.0)
    assert rt.adjustment_locked

    drive(rt, pressure=reading(1.0), t=5.0)

    assert rt.adjustment_locked
    assert rt.adjustment_event == "recontact_started"


def test_sustained_recontact_resumes_adjustment():
    rt = runtime()
    for n in range(20):
        drive(rt, pressure=reading(0.5), t=0.1 * n)
    drive(rt, pressure=reading(0.0, active=False), t=2.5)
    drive(rt, pressure=reading(0.0, active=False), t=4.0)

    drive(rt, pressure=reading(1.0), t=5.0)
    drive(rt, pressure=reading(1.0), t=5.5)

    assert not rt.adjustment_locked
    assert rt.adjustment_event == "recontact_resume"
    assert rt.adjustment_state == "adjusting"


def test_releasing_the_right_hand_clears_the_lock():
    rt = runtime()
    for n in range(20):
        drive(rt, pressure=reading(0.5), t=0.1 * n)
    drive(rt, pressure=reading(0.0, active=False), t=2.5)
    drive(rt, pressure=reading(0.0, active=False), t=4.0)
    assert rt.adjustment_locked

    drive(rt, pressure=reading(0.0, active=False), t=5.0, base=RELEASED)

    assert not rt.adjustment_locked
    assert rt.adjustment_anchor_target is None
    assert rt.adjustment_state == "inactive"


def test_release_survives_a_dead_pv_sender():
    """The right hand's release is a command, not an inference."""
    rt = runtime()
    for n in range(20):
        drive(rt, pressure=reading(0.5), t=0.1 * n)
    grasped = rt._test_last_command
    rt.pressure_source.raises = True

    opening = [
        drive(rt, pressure=None, t=3.0 + 0.1 * n, base=RELEASED).actual_gripper
        for n in range(30)
    ]

    assert rt.last_pressure.status == "pressure_error"
    assert opening[-1] > grasped


def test_a_raising_sender_becomes_an_inactive_reading_rather_than_an_exception():
    rt = runtime()
    rt.pressure_source.raises = True

    decision = drive(rt, pressure=None, t=0.0, command=50.0)

    assert rt.last_pressure.status == "pressure_error"
    assert rt.last_pressure.available is False
    assert decision.actual_gripper == pytest.approx(50.0)


def test_a_silent_sender_becomes_an_inactive_reading():
    rt = runtime()
    drive(rt, pressure=None, t=0.0, command=50.0)
    assert rt.last_pressure.status == "pressure_unavailable"


# -- shadow vs apply ---------------------------------------------------------


def test_shadow_commands_the_callers_legacy_path_and_not_the_proposal():
    rt = runtime(pressure_apply=False, pressure_shadow=True)

    decision = drive_shadow = rt.update(
        base_gripper=GRASPING,
        landmarks=None,
        pinch=0.01,
        enabled=True,
        current_command=50.0,
        observed_at_s=0.0,
        smooth_legacy=lambda: 42.0,
    )

    assert decision.shadow is True
    assert decision.actual_gripper == 42.0
    # The proposal still ran: shadow exists to compare, not to skip.
    assert decision.control.proposed_gripper != 42.0
    assert drive_shadow.control.base_gripper == GRASPING


def test_shadow_without_a_legacy_smoother_is_a_programming_error():
    rt = runtime(pressure_apply=False, pressure_shadow=True)
    with pytest.raises(ValueError, match="legacy smoother"):
        drive(rt, pressure=reading(1.0), t=0.0, command=50.0)


def test_apply_commands_the_proposal():
    rt = runtime()
    decision = drive(rt, pressure=reading(1.0), t=0.0, command=50.0)
    assert decision.shadow is False
    assert decision.actual_gripper == pytest.approx(decision.control.proposed_gripper)


# -- lifecycle ---------------------------------------------------------------


def test_reset_mappers_forgets_the_grasp():
    rt = runtime()
    for n in range(20):
        drive(rt, pressure=reading(0.5), t=0.1 * n)
    drive(rt, pressure=reading(0.0, active=False), t=2.5)
    drive(rt, pressure=reading(0.0, active=False), t=4.0)
    assert rt.adjustment_locked

    rt.reset_mappers(event="middle_reset")

    assert not rt.adjustment_locked
    assert rt.adjustment_anchor_target is None
    assert rt.last_relative_grip is None
    assert rt.last_pressure is None
    assert rt.adjustment_event == "middle_reset"


def test_reset_pressure_source_reports_a_raising_sender_instead_of_propagating():
    class Angry(FakeSource):
        def reset(self):
            raise RuntimeError("nope")

    rt = runtime(Angry(reading(0.0, active=False)))
    assert rt.reset_pressure_source() == "pressure_reset_error:RuntimeError:nope"


def test_reset_pressure_source_is_quiet_on_a_healthy_sender():
    rt = runtime()
    assert rt.reset_pressure_source() is None
    assert rt.pressure_source.resets == 1


def test_reset_control_records_the_transition():
    rt = runtime()
    control = rt.reset_control(GRASPING, 50.0, "middle")
    assert control.actual_gripper == 50.0
    assert rt.last_pressure_control is control


def test_close_closes_the_sender_and_drops_the_reference():
    source = FakeSource(reading(0.0, active=False))
    rt = runtime(source)
    rt.close()
    assert source.closes == 1
    assert rt.pressure_source is None


# -- closure limiter ---------------------------------------------------------


def test_the_closure_limiter_can_override_the_proposal():
    rt = runtime(gripper_closure_limits=LIMITS)
    decision = drive(rt, pressure=reading(1.0), t=0.0, command=50.0)
    # No telemetry has been supplied, so the limiter has nothing to fault on;
    # what matters here is that it is in the path at all.
    assert rt._closure_limiter is not None
    assert decision.control.actual_gripper == decision.actual_gripper
