import ast
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest



def _proposal_module():
    try:
        from lerobot_teleoperator_so101_webcam.grip import proposal as ir_pressure_proposal
    except ImportError as exc:
        pytest.fail(f"pure pressure proposal module is missing: {exc}")
    return ir_pressure_proposal


@dataclass(frozen=True)
class _Reading:
    """A stand-in for whatever reading the caller has.

    Deliberately not any real sensor's type: the proposal machine duck-types on
    these five attributes, and the test would stop proving that if it imported
    one consumer's dataclass.
    """

    pressure_0_1: float
    active: bool
    quality: float
    available: bool
    status: str


def _reading(pressure, *, active=True, available=True, quality=1.0, status=None):
    if status is None:
        status = "active" if active else "baseline"
    return _Reading(
        pressure_0_1=pressure,
        active=active,
        quality=quality,
        available=available,
        status=status,
    )


def test_pure_module_has_no_robot_camera_or_gui_imports():
    module = _proposal_module()
    tree = ast.parse(Path(module.__file__).read_text(encoding="utf-8"))
    imported = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.append(node.module or "")

    forbidden = ("lerobot", "robot", "camera", "serial", "motor", "cv2", "mediapipe", "gui")
    assert not [name for name in imported if any(token in name.lower() for token in forbidden)]


def test_reviewed_baseline_active_and_two_unit_slew_sequence_is_unchanged():
    module = _proposal_module()
    proposal = module.PressureProposalStateMachine(initial_gripper=50.0)

    baseline = proposal.update(60.0, _reading(0.0, active=False))
    active = proposal.update(60.0, _reading(1.0))
    repeated_active = proposal.update(60.0, _reading(1.0))

    assert (baseline.raw_gripper, baseline.proposed_gripper, baseline.state) == (
        52.0,
        52.0,
        "armed",
    )
    assert active.raw_gripper == 50.0
    assert active.proposed_gripper == pytest.approx(50.6)
    assert active.state == "armed"
    assert repeated_active.raw_gripper == 48.0
    assert repeated_active.proposed_gripper == pytest.approx(48.78)
    assert abs(active.raw_gripper - baseline.raw_gripper) <= 2.0
    assert abs(repeated_active.raw_gripper - active.raw_gripper) <= 2.0


def test_fault_latch_holds_exactly_until_valid_inactive_baseline():
    module = _proposal_module()
    proposal = module.PressureProposalStateMachine(initial_gripper=50.0)
    proposal.update(60.0, _reading(0.0, active=False))
    before_fault = proposal.update(60.0, _reading(1.0))

    first_fault = proposal.update(
        60.0,
        _reading(0.0, active=False, available=False, status="thermal_unavailable"),
    )
    repeated_fault = proposal.update(
        60.0,
        _reading(0.0, active=False, available=False, status="thermal_unavailable"),
    )
    active_recovery = proposal.update(60.0, _reading(1.0))
    inactive_non_baseline = proposal.update(
        60.0,
        _reading(0.0, active=False, status="disabled"),
    )

    assert first_fault.proposed_gripper == before_fault.proposed_gripper
    assert repeated_fault.proposed_gripper == before_fault.proposed_gripper
    assert active_recovery.proposed_gripper == before_fault.proposed_gripper
    assert inactive_non_baseline.proposed_gripper == before_fault.proposed_gripper
    assert first_fault.reason == "thermal_unavailable"
    assert repeated_fault.reason == "thermal_unavailable"
    assert active_recovery.reason == "fault_latched"
    assert all(
        decision.state == "fault_latched" and decision.fault_latched
        for decision in (first_fault, repeated_fault, active_recovery, inactive_non_baseline)
    )

    rearmed = proposal.update(60.0, _reading(0.0, active=False, status="baseline"))

    assert rearmed.state == "armed"
    assert not rearmed.fault_latched
    assert rearmed.raw_gripper == 52.0
    assert rearmed.proposed_gripper == pytest.approx(50.81)


def test_pending_thermal_publication_holds_policy_without_advancing_state_or_filters():
    module = _proposal_module()
    proposal = module.PressureProposalStateMachine(initial_gripper=50.0)
    proposal.update(60.0, _reading(0.0, active=False))
    before_pending = proposal.update(60.0, _reading(1.0))
    pending = SimpleNamespace(
        pressure_0_1=0.0,
        active=False,
        quality=0.0,
        available=True,
        status="thermal_pending",
        fresh=False,
    )

    held = proposal.update(20.0, pending)
    held_again = proposal.update(100.0, pending)

    assert held.raw_gripper == before_pending.raw_gripper
    assert held.proposed_gripper == before_pending.proposed_gripper
    assert held_again.raw_gripper == before_pending.raw_gripper
    assert held_again.proposed_gripper == before_pending.proposed_gripper
    assert held.state == before_pending.state == "armed"
    assert held.reason == "thermal_pending"
    assert held.fault_latched is False


def test_first_fault_uses_safe_bounded_target_without_closing():
    module = _proposal_module()
    proposal = module.PressureProposalStateMachine(initial_gripper=50.0)

    decision = proposal.update(
        40.0,
        _reading(0.0, active=False, available=False, status="thermal_unavailable"),
    )

    assert decision.raw_gripper == 50.0
    assert decision.proposed_gripper == 50.0
    assert decision.state == "fault_latched"


def test_hold_preserves_proposal_while_middle_resets_it():
    module = _proposal_module()
    proposal = module.PressureProposalStateMachine(initial_gripper=50.0)
    proposal.update(60.0, _reading(0.0, active=False))
    before_hold = proposal.update(60.0, _reading(1.0))

    held = proposal.reset(60.0, transition="hold", middle_gripper=50.0)
    disarmed_active = proposal.update(60.0, _reading(1.0))

    assert held.state == "disarmed"
    assert held.reason == "hold"
    assert held.raw_gripper == before_hold.proposed_gripper
    assert held.proposed_gripper == before_hold.proposed_gripper
    assert disarmed_active.state == "disarmed"
    assert disarmed_active.reason == "pressure_disarmed"
    assert disarmed_active.raw_gripper == pytest.approx(52.6)
    assert disarmed_active.proposed_gripper == pytest.approx(50.9)

    middle = proposal.reset(60.0, transition="middle", middle_gripper=50.0)
    after_middle = proposal.update(60.0, _reading(1.0))

    assert middle.state == "disarmed"
    assert middle.raw_gripper == 50.0
    assert middle.proposed_gripper == 50.0
    assert proposal.smoothed_gripper == 52.0
    assert after_middle.raw_gripper == 52.0
    assert after_middle.proposed_gripper == 52.0


def test_reset_error_is_preserved_as_fallback_reason():
    module = _proposal_module()
    proposal = module.PressureProposalStateMachine(initial_gripper=50.0)

    decision = proposal.reset(
        60.0,
        transition="hold",
        middle_gripper=50.0,
        reason="pressure_reset_error:RuntimeError:reset failed",
    )

    assert decision.state == "disarmed"
    assert decision.reason == "pressure_reset_error:RuntimeError:reset failed"
