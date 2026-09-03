"""The rejections that have to happen before the arm is energised.

Every timing below reaches a dataclass that raises for it, but only after
`robot.connect()` has powered the joints -- a typo then costs a power cycle
rather than an error message.
"""

import inspect

import pytest

import deploy_so101_grip_ee as deploy

BASE = [
    "--policy", "x",
    "--arm-enabled",
    "--evidence-dir", "/tmp/evidence",
    "--stall-tighten-step", "0.5",
]


def _validate(extra):
    ap = deploy.build_parser()
    return deploy.validate_args(ap, ap.parse_args(BASE + extra))


def test_stall_ramp_baseline_is_accepted():
    assert _validate([]) is False


@pytest.mark.parametrize(
    "flag, value",
    [
        ("--stall-tighten-interval-s", "0"),
        ("--stall-window-s", "0"),
        ("--stall-epsilon", "-1"),
    ],
)
def test_stall_timings_are_rejected_before_the_arm_is_touched(flag, value, capsys):
    with pytest.raises(SystemExit):
        _validate([flag, value])
    assert flag in capsys.readouterr().err


def test_paired_boundaries_refuses_pv_correction_recording(capsys):
    with pytest.raises(SystemExit):
        _validate([
            "--paired-boundaries",
            "--gripper-telemetry-hz", "5",
            "--correction-dataset-root", "/tmp/corrections",
        ])
    assert "--paired-boundaries cannot be combined" in capsys.readouterr().err


def test_ramp_recovery_hint_names_the_key_the_ramp_listens_for():
    # The warning prints exactly when the operator has lost the run's only
    # boundary label, so naming the wrong key loses the next one too.
    hint = inspect.getsource(deploy._print_boundary_summary)
    operator = inspect.getsource(deploy.TightenRampOperator)
    assert "Press 'a' at the lift next time" in hint
    assert 'ord("a")' in operator
