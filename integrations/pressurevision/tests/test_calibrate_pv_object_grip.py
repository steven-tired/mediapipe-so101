import json

import pytest

import calibrate_pv_object_grip as calibrator
from calibrate_pv_object_grip import (
    format_sweep_summary,
    profile_trial_summary,
    prompt_profile_positions,
    scan_targets,
    validate_profile_positions,
    validate_profile_trials,
)


class Sample:
    def __init__(self, actual, current, load=0, temperature=32):
        self.gripper_pos = actual
        self.present_current = current
        self.present_load = load
        self.present_temperature = temperature
        self.t = 0.0
        self.goal_gripper_pos = actual


def test_profile_trial_summary_carries_position_repeatability_and_current():
    result = profile_trial_summary(
        [Sample(28.2, 14), Sample(28.0, 12), Sample(28.1, 13)],
        26.0,
    )
    assert result["median_actual_pos"] == pytest.approx(28.1)
    assert result["position_error"] == pytest.approx(2.1)
    assert result["actual_span"] == pytest.approx(0.2)
    assert result["mean_current"] == pytest.approx(13.0)
    assert result["temperature_over_limit_max_run"] == 0


def test_profile_trial_summary_retains_spike_but_requires_temperature_persistence():
    one_spike = profile_trial_summary([
        Sample(28.0, 13, temperature=33),
        Sample(28.0, 13, temperature=61),
        Sample(28.0, 13, temperature=33),
    ], 28.0)
    sustained = profile_trial_summary([
        Sample(28.0, 13, temperature=33),
        Sample(28.0, 13, temperature=56),
        Sample(28.0, 13, temperature=57),
    ], 28.0)
    assert one_spike["max_temperature"] == 61
    assert one_spike["temperature_over_limit_max_run"] == 1
    assert sustained["temperature_over_limit_max_run"] == 2


def test_profile_selection_accepts_safe_separated_positions():
    result = validate_profile_trials([
        {"target": 28.0, "mean_current": 13.0, "max_current": 20.0,
         "max_temperature": 33.0, "actual_span": 0.4},
        {"target": 26.0, "mean_current": 24.0, "max_current": 35.0,
         "max_temperature": 34.0, "actual_span": 0.5},
    ], light_pos=28.0, hard_pos=26.0)
    assert result["hard"]["target"] == 26.0


def test_profile_selection_does_not_require_a_current_gap_after_operator_teaching():
    result = validate_profile_trials([
        {"target": 28.0, "mean_current": 13.0, "mean_load": 20.0,
         "max_current": 20.0, "max_temperature": 33.0, "actual_span": 0.4},
        {"target": 27.5, "mean_current": 13.5, "mean_load": 21.0,
         "max_current": 21.0, "max_temperature": 34.0, "actual_span": 0.5},
    ], light_pos=28.0, hard_pos=27.5)

    assert result["observed_current_gap"] == pytest.approx(0.5)
    assert result["observed_load_gap"] == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("field", "bad_value"),
    [("max_current", 51.0), ("temperature_over_limit_max_run", 2), ("actual_span", 1.1)],
)
def test_profile_selection_rejects_safety_or_repeatability_failure(field, bad_value):
    records = [
        {"target": 28.0, "mean_current": 13.0, "max_current": 20.0,
         "max_temperature": 33.0, "temperature_over_limit_max_run": 0,
         "actual_span": 0.4},
        {"target": 26.0, "mean_current": 24.0, "max_current": 35.0,
         "max_temperature": 34.0, "temperature_over_limit_max_run": 0,
         "actual_span": 0.5},
    ]
    records[1][field] = bad_value
    with pytest.raises(ValueError):
        validate_profile_trials(records, light_pos=28.0, hard_pos=26.0)


def test_profile_selection_requires_requested_selected_repeats():
    records = [
        {"target": 28.0, "mean_current": 13.0, "max_current": 20.0,
         "max_temperature": 33.0, "actual_span": 0.4, "selected_repeat": 1},
        {"target": 26.0, "mean_current": 24.0, "max_current": 35.0,
         "max_temperature": 34.0, "actual_span": 0.5, "selected_repeat": 1},
    ]
    with pytest.raises(ValueError, match="2"):
        validate_profile_trials(records, light_pos=28.0, hard_pos=26.0, min_repeats=2)


def test_sweep_summary_displays_selection_evidence():
    text = format_sweep_summary({
        "target": 28.0,
        "median_actual_pos": 28.1,
        "mean_current": 13.0,
        "max_current": 20.0,
        "max_temperature": 33.0,
        "temperature_over_limit_max_run": 0,
        "actual_span": 0.4,
    })
    assert text == (
        "[sweep] target=28 actual=28.10 load_mean=0.00 current_mean=13.00 "
        "current_max=20.00 temp_max=33.0C temp_high_run=0 actual_span=0.40"
    )


def test_scan_targets_supports_sub_two_unit_rigid_calibration():
    assert scan_targets(28.0, 26.5, 0.5) == [28.0, 27.5, 27.0, 26.5]


def test_prompt_profile_positions_selects_only_completed_sweep_targets():
    answers = iter(["28", "26"])
    result = prompt_profile_positions(
        [{"target": 30.0}, {"target": 28.0}, {"target": 26.0}],
        input_fn=lambda _prompt: next(answers),
    )
    assert result == (28.0, 26.0)


def test_profile_positions_reject_target_not_completed_before_sweep_stop():
    with pytest.raises(ValueError, match="not completed.*24"):
        validate_profile_positions(
            28.0,
            24.0,
            scanned_targets={30.0, 28.0, 26.0},
        )


def test_interactive_main_opens_after_stopped_sweep_and_writes_profile(monkeypatch, tmp_path):
    moves = []

    class Robot:
        disconnected = False

        def disconnect(self):
            self.disconnected = True

    robot = Robot()

    def fake_hold(_robot, target, _hold_s, _steps, settle_s):
        assert settle_s == pytest.approx(0.25)
        current = {95.0: 1.0, 93.0: 10.0, 91.0: 25.0, 89.0: 50.0}[target]
        return [Sample(target, current) for _ in range(3)], []

    answers = iter(["YES", "93", "91", "YES"])

    def fake_input(prompt):
        if prompt.startswith("Enter light position"):
            assert moves[-1] == 95.0
        return next(answers)

    monkeypatch.setattr(calibrator, "_connect", lambda _port, _arm_id: robot)
    monkeypatch.setattr(calibrator, "_send_gripper", lambda _robot, pos: moves.append(pos))
    monkeypatch.setattr(calibrator, "_hold", fake_hold)
    monkeypatch.setattr(calibrator.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr("builtins.input", fake_input)

    profile_path = tmp_path / "rigid_block_01.profile.json"
    evidence_path = tmp_path / "rigid_block_01.sweep.json"
    result = calibrator.main([
        "--object-id", "rigid_block_01",
        "--out", str(profile_path),
        "--evidence", str(evidence_path),
    ])

    assert result == 0
    assert robot.disconnected is True
    assert moves[-1] == 95.0
    assert profile_path.exists()
    assert evidence_path.exists()
    evidence = json.loads(evidence_path.read_text())
    assert evidence["status"] == "PASS"
    assert evidence["protocol"] == {
        "hold_s": 1.5,
        "settle_s": 0.25,
        "repeats": 3,
        "steps": 20,
        "max_temperature_c": 55.0,
        "max_temperature_high_run": 1,
        "scan_start": 95.0,
        "scan_stop": 20.0,
        "scan_step": 2.0,
    }


def test_interactive_main_retains_no_go_evidence(monkeypatch, tmp_path):
    moves = []

    class Robot:
        disconnected = False

        def disconnect(self):
            self.disconnected = True

    robot = Robot()
    call_counts = {}

    def fake_hold(_robot, target, _hold_s, _steps, settle_s):
        assert settle_s == pytest.approx(0.25)
        call_counts[target] = call_counts.get(target, 0) + 1
        current = {95.0: 1.0, 93.0: 10.0, 91.0: 25.0, 89.0: 50.0}[target]
        actual = [target, target, target]
        if target == 91.0 and call_counts[target] > 1:
            actual = [target, target + 2.0, target + 1.0]
        return [Sample(position, current) for position in actual], []

    answers = iter(["YES", "93", "91", "YES"])
    monkeypatch.setattr(calibrator, "_connect", lambda _port, _arm_id: robot)
    monkeypatch.setattr(calibrator, "_send_gripper", lambda _robot, pos: moves.append(pos))
    monkeypatch.setattr(calibrator, "_hold", fake_hold)
    monkeypatch.setattr(calibrator.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr("builtins.input", lambda _prompt: next(answers))

    profile_path = tmp_path / "rigid_block_01.profile.json"
    evidence_path = tmp_path / "rigid_block_01.sweep.json"
    with pytest.raises(ValueError, match="evidence retained"):
        calibrator.main([
            "--object-id", "rigid_block_01",
            "--out", str(profile_path),
            "--evidence", str(evidence_path),
        ])

    evidence = json.loads(evidence_path.read_text())
    assert robot.disconnected is True
    assert moves[-1] == 95.0
    assert evidence["status"] == "NO_GO"
    assert evidence["protocol"]["settle_s"] == pytest.approx(0.25)
    assert evidence["validation_error"] == "actual position span exceeds 1"
    assert evidence["selected"] is None
    assert not profile_path.exists()


def test_preselected_positions_retain_partial_evidence_when_sweep_stops_early(
    monkeypatch,
    tmp_path,
):
    moves = []

    class Robot:
        disconnected = False

        def disconnect(self):
            self.disconnected = True

    robot = Robot()

    def fake_hold(_robot, target, _hold_s, _steps, _settle_s):
        assert target == 95.0
        return [
            Sample(target, 1, temperature=33),
            Sample(target, 1, temperature=56),
            Sample(target, 1, temperature=57),
        ], []

    monkeypatch.setattr(calibrator, "_connect", lambda _port, _arm_id: robot)
    monkeypatch.setattr(calibrator, "_send_gripper", lambda _robot, pos: moves.append(pos))
    monkeypatch.setattr(calibrator, "_hold", fake_hold)
    monkeypatch.setattr(calibrator.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr("builtins.input", lambda _prompt: "YES")

    profile_path = tmp_path / "rigid_block_01.profile.json"
    evidence_path = tmp_path / "rigid_block_01.sweep.json"
    with pytest.raises(SystemExit):
        calibrator.main([
            "--object-id", "rigid_block_01",
            "--light-pos", "93",
            "--hard-pos", "91",
            "--out", str(profile_path),
            "--evidence", str(evidence_path),
        ])

    evidence = json.loads(evidence_path.read_text())
    assert robot.disconnected is True
    assert moves[-1] == 95.0
    assert evidence["status"] == "NO_GO"
    assert evidence["failure_stage"] == "sweep_before_selected_repeats"
    assert evidence["summaries"][0]["temperature_over_limit_max_run"] == 2
    assert not profile_path.exists()
