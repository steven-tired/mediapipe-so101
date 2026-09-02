"""The no-learning stall-and-tighten baseline any grip head has to beat.

The shapes here are trial05's: the body commands go bit-identical for about
nine seconds while the grasp is inadequate, and resume a large coordinated
motion within a second of the grasp being tightened. That is the whole signal —
no camera, no operator, no model.
"""

import pytest

from lerobot_teleoperator_so101_webcam.grip.runtime import (
    BodyStallDetector,
    StallConfig,
    StallTightenRamp,
    TightenRampConfig,
)

# trial05, 41.2 s to 50.1 s: shoulder_lift and elbow_flex, bit-identical.
STALLED = [-19.32, 35.82]


def lifting(t):
    """The large coordinated motion trial05 resumed with: -19.32 -> -26.46
    on shoulder_lift and 35.82 -> 18.90 on elbow_flex over about six seconds."""
    return [-19.32 - 1.2 * t, 35.82 - 2.8 * t]


def _config(**overrides):
    defaults = dict(step=2.0, interval_s=0.5, floor_pos=20.0)
    defaults.update(overrides)
    return TightenRampConfig(**defaults)


def _run(ramp, *, seconds, body_at, policy_target=28.0, hz=10.0):
    """Drive the ramp at a control rate, letting the jaw follow its target."""
    actual = policy_target
    trace = []
    for index in range(int(seconds * hz)):
        t = index / hz
        target, label = ramp.update(
            t=t, policy_target=policy_target, actual_pos=actual, body_command=body_at(t)
        )
        actual = target
        trace.append((t, target, label))
    return trace


def test_a_moving_body_is_not_a_stall():
    detector = BodyStallDetector(StallConfig(window_s=2.0))
    for index in range(40):
        t = index / 10.0
        state = detector.update(t=t, body_command=[-19.32 + t, 35.82])
    assert state["stalled"] is False


def test_bit_identical_commands_read_as_a_stall_once_the_window_fills():
    detector = BodyStallDetector(StallConfig(window_s=2.0))
    states = [detector.update(t=index / 10.0, body_command=STALLED) for index in range(40)]
    # Nothing fires before the window can show `window_s` of stillness.
    assert states[10]["stalled"] is False
    assert states[10]["still_for_s"] == pytest.approx(1.0)
    assert states[-1]["stalled"] is True
    assert states[-1]["command_spread"] == 0.0


def test_two_identical_samples_are_not_yet_a_stall():
    """An inference hiccup repeats a command; it does not withhold a lift."""
    detector = BodyStallDetector(StallConfig(window_s=2.0))
    detector.update(t=0.0, body_command=STALLED)
    assert detector.update(t=0.1, body_command=STALLED)["stalled"] is False


def test_the_ramp_tightens_only_while_stalled_and_only_at_its_interval():
    ramp = StallTightenRamp(_config(floor_pos=5.0), stall=StallConfig(window_s=2.0))
    trace = _run(ramp, seconds=6.0, body_at=lambda t: STALLED)
    tightens = [label for _, _, label in trace if label["action"] == "tighten"]
    # 6 s of run at 10 Hz (t = 0.0 .. 5.9), 2 s to recognise the stall, then a
    # step every 0.5 s from t = 2.5.
    assert len(tightens) == 7
    assert all(label["delta_q"] == -2.0 for label in tightens)
    assert trace[-1][1] == pytest.approx(28.0 - 7 * 2.0)


def test_the_ramp_stops_at_the_floor_and_says_so():
    ramp = StallTightenRamp(_config(floor_pos=25.0), stall=StallConfig(window_s=2.0))
    trace = _run(ramp, seconds=8.0, body_at=lambda t: STALLED)
    assert trace[-1][1] == pytest.approx(26.0)
    assert trace[-1][2]["at_floor"] is True
    assert any(label["action"] == "at_floor" for _, _, label in trace)


def test_the_depth_that_broke_the_stall_is_recorded_and_control_goes_back():
    """trial05: manual tightening moved readback 31.93 -> 27.28 and the body
    resumed within one second. That 27.28 is the grasp's lift boundary."""
    ramp = StallTightenRamp(_config(), stall=StallConfig(window_s=2.0))
    trace = _run(ramp, seconds=8.0,
                 body_at=lambda t: STALLED if t < 5.0 else lifting(t - 5.0))
    resumed = [label for _, _, label in trace if label["action"] == "resumed_after_tighten"]
    assert len(resumed) == 1
    # The stall is only *seen* to have broken once the window clears, so the
    # boundary is the depth reached, not the depth at the resuming command.
    assert resumed[0]["lift_boundary"] == pytest.approx(ramp.lift_boundary)
    assert ramp.lift_boundary < 28.0
    # After the handback the policy's own target passes through untouched.
    assert trace[-1][1] == pytest.approx(28.0)
    assert trace[-1][2]["action"] == "following_policy"


def test_a_stall_that_breaks_before_the_first_step_leaves_no_boundary():
    """No tightening happened, so there is no depth that a tightening bought.

    A window-based detector spent every other cycle a float hair under its
    threshold, read each as a resume, and wrote the untightened starting depth
    into this column. The column looked populated and meant nothing.
    """
    ramp = StallTightenRamp(_config(interval_s=5.0), stall=StallConfig(window_s=2.0))
    trace = _run(ramp, seconds=4.0,
                 body_at=lambda t: STALLED if t < 2.5 else lifting(t - 2.5))
    assert [label["action"] for _, _, label in trace].count("resumed_untouched") == 1
    assert ramp.lift_boundary is None


def test_a_drift_smaller_than_the_epsilon_still_breaks_the_stall():
    """Measured against where the command settled, not the previous sample."""
    ramp = StallTightenRamp(_config(), stall=StallConfig(window_s=2.0, motion_epsilon=0.05))
    creep = lambda t: [-19.32 + 0.01 * t * 10, 35.82]
    trace = _run(ramp, seconds=6.0, body_at=creep)
    assert not any(label["stalled"] for _, _, label in trace)


def test_the_ramp_never_tightens_into_a_release():
    ramp = StallTightenRamp(_config(), stall=StallConfig(window_s=2.0))
    trace = _run(ramp, seconds=6.0, body_at=lambda t: STALLED, policy_target=99.3)
    assert {label["action"] for _, _, label in trace} == {"release"}
    assert all(target == pytest.approx(99.3) for _, target, _ in trace)


def test_a_step_must_be_sized_from_a_calibration():
    with pytest.raises(ValueError):
        _config(step=0.0)
    with pytest.raises(ValueError):
        _config(interval_s=0.0)
    with pytest.raises(ValueError):
        _config(floor_pos=-1.0)


def test_the_detector_rejects_a_changing_command_width():
    detector = BodyStallDetector()
    detector.update(t=0.0, body_command=STALLED)
    with pytest.raises(ValueError):
        detector.update(t=0.1, body_command=[1.0, 2.0, 3.0])
    with pytest.raises(ValueError):
        BodyStallDetector().update(t=0.0, body_command=[])


def test_the_run_summary_survives_the_handback():
    """The fields the outcome is read from must not be the live ones.

    `_release()` clears steps_applied and target the moment the stall breaks --
    that is, on exactly the trials where the ramp worked -- so a summary read
    from those would report every successful trial as having tightened nothing.
    """
    ramp = StallTightenRamp(_config(floor_pos=5.0), stall=StallConfig(window_s=2.0))
    _run(ramp, seconds=8.0, body_at=lambda t: STALLED if t < 5.0 else lifting(t - 5.0))

    assert ramp.steps_applied == 0, "the live counter is cleared on handback"
    assert ramp.target is None
    assert ramp.total_steps_applied == 6
    assert ramp.deepest_target == pytest.approx(28.0 - 6 * 2.0)
    assert ramp.lift_boundary is not None


def test_the_deepest_target_is_the_squeeze_the_carton_took():
    ramp = StallTightenRamp(_config(floor_pos=25.0), stall=StallConfig(window_s=2.0))
    _run(ramp, seconds=8.0, body_at=lambda t: STALLED)
    assert ramp.deepest_target == pytest.approx(26.0)
    assert ramp.reached_floor is True


def test_the_logged_spread_is_what_was_measured_not_zero_on_reset():
    """The column exists to choose motion_epsilon, so it must show the motion.

    Zeroing it whenever it exceeded the threshold made every moving cycle log
    the same 0.0 as every still one.
    """
    detector = BodyStallDetector(StallConfig(window_s=2.0, motion_epsilon=0.05))
    detector.update(t=0.0, body_command=STALLED)
    moving = detector.update(t=0.1, body_command=[-19.32 + 3.0, 35.82])
    assert moving["command_spread"] == pytest.approx(3.0)
    assert moving["reference_reset"] is True
    still = detector.update(t=0.2, body_command=[-19.32 + 3.0, 35.82])
    assert still["command_spread"] == 0.0
    assert still["reference_reset"] is False


def test_an_operator_can_drive_the_ramp_while_the_body_keeps_moving():
    """attempt04's failure: a grasp held too loosely while ACT keeps moving.

    The detector never fires on it -- the longest still window was 0.50 s
    against a 2 s threshold -- and no automatic lift detector survived
    validation, so a person decides.
    """
    ramp = StallTightenRamp(_config(floor_pos=5.0), stall=StallConfig(window_s=2.0))
    actual = 29.0
    engaged = False
    trace = []
    for index in range(60):
        t = index / 10.0
        if t >= 2.0:
            engaged = True
        target, label = ramp.update(
            t=t, policy_target=29.0, actual_pos=actual,
            body_command=lifting(t), engaged=engaged,
        )
        actual = target
        trace.append(label)
    assert not any(label["stalled"] for label in trace), "the body never stopped moving"
    assert ramp.total_steps_applied == 7
    assert ramp.deepest_target == pytest.approx(29.0 - 7 * 2.0)


def test_the_detector_still_records_while_the_operator_drives():
    ramp = StallTightenRamp(_config(), stall=StallConfig(window_s=2.0))
    labels = [
        ramp.update(t=i / 10.0, policy_target=29.0, actual_pos=29.0,
                    body_command=STALLED, engaged=False)[1]
        for i in range(40)
    ]
    assert labels[-1]["stalled"] is True, "the detector keeps its own verdict"
    assert ramp.total_steps_applied == 0, "but it did not drive the ramp"
