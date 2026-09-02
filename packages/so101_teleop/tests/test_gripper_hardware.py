import itertools

import pytest
from types import SimpleNamespace

from lerobot_teleoperator_so101_webcam.gripper_hardware import (
    GripperClosureLimiter,
    GripperClosureLimits,
    GripperRuntimeTelemetry,
    GripperTelemetrySampler,
    TelemetrySnapshot,
    read_gripper_runtime_telemetry,
    choose_three_grip_targets,
    serialize_telemetry_snapshot,
    slow_close_waypoints,
    summarize_target_current,
)


def _runtime_telemetry(*, t=10.0, pos=30.0, current=20, load=100):
    return GripperRuntimeTelemetry(
        observed_at_s=t,
        observed_gripper_pos=pos,
        present_current=current,
        present_load=load,
        present_temperature=35,
    )


def test_closure_limiter_latches_last_command_and_allows_only_opening():
    limiter = GripperClosureLimiter(GripperClosureLimits(240, 35, 5.0))
    limiter.update(
        requested_pos=27.0,
        last_commanded_pos=27.0,
        telemetry=_runtime_telemetry(t=9.7, pos=27.0, load=100),
        observed_at_s=9.7,
    )

    tripped = limiter.update(
        requested_pos=20.0,
        last_commanded_pos=27.0,
        telemetry=_runtime_telemetry(pos=30.0, load=240),
        observed_at_s=10.1,
    )
    blocked_close = limiter.update(
        requested_pos=18.0,
        last_commanded_pos=27.0,
        telemetry=_runtime_telemetry(t=10.2, pos=29.0, load=100),
        observed_at_s=10.2,
    )
    allowed_open = limiter.update(
        requested_pos=40.0,
        last_commanded_pos=27.0,
        telemetry=_runtime_telemetry(t=10.3, pos=29.0, load=100),
        observed_at_s=10.3,
    )

    assert tripped.actual_pos == 27.0
    assert tripped.reason == "closure_limit_load"
    assert blocked_close.actual_pos == 27.0
    assert allowed_open.actual_pos == 40.0
    assert allowed_open.fault_latched


@pytest.mark.parametrize(
    ("telemetry", "reason"),
    [
        (_runtime_telemetry(current=35), "closure_limit_current"),
        (None, "closure_limit_telemetry_unavailable"),
        (_runtime_telemetry(t=9.0), "closure_limit_telemetry_stale"),
        (
            GripperRuntimeTelemetry(10.0, 30.0, None, 100, 35),
            "closure_limit_telemetry_incomplete",
        ),
    ],
)
def test_closure_limiter_trips_each_feedback_gate(telemetry, reason):
    limiter = GripperClosureLimiter(GripperClosureLimits(240, 35, 5.0))

    decision = limiter.update(
        requested_pos=20.0,
        last_commanded_pos=30.0,
        telemetry=telemetry,
        observed_at_s=10.0,
    )

    assert decision.actual_pos == 30.0
    assert decision.reason == reason
    assert decision.fault_latched


@pytest.mark.parametrize(
    "telemetry",
    [
        None,
        _runtime_telemetry(t=9.0),
        GripperRuntimeTelemetry(10.0, 30.0, None, 100, 35),
    ],
)
def test_closure_limiter_resumes_after_transient_telemetry_block(telemetry):
    limiter = GripperClosureLimiter(GripperClosureLimits(240, 35, 5.0))

    blocked = limiter.update(
        requested_pos=28.0,
        last_commanded_pos=30.0,
        telemetry=telemetry,
        observed_at_s=10.0,
    )
    resumed = limiter.update(
        requested_pos=28.0,
        last_commanded_pos=30.0,
        telemetry=_runtime_telemetry(t=10.01, pos=30.0, current=0, load=24),
        observed_at_s=10.01,
    )

    assert blocked.actual_pos == 30.0
    assert blocked.fault_latched
    assert limiter.latched_pos is None
    assert resumed.actual_pos == 28.0
    assert not resumed.fault_latched


def test_closure_limiter_does_not_compare_cached_feedback_to_moving_command():
    limiter = GripperClosureLimiter(GripperClosureLimits(240, 35, 5.0))
    cached = _runtime_telemetry(t=10.0, pos=99.1475, current=0, load=24)

    for now, last_commanded_pos in (
        (10.000, 99.1475),
        (10.033, 97.1475),
        (10.066, 95.1475),
        (10.099, 93.1475),
    ):
        decision = limiter.update(
            requested_pos=last_commanded_pos - 2.0,
            last_commanded_pos=last_commanded_pos,
            telemetry=cached,
            observed_at_s=now,
        )
        assert not decision.fault_latched

    assert decision.position_lag > 5.0
    assert decision.reason == "within_closure_limits"


def test_closure_limiter_does_not_latch_dynamic_load_from_moving_command():
    limiter = GripperClosureLimiter(GripperClosureLimits(240, 35, 5.0))

    for now, last_commanded_pos, load in (
        (10.000, 85.0, 24),
        (10.033, 83.0, 24),
        (10.066, 81.0, 280),
        (10.099, 79.0, 280),
        (10.132, 77.0, 24),
    ):
        decision = limiter.update(
            requested_pos=last_commanded_pos - 2.0,
            last_commanded_pos=last_commanded_pos,
            telemetry=_runtime_telemetry(
                t=now,
                pos=last_commanded_pos + 7.0,
                current=4,
                load=load,
            ),
            observed_at_s=now,
        )
        assert not decision.fault_latched


def test_closure_limiter_checks_load_after_fresh_settled_feedback():
    limiter = GripperClosureLimiter(GripperClosureLimits(240, 35, 5.0))
    limiter.update(
        requested_pos=30.0,
        last_commanded_pos=30.0,
        telemetry=_runtime_telemetry(t=10.0, pos=30.0, load=100),
        observed_at_s=10.0,
    )

    decision = limiter.update(
        requested_pos=28.0,
        last_commanded_pos=30.0,
        telemetry=_runtime_telemetry(t=10.21, pos=31.0, load=240),
        observed_at_s=10.21,
    )

    assert decision.actual_pos == 30.0
    assert decision.reason == "closure_limit_load"
    assert decision.fault_latched


def test_closure_limiter_checks_position_lag_after_fresh_settled_feedback():
    limiter = GripperClosureLimiter(GripperClosureLimits(240, 35, 5.0))
    limiter.update(
        requested_pos=28.0,
        last_commanded_pos=30.0,
        telemetry=_runtime_telemetry(t=10.0, pos=30.0),
        observed_at_s=10.0,
    )

    decision = limiter.update(
        requested_pos=28.0,
        last_commanded_pos=30.0,
        telemetry=_runtime_telemetry(t=10.21, pos=35.0),
        observed_at_s=10.21,
    )

    assert decision.actual_pos == 30.0
    assert decision.reason == "closure_limit_position_lag"
    assert decision.fault_latched


def test_closure_limiter_explicit_release_clears_latch():
    limiter = GripperClosureLimiter(GripperClosureLimits(240, 35, 5.0))
    limiter.update(
        requested_pos=30.0,
        last_commanded_pos=30.0,
        telemetry=_runtime_telemetry(t=9.7, pos=30.0, load=100),
        observed_at_s=9.7,
    )
    limiter.update(
        requested_pos=20.0,
        last_commanded_pos=30.0,
        telemetry=_runtime_telemetry(load=240),
        observed_at_s=10.0,
    )

    released = limiter.update(
        requested_pos=100.0,
        last_commanded_pos=30.0,
        telemetry=_runtime_telemetry(load=300),
        observed_at_s=10.0,
        release_requested=True,
    )

    assert released.actual_pos == 100.0
    assert not released.fault_latched
    assert limiter.latched_pos is None


def test_runtime_sampler_reads_only_at_requested_cadence():
    reads = []

    def read(register, motor, **_kwargs):
        reads.append((register, motor))
        return {
            "Present_Current": 12,
            "Present_Load": 21,
            "Present_Temperature": 33,
        }[register]

    observations = []

    def get_observation():
        observations.append(True)
        return {"gripper.pos": 27.5}

    robot = SimpleNamespace(
        get_observation=get_observation,
        bus=SimpleNamespace(read=read),
    )
    clock = iter((10.0, 10.01, 10.1, 10.21, 10.22)).__next__
    sampler = GripperTelemetrySampler(interval_s=0.2, clock=clock)

    first = sampler.poll(robot)
    cached = sampler.poll(robot)
    second = sampler.poll(robot)

    assert first.observed_gripper_pos == 27.5
    assert first.present_current == 12
    assert first.present_load == 21
    assert first.present_temperature == 33
    assert cached is first
    assert second.observed_at_s == pytest.approx(10.22)
    assert len(observations) == 2
    assert reads == [
        ("Present_Current", "gripper"),
        ("Present_Temperature", "gripper"),
        ("Present_Load", "gripper"),
        ("Present_Current", "gripper"),
        ("Present_Temperature", "gripper"),
        ("Present_Load", "gripper"),
    ]


def _robot(read):
    return SimpleNamespace(
        get_observation=lambda: {"gripper.pos": 27.5},
        bus=SimpleNamespace(read=read),
    )


def test_a_bus_fault_leaves_the_register_blank_rather_than_zero():
    """The servo did not answer. That is a missing value, not a reading of 0."""

    def read(register, motor, **_kwargs):
        if register == "Present_Load":
            raise ConnectionError("no response from servo")
        return 12

    telemetry = read_gripper_runtime_telemetry(_robot(read), clock=lambda: 10.0)

    assert telemetry.present_load is None
    assert telemetry.present_current == 12


def test_a_register_name_the_control_table_does_not_have_raises():
    """The failure this exists to prevent: a typo'd register name used to be
    swallowed into `None`, so three columns came out permanently blank with
    nothing logged. A bug in this file must be loud."""

    def read(register, motor, **_kwargs):
        raise KeyError(f"Address for '{register}' not found in sts3215 control table.")

    with pytest.raises(KeyError):
        read_gripper_runtime_telemetry(_robot(read), clock=lambda: 10.0)


def test_the_sampler_rides_out_a_bus_fault_but_not_a_wrong_register():
    """`poll` keeps the last good sample across a dropped port -- a control loop
    must survive one -- but a wrong register or a wrong robot is not a hiccup,
    and holding a stale sample would hide it for the whole run."""
    ticks = itertools.count(10.0, 0.1)
    sampler = GripperTelemetrySampler(interval_s=0.05, clock=lambda: next(ticks))

    good = sampler.poll(_robot(lambda register, motor, **_kwargs: 12))
    assert good.present_load == 12

    dropped = SimpleNamespace(
        get_observation=_raise(ConnectionError("port dropped")),
        bus=SimpleNamespace(read=lambda register, motor, **_kwargs: 12),
    )
    assert sampler.poll(dropped) is good

    with pytest.raises(KeyError):
        sampler.poll(_robot(_raise(KeyError("Present_Lodd"))))


def _raise(exc):
    def fail(*_args, **_kwargs):
        raise exc

    return fail


def test_slow_close_waypoints_are_monotonic_and_include_target():
    assert slow_close_waypoints(100.0, 40.0, steps=4) == [85.0, 70.0, 55.0, 40.0]


def test_slow_close_rejects_nonpositive_steps():
    with pytest.raises(ValueError, match="steps must be positive"):
        slow_close_waypoints(100.0, 40.0, steps=0)


def test_summarize_target_current_uses_hold_window_samples():
    samples = [
        TelemetrySnapshot(
            t=0.0,
            gripper_pos=90,
            goal_gripper_pos=80,
            present_current=10,
            present_load=3,
            present_temperature=30,
        ),
        TelemetrySnapshot(
            t=1.0,
            gripper_pos=80,
            goal_gripper_pos=80,
            present_current=20,
            present_load=4,
            present_temperature=31,
        ),
        TelemetrySnapshot(
            t=2.0,
            gripper_pos=80,
            goal_gripper_pos=80,
            present_current=30,
            present_load=5,
            present_temperature=32,
        ),
    ]
    summary = summarize_target_current(samples)
    assert summary["mean_current"] == 20.0
    assert summary["max_current"] == 30.0
    assert summary["mean_load"] == 4.0


def test_serialize_telemetry_snapshot_preserves_raw_fields():
    sample = TelemetrySnapshot(
        t=0.25,
        gripper_pos=81.5,
        goal_gripper_pos=80.0,
        present_current=22,
        present_load=4,
        present_temperature=31,
    )
    assert serialize_telemetry_snapshot(sample, target=80.0, sample_index=3) == {
        "target": 80.0,
        "sample_index": 3,
        "t": 0.25,
        "gripper_pos": 81.5,
        "goal_gripper_pos": 80.0,
        "present_current": 22,
        "present_load": 4,
        "present_temperature": 31,
    }


def test_choose_three_grip_targets_requires_separated_currents():
    records = [
        {"target": 85.0, "mean_current": 10.0},
        {"target": 70.0, "mean_current": 18.0},
        {"target": 55.0, "mean_current": 31.0},
        {"target": 40.0, "mean_current": 45.0},
    ]
    assert choose_three_grip_targets(records, min_current_gap=10.0) == {
        "low": 70.0,
        "med": 55.0,
        "high": 40.0,
    }


def test_choose_three_grip_targets_fails_when_current_gaps_are_small():
    records = [
        {"target": 90.0, "mean_current": 10.0},
        {"target": 80.0, "mean_current": 12.0},
        {"target": 70.0, "mean_current": 13.0},
    ]
    with pytest.raises(ValueError, match="could not find three separated grip targets"):
        choose_three_grip_targets(records, min_current_gap=10.0)
