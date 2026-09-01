import pytest

from carton_fixed_grip_trials import (
    STATIC_CLOSURE_TARGETS,
    build_parser,
    build_trial_plan,
    build_trial_segments,
    compute_lift_pose,
    interpolate_pose,
    parse_outcome_annotation,
    parse_static_annotation,
    parse_targets,
    resolve_protocol_defaults,
    summarize_results,
    summarize_static_results,
)


def test_trial_plan_is_randomized_in_complete_blocks():
    targets = (28.0, 27.0, 26.0, 25.0, 24.0)
    plan = build_trial_plan(targets, repeats=5, seed=42)

    assert len(plan) == 25
    assert plan == build_trial_plan(targets, repeats=5, seed=42)
    for block in range(1, 6):
        block_rows = [row for row in plan if row["block"] == block]
        assert {row["target"] for row in block_rows} == set(targets)


def test_static_protocol_defaults_to_one_ordered_five_level_block():
    args = resolve_protocol_defaults(
        build_parser().parse_args(["--pose-file", "pose.json", "--static-closure"])
    )
    plan = build_trial_plan(args.targets, args.repeats, args.seed, randomize=False)

    assert args.targets == STATIC_CLOSURE_TARGETS
    assert args.repeats == 1
    assert [row["target"] for row in plan] == list(STATIC_CLOSURE_TARGETS)


def test_parse_targets_rejects_duplicates():
    with pytest.raises(Exception, match="unique"):
        parse_targets("28,27,28")


def test_interpolate_pose_keeps_all_joints():
    assert interpolate_pose({"a.pos": 0.0, "b.pos": 10.0}, {"a.pos": 10.0, "b.pos": 20.0}, 0.25) == {
        "a.pos": 2.5,
        "b.pos": 12.5,
    }


def test_compute_lift_pose_requests_vertical_translation_and_preserves_gripper():
    class FakeKinematics:
        def forward_kinematics(self, q):
            transform = __import__("numpy").eye(4)
            transform[:3, 3] = q[:3]
            return transform

        def inverse_kinematics(self, q, desired, **_kwargs):
            result = q.copy()
            result[:3] = desired[:3, 3]
            return result

    motors = ["x", "y", "z", "wrist", "roll", "gripper"]
    pickup = {f"{motor}.pos": float(index) for index, motor in enumerate(motors)}
    lift, pickup_xyz, lift_xyz = compute_lift_pose(FakeKinematics(), motors, pickup, 0.10)

    assert lift["z.pos"] == pytest.approx(2.10)
    assert lift["gripper.pos"] == pickup["gripper.pos"]
    assert lift_xyz[2] - pickup_xyz[2] == pytest.approx(0.10)


def test_parse_outcome_annotation_supports_compact_operator_entry():
    assert parse_outcome_annotation("s,1,0,,small tilt") == {
        "outcome": "success",
        "tilt_grade": 1,
        "crease_grade": 0,
        "first_slip_s_from_lift": "",
        "notes": "small tilt",
    }
    assert parse_outcome_annotation("")["outcome"] == "unlabeled"


def test_parse_static_annotation_records_residual_grade_and_damage():
    assert parse_static_annotation("1,n,mild mark") == {
        "residual_deformation_grade": 1,
        "functional_damage": "no",
        "notes": "mild mark",
    }
    assert parse_static_annotation("")["residual_deformation_grade"] == ""
    with pytest.raises(ValueError, match="functional damage"):
        parse_static_annotation("1,maybe")


def test_static_segments_never_use_lift_pose_and_include_recovery():
    pickup_open = {"arm.pos": 1.0, "gripper.pos": 100.0}
    pickup_closed = {"arm.pos": 1.0, "gripper.pos": 26.0}
    lift_closed = {"arm.pos": 9.0, "gripper.pos": 26.0}
    segments = build_trial_segments(
        static_closure=True,
        pickup_open=pickup_open,
        pickup_closed=pickup_closed,
        lift_closed=lift_closed,
        close_s=2.0,
        settle_s=0.5,
        lift_s=2.0,
        hold_s=3.0,
        lower_s=2.0,
        release_s=1.0,
        recovery_s=10.0,
    )

    assert [phase for phase, *_ in segments] == ["close", "hold", "release", "recovery"]
    assert all(start is not lift_closed and end is not lift_closed for _, start, end, _ in segments)
    assert segments[-1][-1] == pytest.approx(10.0)


def test_summary_checks_success_monotonicity_in_closing_direction():
    plan = [
        {"trial_index": 1, "block": 1, "target": 28.0},
        {"trial_index": 2, "block": 1, "target": 27.0},
        {"trial_index": 3, "block": 1, "target": 26.0},
    ]
    outcomes = [
        {**plan[0], "outcome": "drop"},
        {**plan[1], "outcome": "success"},
        {**plan[2], "outcome": "success"},
    ]
    telemetry = [
        {
            **row,
            "phase": "hold",
            "present_current": 10 + index,
            "present_load": 100 + index,
            "absolute_position_lag": 2 + index,
        }
        for index, row in enumerate(plan)
    ]

    summary = summarize_results(plan, outcomes, telemetry)

    assert summary["success_rate_non_decreasing_with_closure"] is True
    assert summary["per_target"]["28"]["success_rate"] == 0.0
    assert summary["per_target"]["27"]["success_rate"] == 1.0


def test_static_summary_checks_residual_deformation_monotonicity():
    plan = [
        {"trial_index": 1, "block": 1, "target": 32.0},
        {"trial_index": 2, "block": 1, "target": 26.0},
        {"trial_index": 3, "block": 1, "target": 20.0},
    ]
    outcomes = [
        {**plan[0], "residual_deformation_grade": 0, "functional_damage": "no"},
        {**plan[1], "residual_deformation_grade": 1, "functional_damage": "no"},
        {**plan[2], "residual_deformation_grade": 2, "functional_damage": "yes"},
    ]
    telemetry = [
        {
            **row,
            "phase": "hold",
            "present_current": 10 + index,
            "present_load": 100 + index,
            "absolute_position_lag": 2 + index,
        }
        for index, row in enumerate(plan)
    ]

    summary = summarize_static_results(plan, outcomes, telemetry)

    assert summary["mode"] == "static_closure"
    assert summary["residual_deformation_non_decreasing_with_closure"] is True
    assert summary["per_target"]["20"]["functional_damage_count"] == 1
